// SPDX-License-Identifier: Apache-2.0
//! What happens when one job dies.
//!
//! Before WP51 every `?` in the scheduler loop returned to `sched::run` and the
//! process exited, so one bad request — an allocation failure, a transient HIP
//! error, a slot whose state went bad — dropped every in-flight stream. Each
//! failure now has a scope:
//!
//! * **a job's own step** (admit, prefill tile, spec seed/step, adopt,
//!   activate, a corrupt id): fail that job, free its slot, invalidate the
//!   prefix entries it captured, keep serving. The other slots are untouched,
//!   so their ids are exactly what a no-fault run would have produced;
//! * **the shared weight pass**: no slot can be blamed for a launch that
//!   covers every slot, so the batch's jobs fail together and the batch engine
//!   is reset (zeroed SSM state, empty slot set) before the next admission.
//!   The process keeps running;
//! * **the device**: when even that reset fails, or three resets happen with no
//!   successful step between them, every stream is drained with an error and
//!   `run` returns, so a supervisor restarts the process. Nothing is left
//!   mid-write.

use std::sync::mpsc::Sender;

use hipml_model::gpu::{BatchEngine, GpuEngine};

use super::*;

/// Fail one job. Its client gets an `Event::Error`, its own prefix-cache
/// entries go (a job that died mid-prefill cannot vouch for its snapshots),
/// and the counters record a job error rather than a request that completed.
pub fn job(engine: &mut GpuEngine<'_>, metrics: &Metrics, job: &Job, what: &str, e: &str) {
    eprintln!("serve: {what}: {e} (job failed; engine keeps serving)");
    let gone = engine.prefix_invalidate(&job.prompt);
    let _ = job.tx.send(Event::Error(format!("{what}: {e}")));
    metrics.update(|s| {
        s.requests_failed += 1;
        s.job_errors += 1;
        s.prefix_invalidated += gone as u64;
    });
}

/// Fail one job because its own decode step produced nothing usable. Same as
/// [`job`] plus the slot-fault counter, so a poisoned slot is visible on
/// `/metrics` without reading the log.
pub fn slot(engine: &mut GpuEngine<'_>, metrics: &Metrics, job: &Job, why: &str) {
    eprintln!("serve: slot fault: {why} (slot's job failed; other slots continue)");
    let gone = engine.prefix_invalidate(&job.prompt);
    let _ = job.tx.send(Event::Error(format!("decode step: {why}")));
    metrics.update(|s| {
        s.requests_failed += 1;
        s.job_errors += 1;
        s.slot_faults += 1;
        s.prefix_invalidated += gone as u64;
    });
}

/// Return one slot to the pool, moving the last active slot's bookkeeping into
/// the freed index when the engine has to swap device state down.
pub fn release(
    batch: &mut BatchEngine<'_>,
    slots: &mut [Option<Active>],
    slot: usize,
) -> Result<(), String> {
    if let Some(from) = batch.release(slot)? {
        slots[slot] = slots[from].take();
    }
    Ok(())
}

pub fn take_active(slots: &mut [Option<Active>], slot: usize, finish: &str, metrics: &Metrics) {
    if let Some(a) = slots[slot].take() {
        send_done(&a.job.tx, &a.job, finish, a.emitted, a.prefill_ms, a.hit_tokens);
        finished(metrics, finish);
    }
}

/// The shared weight pass failed: fail every job in the batch, then reset the
/// batch engine so the next admission starts from a known state.
///
/// Returns the error only when the reset itself fails; the caller then treats
/// it as a device-level failure and drains everything.
pub fn batch(
    batch: &mut BatchEngine<'_>,
    engine: &mut GpuEngine<'_>,
    slots: &mut [Option<Active>],
    metrics: &Metrics,
    why: &str,
) -> Result<u64, String> {
    let mut failed = 0u64;
    for s in slots.iter_mut() {
        if let Some(a) = s.take() {
            let gone = engine.prefix_invalidate(&a.job.prompt);
            let _ = a.job.tx.send(Event::Error(format!("batch step: {why}")));
            failed += 1;
            metrics.update(|s| s.prefix_invalidated += gone as u64);
        }
    }
    batch.reset_state()?;
    batch.activate(0)?;
    metrics.update(|s| {
        s.requests_failed += failed;
        s.job_errors += failed;
        s.batch_resets += 1;
    });
    eprintln!("serve: batch step failed: {why}; reset after {failed} job(s), engine keeps serving");
    Ok(failed)
}

/// Device-level failure: every stream that is still open gets an error event
/// and closes, and then the process may exit.
pub fn drain(
    metrics: &Metrics,
    queue: &mut std::collections::VecDeque<Job>,
    slots: &mut [Option<Active>],
    prefill: Option<Prefill>,
    spec: Option<Spec>,
    why: &str,
) {
    let mut n = 0u64;
    let mut send = |tx: &Sender<Event>| {
        let _ = tx.send(Event::Error(why.to_string()));
        n += 1;
    };
    for j in queue.drain(..) {
        send(&j.tx);
    }
    for s in slots.iter_mut() {
        if let Some(a) = s.take() {
            send(&a.job.tx);
        }
    }
    if let Some(p) = prefill {
        send(&p.job.tx);
    }
    if let Some(s) = spec {
        send(&s.job.tx);
    }
    metrics.update(|s| s.requests_failed += n);
    eprintln!("serve: drained {n} stream(s) with an error: {why}");
}
