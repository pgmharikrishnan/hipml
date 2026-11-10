// SPDX-License-Identifier: Apache-2.0
//! One batched decode step: the shared weight pass, then each slot's own
//! verdict — a token, a stop, a cancel, or that slot's own failure.
//!
//! The weight pass is all-or-nothing (see `fail.rs`); everything after it is
//! per slot, and that is where a corrupt id or an injected fault is caught.
//! The tokens for the other slots in the same pass are already computed, so a
//! slot that dies here cannot change their ids.

use hipml_model::gpu::{BatchEngine, GpuEngine};

use super::*;

/// A test hook for the isolation path, off unless set.
///
/// `HIPML_FAULT_SLOT=n` makes slot `n`'s decode step fail from
/// `HIPML_FAULT_STEP` (default 4) on, through the same per-slot check that
/// rejects an out-of-vocabulary id. So the probe exercises the real recovery
/// path — fail the job, free the slot, keep the batch — and not a special case
/// compiled in beside it.
pub struct Faults {
    slot: Option<usize>,
    after: u64,
    steps: u64,
    /// Fire once. A released slot is filled by the last active slot's state,
    /// so a fault keyed on the index alone would cascade into every job that
    /// later occupied it; the point is one job's step, not the index.
    fired: bool,
}

impl Faults {
    pub fn new() -> Faults {
        Faults {
            slot: std::env::var("HIPML_FAULT_SLOT")
                .ok()
                .and_then(|v| v.parse().ok()),
            after: std::env::var("HIPML_FAULT_STEP")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4),
            steps: 0,
            fired: false,
        }
    }

    /// Count one batched step.
    pub fn tick(&mut self) {
        self.steps += 1;
    }

    fn hit(&mut self, slot: usize) -> Option<String> {
        if self.fired || self.slot != Some(slot) || self.steps < self.after {
            return None;
        }
        self.fired = true;
        Some("injected fault (HIPML_FAULT_SLOT)".to_string())
    }
}

/// Deliver `ids[i]` to `order[i]`'s client and retire the slots that finished.
///
/// Returns `Err` only when a slot could not be returned to the pool: the active
/// set and the scheduler's table would then disagree, and the caller resets the
/// batch engine instead of guessing.
pub fn deliver(
    batch: &mut BatchEngine<'_>,
    engine: &mut GpuEngine<'_>,
    slots: &mut [Option<Active>],
    order: &[usize],
    ids: &[u32],
    metrics: &Metrics,
    faults: &mut Faults,
) -> Result<(), String> {
    let n_vocab = batch.cfg.n_vocab;
    // (slot, fault, finish reason)
    let mut out: Vec<(usize, Option<String>, &'static str)> = Vec::new();
    for (i, &slot) in order.iter().enumerate() {
        let a = slots[slot].as_mut().expect("active slot has a job");
        let id = ids[i];
        if let Some(reason) = a.job.cancelled() {
            out.push((slot, None, reason));
            continue;
        }
        let fault = faults
            .hit(slot)
            .or_else(|| (id as usize >= n_vocab).then(|| format!("id {id} outside the vocabulary")));
        if let Some(why) = fault {
            out.push((slot, Some(why), ""));
            continue;
        }
        if a.job.tx.send(Event::Tokens(vec![id])).is_err() {
            out.push((slot, None, "cancelled"));
            continue;
        }
        a.feed = id;
        a.emitted += 1;
        metrics.update(|s| s.generated_tokens += 1);
        if a.job.stop_ids.contains(&id) {
            out.push((slot, None, "stop"));
        } else if a.emitted >= a.job.max_tokens {
            out.push((slot, None, "length"));
        }
    }
    // Release from the highest index down. A lower release swaps the last
    // active slot into the freed index, and all unprocessed retirees are below
    // it, so no retiring entry ever moves.
    out.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    for (slot, fault, reason) in out {
        if let Some(why) = fault {
            if let Some(a) = slots[slot].take() {
                fail::slot(engine, metrics, &a.job, &why);
            }
        } else {
            fail::take_active(slots, slot, reason, metrics);
        }
        fail::release(batch, slots, slot)?;
    }
    Ok(())
}
