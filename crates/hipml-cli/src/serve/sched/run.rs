//! The event loop. A child of `sched` so it can see the state structs'
//! private fields without making them part of the module's API.
//!
//! Every fallible step is scoped (WP51). A failure inside one job's prefill,
//! speculation, adoption or per-slot decode fails that job, frees its slot and
//! leaves the engine serving; only a shared weight pass fails a whole batch,
//! and only a reset that itself fails drains every stream and exits. See
//! `fail.rs` for the three scopes.

use super::*;
use super::{fail, warm};
use std::collections::VecDeque;
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::time::Duration;

use hipml_model::gpu::{BatchEngine, GpuEngine, KvMode, MtpGpu};
use hipml_model::Model;

/// A test hook for the isolation path, off unless set.
///
/// `HIPML_FAULT_SLOT=n` makes slot `n`'s decode step fail from
/// `HIPML_FAULT_STEP` (default 4) on, through the same per-slot check that
/// rejects an out-of-vocabulary id. So the probe exercises the real recovery
/// path — fail the job, free the slot, keep the batch — and not a special
/// case compiled in beside it.
struct Faults {
    slot: Option<usize>,
    after: u64,
    steps: u64,
}

impl Faults {
    fn new() -> Faults {
        Faults {
            slot: std::env::var("HIPML_FAULT_SLOT")
                .ok()
                .and_then(|v| v.parse().ok()),
            after: std::env::var("HIPML_FAULT_STEP")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4),
            steps: 0,
        }
    }

    fn hit(&self, slot: usize) -> Option<String> {
        (self.slot == Some(slot) && self.steps >= self.after)
            .then(|| "injected fault (HIPML_FAULT_SLOT)".to_string())
    }
}

pub fn run(
    model: &Model,
    rx: Receiver<Job>,
    cfg: SchedCfg,
    metrics: Arc<Metrics>,
) -> Result<(), String> {
    let mut engine = GpuEngine::new(model, cfg.ctx, KvMode::Q8)?;
    engine.enable_prefix_cache(cfg.prefix_entries, cfg.prefix_bytes);
    let mut mtp = if cfg.mtp {
        Some(MtpGpu::new(model, cfg.ctx)?)
    } else {
        None
    };
    let mut batch = BatchEngine::new(model, cfg.ctx, KvMode::Q8, cfg.slots)?;
    // Resolve every kernel and touch every weight page before the first
    // request, so TTFT is not charged for graph construction and first touch.
    warm::warm_up(&mut engine, &mut batch, mtp.as_mut(), &cfg, &metrics)?;

    let mut queue: VecDeque<Job> = VecDeque::new();
    let mut slots: Vec<Option<Active>> = (0..batch.slots).map(|_| None).collect();
    let mut prefill: Option<Prefill> = None;
    let mut spec: Option<Spec> = None;
    let mut faults = Faults::new();
    // Batch resets with no successful step between them: a device that cannot
    // finish one weight pass is not coming back on its own.
    let mut reset_streak = 0u32;

    // Device-level failure: drain every open stream with an error, then let the
    // process exit so a supervisor can restart it.
    macro_rules! fatal {
        ($why:expr) => {{
            let why = $why;
            fail::drain(
                &metrics,
                &mut queue,
                &mut slots,
                prefill.take(),
                spec.take(),
                &why,
            );
            return Err(why);
        }};
    }

    loop {
        while let Ok(j) = rx.try_recv() {
            metrics.update(|s| s.requests_total += 1);
            queue.push_back(j);
        }
        // A request whose client left while it waited never reaches the GPU.
        let before = queue.len();
        queue.retain(|j| j.cancelled().is_none());
        let dropped = (before - queue.len()) as u64;
        let live = slots.iter().filter(|s| s.is_some()).count()
            + spec.is_some() as usize
            + prefill.is_some() as usize;
        metrics.update(|s| {
            s.requests_cancelled += dropped;
            s.active_slots = live as u64;
            s.queue_depth = queue.len() as u64;
        });
        let busy = prefill.is_some()
            || spec.is_some()
            || slots.iter().any(|s| s.is_some());
        if !busy && queue.is_empty() {
            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(j) => {
                    metrics.update(|s| s.requests_total += 1);
                    queue.push_back(j);
                }
                Err(_) => continue,
            }
        }

        // WP47: gather prefix-hit tails for one multi-row prefill. Each tail's
        // restored state is copied into a quiet slot; one block then advances
        // every tail's suffix in a single weight pass (instead of one tiled
        // prefill per tail, each streaming the whole model).
        if prefill.is_none() && spec.is_none() {
            let row_max = batch.row_block_max();
            let min_tail = batch.row_min_tail();
            let max_tail = batch.row_tail_max();
            let gather_ms: u64 = std::env::var("HIPML_ROW_GATHER_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4);
            let mut gathered: Vec<RowTail> = Vec::new();
            let mut rows_used = 0usize;
            let mut waited = false;
            let mut flush = false;
            loop {
                let slots_left = batch.available_slots().saturating_sub(gathered.len());
                if slots_left == 0 || row_max.saturating_sub(rows_used) < min_tail {
                    flush = !gathered.is_empty();
                    break;
                }
                if queue.is_empty() {
                    if gathered.is_empty() || waited || gather_ms == 0 {
                        flush = !gathered.is_empty();
                        break;
                    }
                    waited = true;
                    match rx.recv_timeout(Duration::from_millis(gather_ms)) {
                        Ok(j) => {
                            metrics.update(|s| s.requests_total += 1);
                            queue.push_back(j);
                            continue;
                        }
                        Err(_) => {
                            flush = !gathered.is_empty();
                            break;
                        }
                    }
                }
                let job = queue.pop_front().expect("checked");
                let plen = job.prompt.len();
                let alone = slots.iter().all(|s| s.is_none())
                    && queue.is_empty()
                    && gathered.is_empty();
                let mode = if alone && cfg.mtp { Mode::Spec } else { Mode::Batch };
                match begin(&mut engine, job, &cfg, mode, &metrics) {
                    Ok(pf) => {
                        let tail = pf.job.prompt.len().saturating_sub(pf.done);
                        let qualifies = pf.mode == Mode::Batch
                            && pf.done > 0
                            && tail >= min_tail
                            && tail <= max_tail
                            && tail <= row_max - rows_used
                            && pf.cuts.is_empty();
                        if qualifies {
                            let slot = batch.next_slot() + gathered.len();
                            match batch.adopt_quiet(&engine, slot, pf.st.pos) {
                                Ok(()) => {
                                    let ids = pf.job.prompt[pf.done..].to_vec();
                                    metrics.update(|s| s.prompt_tokens += plen as u64);
                                    rows_used += tail;
                                    gathered.push(RowTail {
                                        job: pf.job,
                                        slot,
                                        ids,
                                        hit_tokens: pf.done,
                                    });
                                    continue;
                                }
                                Err(e) => {
                                    // The failing slot is always the last
                                    // gathered one, so the rest stay dense.
                                    fail::job(&mut engine, &metrics, &pf.job, "row adopt", &e);
                                    continue;
                                }
                            }
                        }
                        if gathered.is_empty() {
                            metrics.update(|s| s.prompt_tokens += plen as u64);
                            prefill = Some(pf);
                        } else {
                            // Let the queued request keep its place; the block
                            // runs first so it is not delayed behind it.
                            queue.push_front(pf.job);
                            flush = true;
                        }
                        break;
                    }
                    Err((job, e)) => fail::job(&mut engine, &metrics, &job, "admit", &e),
                }
            }

            if flush && !gathered.is_empty() {
                let tails: Vec<(usize, &[u32])> = gathered
                    .iter()
                    .map(|g| (g.slot, g.ids.as_slice()))
                    .collect();
                match batch.prefill_rows(&tails) {
                    Ok(ids) => {
                        let ms = batch.last_row_block().1;
                        let tails_n = gathered.len();
                        metrics.update(|s| {
                            s.tail_row_blocks += 1;
                            s.tail_rows += rows_used as u64;
                            s.tail_rows_ms += ms;
                            s.batch_requests += tails_n as u64;
                            s.generated_tokens += tails_n as u64;
                        });
                        // (slot, fault, finish reason)
                        let mut out: Vec<(usize, Option<String>, &'static str)> = Vec::new();
                        let rows: Vec<RowTail> = gathered.drain(..).collect();
                        let mut blocked: Option<String> = None;
                        for (k, g) in rows.into_iter().enumerate() {
                            if let Some(e) = blocked.take() {
                                fail::job(&mut engine, &metrics, &g.job, "row activate", &e);
                                blocked = Some(e);
                                continue;
                            }
                            if let Err(e) = batch.activate_slot(g.slot) {
                                // The active set is dense, so a failure here
                                // leaves the later quiet slots unactivatable.
                                fail::job(&mut engine, &metrics, &g.job, "row activate", &e);
                                blocked = Some(e);
                                continue;
                            }
                            let id = ids[k];
                            let _ = g.job.tx.send(Event::Tokens(vec![id]));
                            let finish = if g.job.stop_ids.contains(&id) {
                                "stop"
                            } else if g.job.max_tokens <= 1 {
                                "length"
                            } else {
                                ""
                            };
                            if !finish.is_empty() {
                                out.push((g.slot, None, finish));
                            }
                            slots[g.slot] = Some(Active {
                                job: g.job,
                                feed: id,
                                emitted: 1,
                                prefill_ms: ms,
                                hit_tokens: g.hit_tokens,
                            });
                        }
                        // Release from the highest index down: a lower release
                        // swaps the last active slot in, and every unprocessed
                        // entry is below it, so nothing moves under our feet.
                        out.sort_unstable_by(|a, b| b.0.cmp(&a.0));
                        for (slot, fault, reason) in out {
                            if let Some(why) = fault {
                                if let Some(a) = slots[slot].take() {
                                    fail::slot(&mut engine, &metrics, &a.job, &why);
                                }
                            } else {
                                fail::take_active(&mut slots, slot, reason, &metrics);
                            }
                            if let Err(e) = fail::release(&mut batch, &mut slots, slot) {
                                let why = format!("slot release: {e}");
                                if let Err(e2) =
                                    fail::batch(&mut batch, &mut engine, &mut slots, &metrics, &why)
                                {
                                    fatal!(format!("{why}; reset failed: {e2}"));
                                }
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        for g in gathered.drain(..) {
                            fail::job(&mut engine, &metrics, &g.job, "row prefill", &e);
                        }
                    }
                }
            }
        }

        // Start one prefill when the engine is free. A request is only allowed
        // to become speculative when it is alone and nothing is queued.
        let alone_next = slots.iter().all(|s| s.is_none()) && queue.len() == 1;
        let can_admit = (alone_next && cfg.mtp) || batch.available_slots() > 0;
        if prefill.is_none() && spec.is_none() && !queue.is_empty() && can_admit {
            let job = queue.pop_front().expect("checked");
            let alone = slots.iter().all(|s| s.is_none()) && queue.is_empty();
            let mode = if alone && cfg.mtp { Mode::Spec } else { Mode::Batch };
            let plen = job.prompt.len();
            match begin(&mut engine, job, &cfg, mode, &metrics) {
                Ok(pf) => {
                    metrics.update(|s| s.prompt_tokens += plen as u64);
                    prefill = Some(pf);
                }
                Err((job, e)) => fail::job(&mut engine, &metrics, &job, "admit", &e),
            }
        }

        // One prefill tile, then the batch keeps going on the next line.
        if let Some(pf) = prefill.as_mut() {
            if pf.job.cancelled().is_some() {
                // The client left mid-prefill: a 100K prompt would otherwise
                // hold the engine for ~2 minutes for nobody.
                prefill = None;
                metrics.update(|s| s.requests_cancelled += 1);
                continue;
            }
            let batch_active = slots.iter().any(|s| s.is_some());
            if let Err(e) = prefill_step(pf, &mut engine, &cfg, batch_active, &metrics) {
                let pf = prefill.take().expect("checked");
                fail::job(&mut engine, &metrics, &pf.job, "prefill", &e);
                continue;
            }
            if pf.job.prompt.len() == pf.done {
                let mut pf = prefill.take().expect("checked");
                let job = pf.job;
                let first = pf.first.expect("full prompt has a first token");
                // A request that joined while this prompt was being prefilled
                // must not wait for a whole MTP generation.
                let alone = slots.iter().all(|s| s.is_none()) && queue.is_empty();
                let mode = if pf.mode == Mode::Spec && alone && cfg.mtp {
                    Mode::Spec
                } else {
                    Mode::Batch
                };
                if job.stop_ids.contains(&first) {
                    let _ = job.tx.send(Event::Tokens(vec![first]));
                    send_done(&job.tx, &job, "stop", 1, pf.prefill_ms, pf.hit_tokens);
                    metrics.update(|s| s.requests_done += 1);
                    continue;
                }
                if job.max_tokens <= 1 {
                    let _ = job.tx.send(Event::Tokens(vec![first]));
                    send_done(&job.tx, &job, "length", 1, pf.prefill_ms, pf.hit_tokens);
                    metrics.update(|s| s.requests_done += 1);
                    continue;
                }
                match mode {
                    Mode::Spec => {
                        let capture = pf.capture.take().expect("spec prefill captures");
                        let seeded = match mtp.as_mut() {
                            Some(d) => engine.spec_seed(d, &job.prompt, &capture),
                            None => Err("MTP drafter is not loaded".to_string()),
                        };
                        if let Err(e) = seeded {
                            fail::job(&mut engine, &metrics, &job, "spec seed", &e);
                            continue;
                        }
                        let snaps = match engine.alloc_snapshots(cfg.drafts) {
                            Ok(s) => s,
                            Err(e) => {
                                fail::job(&mut engine, &metrics, &job, "snapshots", &e);
                                continue;
                            }
                        };
                        let _ = job.tx.send(Event::Tokens(vec![first]));
                        metrics.update(|s| {
                            s.mtp_requests += 1;
                            s.generated_tokens += 1;
                        });
                        spec = Some(Spec {
                            job,
                            st: pf.st,
                            snaps,
                            next: first,
                            stats: SpecStats::default(),
                            emitted: 1,
                            prefill_ms: pf.prefill_ms,
                            hit_tokens: pf.hit_tokens,
                        });
                    }
                    Mode::Batch => match batch.adopt(&engine, pf.st.pos) {
                        Ok(slot) => {
                            let _ = job.tx.send(Event::Tokens(vec![first]));
                            metrics.update(|s| {
                                s.batch_requests += 1;
                                s.generated_tokens += 1;
                            });
                            slots[slot] = Some(Active {
                                job,
                                feed: first,
                                emitted: 1,
                                prefill_ms: pf.prefill_ms,
                                hit_tokens: pf.hit_tokens,
                            });
                        }
                        Err(e) => fail::job(&mut engine, &metrics, &job, "adopt", &e),
                    },
                }
            }
        }

        // Lone speculative step. Arrivals abort it and hand the state to the
        // batch engine; the emitted tokens stay exactly greedy.
        if spec.is_some() {
            let joined = !queue.is_empty() || slots.iter().any(|s| s.is_some());
            if joined {
                let sp = spec.take().expect("checked");
                match batch.adopt(&engine, sp.st.pos) {
                    Ok(slot) => {
                        slots[slot] = Some(Active {
                            job: sp.job,
                            feed: sp.next,
                            emitted: sp.emitted,
                            prefill_ms: sp.prefill_ms,
                            hit_tokens: sp.hit_tokens,
                        });
                    }
                    Err(e) => fail::job(&mut engine, &metrics, &sp.job, "spec join", &e),
                }
            } else if let Some(reason) = spec.as_ref().and_then(|sp| sp.job.cancelled()) {
                let sp = spec.take().expect("checked");
                send_done(&sp.job.tx, &sp.job, reason, sp.emitted, sp.prefill_ms, sp.hit_tokens);
                finished(&metrics, reason);
            } else {
                let step = match mtp.as_mut() {
                    Some(d) => {
                        let sp = spec.as_mut().expect("checked");
                        engine.spec_step(
                            d,
                            &mut sp.st,
                            &mut sp.snaps,
                            sp.next,
                            sp.job.drafts,
                            &sp.job.stop_ids,
                            &mut sp.stats,
                        )
                    }
                    None => Err("MTP drafter is not loaded".to_string()),
                };
                let step = match step {
                    Ok(step) => step,
                    Err(e) => {
                        let sp = spec.take().expect("checked");
                        fail::job(&mut engine, &metrics, &sp.job, "spec step", &e);
                        continue;
                    }
                };
                let sp = spec.as_mut().expect("checked");
                metrics.update(|s| {
                    s.spec_steps += 1;
                    s.spec_drafted += sp.job.drafts as u64;
                    s.spec_accepted += step.accepted as u64;
                });
                // The reason is the first of these the emitted ids reach: a
                // step can carry a stop id past `max_tokens`, and that request
                // finished on length, not on the stop.
                let mut finish = None;
                for id in step.ids {
                    if sp.job.tx.send(Event::Tokens(vec![id])).is_err() {
                        finish = Some("cancelled");
                        break;
                    }
                    sp.next = id;
                    sp.emitted += 1;
                    metrics.update(|s| s.generated_tokens += 1);
                    if sp.job.stop_ids.contains(&id) {
                        finish = Some("stop");
                        break;
                    }
                    if sp.emitted >= sp.job.max_tokens {
                        finish = Some("length");
                        break;
                    }
                }
                if finish.is_none() && step.stopped {
                    finish = Some("stop");
                }
                if let Some(finish) = finish {
                    let sp = spec.take().expect("checked");
                    send_done(
                        &sp.job.tx,
                        &sp.job,
                        finish,
                        sp.emitted,
                        sp.prefill_ms,
                        sp.hit_tokens,
                    );
                    finished(&metrics, finish);
                }
            }
        }

        // One batched step for every active slot.
        let order: Vec<usize> = batch.active_slots().to_vec();
        if !order.is_empty() {
            faults.steps += 1;
            let toks: Vec<u32> = order
                .iter()
                .map(|&s| slots[s].as_ref().expect("active slot has a job").feed)
                .collect();
            let ids = match batch.forward(&toks) {
                Ok(ids) => {
                    reset_streak = 0;
                    ids
                }
                Err(e) => {
                    // One weight pass covers every slot, so a launch that fails
                    // cannot be blamed on one of them: fail the batch, reset
                    // the engine, keep serving. Repeated resets are fatal.
                    let why = format!("batch step: {e}");
                    if let Err(e2) =
                        fail::batch(&mut batch, &mut engine, &mut slots, &metrics, &why)
                    {
                        fatal!(format!("{why}; reset failed: {e2}"));
                    }
                    reset_streak += 1;
                    if reset_streak >= 3 {
                        fatal!(format!("{reset_streak} batch resets with no step in between"));
                    }
                    continue;
                }
            };
            metrics.update(|s| {
                s.decode_steps += 1;
                s.batch_steps += 1;
                // B>=4 is where the serving path switches to the WMMA tile.
                if cfg.skinny && order.len() >= 4 {
                    s.tile_steps += 1;
                }
            });
            let n_vocab = batch.cfg.n_vocab;
            // (slot, fault, finish reason). A slot's own step is validated
            // here: a corrupt id is that slot's problem, not the batch's.
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
            // Release from the highest index down. A lower release swaps the
            // last active slot into the freed index, and all unprocessed
            // retirees are below it, so no retiring entry ever moves.
            out.sort_unstable_by(|a, b| b.0.cmp(&a.0));
            for (slot, fault, reason) in out {
                if let Some(why) = fault {
                    if let Some(a) = slots[slot].take() {
                        fail::slot(&mut engine, &metrics, &a.job, &why);
                    }
                } else {
                    fail::take_active(&mut slots, slot, reason, &metrics);
                }
                if let Err(e) = fail::release(&mut batch, &mut slots, slot) {
                    let why = format!("slot release: {e}");
                    if let Err(e2) = fail::batch(&mut batch, &mut engine, &mut slots, &metrics, &why) {
                        fatal!(format!("{why}; reset failed: {e2}"));
                    }
                    break;
                }
            }
        }
    }
}
