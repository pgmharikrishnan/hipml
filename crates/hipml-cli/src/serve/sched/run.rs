//! The event loop. A child of `sched` so it can see the state structs'
//! private fields without making them part of the module's API.

use super::*;
use std::collections::VecDeque;
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::time::Duration;

use hipml_model::gpu::{BatchEngine, GpuEngine, KvMode, MtpGpu};
use hipml_model::Model;

fn take_active(
    slots: &mut [Option<Active>],
    slot: usize,
    finish: &str,
    metrics: &Metrics,
) {
    if let Some(a) = slots[slot].take() {
        send_done(&a.job.tx, &a.job, finish, a.emitted, a.prefill_ms, a.hit_tokens);
        metrics.update(|s| s.requests_done += 1);
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
    // Resolve every kernel and allocate the chunk scratch before the first
    // request, so TTFT is not charged for graph construction.
    {
        let warm = vec![0u32, 1u32];
        let mut st = engine.new_state();
        let _ = engine.prefill_tile(&warm, &mut st, cfg.chunk, false);
    }

    let mut queue: VecDeque<Job> = VecDeque::new();
    let mut slots: Vec<Option<Active>> = (0..batch.slots).map(|_| None).collect();
    let mut prefill: Option<Prefill> = None;
    let mut spec: Option<Spec> = None;

    loop {
        while let Ok(j) = rx.try_recv() {
            metrics.update(|s| s.requests_total += 1);
            queue.push_back(j);
        }
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
                let tx = job.tx.clone();
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
                                    eprintln!("serve: row adopt: {e}");
                                    let _ = tx.send(Event::Error(e));
                                    continue;
                                }
                            }
                        }
                        if gathered.is_empty() {
                            prefill = Some(pf);
                        } else {
                            // Let the queued request keep its place; the block
                            // runs first so it is not delayed behind it.
                            queue.push_front(pf.job);
                            flush = true;
                        }
                        break;
                    }
                    Err(e) => {
                        eprintln!("serve: admit: {e}");
                        let _ = tx.send(Event::Error(e));
                    }
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
                        let mut finish: Vec<(usize, &'static str)> = Vec::new();
                        for (k, g) in gathered.drain(..).enumerate() {
                            let id = ids[k];
                            batch.activate_slot(g.slot)?;
                            let _ = g.job.tx.send(Event::Tokens(vec![id]));
                            if g.job.stop_ids.contains(&id) {
                                finish.push((g.slot, "stop"));
                            } else if g.job.max_tokens <= 1 {
                                finish.push((g.slot, "length"));
                            }
                            slots[g.slot] = Some(Active {
                                job: g.job,
                                feed: id,
                                emitted: 1,
                                prefill_ms: ms,
                                hit_tokens: g.hit_tokens,
                            });
                        }
                        finish.sort_unstable_by(|a, b| b.0.cmp(&a.0));
                        for (slot, reason) in finish {
                            take_active(&mut slots, slot, reason, &metrics);
                            if let Some(from) = batch.release(slot)? {
                                slots[slot] = slots[from].take();
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("serve: row prefill: {e}");
                        for g in gathered.drain(..) {
                            let _ = g.job.tx.send(Event::Error(e.clone()));
                        }
                    }
                }
            }
        }

        // Start one prefill when the engine is free. A request is only allowed
        // to become speculative when it is alone and nothing is queued.
        let alone_next = slots.iter().all(|s| s.is_none()) && queue.len() == 1;
        let can_admit =
            (alone_next && cfg.mtp) || batch.available_slots() > 0;
        if prefill.is_none() && spec.is_none() && !queue.is_empty() && can_admit {
            let job = queue.pop_front().expect("checked");
            let alone = slots.iter().all(|s| s.is_none()) && queue.is_empty();
            let mode = if alone && cfg.mtp { Mode::Spec } else { Mode::Batch };
            let tx = job.tx.clone();
            let plen = job.prompt.len();
            match begin(&mut engine, job, &cfg, mode, &metrics) {
                Ok(pf) => {
                    metrics.update(|s| s.prompt_tokens += plen as u64);
                    prefill = Some(pf);
                }
                Err(e) => {
                    eprintln!("serve: admit: {e}");
                    let _ = tx.send(Event::Error(e));
                }
            }
        }

        // One prefill tile, then the batch keeps going on the next line.
        if let Some(pf) = prefill.as_mut() {
            let batch_active = slots.iter().any(|s| s.is_some());
            if let Err(e) = prefill_step(pf, &mut engine, &cfg, batch_active, &metrics) {
                eprintln!("serve: prefill: {e}");
                prefill = None;
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
                        let drafter = mtp.as_mut().ok_or("MTP drafter is not loaded")?;
                        engine.spec_seed(drafter, &job.prompt, &capture)?;
                        let snaps = engine.alloc_snapshots(cfg.drafts)?;
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
                    Mode::Batch => {
                        let slot = batch.adopt(&engine, pf.st.pos)?;
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
                }
            }
        }

        // Lone speculative step. Arrivals abort it and hand the state to the
        // batch engine; the emitted tokens stay exactly greedy.
        if spec.is_some() {
            let joined = !queue.is_empty() || slots.iter().any(|s| s.is_some());
            if joined {
                let sp = spec.take().expect("checked");
                let slot = batch.adopt(&engine, sp.st.pos)?;
                slots[slot] = Some(Active {
                    job: sp.job,
                    feed: sp.next,
                    emitted: sp.emitted,
                    prefill_ms: sp.prefill_ms,
                    hit_tokens: sp.hit_tokens,
                });
            } else {
                let sp = spec.as_mut().expect("checked");
                let drafter = mtp.as_mut().ok_or("MTP drafter is not loaded")?;
                let step = engine.spec_step(
                    drafter,
                    &mut sp.st,
                    &mut sp.snaps,
                    sp.next,
                    sp.job.drafts,
                    &sp.job.stop_ids,
                    &mut sp.stats,
                )?;
                metrics.update(|s| {
                    s.spec_steps += 1;
                    s.spec_drafted += sp.job.drafts as u64;
                    s.spec_accepted += step.accepted as u64;
                });
                let mut stop = false;
                for id in step.ids {
                    if !sp.job.tx.send(Event::Tokens(vec![id])).is_ok() {
                        stop = true;
                        break;
                    }
                    sp.next = id;
                    sp.emitted += 1;
                    metrics.update(|s| s.generated_tokens += 1);
                    if sp.job.stop_ids.contains(&id) {
                        stop = true;
                        break;
                    }
                    if sp.emitted >= sp.job.max_tokens {
                        stop = true;
                        break;
                    }
                }
                if stop || step.stopped {
                    let sp = spec.take().expect("checked");
                    let finish = if step.stopped { "stop" } else { "length" };
                    send_done(
                        &sp.job.tx,
                        &sp.job,
                        finish,
                        sp.emitted,
                        sp.prefill_ms,
                        sp.hit_tokens,
                    );
                    metrics.update(|s| s.requests_done += 1);
                }
            }
        }

        // One batched step for every active slot.
        let order: Vec<usize> = batch.active_slots().to_vec();
        if !order.is_empty() {
            let toks: Vec<u32> = order
                .iter()
                .map(|&s| slots[s].as_ref().expect("active slot has a job").feed)
                .collect();
            let ids = batch.forward(&toks)?;
            metrics.update(|s| {
                s.decode_steps += 1;
                s.batch_steps += 1;
            });
            let mut retire = Vec::new();
            for (i, &slot) in order.iter().enumerate() {
                let a = slots[slot].as_mut().expect("active slot has a job");
                let id = ids[i];
                if a.job.tx.send(Event::Tokens(vec![id])).is_err() {
                    retire.push((slot, "cancelled"));
                    continue;
                }
                a.feed = id;
                a.emitted += 1;
                metrics.update(|s| s.generated_tokens += 1);
                if a.job.stop_ids.contains(&id) {
                    retire.push((slot, "stop"));
                } else if a.emitted >= a.job.max_tokens {
                    retire.push((slot, "length"));
                }
            }
            // Release from the highest index down. A lower release swaps the
            // last active slot into the freed index, and all unprocessed
            // retirees are below it, so no retiring entry ever moves.
            retire.sort_unstable_by(|a, b| b.0.cmp(&a.0));
            for (slot, finish) in retire {
                take_active(&mut slots, slot, finish, &metrics);
                if let Some(from) = batch.release(slot)? {
                    slots[slot] = slots[from].take();
                }
            }
        }
    }
}
