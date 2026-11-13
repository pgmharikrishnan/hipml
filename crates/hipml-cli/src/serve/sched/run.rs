// SPDX-License-Identifier: Apache-2.0
//! The event loop. A child of `sched` so it can see the state structs'
//! private fields without making them part of the module's API.
//!
//! Every fallible step is scoped (WP51). A failure inside one job's prefill,
//! speculation, adoption or per-slot decode fails that job, frees its slot and
//! leaves the engine serving; only a shared weight pass fails a whole batch,
//! and only a reset that itself fails drains every stream and exits. See
//! `fail.rs` for the three scopes, `step.rs` for the per-slot verdicts and
//! `rows.rs` for the row-block admission.

use super::*;
use super::{fail, rows, step, warm};
use std::collections::VecDeque;
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::time::Duration;

use hipml_model::gpu::{BatchEngine, GpuEngine, KvMode, MtpGpu};
use hipml_model::Model;

pub fn run(
    model: &Model,
    rx: Receiver<Job>,
    cfg: SchedCfg,
    metrics: Arc<Metrics>,
    inflight: Arc<crate::serve::admit::Bound>,
    ready: std::sync::mpsc::Sender<()>,
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
    // Only now may the listener accept: until this signal the port is bound
    // but nothing is answered, so `/health` means "warm and serving" and the
    // first real request does not queue behind the warm-up.
    let _ = ready.send(());

    let mut queue: VecDeque<Job> = VecDeque::new();
    let mut slots: Vec<Option<Active>> = (0..batch.slots).map(|_| None).collect();
    let mut prefill: Option<Prefill> = None;
    let mut spec: Option<Spec> = None;
    let mut faults = step::Faults::new();
    // Test hooks for the batch and device scopes, off unless set.
    // `HIPML_FAULT_BATCH=n` fails shared pass n once (one batch reset, then the
    // engine serves again); `HIPML_FAULT_BATCH_ALL=1` fails every pass, which
    // is what drives three resets and the drain-and-exit path.
    let batch_fault_at: Option<u64> = std::env::var("HIPML_FAULT_BATCH")
        .ok()
        .and_then(|v| v.parse().ok());
    let batch_fault_all = std::env::var_os("HIPML_FAULT_BATCH_ALL").is_some();
    let mut batch_steps: u64 = 0;
    // Batch resets with no successful step between them: a device that cannot
    // finish one weight pass is not coming back on its own.
    let mut reset_streak = 0u32;
    // Measurement hooks, off unless set. `HIPML_SERVE_STEP_LOG=1` prints one
    // line per batched step: the weight pass's own wall time and the wall gap
    // since the previous step began. `HIPML_SERVE_SPLIT=1` prints the engine's
    // syncing stage buckets every 16 steps. Both are how the one-slot
    // attribution in `docs/notes/serve-ud-batch.md` was measured; neither
    // changes a step when unset.
    let step_log = std::env::var_os("HIPML_SERVE_STEP_LOG").is_some();
    let mut step_started = std::time::Instant::now();
    let split_log = std::env::var_os("HIPML_SERVE_SPLIT").is_some();
    const SPLIT_NAMES: [&str; 9] = [
        "embed",
        "attn_proj",
        "attn_kv",
        "attn_out",
        "ssm_proj",
        "ssm_rec",
        "ssm_out",
        "ffn",
        "final",
    ];
    const SPLIT_EVERY: u64 = 16;
    let mut split_prev = [0.0f64; 9];
    if split_log {
        batch.trace = true;
    }

    // Device-level failure: drain every open stream with an error, give the
    // connection threads a moment to write it, then let the process exit so a
    // supervisor can restart it.
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
            crate::serve::admit::await_close(&inflight, Duration::from_secs(2));
            return Err(why);
        }};
    }

    // A release that failed leaves the batch engine's slot table and ours out
    // of step; the only honest repair is a reset, and a reset that fails is
    // fatal.
    macro_rules! reset_batch {
        ($why:expr) => {{
            let why = $why;
            if let Err(e) = fail::batch(&mut batch, &mut engine, &mut slots, &metrics, &why) {
                fatal!(format!("{why}; reset failed: {e}"));
            }
            reset_streak += 1;
            if reset_streak >= 3 {
                fatal!(format!(
                    "{reset_streak} batch resets with no step in between"
                ));
            }
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
        let busy = prefill.is_some() || spec.is_some() || slots.iter().any(|s| s.is_some());
        if !busy && queue.is_empty() {
            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(j) => {
                    metrics.update(|s| s.requests_total += 1);
                    queue.push_back(j);
                }
                Err(_) => continue,
            }
        }

        // WP47: gather prefix-hit tails for one multi-row prefill.
        if prefill.is_none() && spec.is_none() {
            if let Err(e) = rows::gather(
                &mut engine,
                &mut batch,
                &mut slots,
                &mut queue,
                &rx,
                &cfg,
                &metrics,
                &mut prefill,
            ) {
                reset_batch!(e);
            }
        }

        // Start one prefill when the engine is free. A request is only allowed
        // to become speculative when it is alone and nothing is queued.
        let alone_next = slots.iter().all(|s| s.is_none()) && queue.len() == 1;
        let can_admit = (alone_next && cfg.mtp) || batch.available_slots() > 0;
        if prefill.is_none() && spec.is_none() && !queue.is_empty() && can_admit {
            let job = queue.pop_front().expect("checked");
            let alone = slots.iter().all(|s| s.is_none()) && queue.is_empty();
            let mode = if alone && cfg.mtp {
                Mode::Spec
            } else {
                Mode::Batch
            };
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
                send_done(
                    &sp.job.tx,
                    &sp.job,
                    reason,
                    sp.emitted,
                    sp.prefill_ms,
                    sp.hit_tokens,
                );
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
            faults.tick();
            if split_log && batch_steps > 0 && batch_steps % SPLIT_EVERY == 0 {
                let cur = batch.stages;
                let mut line = String::new();
                for i in 0..9 {
                    line.push_str(&format!(
                        "{}={:.3} ",
                        SPLIT_NAMES[i],
                        (cur[i] - split_prev[i]) * 1e3 / SPLIT_EVERY as f64
                    ));
                }
                eprintln!("split step {batch_steps}: {line}");
                split_prev = cur;
            }
            let toks: Vec<u32> = order
                .iter()
                .map(|&s| slots[s].as_ref().expect("active slot has a job").feed)
                .collect();
            batch_steps += 1;
            let stepped = if batch_fault_all || batch_fault_at == Some(batch_steps) {
                Err("injected batch-step failure (HIPML_FAULT_BATCH)".to_string())
            } else {
                let t0 = step_log.then(std::time::Instant::now);
                let gap = t0.map(|t| t.duration_since(step_started).as_secs_f64() * 1e3);
                if let Some(t) = t0 {
                    step_started = t;
                }
                let r = batch.forward(&toks);
                if let (Some(t), Some(gap)) = (t0, gap) {
                    eprintln!(
                        "step {} b={} gap {gap:.3} ms forward {:.3} ms",
                        batch_steps,
                        toks.len(),
                        t.elapsed().as_secs_f64() * 1e3
                    );
                }
                r
            };
            let ids = match stepped {
                Ok(ids) => {
                    reset_streak = 0;
                    ids
                }
                Err(e) => {
                    // One weight pass covers every slot, so a launch that fails
                    // cannot be blamed on one of them: fail the batch, reset
                    // the engine, keep serving.
                    reset_batch!(format!("batch step: {e}"));
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
            if let Err(e) = step::deliver(
                &mut batch,
                &mut engine,
                &mut slots,
                &order,
                &ids,
                &metrics,
                &mut faults,
            ) {
                reset_batch!(e);
            }
        }
    }
}
