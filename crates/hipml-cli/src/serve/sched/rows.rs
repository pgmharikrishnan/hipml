// SPDX-License-Identifier: Apache-2.0
//! WP47 row-block admission: gather prefix-hit tails and advance them all in
//! one weight pass.
//!
//! A tail is a request whose prompt already matches a cached prefix: its
//! restored state is copied into a quiet batch slot, and one `prefill_rows`
//! block advances every gathered tail's suffix at once instead of streaming the
//! whole model once per tail. This module is the gather-and-flush half of the
//! scheduler loop; `run.rs` keeps the loop itself.

use std::collections::VecDeque;
use std::sync::mpsc::Receiver;
use std::time::Duration;

use hipml_model::gpu::{BatchEngine, GpuEngine};

use super::*;

/// Gather tails until the row budget, the slot budget or the wait expires, then
/// run one block. Returns `Err` only when a failed release could not be
/// repaired by resetting the batch engine.
pub fn gather<'a>(
    engine: &mut GpuEngine<'a>,
    batch: &mut BatchEngine<'a>,
    slots: &mut [Option<Active>],
    queue: &mut VecDeque<Job>,
    rx: &Receiver<Job>,
    cfg: &SchedCfg,
    metrics: &Metrics,
    prefill: &mut Option<Prefill>,
) -> Result<(), String> {
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
        let alone = slots.iter().all(|s| s.is_none()) && queue.is_empty() && gathered.is_empty();
        let mode = if alone && cfg.mtp {
            Mode::Spec
        } else {
            Mode::Batch
        };
        match begin(engine, job, cfg, mode, metrics) {
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
                    match batch.adopt_quiet(engine, slot, pf.st.pos) {
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
                            // The failing slot is always the last gathered
                            // one, so the rest stay dense.
                            fail::job(engine, metrics, &pf.job, "row adopt", &e);
                            continue;
                        }
                    }
                }
                if gathered.is_empty() {
                    metrics.update(|s| s.prompt_tokens += plen as u64);
                    *prefill = Some(pf);
                } else {
                    // Let the queued request keep its place; the block runs
                    // first so it is not delayed behind it.
                    queue.push_front(pf.job);
                    flush = true;
                }
                break;
            }
            Err((job, e)) => fail::job(engine, metrics, &job, "admit", &e),
        }
    }

    if flush && !gathered.is_empty() {
        flush_rows(engine, batch, slots, gathered, rows_used, metrics)?;
    }
    Ok(())
}

fn flush_rows<'a>(
    engine: &mut GpuEngine<'a>,
    batch: &mut BatchEngine<'a>,
    slots: &mut [Option<Active>],
    gathered: Vec<RowTail>,
    rows_used: usize,
    metrics: &Metrics,
) -> Result<(), String> {
    let tails: Vec<(usize, &[u32])> = gathered
        .iter()
        .map(|g| (g.slot, g.ids.as_slice()))
        .collect();
    let ids = match batch.prefill_rows(&tails) {
        Ok(ids) => ids,
        Err(e) => {
            for g in gathered {
                fail::job(engine, metrics, &g.job, "row prefill", &e);
            }
            return Ok(());
        }
    };
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
    let mut blocked: Option<String> = None;
    for (k, g) in gathered.into_iter().enumerate() {
        if let Some(e) = blocked.take() {
            fail::job(engine, metrics, &g.job, "row activate", &e);
            blocked = Some(e);
            continue;
        }
        if let Err(e) = batch.activate_slot(g.slot) {
            // The active set is dense, so a failure here leaves the later
            // quiet slots unactivatable; fail them too instead of wedging.
            fail::job(engine, metrics, &g.job, "row activate", &e);
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
    // Highest index first: a lower release swaps the last active slot in, and
    // every unprocessed entry is below it.
    out.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    for (slot, fault, reason) in out {
        if let Some(why) = fault {
            if let Some(a) = slots[slot].take() {
                fail::slot(engine, metrics, &a.job, &why);
            }
        } else {
            fail::take_active(slots, slot, reason, metrics);
        }
        if let Err(e) = fail::release(batch, slots, slot) {
            return Err(format!("slot release: {e}"));
        }
    }
    Ok(())
}
