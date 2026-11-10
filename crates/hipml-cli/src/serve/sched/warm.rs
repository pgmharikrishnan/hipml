// SPDX-License-Identifier: Apache-2.0
//! Load-time warm-up.
//!
//! The first prefill after a cold load pays for things no later request pays
//! for: the kernel module's first launches, the host-side weight pages being
//! faulted in, and the first-touch of every device scratch buffer. Measured on
//! the R9700 that is ~5 s, charged to the first client's TTFT.
//!
//! The warm-up below runs the three paths a request can take — one chunked
//! prefill tile, one batched decode step on two synthetic slots, and one
//! drafter step — on synthetic ids, then resets the state. It is deliberately
//! cheap: one weight stream per path, no prefix capture, no sampling.
//!
//! `HIPML_NO_WARMUP=1` skips it, which is how the cost is measured.

use std::time::Instant;

use hipml_model::gpu::{BatchEngine, GpuEngine, MtpGpu};

use super::*;

pub fn warm_up(
    engine: &mut GpuEngine<'_>,
    batch: &mut BatchEngine<'_>,
    mut mtp: Option<&mut MtpGpu<'_>>,
    cfg: &SchedCfg,
    metrics: &Metrics,
) -> Result<(), String> {
    if std::env::var_os("HIPML_NO_WARMUP").is_some() {
        return Ok(());
    }
    let t = Instant::now();
    // 1. One full prefill tile: every weight page and every prefill kernel the
    //    admission path will use. Ids are in-vocabulary and never emitted.
    let n = cfg.chunk.max(2).min(cfg.ctx.max(2));
    let toks: Vec<u32> = (0..n).map(|i| (i % 1000) as u32 + 1).collect();
    engine.reset_state()?;
    let mut st = engine.new_state();
    engine.prefill_tile(&toks, &mut st, cfg.chunk, false)?;

    // 2. One batched step: the batch kernels are a separate set of launches
    //    from the single-column ones, so they need their own first touch.
    let b = batch.slots.min(2).max(1);
    batch.activate(b)?;
    batch.set_positions(&vec![0; b])?;
    batch.forward(&vec![1u32; b])?;
    batch.activate(0)?;
    batch.reset_state()?;

    // 3. The drafter's weights, if one is loaded. A failure here is not fatal:
    //    the drafter's own path reports its errors when it is used.
    if let Some(m) = mtp.as_mut() {
        let h = vec![0.0f32; engine.n_embd()];
        let _ = m
            .set_hidden_host(&h)
            .and_then(|_| m.prefill_kv(1, 0))
            .and_then(|_| m.step(1, 1));
    }
    engine.reset_state()?;
    metrics.update(|s| s.warmup_ms += t.elapsed().as_secs_f64() * 1e3);
    eprintln!("serve: warm-up {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
    Ok(())
}
