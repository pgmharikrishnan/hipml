// SPDX-License-Identifier: Apache-2.0
//! `attn-bench --chunk POS0,TOK`: one layer's chunk attention at real depth.
//!
//! The full-model prefill takes 136 s at 100K, which is too slow to iterate a
//! kernel on. One layer of one chunk at the same position is the same kernel
//! launch and the same cache layout, so it isolates the attention cost in a few
//! hundred milliseconds and reports the two numbers that matter: wall time and
//! effective staged bandwidth (the kernel re-reads the prefix 1.5 times per
//! query tile, so `GB/s` here is the *staged* rate, not the unique one).
//!
//! The cache is filled with format-valid synthetic q8/q4 blocks, not with
//! tokens: attention time does not depend on the values, and a 100K f32 cache
//! would need 800 MB of host traffic per rep.

use hipml_core::config::Config;
use hipml_core::kv::{self, KvDtype};
use hipml_kernels::{AttnLayer, Launcher};
use std::time::Instant;

/// Position range and chunk length from `--chunk`.
pub struct Args {
    pub pos0: usize,
    pub tok: usize,
    pub splits: usize,
    pub dtype: KvDtype,
    pub reps: usize,
}

pub fn parse(s: &str) -> Result<(usize, usize), String> {
    let mut it = s.split(',');
    let pos0 = it
        .next()
        .ok_or("--chunk needs POS0,TOK")?
        .parse()
        .map_err(|e| format!("bad --chunk position: {e}"))?;
    let tok = it
        .next()
        .ok_or("--chunk needs POS0,TOK")?
        .parse()
        .map_err(|e| format!("bad --chunk length: {e}"))?;
    if tok == 0 || tok > 2048 {
        return Err("--chunk TOK must be 1..=2048".to_string());
    }
    Ok((pos0, tok))
}

pub fn run(
    l: &Launcher,
    cfg: &Config,
    w: &hipml_core::attention::FullAttnW,
    a: &Args,
) -> Result<(), String> {
    let hd = cfg.head_dim;
    let nkv = cfg.n_head_kv;
    let nhd = cfg.n_head * hd;
    let seq = a.pos0 + a.tok;
    let (dtype_id, row) = match a.dtype {
        KvDtype::Q8 => (1usize, KvDtype::Q8.row_bytes(hd)),
        KvDtype::Q4 => (2usize, KvDtype::Q4.row_bytes(hd)),
    };

    // Format-valid blocks, one full cache per head.
    let blocks = seq * (hd / kv::BLOCK);
    let pat = kv::synthetic_blocks(a.dtype, blocks, 0x51ED_0001);
    let kb = hipml_hip::Buffer::alloc((nkv * blocks * pat.len() / blocks).div_ceil(4))?;
    let vb = hipml_hip::Buffer::alloc((nkv * blocks * pat.len() / blocks).div_ceil(4))?;
    let per_head = seq * row;
    for h in 0..nkv {
        kb.upload_bytes(h * per_head, &pat)?;
        vb.upload_bytes(h * per_head, &pat)?;
    }

    let mut qc = vec![0.0f32; a.tok * nhd];
    let mut s = 0x1234_5678u64;
    for v in qc.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *v = ((s >> 40) as f32 / 16777216.0 - 0.5) * 0.2;
    }
    let qb = hipml_hip::Buffer::from_host(&qc)?;
    let ob = hipml_hip::Buffer::alloc(a.tok * nhd)?;

    let qslots = (a.tok + 31) / 32 * 32;
    let apart = hipml_hip::Buffer::alloc(a.splits * qslots * cfg.n_head * hd)?;
    let aml = hipml_hip::Buffer::alloc(a.splits * qslots * cfg.n_head * 2)?;

    let layer = AttnLayer::new(l, w, cfg, seq + 8)?;
    let launch = |layer: &AttnLayer| -> Result<(), String> {
        layer.chunk_attention_q8_split(
            l, &qb, &kb, &vb, &ob, a.pos0, a.tok, dtype_id, seq, &apart, &aml, a.splits,
        )
    };
    launch(&layer)?;
    l.synchronize()?;

    let t0 = Instant::now();
    for _ in 0..a.reps {
        launch(&layer)?;
    }
    l.synchronize()?;
    let ms = t0.elapsed().as_secs_f64() * 1e3 / a.reps as f64;

    // Useful work in this launch.
    let pairs: f64 = (0..a.tok).map(|t| (a.pos0 + t + 1) as f64).sum::<f64>() * cfg.n_head as f64;
    let flops = pairs * 2.0 * hd as f64 * 2.0;
    // Staged bytes: K in full plus V at half width per (query tile, d half).
    let mut staged = 0.0f64;
    let mut y = 0;
    while y < a.tok {
        let keys = (a.pos0 + y + 32).min(a.pos0 + a.tok) as f64;
        let kt = (keys / 16.0).ceil();
        staged += nkv as f64 * 2.0 * kt * (16.0 * row as f64 + 16.0 * (row / 2) as f64);
        y += 32;
    }
    println!(
        "chunk attn {}@{} [{} cache, splits {}]: {:.3} ms  {:.2} TFLOP/s useful  {:.1} GB/s staged  {:.1} MiB scratch",
        a.tok,
        a.pos0,
        a.dtype.name(),
        a.splits,
        ms,
        flops / (ms * 1e-3) / 1e12,
        staged / (ms * 1e-3) / 1e9,
        (apart.len() + aml.len()) as f64 * 4.0 / (1024.0 * 1024.0),
    );
    Ok(())
}
