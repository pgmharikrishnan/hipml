// SPDX-License-Identifier: Apache-2.0
//! The chunked-prefill attention kernel against the CPU reference.
//!
//! One case: 37 queries at position 29 over a 66-row cache, compared row by
//! row. It allocates its own q/k/v buffers, so it lives outside the main parity
//! table. Position 29 (not 0) is deliberate: the causal mask compares the cache
//! row index against `pos0 + chunk_row`, and a `pos0 = 0` case cannot see a
//! double-counted position.

use hipml_core::attention::{attn_causal_f32, FullAttnW};
use hipml_core::config::Config;
use hipml_core::kv::{self, KvDtype};
use hipml_core::parity::{compare, Stats, Tol};
use hipml_core::rng::Rng;
use hipml_kernels::{AttnLayer, Launcher};

/// Tolerance: the kernel runs fp16 operands with f32 accumulation.
pub fn tolerance() -> Tol {
    Tol::new(2e-3, 2e-2)
}

/// `(passed, stats)` for the one case; `Err` only for a launch failure.
pub fn run(l: &Launcher, cfg: &Config, w: &FullAttnW) -> Result<(bool, Stats), String> {
    const TOK: usize = 37;
    const POS0: usize = 29;
    let hd = cfg.head_dim;
    let width = cfg.n_head_kv * hd;
    let nhd = cfg.n_head * hd;
    let seq = POS0 + TOK;

    let mut rng = Rng::new(0xC4A0_1234_5678);
    let qc: Vec<f32> = (0..TOK * nhd).map(|_| rng.normal()).collect();
    let kc: Vec<f32> = (0..seq * width).map(|_| rng.normal()).collect();
    let vc: Vec<f32> = (0..seq * width).map(|_| rng.normal()).collect();
    let qb = hipml_hip::Buffer::from_host(&qc)?;
    let kb = hipml_hip::Buffer::from_host(&kc)?;
    let vb = hipml_hip::Buffer::from_host(&vc)?;
    let ob = hipml_hip::Buffer::alloc(TOK * nhd)?;

    let layer = AttnLayer::new(l, w, cfg, seq + 8)?;
    layer.chunk_attention(l, &qb, &kb, &vb, &ob, POS0, TOK)?;
    let gpu = ob.to_host()?;

    let scale = 1.0f32 / (hd as f32).sqrt();
    let mut cpu = vec![0.0f32; TOK * nhd];
    for t in 0..TOK {
        let row = attn_causal_f32(
            &qc[t * nhd..(t + 1) * nhd],
            &kc,
            &vc,
            cfg.n_head,
            cfg.n_head_kv,
            hd,
            POS0 + t + 1,
            scale,
        );
        cpu[t * nhd..(t + 1) * nhd].copy_from_slice(&row);
    }
    let stats = compare(&cpu, &gpu, tolerance());
    if stats.is_err() {
        // Where the error lives, so a layout regression is one line away.
        let (mut worst, mut at) = (0.0f32, (0usize, 0usize, 0usize));
        let (mut half0, mut half1) = (0.0f32, 0.0f32);
        for t in 0..TOK {
            for h in 0..cfg.n_head {
                for d in 0..hd {
                    let i = t * nhd + h * hd + d;
                    let e = (cpu[i] - gpu[i]).abs();
                    if d < hd / 2 {
                        half0 = half0.max(e);
                    } else {
                        half1 = half1.max(e);
                    }
                    if e > worst {
                        worst = e;
                        at = (t, h, d);
                    }
                }
            }
        }
        eprintln!(
            "  diag: max|d| half0 {half0:.3e} half1 {half1:.3e}; worst at row {} head {} d {} \
             (cpu {:.4} gpu {:.4})",
            at.0,
            at.1,
            at.2,
            cpu[at.0 * nhd + at.1 * hd + at.2],
            gpu[at.0 * nhd + at.1 * hd + at.2]
        );
    }
    Ok(match stats {
        Ok(st) => (true, st),
        Err(st) => (false, st),
    })
}

/// The same case over a q8_0 cache: the matrix-unit kernel dequantizes each KV
/// tile from the head-major quantized rows instead of reading an f32 mirror.
///
/// The cache is encoded here and both the reference and the kernel read those
/// bytes, so the comparison isolates accumulation order and the f16 operand
/// conversion; the 8-bit quantization error is common to both sides.
pub fn run_q8(l: &Launcher, cfg: &Config, w: &FullAttnW) -> Result<(bool, Stats), String> {
    run_q8_splits(l, cfg, w, 1)
}

/// The same case with the key axis split across blocks: `attn_chunk_split_f16`
/// writes per-slice partials and `attn_chunk_combine_f32` folds them, so this
/// case pins the combine pass against the CPU reference at a non-zero position
/// (where one slice can sit wholly above the diagonal).
pub fn run_q8_split(l: &Launcher, cfg: &Config, w: &FullAttnW) -> Result<(bool, Stats), String> {
    run_q8_splits(l, cfg, w, 4)
}

fn run_q8_splits(
    l: &Launcher,
    cfg: &Config,
    w: &FullAttnW,
    splits: usize,
) -> Result<(bool, Stats), String> {
    // Default is the 37@29 case; the env knobs run the same comparison deep in
    // a cache whose per-head stride is larger than the live length, which is
    // how the engine lays the cache out (stride = n_ctx).
    let tok = env_usize("HIPML_CHUNK_PARITY_TOK", 37);
    let pos0 = env_usize("HIPML_CHUNK_PARITY_POS0", 29);
    let stride = pos0 + tok + env_usize("HIPML_CHUNK_PARITY_PAD", 0);
    let (tok, pos0) = (tok.max(1), pos0);
    let hd = cfg.head_dim;
    let nkv = cfg.n_head_kv;
    let width = nkv * hd;
    let nhd = cfg.n_head * hd;
    // The cache is allocated with the full per-head stride.
    let seq = stride;
    if hd % kv::BLOCK != 0 {
        return Err(format!(
            "q8 chunk parity: head_dim {hd} is not a block multiple"
        ));
    }

    let mut rng = Rng::new(0xC4A0_9876_5432);
    let qc: Vec<f32> = (0..tok * nhd).map(|_| rng.normal()).collect();
    let kc: Vec<f32> = (0..seq * width).map(|_| rng.normal()).collect();
    let vc: Vec<f32> = (0..seq * width).map(|_| rng.normal()).collect();

    // Re-pack position-major f32 rows into the head-major cache layout, then
    // quantize. The dequantized result feeds the reference, so both sides read
    // the same 8-bit values.
    let mut kh: Vec<f32> = vec![0.0; seq * width];
    let mut vh: Vec<f32> = vec![0.0; seq * width];
    for h in 0..nkv {
        for p in 0..seq {
            let src = p * width + h * hd;
            let dst = (h * seq + p) * hd;
            kh[dst..dst + hd].copy_from_slice(&kc[src..src + hd]);
            vh[dst..dst + hd].copy_from_slice(&vc[src..src + hd]);
        }
    }
    let kb = kv::quantize(KvDtype::Q8, &kh);
    let vb = kv::quantize(KvDtype::Q8, &vh);
    let kd = kv::dequantize(KvDtype::Q8, &kb);
    let vd = kv::dequantize(KvDtype::Q8, &vb);
    let mut kq: Vec<f32> = vec![0.0; seq * width];
    let mut vq: Vec<f32> = vec![0.0; seq * width];
    for h in 0..nkv {
        for p in 0..seq {
            let src = (h * seq + p) * hd;
            let dst = p * width + h * hd;
            kq[dst..dst + hd].copy_from_slice(&kd[src..src + hd]);
            vq[dst..dst + hd].copy_from_slice(&vd[src..src + hd]);
        }
    }

    let qb = hipml_hip::Buffer::from_host(&qc)?;
    let kbb = hipml_hip::Buffer::alloc(kb.len().div_ceil(4))?;
    let vbb = hipml_hip::Buffer::alloc(vb.len().div_ceil(4))?;
    kbb.upload_bytes(0, &kb)?;
    vbb.upload_bytes(0, &vb)?;
    let ob = hipml_hip::Buffer::alloc(tok * nhd)?;

    let layer = AttnLayer::new(l, w, cfg, seq + 8)?;
    if splits > 1 {
        let qslots = (tok + 31) / 32 * 32;
        let apart = hipml_hip::Buffer::alloc(splits * qslots * cfg.n_head * hd)?;
        let aml = hipml_hip::Buffer::alloc(splits * qslots * cfg.n_head * 2)?;
        layer.chunk_attention_q8_split(
            l, &qb, &kbb, &vbb, &ob, pos0, tok, 1, seq, &apart, &aml, splits,
        )?;
    } else {
        layer.chunk_attention_q8(l, &qb, &kbb, &vbb, &ob, pos0, tok, 1, seq)?;
    }
    let gpu = ob.to_host()?;

    let scale = 1.0f32 / (hd as f32).sqrt();
    let mut cpu = vec![0.0f32; tok * nhd];
    for t in 0..tok {
        let row = attn_causal_f32(
            &qc[t * nhd..(t + 1) * nhd],
            &kq,
            &vq,
            cfg.n_head,
            cfg.n_head_kv,
            hd,
            pos0 + t + 1,
            scale,
        );
        cpu[t * nhd..(t + 1) * nhd].copy_from_slice(&row);
    }
    let stats = compare(&cpu, &gpu, tolerance());
    if stats.is_err() {
        let (mut worst, mut at) = (0.0f32, (0usize, 0usize, 0usize));
        for t in 0..tok {
            for h in 0..cfg.n_head {
                for d in 0..hd {
                    let i = t * nhd + h * hd + d;
                    let e = (cpu[i] - gpu[i]).abs();
                    if e > worst {
                        worst = e;
                        at = (t, h, d);
                    }
                }
            }
        }
        eprintln!(
            "  diag: q8 worst at row {} head {} d {} (cpu {:.5} gpu {:.5}, max|d| {worst:.3e})",
            at.0,
            at.1,
            at.2,
            cpu[at.0 * nhd + at.1 * hd + at.2],
            gpu[at.0 * nhd + at.1 * hd + at.2]
        );
    }
    Ok(match stats {
        Ok(st) => (true, st),
        Err(st) => (false, st),
    })
}

/// Integer env override, used only by the parity diagnostics above.
fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
