//! The chunked-prefill attention kernel against the CPU reference.
//!
//! One case: 37 queries at position 29 over a 66-row cache, compared row by
//! row. It allocates its own q/k/v buffers, so it lives outside the main parity
//! table. Position 29 (not 0) is deliberate: the causal mask compares the cache
//! row index against `pos0 + chunk_row`, and a `pos0 = 0` case cannot see a
//! double-counted position.

use hipml_core::attention::{attn_causal_f32, FullAttnW};
use hipml_core::config::Config;
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
