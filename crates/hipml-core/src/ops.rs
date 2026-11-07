//! Reference kernels: the definition of correct, in plain Rust.
//!
//! Each function here mirrors one kernel in `hipml-kernels/kernels/*.hip` — same
//! name, same argument order, same layout. If a signature changes here it must
//! change there, and the parity harness will notice.
//!
//! Conventions: everything is row-major f32, `eps` guards normalisation, and
//! `rope` is the NeoX (half-split) variant used by Llama/Qwen-style models.

use crate::quant::{
    dequant_q4_k, dequant_q6_k, dequant_q8_0, Q4_K_BYTES, Q6_K_BYTES, Q8_0_BYTES, Q8_0_VALUES, QK_K,
};

/// `y[m] = W[m,k] · x[k]`
///
/// Kernel: `gemv_f32`, one block per output row, `k` reduced across the block.
pub fn gemv_f32(w: &[f32], x: &[f32], m: usize, k: usize) -> Vec<f32> {
    assert_eq!(w.len(), m * k, "W must be m*k");
    assert_eq!(x.len(), k, "x must be k");
    let mut y = vec![0.0f32; m];
    for (i, out) in y.iter_mut().enumerate() {
        let row = &w[i * k..(i + 1) * k];
        let mut acc = 0.0f32;
        for j in 0..k {
            acc += row[j] * x[j];
        }
        *out = acc;
    }
    y
}

/// `y[r,m] = W[m,k] · x[r,k]`, with all tensors row-major.
///
/// Kernel: `gemm_f32`, one block per output column, reusing W across rows.
pub fn gemm_f32(w: &[f32], x: &[f32], r: usize, m: usize, k: usize) -> Vec<f32> {
    assert_eq!(w.len(), m * k, "W must be m*k");
    assert_eq!(x.len(), r * k, "x must be r*k");
    let mut y = vec![0.0f32; r * m];
    for batch in 0..r {
        for row in 0..m {
            let mut acc = 0.0f32;
            for j in 0..k {
                acc += w[row * k + j] * x[batch * k + j];
            }
            y[batch * m + row] = acc;
        }
    }
    y
}

/// `out[i] = sigmoid(x[i]) = 1 / (1 + exp(-x[i]))`
///
/// Kernel: `sigmoid_f32`, elementwise.
pub fn sigmoid_f32(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect()
}

/// `log(1 + exp(x))`, with the `x > 20` shortcut from
/// `ggml_compute_softplus_f32` so the exponential cannot overflow.
///
/// Kernel: `softplus_f32`, one value (the GDN time-step gate).
pub fn softplus_f32(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// `y[t,m] = W[m,k] · x[t,k]` with `W` stored as Q4_K blocks, row-major.
///
/// This is the prefill (multi-token) twin of `quant::gemv_q4_k`. The GPU kernel
/// dequantises a weight tile once into shared memory and reuses it for every
/// token; this reference dequantises one whole row and then dots it against
/// every token, which is the same arithmetic in the same k order.
pub fn gemm_q4_k(w: &[u8], x: &[f32], tokens: usize, m: usize, k: usize) -> Vec<f32> {
    gemm_quant_blocks(w, x, tokens, m, k, QK_K, Q4_K_BYTES, dequant_q4_k)
}

/// `y[t,m] = W[m,k] · x[t,k]` with `W` stored as Q6_K blocks, row-major.
pub fn gemm_q6_k(w: &[u8], x: &[f32], tokens: usize, m: usize, k: usize) -> Vec<f32> {
    gemm_quant_blocks(w, x, tokens, m, k, QK_K, Q6_K_BYTES, dequant_q6_k)
}

/// `y[t,m] = W[m,k] · x[t,k]` with `W` stored as Q8_0 blocks, row-major.
pub fn gemm_q8_0(w: &[u8], x: &[f32], tokens: usize, m: usize, k: usize) -> Vec<f32> {
    gemm_quant_blocks(
        w,
        x,
        tokens,
        m,
        k,
        Q8_0_VALUES,
        Q8_0_BYTES,
        dequant_q8_0,
    )
}

/// One dequantise-a-row-then-dot implementation for every block format.
fn gemm_quant_blocks(
    w: &[u8],
    x: &[f32],
    tokens: usize,
    m: usize,
    k: usize,
    block_values: usize,
    block_bytes: usize,
    dequant: fn(&[u8], &mut [f32]),
) -> Vec<f32> {
    assert_eq!(k % block_values, 0, "k must be a whole number of blocks");
    let blocks_per_row = k / block_values;
    let row_bytes = blocks_per_row * block_bytes;
    assert_eq!(w.len(), m * row_bytes, "W must be m rows of quantized blocks");
    assert_eq!(x.len(), tokens * k, "x must be tokens*k");

    let mut y = vec![0.0f32; tokens * m];
    let mut row = vec![0.0f32; k];
    for r in 0..m {
        let data = &w[r * row_bytes..(r + 1) * row_bytes];
        for b in 0..blocks_per_row {
            dequant(
                &data[b * block_bytes..(b + 1) * block_bytes],
                &mut row[b * block_values..(b + 1) * block_values],
            );
        }
        for t in 0..tokens {
            let xr = &x[t * k..(t + 1) * k];
            let mut acc = 0.0f32;
            for j in 0..k {
                acc += row[j] * xr[j];
            }
            y[t * m + r] = acc;
        }
    }
    y
}

/// `y[n] = x[n] / sqrt(mean(x²) + eps) * w[n]`
///
/// Kernel: `rmsnorm_f32`, one block per row, block-wide reduction for the mean.
pub fn rmsnorm_f32(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), w.len(), "x and w must match");
    let n = x.len();
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let scale = 1.0 / (mean_sq + eps).sqrt();
    x.iter().zip(w).map(|(v, g)| v * scale * g).collect()
}

/// NeoX rope applied to every head at one position.
///
/// `x` is `[n_heads, head_dim]`; for each head the first half is rotated against
/// the second half. Frequencies: `theta_base^(-2i/head_dim)`.
///
/// Kernel: `rope_neox_f32`, one warp per head.
pub fn rope_neox_f32(
    x: &[f32],
    n_heads: usize,
    head_dim: usize,
    pos: usize,
    theta_base: f32,
) -> Vec<f32> {
    assert_eq!(x.len(), n_heads * head_dim, "x must be n_heads*head_dim");
    assert_eq!(head_dim % 2, 0, "head_dim must be even");
    let half = head_dim / 2;
    let mut y = x.to_vec();
    for h in 0..n_heads {
        let base = h * head_dim;
        for i in 0..half {
            // The angle is built in f64 on purpose: at position 37 it reaches
            // ~37 rad, and computing it in f32 makes two libms disagree by
            // ~7e-5. The kernel does the same, so parity compares the rotation
            // and not two implementations of sinf.
            let freq = (theta_base as f64).powf(-2.0 * i as f64 / head_dim as f64);
            let ang = pos as f64 * freq;
            let (sin, cos) = ang.sin_cos();
            let (sin, cos) = (sin as f32, cos as f32);
            let x1 = x[base + i];
            let x2 = x[base + half + i];
            y[base + i] = x1 * cos - x2 * sin;
            y[base + half + i] = x1 * sin + x2 * cos;
        }
    }
    y
}

/// `out[i] = silu(gate[i]) * up[i]`, `silu(v) = v / (1 + exp(-v))`
///
/// Kernel: `silu_mul_f32`, elementwise.
pub fn silu_mul_f32(gate: &[f32], up: &[f32]) -> Vec<f32> {
    assert_eq!(gate.len(), up.len(), "gate and up must match");
    gate.iter()
        .zip(up)
        .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
        .collect()
}

/// Numerically stable softmax over one row.
///
/// Kernel: `softmax_f32`, one block per row, max-subtracted.
pub fn softmax_f32(x: &[f32]) -> Vec<f32> {
    assert!(!x.is_empty(), "softmax needs at least one value");
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|v| v / sum).collect()
}

/// M-RoPE (Qwen3.5) applied to every head at one position.
///
/// Port of `ggml_compute_forward_rope_flt` (mode `GGML_ROPE_TYPE_MROPE`,
/// `ggml-mrope-cache-init`): only the first `n_rot` channels rotate, NeoX
/// pair `i` against `i + n_rot/2`, and the frequency of pair `k` is
/// `theta_base^(-2k/n_rot)`. `sections` chooses which of the four position ids
/// feeds pair `k`; for text all four ids are the token position, so this
/// degenerates to plain NeoX rope over `n_rot` channels.
///
/// `pos` is `[temporal, height, width, extra]`; pass the same value four times
/// for text. Kernel: `rope_multi_f32`, one warp per head.
#[allow(clippy::too_many_arguments)]
pub fn rope_multi_f32(
    x: &[f32],
    n_heads: usize,
    head_dim: usize,
    n_rot: usize,
    sections: [usize; 4],
    pos: [usize; 4],
    theta_base: f32,
) -> Vec<f32> {
    assert_eq!(x.len(), n_heads * head_dim, "x must be n_heads*head_dim");
    assert!(n_rot <= head_dim && n_rot % 2 == 0, "n_rot must be even and <= head_dim");
    let half = n_rot / 2;
    let sect: usize = sections.iter().sum();
    let mut y = x.to_vec();
    for h in 0..n_heads {
        let base = h * head_dim;
        for k in 0..half {
            // Which position id: sections[0] temporal, [1] height, [2] width,
            // [3] extra, matching ggml_mrope_cache_init's sector walk.
            let p = if sect == 0 {
                pos[0]
            } else {
                let s = k % sect;
                if s < sections[0] {
                    pos[0]
                } else if s < sections[0] + sections[1] {
                    pos[1]
                } else if s < sections[0] + sections[1] + sections[2] {
                    pos[2]
                } else {
                    pos[3]
                }
            };
            // f64 for the angle: at position 37 the angle reaches ~37 rad and
            // an f32 powf differs from the reference libm by ~7e-5.
            let freq = (theta_base as f64).powf(-2.0 * k as f64 / n_rot as f64);
            let ang = p as f64 * freq;
            let (sin, cos) = ang.sin_cos();
            let (sin, cos) = (sin as f32, cos as f32);
            let x1 = x[base + k];
            let x2 = x[base + half + k];
            y[base + k] = x1 * cos - x2 * sin;
            y[base + half + k] = x1 * sin + x2 * cos;
        }
    }
    y
}

/// `c[i] = a[i] + b[i]` — the residual add.
///
/// Kernel: `residual_add_f32`, elementwise.
pub fn residual_add_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "a and b must match");
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

/// Top-p (nucleus) sampling from probabilities, driven by an explicit `u` in
/// `[0,1)` so the reference is deterministic.
///
/// Sorts descending, keeps the shortest prefix whose mass reaches `top_p`
/// (always at least one candidate), then walks that prefix with `u`.
///
/// Kernel: `top_p_cdf_f32` builds the prefix on device; the walk stays on the
/// host because it is one token per step.
pub fn sample_top_p_f32(probs: &[f32], top_p: f32, u: f32) -> usize {
    assert!(!probs.is_empty(), "need at least one candidate");
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut mass = 0.0f32;
    let mut keep = 0usize;
    for &idx in &order {
        mass += probs[idx];
        keep += 1;
        if mass >= top_p {
            break;
        }
    }
    let mass = mass.max(1e-12);
    let mut target = u * mass;
    for &idx in order.iter().take(keep) {
        target -= probs[idx];
        if target <= 0.0 {
            return idx;
        }
    }
    order[keep - 1]
}

/// Decode-time causal attention for one query position against the whole cache.
///
/// `q` is `[n_heads, head_dim]`, `k` and `v` are `[n_kv, seq, head_dim]`, and
/// GQA is handled by pointing `n_heads / n_kv` query heads at one KV head. This
/// is the shape a dense model runs per generated token, and the one llama.cpp
/// cannot do on the GPU for an architecture without a Flash Attention path.
///
/// Kernel: `attn_decode_f32`, one block per query head, online softmax so the
/// whole cache is never materialised.
#[allow(clippy::too_many_arguments)]
pub fn attn_decode_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
    seq: usize,
    scale: f32,
) -> Vec<f32> {
    assert_eq!(q.len(), n_heads * head_dim, "q must be n_heads*head_dim");
    assert_eq!(
        k.len(),
        n_kv * seq * head_dim,
        "k must be n_kv*seq*head_dim"
    );
    assert_eq!(v.len(), k.len(), "v must match k");
    assert!(n_heads % n_kv == 0, "n_heads must be a multiple of n_kv");
    let group = n_heads / n_kv;

    let mut out = vec![0.0f32; n_heads * head_dim];
    for h in 0..n_heads {
        let kvh = h / group;
        let qh = &q[h * head_dim..(h + 1) * head_dim];
        let kbase = kvh * seq * head_dim;
        let obase = h * head_dim;

        // Pass 1: scores with the max folded in, so exp cannot overflow.
        let mut scores = vec![0.0f32; seq];
        let mut max = f32::NEG_INFINITY;
        for row in 0..seq {
            let kr = &k[kbase + row * head_dim..kbase + (row + 1) * head_dim];
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += qh[d] * kr[d];
            }
            let s = dot * scale;
            scores[row] = s;
            max = max.max(s);
        }

        // Pass 2: softmax weights and the weighted sum of V.
        let mut sum = 0.0f32;
        for row in 0..seq {
            let w = (scores[row] - max).exp();
            scores[row] = w;
            sum += w;
        }
        let inv = 1.0 / sum;
        for row in 0..seq {
            let w = scores[row] * inv;
            let vr = &v[kbase + row * head_dim..kbase + (row + 1) * head_dim];
            for d in 0..head_dim {
                out[obase + d] += w * vr[d];
            }
        }
    }
    out
}
