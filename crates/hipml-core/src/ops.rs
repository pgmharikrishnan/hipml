//! Reference kernels: the definition of correct, in plain Rust.
//!
//! Each function here mirrors one kernel in `hipml-kernels/kernels/*.hip` — same
//! name, same argument order, same layout. If a signature changes here it must
//! change there, and the parity harness will notice.
//!
//! Conventions: everything is row-major f32, `eps` guards normalisation, and
//! `rope` is the NeoX (half-split) variant used by Llama/Qwen-style models.

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
