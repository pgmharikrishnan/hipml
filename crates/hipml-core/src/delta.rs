//! The recurrent half of Qwen3.5: causal conv1d + gated delta net.
//!
//! Ports the single-token path of `build_layer_attn_linear` in
//! `src/models/qwen35.cpp` and `build_delta_net_autoregressive` in
//! `src/models/delta-net-base.cpp`, with the fused-kernel semantics from
//! `ggml/src/ggml-et/et-kernels/src/gated_delta_net_f32.c`:
//!
//! ```text
//!   S[i][j] <- S[i][j] * exp(g[h]) + k[i] * (v[j] - sum_i S[i][j]*exp(g[h])*k[i]) * beta[h]
//!   o[j]    <- sum_i S[i][j] * q[i] * 1/sqrt(S)
//! ```
//!
//! `i` is the key/state axis (ggml dim 0) and `j` the value axis (dim 1), so
//! `state[h][j][i]` is contiguous in `i`. Getting this transposition wrong is
//! the most likely way to be silently wrong, hence the explicit naming.

use crate::mat::Mat;
use crate::ops::{rmsnorm_f32, silu_mul_f32, softplus_f32};

/// Weights of one gated-delta-net layer.
pub struct SsmW {
    pub wqkv: Mat,
    /// `attn_gate` in GGUF: the z projection whose silu gates the output norm.
    pub wgate: Mat,
    pub conv1d: Vec<f32>,
    /// [n_v, n_embd], row per value head.
    pub alpha: Mat,
    /// [n_v, n_embd], row per value head.
    pub beta: Mat,
    pub dt_bias: Vec<f32>,
    /// `ssm_a`, already negative in the file (the decay log-rate).
    pub a: Vec<f32>,
    pub norm: Vec<f32>,
    pub wout: Mat,
}

/// Recurrent state carried across tokens.
pub struct SsmState {
    /// Last `d_conv - 1` input columns: `[(d_conv-1)][channels]`.
    pub conv: Vec<f32>,
    /// Delta-net state: `[n_v][head_v][head_v]`, `state[h][j][i]`.
    pub s: Vec<f32>,
}

impl SsmState {
    pub fn new(d_conv: usize, channels: usize, n_v: usize, head_v: usize) -> SsmState {
        SsmState {
            conv: vec![0.0; (d_conv - 1) * channels],
            s: vec![0.0; n_v * head_v * head_v],
        }
    }
}

/// `y[c] = sum_k conv_in[t+k][c] * w[c][k]`, with the history kept in `state`.
///
/// Port of `ggml_compute_forward_ssm_conv_f32`: the kernel is per channel, and
/// the row fed to it is `[d_conv-1 history columns; the current token]`.
/// Kernel: `ssm_conv1d_f32`.
pub fn ssm_conv1d_f32(state: &mut [f32], x: &[f32], w: &[f32], d_conv: usize) -> Vec<f32> {
    let channels = x.len();
    let hist = d_conv - 1;
    assert_eq!(state.len(), hist * channels, "conv state shape");
    assert_eq!(w.len(), d_conv * channels, "conv kernel shape");

    let mut y = vec![0.0f32; channels];
    for (c, out) in y.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for k in 0..d_conv {
            let v = if k < hist {
                state[k * channels + c]
            } else {
                x[c]
            };
            acc += v * w[c * d_conv + k];
        }
        *out = acc;
    }

    // Slide the window: drop the oldest column, append the current input.
    if hist > 1 {
        state.copy_within(channels..hist * channels, 0);
    }
    if hist > 0 {
        state[(hist - 1) * channels..].copy_from_slice(x);
    }
    y
}

/// Gated delta net, one token, autoregressive form.
///
/// `q`,`k` are `[n_k][sd]`, `v` is `[n_v][sd]`, `g`/`beta` are `[n_v]`
/// (already `exp`-ready and sigmoid respectively), `s` is `[n_v][sd][sd]`.
/// Value head `h` reads key head `h % n_k` (`ggml_repeat_4d` broadcast).
/// Returns `[n_v][sd]`.
/// Kernel: `gdn_ar_f32`.
#[allow(clippy::too_many_arguments)]
pub fn gdn_ar_f32(
    s: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    n_v: usize,
    n_k: usize,
    sd: usize,
) -> Vec<f32> {
    assert_eq!(q.len(), n_k * sd, "q shape");
    assert_eq!(k.len(), n_k * sd, "k shape");
    assert_eq!(v.len(), n_v * sd, "v shape");
    assert_eq!(s.len(), n_v * sd * sd, "state shape");
    let scale = 1.0 / (sd as f32).sqrt();

    let mut out = vec![0.0f32; n_v * sd];
    for h in 0..n_v {
        let hk = h % n_k;
        let decay = g[h].exp();
        let qh = &q[hk * sd..(hk + 1) * sd];
        let kh = &k[hk * sd..(hk + 1) * sd];
        let vh = &v[h * sd..(h + 1) * sd];
        let b = beta[h];
        let head = &mut s[h * sd * sd..(h + 1) * sd * sd];
        for j in 0..sd {
            let row = &mut head[j * sd..(j + 1) * sd];
            // dot(S[:,j] * decay, k)
            let mut dot = 0.0f32;
            for i in 0..sd {
                dot += row[i] * decay * kh[i];
            }
            let delta = (vh[j] - dot) * b;
            let mut acc = 0.0f32;
            for i in 0..sd {
                let nv = row[i] * decay + kh[i] * delta;
                row[i] = nv;
                acc += nv * qh[i];
            }
            out[h * sd + j] = acc * scale;
        }
    }
    out
}

/// One gated-delta-net layer for one token.
///
/// Mirrors `build_layer_attn_linear`: project qkv and z, alpha/beta, conv1d,
/// silu, split q|k|v, L2-normalise q/k, run the recurrence, gated RMS norm,
/// output projection. Returns the block output (before the residual add).
pub fn ssm_layer_f32(
    w: &SsmW,
    st: &mut SsmState,
    x: &[f32],
    cfg: &crate::config::Config,
) -> Vec<f32> {
    let head_k = cfg.ssm_head_k();
    let head_v = cfg.ssm_head_v();
    let n_k = cfg.ssm_n_group;
    let n_v = cfg.ssm_dt_rank;
    let key_dim = head_k * n_k;

    // Projections (qwen35.cpp build_qkvz / ssm_alpha / ssm_beta).
    let mut qkv = w.wqkv.gemv(x);
    let z = w.wgate.gemv(x);
    let beta = crate::ops::sigmoid_f32(&w.beta.gemv(x));
    let alpha = w.alpha.gemv(x);
    let g: Vec<f32> = (0..n_v)
        .map(|h| softplus_f32(alpha[h] + w.dt_bias[h]) * w.a[h])
        .collect();

    // conv1d over the q|k|v channels, then silu (qwen35.cpp build_conv_state).
    let conv = ssm_conv1d_f32(&mut st.conv, &qkv, &w.conv1d, cfg.ssm_d_conv);
    for (v, a) in qkv.iter_mut().zip(&conv) {
        *v = a / (1.0 + (-a).exp());
    }

    // Split q | k | v and L2-normalise each head (build_gdn_l2_norm:
    // rms_norm with eps/n, then scale by 1/sqrt(n)).
    let q = l2norm_heads(&qkv[..key_dim], n_k, head_k, cfg.eps);
    let k = l2norm_heads(&qkv[key_dim..2 * key_dim], n_k, head_k, cfg.eps);
    let v = &qkv[2 * key_dim..];

    let o = gdn_ar_f32(&mut st.s, &q, &k, v, &g, &beta, n_v, n_k, head_v);

    // Gated normalisation: rms_norm(o, ssm_norm) * silu(z), per 128-value head.
    let mut normed = vec![0.0f32; o.len()];
    for h in 0..n_v {
        let r = rmsnorm_f32(
            &o[h * head_v..(h + 1) * head_v],
            &w.norm[..head_v],
            cfg.eps,
        );
        normed[h * head_v..(h + 1) * head_v].copy_from_slice(&r);
    }
    let gated = silu_mul_f32(&z, &normed);
    w.wout.gemv(&gated)
}

/// Per-head L2 normalisation used by the delta net.
///
/// `ggml_rms_norm(x, eps/n) * 1/sqrt(n)` is exactly `x / sqrt(sum(x^2)+eps)`.
/// Kernel: `l2norm_f32`.
fn l2norm_heads(x: &[f32], n_heads: usize, head_dim: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for h in 0..n_heads {
        let src = &x[h * head_dim..(h + 1) * head_dim];
        let sum: f64 = src.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let inv = 1.0 / ((sum + eps as f64).sqrt() as f32);
        for d in 0..head_dim {
            out[h * head_dim + d] = src[d] * inv;
        }
    }
    out
}
