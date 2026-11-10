// SPDX-License-Identifier: Apache-2.0
//! Full-attention layers: Q projection with gate, per-head q/k RMS norm, M-RoPE,
//! causal attention over a per-layer KV cache, and the output projection.
//!
//! Ports `build_layer_attn` in `src/models/qwen35.cpp`:
//!
//! ```text
//!   qg = Wq x                      # 2 * head_dim per head: [q | gate]
//!   q  = rmsnorm(q, q_norm)        # per head
//!   k  = rmsnorm(Wk x, k_norm)     # per head
//!   q, k = rope_multi(q, k)
//!   o  = softmax(q k^T / sqrt(d)) v
//!   y  = Wo (o * sigmoid(gate))
//! ```

use crate::config::Config;
use crate::mat::Mat;
use crate::ops::{rmsnorm_f32, rope_multi_f32, sigmoid_f32};

/// Weights of one full-attention layer.
pub struct FullAttnW {
    /// `attn_q`: rows are `[q(head_dim) | gate(head_dim)]` per head.
    pub wq: Mat,
    pub wk: Mat,
    pub wv: Mat,
    pub wo: Mat,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
}

/// Per-layer KV cache, `[n_ctx][n_kv * head_dim]`.
pub struct KvState {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub n_ctx: usize,
}

impl KvState {
    pub fn new(n_ctx: usize, width: usize) -> KvState {
        KvState {
            k: vec![0.0; n_ctx * width],
            v: vec![0.0; n_ctx * width],
            n_ctx,
        }
    }
}

/// Causal attention for one query position.
///
/// `cache` is `[n_ctx][n_kv*head_dim]`, so the visible prefix is the first
/// `seq` rows. Query head `h` reads KV head `h / (n_heads / n_kv)`.
/// Kernel: `attn_causal_f32` (decode shape; the existing `attn_decode_f32`
/// assumes a kv-major cache, this one is position-major because it is append
/// only).
#[allow(clippy::too_many_arguments)]
pub fn attn_causal_f32(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
    seq: usize,
    scale: f32,
) -> Vec<f32> {
    let width = n_kv * head_dim;
    let group = n_heads / n_kv;
    let mut out = vec![0.0f32; n_heads * head_dim];
    let mut scores = vec![0.0f32; seq];

    for h in 0..n_heads {
        let kvh = h / group;
        let qh = &q[h * head_dim..(h + 1) * head_dim];
        let mut max = f32::NEG_INFINITY;
        for t in 0..seq {
            let kr = &k_cache[t * width + kvh * head_dim..t * width + (kvh + 1) * head_dim];
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += qh[d] * kr[d];
            }
            let s = dot * scale;
            scores[t] = s;
            max = max.max(s);
        }
        let mut sum = 0.0f32;
        for s in scores.iter_mut().take(seq) {
            let w = (*s - max).exp();
            *s = w;
            sum += w;
        }
        let inv = 1.0 / sum;
        let ob = &mut out[h * head_dim..(h + 1) * head_dim];
        for t in 0..seq {
            let w = scores[t] * inv;
            let vr = &v_cache[t * width + kvh * head_dim..t * width + (kvh + 1) * head_dim];
            for d in 0..head_dim {
                ob[d] += w * vr[d];
            }
        }
    }
    out
}

/// One full-attention layer for one token at absolute position `pos`.
pub fn attn_layer_f32(
    w: &FullAttnW,
    kv: &mut KvState,
    x: &[f32],
    cfg: &Config,
    pos: usize,
) -> Vec<f32> {
    let hd = cfg.head_dim;
    let n_h = cfg.n_head;
    let n_kv = cfg.n_head_kv;
    let width = n_kv * hd;

    // Joint q|gate projection, then the per-head interleaved split.
    let qg = w.wq.gemv(x);
    let mut q = vec![0.0f32; n_h * hd];
    let mut gate = vec![0.0f32; n_h * hd];
    for h in 0..n_h {
        let src = &qg[h * 2 * hd..(h + 1) * 2 * hd];
        q[h * hd..(h + 1) * hd].copy_from_slice(&src[..hd]);
        gate[h * hd..(h + 1) * hd].copy_from_slice(&src[hd..]);
    }

    let mut k = w.wk.gemv(x);
    let v = w.wv.gemv(x);

    // Per-head RMS norm (build_norm over a 4d view: last dim is head_dim).
    for h in 0..n_h {
        let r = rmsnorm_f32(&q[h * hd..(h + 1) * hd], &w.q_norm, cfg.eps);
        q[h * hd..(h + 1) * hd].copy_from_slice(&r);
    }
    for h in 0..n_kv {
        let r = rmsnorm_f32(&k[h * hd..(h + 1) * hd], &w.k_norm, cfg.eps);
        k[h * hd..(h + 1) * hd].copy_from_slice(&r);
    }

    // M-RoPE. For text all four position ids are the same token position.
    let p = [pos; 4];
    let q = rope_multi_f32(&q, n_h, hd, cfg.n_rot, cfg.rope_sections, p, cfg.rope_base);
    let k = rope_multi_f32(&k, n_kv, hd, cfg.n_rot, cfg.rope_sections, p, cfg.rope_base);

    // Append to the cache.
    let off = pos * width;
    assert!(pos < kv.n_ctx, "position {pos} beyond KV cache {}", kv.n_ctx);
    kv.k[off..off + width].copy_from_slice(&k);
    kv.v[off..off + width].copy_from_slice(&v);

    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = attn_causal_f32(
        &q,
        &kv.k,
        &kv.v,
        n_h,
        n_kv,
        hd,
        pos + 1,
        scale,
    );

    let g = sigmoid_f32(&gate);
    for (o, g) in out.iter_mut().zip(&g) {
        *o *= g;
    }
    w.wo.gemv(&out)
}
