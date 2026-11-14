// SPDX-License-Identifier: Apache-2.0
// Portions of this file are derived from llama.cpp
// (https://github.com/ggml-org/llama.cpp, commit d7725699 unless noted),
// Copyright (c) the ggml authors, used under the MIT License.
// See NOTICE and docs/reference/LICENSE.
//! The MoE FFN, as a CPU reference.
//!
//! The authority is `llama-graph.cpp::build_moe_ffn` (vendored at
//! `docs/reference/llama-graph.cpp`) plus `qwen35moe.cpp::build_layer_ffn` for
//! the shared-expert wiring (`docs/reference/qwen35moe.cpp`). The conventions
//! below are the ones a GPU kernel must reproduce, and each one is stated where
//! it is implemented:
//!
//! 1. **Router logits** are `ffn_gate_inp · x`, no bias.
//! 2. **Softmax is over all `n_expert`** values, not over the selected k.
//! 3. **Top-k = 8, ordered by value descending, ties by expert id ascending.**
//!    llama.cpp's CPU argsort is `std::sort` with a strict `>` comparator, so
//!    equal values are left in an unspecified order; id-ascending is this
//!    engine's deterministic refinement of that, and it is the same rule the
//!    GPU kernel must use. Real weights never tie at f32.
//! 4. **The 8 weights are the unbiased softmax probabilities renormalised by
//!    their sum, with the sum clamped at `6.103515625e-5`** (the smallest
//!    normal f16), and no `expert_weights_scale` — the key is absent from this
//!    architecture's files, so scaling is off.
//! 5. **Per expert:** `up = up_exps[e]·x`, `gate = gate_exps[e]·x`,
//!    `act = silu(gate) * up`, `down = down_exps[e]·act`, then **multiply by the
//!    router weight after the down projection**.
//! 6. **The combine is a left fold in top-k rank order**:
//!    `((out[0] + out[1]) + out[2]) + …`, not a sum over expert id.
//! 7. **The shared expert** is a plain dense SwiGLU added after the MoE sum,
//!    scaled by `sigmoid(dot(x, ffn_gate_inp_shexp))` — one scalar for the
//!    whole vector.
//!
//! Arithmetic follows llama.cpp's CPU kernels: f32 projections and f32
//! elementwise work, with f64 only where llama.cpp accumulates into
//! `ggml_float` (a `double`) — the softmax sum and the router-weight sum. That
//! is what makes this a reference and not a second implementation: the id gate
//! in `moe-gate` compares it with llama.cpp itself.

use std::cmp::Ordering;

use crate::config::Config;
use crate::mat::Mat;
use crate::ops::{gemv_f32, silu_mul_f32};

/// The smallest positive normal f16. `build_moe_ffn` clamps the router-weight
/// sum here before dividing (`llama-graph.cpp`, `ggml_clamp(..., 6.103515625e-5,
/// INFINITY)`).
pub const WEIGHTS_SUM_MIN: f32 = 6.103515625e-5;

/// A dense SwiGLU FFN: `down(silu(gate·x) * (up·x))`.
pub struct DenseFfn {
    pub gate: Mat,
    pub up: Mat,
    pub down: Mat,
}

impl DenseFfn {
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let gate = self.gate.gemv(x);
        let up = self.up.gemv(x);
        let act = silu_mul_f32(&gate, &up);
        self.down.gemv(&act)
    }
}

/// The routed-expert tensors and the shared expert of one layer.
///
/// Each `*_exps` tensor is one matrix with the experts stacked along the row
/// axis: a GGML `[n_embd, n_ff_exp, n_expert]` is `n_expert * n_ff_exp` rows of
/// `n_embd` values, and `[n_ff_exp, n_embd, n_expert]` is `n_expert * n_embd`
/// rows of `n_ff_exp`. Expert `e` is therefore a row range, which is what
/// [`Mat::gemv_range`] indexes.
pub struct MoeW {
    /// `ffn_gate_inp`: `n_expert` rows of `n_embd`.
    pub router: Mat,
    /// `ffn_gate_inp_shexp`: one vector, `n_embd` values.
    pub gate_inp_shexp: Vec<f32>,
    /// `ffn_gate_exps`: `n_expert * n_ff_exp` rows of `n_embd`.
    pub gate_exps: Mat,
    /// `ffn_up_exps`: `n_expert * n_ff_exp` rows of `n_embd`.
    pub up_exps: Mat,
    /// `ffn_down_exps`: `n_expert * n_embd` rows of `n_ff_exp`.
    pub down_exps: Mat,
    /// `ffn_{gate,up,down}_shexp`, when the file has a shared expert.
    pub shexp: Option<DenseFfn>,
}

/// Router probabilities: softmax over every expert, max-subtracted.
///
/// f32 `exp`, f64 sum, f32 reciprocal-scaled — `ggml_vec_soft_max_f32` and
/// `ggml_compute_forward_soft_max_f32` in the vendored `ggml-cpu-ops.cpp`.
pub fn softmax_experts(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
    let sum: f64 = probs.iter().map(|v| *v as f64).sum();
    let scale = (1.0 / sum) as f32;
    for p in &mut probs {
        *p *= scale;
    }
    probs
}

/// Top-k expert ids by **value descending, id ascending**, in rank order.
///
/// This is the one tie rule in the engine; the GPU kernel must use it too.
/// `partial_cmp` returning `None` (a NaN probability) is treated as a tie, so
/// the id still decides.
pub fn top_k(probs: &[f32], k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(Ordering::Equal)
            .then(a.cmp(&b))
    });
    order.truncate(k);
    order
}

/// The 8 renormalised router weights for `selected`, in rank order.
///
/// The sum is clamped at [`WEIGHTS_SUM_MIN`] before dividing; `w_scale` is
/// applied by the caller only when the file declares it, and this architecture
/// does not.
pub fn router_weights(probs: &[f32], selected: &[usize]) -> Vec<f32> {
    let mut sum = 0.0f64;
    for &e in selected {
        sum += probs[e] as f64;
    }
    let sum = (sum as f32).max(WEIGHTS_SUM_MIN);
    selected.iter().map(|&e| probs[e] / sum).collect()
}

/// The whole MoE FFN for one token: routed experts plus shared expert.
///
/// Returns `n_embd` values, the FFN block's output before the residual add.
pub fn moe_ffn_f32(x: &[f32], w: &MoeW, cfg: &Config) -> Vec<f32> {
    let n_embd = cfg.n_embd;
    let n_ff_exp = cfg.n_ff_exp;
    let k = cfg.n_expert_used;
    assert_eq!(x.len(), n_embd, "moe: x must be n_embd");
    assert!(k > 0 && k <= cfg.n_expert, "moe: bad expert_used_count");

    let probs = softmax_experts(&w.router.gemv(x));
    let selected = top_k(&probs, k);
    let weights = router_weights(&probs, &selected);

    // Left fold in rank order, starting from the first expert's output.
    let mut moe: Option<Vec<f32>> = None;
    for (rank, &e) in selected.iter().enumerate() {
        let gate = w.gate_exps.gemv_range(e * n_ff_exp, n_ff_exp, x);
        let up = w.up_exps.gemv_range(e * n_ff_exp, n_ff_exp, x);
        let act = silu_mul_f32(&gate, &up);
        let down = w.down_exps.gemv_range(e * n_embd, n_embd, &act);
        // The router weight applies after the down projection.
        let scaled: Vec<f32> = down.iter().map(|v| v * weights[rank]).collect();
        match moe.as_mut() {
            None => moe = Some(scaled),
            Some(acc) => {
                for (a, v) in acc.iter_mut().zip(&scaled) {
                    *a += v;
                }
            }
        }
    }
    let mut out = moe.unwrap_or_else(|| vec![0.0; n_embd]);

    if let Some(shexp) = &w.shexp {
        let shared = shexp.forward(x);
        // One dot product, one sigmoid, one scalar for the whole vector.
        let gate = gemv_f32(&w.gate_inp_shexp, x, 1, n_embd)[0];
        let scale = 1.0 / (1.0 + (-gate).exp());
        for (o, s) in out.iter_mut().zip(&shared) {
            *o += s * scale;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_breaks_ties_by_ascending_id() {
        // 2.0 at ids 1, 3 and 4; 1.0 at 0; 0.0 at 2.
        let probs = [1.0f32, 2.0, 0.0, 2.0, 2.0];
        assert_eq!(top_k(&probs, 3), vec![1, 3, 4]);
        assert_eq!(top_k(&probs, 5), vec![1, 3, 4, 0, 2]);
    }

    #[test]
    fn softmax_is_over_every_expert() {
        // Two equal logits out of four: each is 0.5 of the total, not 1/2 of a
        // selected pair.
        let p = softmax_experts(&[0.0, 0.0, -100.0, -100.0]);
        assert!((p[0] - 0.5).abs() < 1e-6 && (p[1] - 0.5).abs() < 1e-6);
        assert!((p.iter().map(|v| *v as f64).sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn router_weights_renormalise_and_clamp() {
        let probs = [0.5f32, 0.25, 0.25, 0.0];
        let w = router_weights(&probs, &[0, 1]);
        assert!((w[0] - 2.0 / 3.0).abs() < 1e-6 && (w[1] - 1.0 / 3.0).abs() < 1e-6);
        // An all-zero selection cannot divide by zero.
        let w = router_weights(&[0.0, 0.0], &[0, 1]);
        assert_eq!(w, vec![0.0, 0.0]);
    }
}
