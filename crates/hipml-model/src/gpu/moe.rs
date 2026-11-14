// SPDX-License-Identifier: Apache-2.0
// Portions of this file are derived from llama.cpp
// (https://github.com/ggml-org/llama.cpp, commit d7725699 unless noted),
// Copyright (c) the ggml authors, used under the MIT License.
// See NOTICE and docs/reference/LICENSE.
//! The mixture-of-experts FFN on the device.
//!
//! One layer's FFN is a router, a bank of 256 experts of which 8 run, and a
//! dense shared expert added outside the routed sum. The conventions are
//! `hipml_core::moe`'s and the kernel's (`moe.hip`); this file is the wiring.
//!
//! ```text
//!   logits  = ffn_gate_inp · t                    [n_expert]   F32 GEMV
//!   ids, w  = topk(softmax(logits), 8)            device only
//!   gate[j] = ffn_gate_exps[ids[j]] · t           [8][n_ff_exp]
//!   up[j]   = ffn_up_exps[ids[j]] · t             [8][n_ff_exp]
//!   act[j]  = silu(gate[j]) * up[j]
//!   down[j] = ffn_down_exps[ids[j]] · act[j]      [8][n_embd]
//!   out     = fold_rank(w[j] * down[j])           [n_embd]
//!   out    += shexp(t) * sigmoid(dot(t, ffn_gate_inp_shexp))
//! ```
//!
//! Every step is a device launch with no host readback: the ids are only ever
//! device values, which is what lets the whole token stay inside one captured
//! hipGraph. The expert gather is 8 of 256 slices per layer — 24 scattered
//! ~600 KB reads a layer, and the entire reason a 35B model can decode faster
//! than a 27B dense one.

use super::*;
use crate::gpu_weights::Experts;

/// One layer's router, expert bank, shared expert and the scratch a decode step
/// writes.
pub(crate) struct MoeBlock<'a> {
    /// `ffn_gate_inp`: `[n_expert, n_embd]`, one row per expert.
    router: Proj<'a>,
    /// `ffn_gate_inp_shexp`: `[n_embd]` as a one-row projection.
    gate_inp_shexp: Option<Proj<'a>>,
    /// `ffn_gate_exps`: `n_expert` slices of `[n_ff_exp, n_embd]`.
    gate_exps: Experts<'a>,
    /// `ffn_up_exps`: the same shape.
    up_exps: Experts<'a>,
    /// `ffn_down_exps`: `n_expert` slices of `[n_embd, n_ff_exp]`.
    down_exps: Experts<'a>,
    /// `ffn_{gate,up,down}_shexp`, three dense projections.
    shexp: Option<(Proj<'a>, Proj<'a>, Proj<'a>)>,
    /// `gemv_jobs` kind ids for the three banks.
    kind_gate: i32,
    kind_up: i32,
    kind_down: i32,
    n_expert: usize,
    n_used: usize,
    n_embd: usize,
    n_ff_exp: usize,
    n_ff_shexp: usize,
    /// Router logits, `[n_expert]`.
    logits: Buffer,
    /// Selected expert ids, `[n_used]` u32, written and read on the device.
    ids: Buffer,
    /// Renormalised router weights, `[n_used]` f32.
    weights: Buffer,
    /// Per-rank SwiGLU inputs and output: `[n_used][n_ff_exp]` twice, then the
    /// activation, then `[n_used][n_embd]` for the down projections.
    gate: Buffer,
    up: Buffer,
    act: Buffer,
    down: Buffer,
    /// The routed sum before the shared expert, `[n_embd]`.
    out: Buffer,
    /// Shared-expert scratch: `[n_ff_shexp]` twice, the activation, `[n_embd]`
    /// output, the `[1]` gate dot product and its sigmoid.
    shexp_gate: Buffer,
    shexp_up: Buffer,
    shexp_act: Buffer,
    shexp_out: Buffer,
    shexp_dot: Buffer,
    shexp_scale: Buffer,
}

impl<'a> MoeBlock<'a> {
    pub(crate) fn new(model: &'a Model, cfg: &Config, i: usize) -> Result<MoeBlock<'a>, String> {
        let name = |s: &str| format!("blk.{i}.{s}");
        let router = Proj::of(model, &name("ffn_gate_inp.weight"))?;
        if router.rows != cfg.n_expert || router.cols != cfg.n_embd {
            return Err(format!(
                "blk.{i}.ffn_gate_inp.weight: expected [{}x{}] (experts x n_embd), \
                 got [{}x{}]",
                cfg.n_expert, cfg.n_embd, router.rows, router.cols
            ));
        }
        let gate_exps = Experts::of(model, &name("ffn_gate_exps.weight"))?;
        let up_exps = Experts::of(model, &name("ffn_up_exps.weight"))?;
        let down_exps = Experts::of(model, &name("ffn_down_exps.weight"))?;
        for (label, e, rows, cols) in [
            ("ffn_gate_exps", &gate_exps, cfg.n_ff_exp, cfg.n_embd),
            ("ffn_up_exps", &up_exps, cfg.n_ff_exp, cfg.n_embd),
            ("ffn_down_exps", &down_exps, cfg.n_embd, cfg.n_ff_exp),
        ] {
            if e.n_expert != cfg.n_expert || e.rows != rows || e.cols != cols {
                return Err(format!(
                    "blk.{i}.{label}.weight: expected {} x [{rows}x{cols}], got {} x [{}x{}]",
                    cfg.n_expert, e.n_expert, e.rows, e.cols
                ));
            }
        }
        let shexp = if cfg.has_shexp {
            let g = Proj::of(model, &name("ffn_gate_shexp.weight"))?;
            let u = Proj::of(model, &name("ffn_up_shexp.weight"))?;
            let d = Proj::of(model, &name("ffn_down_shexp.weight"))?;
            if g.rows != cfg.n_ff_shexp || g.cols != cfg.n_embd {
                return Err(format!(
                    "blk.{i}.ffn_gate_shexp.weight: expected [{}x{}], got [{}x{}]",
                    cfg.n_ff_shexp, cfg.n_embd, g.rows, g.cols
                ));
            }
            if d.rows != cfg.n_embd || d.cols != cfg.n_ff_shexp {
                return Err(format!(
                    "blk.{i}.ffn_down_shexp.weight: expected [{}x{}], got [{}x{}]",
                    cfg.n_embd, cfg.n_ff_shexp, d.rows, d.cols
                ));
            }
            Some((g, u, d))
        } else {
            None
        };
        let gate_inp_shexp = match cfg.has_shexp {
            true => Some(Proj::vector(model, &name("ffn_gate_inp_shexp.weight"))?),
            false => None,
        };
        let kind = |e: &Experts<'_>, label: &str| {
            hipml_kernels::gemv_job_kind(e.entry())
                .ok_or_else(|| format!("blk.{i}.{label}: {} has no expert GEMV", e.entry()))
        };
        let kind_gate = kind(&gate_exps, "ffn_gate_exps")?;
        let kind_up = kind(&up_exps, "ffn_up_exps")?;
        let kind_down = kind(&down_exps, "ffn_down_exps")?;

        let zeros = |n: usize| Buffer::alloc(n);
        let n_used = cfg.n_expert_used;
        Ok(MoeBlock {
            router,
            gate_inp_shexp,
            gate_exps,
            up_exps,
            down_exps,
            shexp,
            kind_gate,
            kind_up,
            kind_down,
            n_expert: cfg.n_expert,
            n_used,
            n_embd: cfg.n_embd,
            n_ff_exp: cfg.n_ff_exp,
            n_ff_shexp: cfg.n_ff_shexp,
            logits: zeros(cfg.n_expert)?,
            ids: zeros(n_used)?,
            weights: zeros(n_used)?,
            gate: zeros(n_used * cfg.n_ff_exp)?,
            up: zeros(n_used * cfg.n_ff_exp)?,
            act: zeros(n_used * cfg.n_ff_exp)?,
            down: zeros(n_used * cfg.n_embd)?,
            out: zeros(cfg.n_embd)?,
            shexp_gate: zeros(cfg.n_ff_shexp.max(1))?,
            shexp_up: zeros(cfg.n_ff_shexp.max(1))?,
            shexp_act: zeros(cfg.n_ff_shexp.max(1))?,
            shexp_out: zeros(cfg.n_embd)?,
            shexp_dot: zeros(1)?,
            shexp_scale: zeros(1)?,
        })
    }

    /// Weight bytes this layer streams per token: the router, the 8 experts the
    /// token is routed to (not all 256), and the shared expert.
    pub(crate) fn bytes(&self) -> usize {
        // `Experts::stride` is one expert's byte length, so the routed gather
        // streams exactly `n_used` of them per bank.
        let per_token_expert = |e: &Experts<'_>| e.stride * self.n_used;
        let mut b = self.router.bytes()
            + per_token_expert(&self.gate_exps)
            + per_token_expert(&self.up_exps)
            + per_token_expert(&self.down_exps);
        if let Some((g, u, d)) = &self.shexp {
            b += g.bytes() + u.bytes() + d.bytes();
        }
        if let Some(g) = &self.gate_inp_shexp {
            b += g.bytes();
        }
        b
    }

    /// One token through the FFN, writing the block output into [`Self::out`].
    pub(crate) fn forward(&self, l: &Launcher, x: &Buffer) -> Result<(), String> {
        // Router: one GEMV to n_expert logits, then softmax + top-k on device.
        self.router.gemv(l, x, &self.logits)?;
        l.moe_topk(
            &self.logits,
            self.n_expert,
            self.n_used,
            &self.ids,
            &self.weights,
        )?;

        // The gather. `x_stride` is 0: every rank reads the same activation.
        let g = &self.gate_exps;
        l.moe_gemv(
            g.base(),
            &self.ids,
            g.stride,
            x,
            0,
            &self.gate,
            self.n_ff_exp,
            self.n_embd,
            self.kind_gate,
            self.n_used,
        )?;
        let u = &self.up_exps;
        l.moe_gemv(
            u.base(),
            &self.ids,
            u.stride,
            x,
            0,
            &self.up,
            self.n_ff_exp,
            self.n_embd,
            self.kind_up,
            self.n_used,
        )?;
        l.silu_mul_buf(&self.gate, &self.up, &self.act, self.n_used * self.n_ff_exp)?;

        // Down: rank j reads its own expert's activation row.
        let d = &self.down_exps;
        l.moe_gemv(
            d.base(),
            &self.ids,
            d.stride,
            &self.act,
            self.n_ff_exp,
            &self.down,
            self.n_embd,
            self.n_ff_exp,
            self.kind_down,
            self.n_used,
        )?;
        // The weight applies after the down projection, and the sum is a left
        // fold in rank order.
        l.moe_combine(
            &self.out,
            &self.down,
            &self.weights,
            self.n_embd,
            self.n_used,
        )?;

        // The shared expert: a dense SwiGLU scaled by one sigmoid scalar, added
        // outside the routed sum.
        if let (Some((sg, su, sd)), Some(gate_in)) = (&self.shexp, &self.gate_inp_shexp) {
            sg.gemv(l, x, &self.shexp_gate)?;
            su.gemv(l, x, &self.shexp_up)?;
            l.silu_mul_buf(
                &self.shexp_gate,
                &self.shexp_up,
                &self.shexp_act,
                self.n_ff_shexp,
            )?;
            sd.gemv(l, &self.shexp_act, &self.shexp_out)?;
            gate_in.gemv(l, x, &self.shexp_dot)?;
            l.sigmoid_buf(&self.shexp_dot, &self.shexp_scale, 1)?;
            l.moe_shared_add(&self.out, &self.shexp_out, &self.shexp_scale, self.n_embd)?;
        }
        Ok(())
    }

    /// The FFN output of the last step (`[n_embd]`).
    pub(crate) fn out(&self) -> &Buffer {
        &self.out
    }

    /// The router's decision of the last step, read back to the host.
    ///
    /// This is a synchronising readback, so it belongs to the routing gate and
    /// nowhere else: on the decode path the ids never leave the device, which
    /// is what keeps the token capturable into one hipGraph.
    pub fn route_host(&self) -> Result<(Vec<u32>, Vec<f32>), String> {
        let raw = self.ids.download_bytes(0, self.n_used * 4)?;
        let ids = raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok((ids, self.weights.to_host()?))
    }

    /// Zero the step scratch, so an ablated cell reads zeros and not the
    /// previous token's state.
    pub(crate) fn zero_scratch(&self) -> Result<(), String> {
        for b in [
            &self.logits,
            &self.ids,
            &self.weights,
            &self.gate,
            &self.up,
            &self.act,
            &self.down,
            &self.out,
            &self.shexp_gate,
            &self.shexp_up,
            &self.shexp_act,
            &self.shexp_out,
            &self.shexp_dot,
            &self.shexp_scale,
        ] {
            b.zero()?;
        }
        Ok(())
    }
}
