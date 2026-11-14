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

use super::prefill::{gemm_v2, Chunk};

/// One chunk's bucketed-path scratch: the router's logits and decisions, the
/// counting sort's output, and the slot-major tiles the grouped GEMM writes.
///
/// Slot-major means `[n_tokens * n_used][...]`, i.e. one row per (token, rank)
/// decision in top-k rank order — llama.cpp's `[n_embd, n_expert_used,
/// n_tokens]` layout. The gate and up projections write it in that order and
/// the down projection scatters into it, which is what lets the combine be the
/// same left fold the decode path does.
pub(crate) struct MoeChunk {
    /// Token capacity this scratch was sized for.
    pub cap: usize,
    /// Router logits, `[cap][n_expert]`.
    pub logits: Buffer,
    /// Selected expert ids and renormalised weights, `[cap][n_used]`.
    pub ids: Buffer,
    pub weights: Buffer,
    /// Per-expert run offsets, `[n_expert + 1]` u32.
    pub off: Buffer,
    /// Slot -> activation row (the token), and slot -> output row (its own
    /// token-major index). Both `[cap * n_used]` u32.
    pub xtok: Buffer,
    pub yslot: Buffer,
    /// `0..cap*n_used`: the identity map, for the tiles that are already in
    /// slot order (the gate/up output and the down input).
    pub ident: Buffer,
    /// `up`'s SwiGLU inputs and output, `[cap * n_used][n_ff_exp]`; `acth` is
    /// the fp16 copy the down projection reads.
    pub gate: Buffer,
    pub act: Buffer,
    pub acth: Buffer,
    /// Down projection output, `[cap * n_used][n_embd]`.
    pub down: Buffer,
    /// Shared expert, `[cap][n_ff_shexp]` twice, its activation, the fp16 copy,
    /// the `[cap][n_embd]` output, and the per-token gate dot and its sigmoid.
    pub shexp_gate: Buffer,
    pub shexp_up: Buffer,
    pub shexp_act: Buffer,
    pub shexp_acth: Buffer,
    pub shexp_out: Buffer,
    pub shexp_dot: Buffer,
    pub shexp_scale: Buffer,
}

impl MoeChunk {
    pub(super) fn new(cfg: &hipml_core::config::Config, cap: usize) -> Result<MoeChunk, String> {
        let nslot = cap * cfg.n_expert_used;
        let nff = cfg.n_ff_exp;
        let shexp = cfg.n_ff_shexp.max(1);
        let zeros = |n: usize| Buffer::alloc(n);
        let ident = zeros(nslot)?;
        let bytes: Vec<u8> = (0..nslot as u32).flat_map(|v| v.to_le_bytes()).collect();
        ident.upload_bytes(0, &bytes)?;
        Ok(MoeChunk {
            cap,
            logits: zeros(cap * cfg.n_expert)?,
            ids: zeros(nslot)?,
            weights: zeros(nslot)?,
            off: zeros(cfg.n_expert + 1)?,
            xtok: zeros(nslot)?,
            yslot: zeros(nslot)?,
            ident,
            gate: zeros(nslot * nff)?,
            act: zeros(nslot * nff)?,
            acth: zeros((nslot * nff + 1) / 2)?,
            down: zeros(nslot * cfg.n_embd)?,
            shexp_gate: zeros(cap * shexp)?,
            shexp_up: zeros(cap * shexp)?,
            shexp_act: zeros(cap * shexp)?,
            shexp_acth: zeros((cap * shexp + 1) / 2)?,
            shexp_out: zeros(cap * cfg.n_embd)?,
            shexp_dot: zeros(cap)?,
            shexp_scale: zeros(cap)?,
        })
    }

    pub(crate) fn bytes(&self) -> usize {
        [
            &self.logits,
            &self.ids,
            &self.weights,
            &self.off,
            &self.xtok,
            &self.yslot,
            &self.ident,
            &self.gate,
            &self.act,
            &self.acth,
            &self.down,
            &self.shexp_gate,
            &self.shexp_up,
            &self.shexp_act,
            &self.shexp_acth,
            &self.shexp_out,
            &self.shexp_dot,
            &self.shexp_scale,
        ]
        .iter()
        .map(|b| b.len())
        .sum::<usize>()
            * 4
    }

    /// Per-expert slot counts of the last routed tile, read back to the host.
    /// The routing diagnostic and the note's bucketing table; nothing on the
    /// forward path calls it.
    pub fn run_lengths(
        &self,
        n_expert: usize,
        n_tokens: usize,
        n_used: usize,
    ) -> Result<Vec<u32>, String> {
        let raw = self.off.download_bytes(0, (n_expert + 1) * 4)?;
        let off: Vec<u32> = raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let total = (n_tokens * n_used) as u32;
        if off[n_expert] != total {
            return Err(format!(
                "moe bucket: offsets end at {} but the tile has {total} slots",
                off[n_expert]
            ));
        }
        Ok((0..n_expert).map(|e| off[e + 1] - off[e]).collect())
    }
}

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
    /// `prefill_moe.hip`'s decode kinds (`pf_dequant8`'s numbering) for the
    /// same three banks: the grouped GEMM of a prefill chunk.
    pf_gate: i32,
    pf_up: i32,
    pf_down: i32,
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
        let pf = |e: &Experts<'_>, label: &str| {
            hipml_kernels::moe_prefill_kind(e.entry()).ok_or_else(|| {
                format!("blk.{i}.{label}: {} has no grouped prefill GEMM", e.entry())
            })
        };
        let pf_gate = pf(&gate_exps, "ffn_gate_exps")?;
        let pf_up = pf(&up_exps, "ffn_up_exps")?;
        let pf_down = pf(&down_exps, "ffn_down_exps")?;

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
            pf_gate,
            pf_up,
            pf_down,
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
    ///
    /// `drop` is the decode ablation set. Each group below is one launch site
    /// or a short run of them, so `decode-ablate` can remove a group and time
    /// the rest untraced. Nothing on the shipped path drops anything.
    pub(crate) fn forward(&self, l: &Launcher, x: &Buffer, drop: Ablate) -> Result<(), String> {
        let skip = |s: Stage| drop.dropped(s);

        // Router: one GEMV to n_expert logits, then softmax + top-k on device.
        if !skip(Stage::MoeRouter) {
            self.router.gemv(l, x, &self.logits)?;
        }
        if !skip(Stage::MoeTopk) {
            l.moe_topk(
                &self.logits,
                self.n_expert,
                self.n_used,
                &self.ids,
                &self.weights,
            )?;
        }

        // The gather. `x_stride` is 0: every rank reads the same activation.
        if !skip(Stage::MoeBank) {
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
        }
        if !skip(Stage::MoeAct) {
            l.silu_mul_buf(&self.gate, &self.up, &self.act, self.n_used * self.n_ff_exp)?;
        }

        // Down: rank j reads its own expert's activation row.
        if !skip(Stage::MoeBank) {
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
            if drop.doubled(Stage::MoeBank) {
                // The same launch again at the same site: its marginal cost is
                // what one more 4.7 MB bank read costs where it runs, which is
                // the launch-price control for a routed FFN.
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
            }
        }
        // The weight applies after the down projection, and the sum is a left
        // fold in rank order.
        if !skip(Stage::MoeCombine) {
            l.moe_combine(
                &self.out,
                &self.down,
                &self.weights,
                self.n_embd,
                self.n_used,
            )?;
        }

        // The shared expert: a dense SwiGLU scaled by one sigmoid scalar, added
        // outside the routed sum.
        if !skip(Stage::MoeShared) {
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
        }
        Ok(())
    }

    /// One prompt tile through the FFN as batched GEMMs: route every token,
    /// bucket the slots by expert, and read each expert once.
    ///
    /// The output lands in `c.down` (`[n_tokens][n_embd]`), which is where the
    /// dense path's `down` projection writes, so the residual that follows is
    /// the same one.
    ///
    /// The conventions are [`MoeBlock::forward`]'s — `moe_topk_rows_f32` is the
    /// same selection body — and the arithmetic is llama.cpp's
    /// `build_moe_ffn` over a tile instead of a token: one grouped GEMM per
    /// bank, the weight applied after the down projection, a left fold in rank
    /// order, and the shared expert added outside the routed sum.
    pub(crate) fn forward_chunk(
        &self,
        l: &Launcher,
        c: &Chunk,
        n_tok: usize,
    ) -> Result<(), String> {
        let mc = c
            .moe
            .as_ref()
            .ok_or_else(|| "moe chunk scratch not allocated".to_string())?;
        if mc.cap < n_tok {
            return Err(format!(
                "moe chunk of {n_tok} tokens exceeds scratch {}",
                mc.cap
            ));
        }
        let n = self.n_embd;
        let nff = self.n_ff_exp;

        // One fp16 copy of the block input: every projection below reads it,
        // and the grouped GEMM stages it a row at a time.
        ::hipml_kernels::prefill_xhalf(l.module(), l.stream(), &c.t, &c.xh, n_tok * n)?;

        // 1. Route the whole tile: `[n_tokens][n_expert]` logits, then the
        //    same softmax / top-k / renormalise the decode step runs.
        self.router.gemm(l, &c.t, &mc.logits, n_tok)?;
        l.moe_topk_rows(
            &mc.logits,
            self.n_expert,
            self.n_used,
            n_tok,
            &mc.ids,
            &mc.weights,
        )?;

        // 2. Counting sort: `off[e]..off[e+1]` is expert e's run of slots.
        l.moe_bucket(
            &mc.ids,
            n_tok,
            self.n_used,
            self.n_expert,
            &mc.off,
            &mc.xtok,
            &mc.yslot,
        )?;

        // 3. The routed bank, one launch for all 256 experts. The gate tile is
        //    slot-major; `up` applies the SwiGLU in its epilogue and writes the
        //    fp16 activation the down projection reads.
        l.moe_prefill_gemm(
            self.gate_exps.base(),
            self.gate_exps.stride,
            &mc.off,
            &c.xh,
            &mc.xtok,
            &mc.gate,
            &mc.ident,
            &mc.gate,
            &mc.acth,
            nff,
            n,
            self.n_expert,
            self.pf_gate,
            false,
        )?;
        l.moe_prefill_gemm(
            self.up_exps.base(),
            self.up_exps.stride,
            &mc.off,
            &c.xh,
            &mc.xtok,
            &mc.act,
            &mc.ident,
            &mc.gate,
            &mc.acth,
            nff,
            n,
            self.n_expert,
            self.pf_up,
            true,
        )?;
        // 4. Down, scattered into the token-major slot order the fold reads.
        l.moe_prefill_gemm(
            self.down_exps.base(),
            self.down_exps.stride,
            &mc.off,
            &mc.acth,
            &mc.ident,
            &mc.down,
            &mc.yslot,
            &mc.gate,
            &mc.acth,
            n,
            nff,
            self.n_expert,
            self.pf_down,
            false,
        )?;

        // 5. `out[t] = fold_rank(w[t][j] * down[t][j])`.
        l.moe_combine_rows(&c.down, &mc.down, &mc.weights, n, self.n_used, n_tok)?;

        // 6. The shared expert: a dense SwiGLU over the whole tile, scaled by
        //    one sigmoid per token and added outside the routed sum.
        if let (Some((sg, su, sd)), Some(gate_in)) = (&self.shexp, &self.gate_inp_shexp) {
            let nsh = self.n_ff_shexp;
            let sk = &c.sk;
            // `up`'s fused epilogue reads the gate tile, so the gate runs
            // first either way.
            gemm_v2(l, sg, &c.t, &c.xh, &mc.shexp_gate, sk, n_tok)?;
            let fused = su.gemm_half_silu(
                l,
                &c.xh,
                &mc.shexp_gate,
                &mc.shexp_act,
                &mc.shexp_acth,
                n_tok,
            )?;
            if !fused {
                gemm_v2(l, su, &c.t, &c.xh, &mc.shexp_up, sk, n_tok)?;
                l.silu_mul_buf(&mc.shexp_gate, &mc.shexp_up, &mc.shexp_act, n_tok * nsh)?;
                ::hipml_kernels::prefill_xhalf(
                    l.module(),
                    l.stream(),
                    &mc.shexp_act,
                    &mc.shexp_acth,
                    n_tok * nsh,
                )?;
            }
            gemm_v2(
                l,
                sd,
                &mc.shexp_act,
                &mc.shexp_acth,
                &mc.shexp_out,
                sk,
                n_tok,
            )?;
            hipml_kernels::gemm_rows_f32(
                l.module(),
                l.stream(),
                gate_in.ptr() as *mut std::ffi::c_void,
                c.t.as_ptr(),
                mc.shexp_dot.as_ptr(),
                n_tok,
                1,
                n,
            )?;
            l.sigmoid_buf(&mc.shexp_dot, &mc.shexp_scale, n_tok)?;
            l.moe_shared_add_rows(&c.down, &mc.shexp_out, &mc.shexp_scale, n, n_tok)?;
        }

        // The routing diagnostic (`moe-gate --routes`) reads the decode step's
        // `ids`/`weights`, which this path never writes. Publish the tile's last
        // token — the prompt's last token, which is the step the gate asks
        // about — into them, so the per-layer diff keeps its meaning. Two
        // 8-word device copies per layer per chunk; the next decode step
        // overwrites them anyway.
        mc.ids.copy_range_on(
            &self.ids,
            (n_tok - 1) * self.n_used,
            0,
            self.n_used,
            l.stream(),
        )?;
        mc.weights.copy_range_on(
            &self.weights,
            (n_tok - 1) * self.n_used,
            0,
            self.n_used,
            l.stream(),
        )?;
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
