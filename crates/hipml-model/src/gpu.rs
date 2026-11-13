// SPDX-License-Identifier: Apache-2.0
// Portions of this file are derived from llama.cpp
// (https://github.com/ggml-org/llama.cpp, tag b12126 unless noted),
// Copyright (c) the ggml authors, used under the MIT License.
// See NOTICE and docs/reference/LICENSE.
//! One GPU graph for the qwen35 hybrid decoder.
//!
//! ```text
//!   x = embed(token)
//!   for layer:
//!       h = rmsnorm(x, attn_norm)
//!       x = x + block(h)                  # full attention (3,7,..63) or GDN
//!       t = rmsnorm(x, post_attention_norm)
//!       x = x + down(silu(gate(t)) * up(t))
//!   logits = output(rmsnorm(x, output_norm))
//! ```
//!
//! Every weight matvec goes through the quantized GEMV over the loader's
//! resident `DeviceTensor`s: nothing is dequantized on the host and no weight
//! is uploaded twice. Launches are chained on one stream and the token is
//! synchronized once, before the sampled id comes back.
//!
//! Nothing on the decode path round-trips to the host: the embedding row is
//! dequantized into `x` on the device and the 248,320-way argmax runs on the
//! device, returning one u32. `prefill.rs` adds the chunked prompt path.
//!
//! The full-attention block borrows the resident projections through
//! `AttnLayer::new_resident`; the SSM block borrows them through
//! `SsmGpuLayer::new_resident`, so both reuse the same kernel families WP11 and
//! WP12 validated.

use hipml_core::config::Config;
use hipml_core::gguf::Gguf;
use hipml_core::kv::{self, KvDtype};
use hipml_core::mat::Kind;
use hipml_hip::{Buffer, ByteBuffer};
use hipml_kernels::launch_kv;
use hipml_kernels::ssm::{SsmDims, SsmGpuLayer, SsmResidentRefs};
use hipml_kernels::{AttnLayer, Launcher};

use crate::gpu_weights::{gemv_entry, mat_kind, Proj};
use crate::Model;

#[path = "gpu/ablate.rs"]
mod ablate;
#[path = "gpu/attn_tile.rs"]
mod attn_tile;
#[path = "gpu/deep_kv.rs"]
mod deep_kv;
#[path = "gpu/layers.rs"]
mod layers;
/// Chunked prefill (a prompt tile as batched GEMMs). A child module so it can
/// reach the engine's private scratch and layer list.
#[path = "gpu/prefill.rs"]
mod prefill;
#[path = "gpu/trace.rs"]
mod trace;

#[path = "gpu/graph.rs"]
mod graph;
pub use ablate::{Ablate, Stage};
pub use graph::{launches_per_token, GraphTiming};
#[path = "gpu/batch.rs"]
mod batch;
pub use batch::{BatchEngine, MtpBatch, SlotPool, VerifySnap};
#[path = "gpu/columns.rs"]
mod columns;
#[path = "gpu/mtp.rs"]
mod mtp;
#[path = "gpu/prefix.rs"]
mod prefix;
#[path = "gpu/spec.rs"]
mod spec;

use layers::{build_attn, build_ssm};
use trace::mark;

pub use columns::Snap;
pub use mtp::MtpGpu;
pub use prefill::PrefillReport;
pub use prefix::PrefixHit;
pub use spec::{SpecRun, SpecStats, SpecStep};

pub use deep_kv::DeepKv;
pub use trace::Trace;

struct Block<'a> {
    attn_norm: Buffer,
    post_norm: Buffer,
    gate: Proj<'a>,
    up: Proj<'a>,
    down: Proj<'a>,
    kind: BlockKind<'a>,
}

/// Attention-cache choice. `F32` is the short-depth oracle path; `Q8`/`Q4` use
/// the quantized-KV decode pair from `feat/kv-100k`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum KvMode {
    F32,
    Q8,
    Q4,
}

impl KvMode {
    pub fn parse(s: &str) -> Result<KvMode, String> {
        match s {
            "f32" => Ok(KvMode::F32),
            "q8" => Ok(KvMode::Q8),
            "q4" => Ok(KvMode::Q4),
            other => Err(format!("unknown --kv {other} (use f32, q8 or q4)")),
        }
    }

    fn dtype(self) -> Option<KvDtype> {
        match self {
            KvMode::F32 => None,
            KvMode::Q8 => Some(KvDtype::Q8),
            KvMode::Q4 => Some(KvDtype::Q4),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            KvMode::F32 => "f32",
            KvMode::Q8 => "q8",
            KvMode::Q4 => "q4",
        }
    }
}

enum BlockKind<'a> {
    Full {
        attn: AttnLayer<'a>,
        deep: Option<DeepKv>,
    },
    Ssm(SsmGpuLayer<'a>),
}

/// Per-sequence state: position and the per-layer caches held by the blocks.
pub struct GpuState {
    pub pos: usize,
}

pub struct GpuEngine<'a> {
    model: &'a Model,
    pub cfg: Config,
    /// Positions the caches were sized for.
    pub n_ctx: usize,
    pub kv_mode: KvMode,
    layers: Vec<Block<'a>>,
    output_norm: Buffer,
    x: Buffer,
    x1: Buffer,
    t: Buffer,
    gate: Buffer,
    up: Buffer,
    act: Buffer,
    down: Buffer,
    final_h: Buffer,
    logits: Buffer,
    /// The embedding tensor's resident payload, its row stride and its block
    /// format, so a token id becomes a device row lookup instead of a host
    /// round-trip.
    embed: (&'a ByteBuffer, usize, hipml_core::quant::TensorType),
    /// Argmax scratch: `parts` partial values, `parts` partial indices, one out.
    arg_v: Buffer,
    arg_i: Buffer,
    arg_out: Buffer,
    /// Chunked-prefill scratch, allocated on first use and reused after.
    chunk: Option<(usize, prefill::Chunk)>,
    /// The captured token graph, built on first use: one `hipGraphLaunch`
    /// submits the whole step instead of ~1,300 kernel launches.
    graph: Option<hipml_hip::Graph>,
    /// Stages to drop, for `decode-ablate`. Empty on the shipped path.
    ablate: Ablate,
    /// Stage timing; `None` unless the caller asked for a trace.
    pub trace: Option<Trace>,
    /// Prefix snapshots (SSM recurrent state + f32 KV rows), when enabled.
    prefix: Option<hipml_core::prefix_cache::PrefixCache<prefix::Snapshot>>,
    /// R-column verify scratch, allocated on first use and reused after.
    cols: Option<columns::Cols>,
}

/// Blocks the device argmax reduces over before the one-block fold. 64 blocks
/// is one wave on this card and turns the 993 KB logit pass into a single sweep.
const ARGMAX_PARTS: usize = 64;

/// Quantized-KV engines carry no f32 prefill mirror.
///
/// The mirror existed so chunked prefill could read f32 KV tiles. The
/// matrix-unit kernel now stages a q8_0/q4_0 tile in place, and that is no
/// slower than the mirror it replaces: 8K prefills at 998 tok/s (993 through
/// the mirror, which reads 4x the bytes) and 32K is level (867 vs 871). A 32K
/// mirror is 4.3 GiB and 100K's would be 13.4 GiB, so the engine never
/// allocates one; `--kv f32` keeps the f32 cache and the token loop, which is
/// the parity reference.
const MIRROR_ROWS: usize = 1;

impl<'a> GpuEngine<'a> {
    /// Build every resident layer, borrowing the loader's weight payloads.
    pub fn new(model: &'a Model, n_ctx: usize, kv_mode: KvMode) -> Result<GpuEngine<'a>, String> {
        let cfg = {
            let mut g = Gguf::open(model.path()).map_err(|e| format!("{e}"))?;
            Config::from_gguf(&mut g).map_err(|e| format!("config: {e}"))?
        };
        let l = model.launcher();
        let n = cfg.n_embd;
        let embed = model
            .tensor(crate::TOKEN_EMBD)
            .ok_or_else(|| "model has no token_embd.weight".to_string())?;
        if embed.ty() != hipml_core::quant::TensorType::Q4K
            && embed.ty() != hipml_core::quant::TensorType::Q6K
        {
            return Err(format!(
                "{}: expected Q4_K or Q6_K, found {}",
                crate::TOKEN_EMBD,
                embed.ty().name()
            ));
        }
        let embed = (
            embed.buffer(),
            (embed.nbytes() / embed.rows()) as usize,
            embed.ty(),
        );

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let attn_norm =
                Buffer::from_host(&model.f32_vec(&format!("blk.{i}.attn_norm.weight"))?)?;
            let post_norm =
                Buffer::from_host(&model.f32_vec(&format!("blk.{i}.post_attention_norm.weight"))?)?;
            let gate = Proj::of(model, &format!("blk.{i}.ffn_gate.weight"))?;
            let up = Proj::of(model, &format!("blk.{i}.ffn_up.weight"))?;
            let down = Proj::of(model, &format!("blk.{i}.ffn_down.weight"))?;

            let kind = if cfg.is_recr[i] {
                BlockKind::Ssm(build_ssm(model, l, &cfg, i, false)?)
            } else if let Some(dtype) = kv_mode.dtype() {
                // No f32 mirror: prefill and decode share the quantized cache
                // (see MIRROR_ROWS).
                BlockKind::Full {
                    attn: build_attn(model, l, &cfg, i, MIRROR_ROWS)?,
                    deep: Some(DeepKv::new(&cfg, n_ctx, dtype)?),
                }
            } else {
                BlockKind::Full {
                    attn: build_attn(model, l, &cfg, i, n_ctx)?,
                    deep: None,
                }
            };
            layers.push(Block {
                attn_norm,
                post_norm,
                gate,
                up,
                down,
                kind,
            });
        }

        let output_norm = Buffer::from_host(&model.f32_vec("output_norm.weight")?)?;
        let zeros = |n: usize| Buffer::alloc(n);
        // `decode-ablate` seeds the set through the environment so a cell can be
        // reproduced in a fresh process; the shipped path leaves it unset.
        let ablate = match std::env::var("HIPML_ABLATE") {
            Ok(v) => Ablate::parse(&v)?,
            Err(_) => Ablate::none(),
        };
        Ok(GpuEngine {
            model,
            cfg: cfg.clone(),
            n_ctx,
            kv_mode,
            layers,
            output_norm,
            x: zeros(n)?,
            x1: zeros(n)?,
            t: zeros(n)?,
            gate: zeros(cfg.n_ff)?,
            up: zeros(cfg.n_ff)?,
            act: zeros(cfg.n_ff)?,
            down: zeros(n)?,
            final_h: zeros(n)?,
            logits: zeros(cfg.n_vocab)?,
            embed,
            arg_v: zeros(ARGMAX_PARTS)?,
            arg_i: zeros(ARGMAX_PARTS)?,
            arg_out: zeros(1)?,
            chunk: None,
            graph: None,
            ablate,
            trace: None,
            prefix: None,
            cols: None,
        })
    }

    pub fn new_state(&self) -> GpuState {
        GpuState { pos: 0 }
    }

    /// Zero every SSM layer's carried state (conv window + delta-net matrix).
    ///
    /// `GpuState` only holds the position; the recurrent buffers live in the
    /// layers and outlive it. A cold prefill on a reused engine must clear them
    /// or it starts from the previous request's summary. The KV rows do not
    /// need clearing: position `t` is appended before it is read.
    pub fn reset_state(&self) -> Result<(), String> {
        for b in &self.layers {
            if let BlockKind::Ssm(s) = &b.kind {
                s.conv_state_buf().zero()?;
                s.s_state_buf().zero()?;
            }
        }
        self.model.launcher().synchronize()
    }

    /// Bytes of resident weights actually streamed per token.
    pub fn weight_bytes(&self) -> usize {
        let mut b = 0usize;
        for l in &self.layers {
            b += l.gate.bytes() + l.up.bytes() + l.down.bytes();
            b += match &l.kind {
                BlockKind::Full { attn, .. } => attn.weight_bytes(),
                BlockKind::Ssm(s) => s.weight_bytes(),
            };
        }
        b + self
            .model
            .tensor(crate::OUTPUT)
            .map(|t| t.nbytes() as usize)
            .unwrap_or(0)
    }

    /// One token through the whole graph; returns the `n_vocab` logits.
    pub fn forward_token(&mut self, token: u32, st: &mut GpuState) -> Result<Vec<f32>, String> {
        self.run_token(token, st, true)?;
        self.model.launcher().synchronize()?;
        self.logits.to_host()
    }

    /// One token, greedy: the argmax runs on the device and only the id comes
    /// back. This is the decode loop; `forward_token` stays for the oracle
    /// comparison, which needs the whole logit vector.
    pub fn forward_token_id(&mut self, token: u32, st: &mut GpuState) -> Result<u32, String> {
        self.run_token(token, st, true)?;
        let l = self.model.launcher();
        if !self.ablate.dropped(Stage::Argmax) {
            l.argmax_buf(
                &self.logits,
                &self.arg_v,
                &self.arg_i,
                &self.arg_out,
                self.cfg.n_vocab,
                ARGMAX_PARTS,
            )?;
        }
        l.synchronize()?;
        let bytes = self.arg_out.download_bytes(0, 4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    /// Everything up to and including `lm_head`; no synchronize.
    ///
    /// `set_step` writes `[pos, seq, token]` into the launcher's device step
    /// buffer. It is false inside a capture: the graph reads that buffer at
    /// launch time, so the host writes it once per token instead and the captured
    /// launch has no host arguments left.
    ///
    /// Every stage below is skippable through `self.ablate`, which
    /// `decode-ablate` sets and nothing else does. A dropped stage issues no
    /// launch at all, so the difference between two runs is the stage's real
    /// in-situ cost and cannot include an instrument's.
    fn run_token(&mut self, token: u32, st: &mut GpuState, set_step: bool) -> Result<(), String> {
        let l = self.model.launcher();
        let cfg = &self.cfg;
        let drop = self.ablate;
        let pos = st.pos;
        if set_step {
            l.set_step(pos, pos + 1, token as usize)?;
        }

        if !drop.dropped(Stage::Embed) {
            mark(&mut self.trace, l, "embed")?;
            l.embed_row_dev(
                self.embed.2,
                self.embed.0,
                self.embed.1,
                &self.x,
                cfg.n_embd,
            )?;
        }

        for i in 0..layer_limit(self.layers.len()) {
            let block = &self.layers[i];
            let block_out: &Buffer = match &block.kind {
                BlockKind::Ssm(s) => {
                    if !drop.dropped(Stage::Norm) {
                        mark(&mut self.trace, l, "attn_norm")?;
                        l.rmsnorm_buf(&self.x, &block.attn_norm, s.input(), cfg.n_embd, cfg.eps)?;
                    }
                    if !drop.dropped(Stage::SsmGemv) {
                        mark(&mut self.trace, l, "ssm_proj")?;
                        s.gemvs()?;
                    }
                    if drop.splits_ssm_scan() {
                        mark(&mut self.trace, l, "ssm_step")?;
                        if !drop.dropped(Stage::SsmGates) {
                            s.step_gates()?;
                        }
                        if !drop.dropped(Stage::SsmConv) {
                            s.step_conv()?;
                        }
                        if !drop.dropped(Stage::SsmL2) {
                            s.step_l2()?;
                        }
                        if !drop.dropped(Stage::SsmRec) {
                            s.step_rec()?;
                        }
                        if !drop.dropped(Stage::SsmGateNorm) {
                            s.step_gate_norm()?;
                        }
                    } else {
                        mark(&mut self.trace, l, "ssm_step")?;
                        s.recurrent()?;
                    }
                    if drop.doubled(Stage::SsmRec) {
                        // The same launch again at the same site: its marginal
                        // cost is what the site costs, not what the kernel does.
                        s.step_rec()?;
                    }
                    if !drop.dropped(Stage::SsmGemv) {
                        mark(&mut self.trace, l, "ssm_out")?;
                        s.output_gemv()?;
                    }
                    s.output()
                }
                BlockKind::Full { attn, deep: None } => {
                    if !drop.dropped(Stage::Norm) {
                        mark(&mut self.trace, l, "attn_norm")?;
                        l.rmsnorm_buf(&self.x, &block.attn_norm, &self.t, cfg.n_embd, cfg.eps)?;
                    }
                    // The stage marks keep the names and boundaries the shipped
                    // path had, so `gpu-profile`'s table is comparable across
                    // trees: `attn_qkv` covers the projections and the front end
                    // together, `attn_wo` covers `q_gate` and the output
                    // projection together.
                    if !drop.dropped(Stage::AttnGemv) || !drop.dropped(Stage::AttnFront) {
                        mark(&mut self.trace, l, "attn_qkv")?;
                    }
                    if !drop.dropped(Stage::AttnGemv) {
                        attn.qkv_gemvs(l, &self.t)?;
                    }
                    if !drop.dropped(Stage::AttnFront) {
                        attn.front_only(l)?;
                    }
                    if !drop.dropped(Stage::KvAppend) {
                        mark(&mut self.trace, l, "attn_kv")?;
                        attn.append_at(l, attn.k_buf().as_ptr(), attn.v_buf().as_ptr())?;
                    }
                    if !drop.dropped(Stage::AttnReduce) {
                        mark(&mut self.trace, l, "attn_core")?;
                        attn.attention_at(l, attn.q_buf().as_ptr(), attn.attn_out().as_ptr(), pos)?;
                    }
                    if drop.doubled(Stage::AttnReduce) {
                        attn.attention_at(l, attn.q_buf().as_ptr(), attn.attn_out().as_ptr(), pos)?;
                    }
                    if !drop.dropped(Stage::QGate) || !drop.dropped(Stage::AttnGemv) {
                        mark(&mut self.trace, l, "attn_wo")?;
                    }
                    if !drop.dropped(Stage::QGate) {
                        attn.q_gate_only(l)?;
                    }
                    if !drop.dropped(Stage::AttnGemv) {
                        attn.out_gemv(l)?;
                    }
                    attn.output()
                }
                BlockKind::Full {
                    attn,
                    deep: Some(dk),
                } => {
                    if !drop.dropped(Stage::Norm) {
                        mark(&mut self.trace, l, "attn_norm")?;
                        l.rmsnorm_buf(&self.x, &block.attn_norm, &self.t, cfg.n_embd, cfg.eps)?;
                    }
                    if !drop.dropped(Stage::AttnGemv) || !drop.dropped(Stage::AttnFront) {
                        mark(&mut self.trace, l, "attn_qkv")?;
                    }
                    if !drop.dropped(Stage::AttnGemv) {
                        attn.qkv_gemvs(l, &self.t)?;
                    }
                    if !drop.dropped(Stage::AttnFront) {
                        attn.front_only(l)?;
                    }
                    if !drop.dropped(Stage::KvAppend) {
                        mark(&mut self.trace, l, "attn_kv")?;
                        // Quantize and append the new K/V row on the device: one
                        // launch instead of a readback and eight uploads.
                        l.kv_encode(
                            dk.dtype.id(),
                            attn.k_buf(),
                            attn.v_buf(),
                            &dk.k,
                            &dk.v,
                            cfg.n_head_kv,
                            cfg.head_dim,
                            dk.seq,
                        )?;
                    }
                    if !drop.dropped(Stage::AttnReduce) {
                        mark(&mut self.trace, l, "attn_core")?;
                        deep_attention(l, cfg, attn, dk, pos)?;
                    }
                    if drop.doubled(Stage::AttnReduce) {
                        deep_attention(l, cfg, attn, dk, pos)?;
                    }
                    if !drop.dropped(Stage::QGate) || !drop.dropped(Stage::AttnGemv) {
                        mark(&mut self.trace, l, "attn_wo")?;
                    }
                    if !drop.dropped(Stage::QGate) {
                        attn.q_gate_only(l)?;
                    }
                    if !drop.dropped(Stage::AttnGemv) {
                        attn.out_gemv(l)?;
                    }
                    attn.output()
                }
            };
            if !drop.dropped(Stage::Ew) {
                mark(&mut self.trace, l, "residual_in")?;
                l.residual_buf(&self.x, block_out, &self.x1, cfg.n_embd)?;
            }
            if !drop.dropped(Stage::Norm) {
                mark(&mut self.trace, l, "post_norm")?;
                l.rmsnorm_buf(&self.x1, &block.post_norm, &self.t, cfg.n_embd, cfg.eps)?;
            }

            if !drop.dropped(Stage::FfnGemv) {
                mark(&mut self.trace, l, "ffn_gate")?;
                block.gate.gemv(l, &self.t, &self.gate)?;
                mark(&mut self.trace, l, "ffn_up")?;
                block.up.gemv(l, &self.t, &self.up)?;
            }
            if !drop.dropped(Stage::Ew) {
                mark(&mut self.trace, l, "ffn_act")?;
                l.silu_mul_buf(&self.gate, &self.up, &self.act, cfg.n_ff)?;
            }
            if !drop.dropped(Stage::FfnGemv) {
                mark(&mut self.trace, l, "ffn_down")?;
                block.down.gemv(l, &self.act, &self.down)?;
            }
            if !drop.dropped(Stage::Ew) {
                mark(&mut self.trace, l, "residual_out")?;
                l.residual_buf(&self.down, &self.x1, &self.x, cfg.n_embd)?;
            }
        }

        st.pos += 1;
        if !drop.dropped(Stage::Norm) {
            mark(&mut self.trace, l, "out_norm")?;
            l.rmsnorm_buf(
                &self.x,
                &self.output_norm,
                &self.final_h,
                cfg.n_embd,
                cfg.eps,
            )?;
        }
        if drop.dropped(Stage::LmHead) {
            return Ok(());
        }
        mark(&mut self.trace, l, "lm_head")?;
        let out = self
            .model
            .tensor(crate::OUTPUT)
            .ok_or_else(|| "model has no output.weight".to_string())?;
        l.run_quant_resident(
            gemv_entry(out.ty())?,
            out.buffer(),
            &self.final_h,
            &self.logits,
            cfg.n_vocab,
            cfg.n_embd,
        )
    }

    /// The stages `decode-ablate` is currently dropping (empty on the shipped
    /// path).
    pub fn ablate(&self) -> Ablate {
        self.ablate
    }

    /// Change the dropped set. Only `decode-ablate` calls this; a graph built
    /// under one set replays that set, so build it after the last change.
    pub fn set_ablate(&mut self, a: Ablate) {
        self.ablate = a;
    }

    /// Zero the decode scratch, so an ablated cell reads zeros rather than
    /// whatever the previous cell left in an uninitialized buffer.
    ///
    /// This matters for the measurement, not just for tidiness: a dropped norm
    /// leaves its consumer reading the same buffer, and a buffer that is
    /// uninitialized can hold subnormals, which this card executes an order of
    /// magnitude slower than zeros. `decode-ablate` calls this once per cell.
    pub fn reset_scratch(&self) -> Result<(), String> {
        for b in [
            &self.x,
            &self.x1,
            &self.t,
            &self.gate,
            &self.up,
            &self.act,
            &self.down,
            &self.final_h,
            &self.logits,
            &self.arg_v,
            &self.arg_i,
            &self.arg_out,
        ] {
            b.zero()?;
        }
        for b in &self.layers {
            match &b.kind {
                BlockKind::Full { attn, .. } => attn.zero_activations()?,
                BlockKind::Ssm(s) => s.zero_activations()?,
            }
        }
        self.model.launcher().synchronize()
    }

    pub fn model(&self) -> &Model {
        self.model
    }

    /// The logits row of the last plain decode step (`[n_vocab]`).
    pub fn logits(&self) -> &Buffer {
        &self.logits
    }

    /// Vocabulary width (`n_vocab`), the row width of `logits`.
    pub fn n_vocab(&self) -> usize {
        self.cfg.n_vocab
    }

    /// Hidden width (`n_embd`), the row width of `final_hidden`.
    pub fn n_embd(&self) -> usize {
        self.cfg.n_embd
    }

    /// The residual stream after the last layer of the last plain decode step.
    pub fn residual_x(&self) -> &Buffer {
        &self.x
    }

    /// The post-`output_norm` hidden row of the last plain decode step.
    pub fn final_hidden(&self) -> &Buffer {
        &self.final_h
    }

    /// Resident attention-cache bytes at the configured depth.
    pub fn kv_bytes(&self, depth: usize) -> usize {
        self.layers
            .iter()
            .map(|b| match &b.kind {
                BlockKind::Full { attn, deep: None } => 2 * depth * attn.dims.width * 4,
                BlockKind::Full {
                    attn: _,
                    deep: Some(d),
                } => d.bytes(),
                BlockKind::Ssm(_) => 0,
            })
            .sum()
    }

    /// Pre-fill the quantized caches to `depth` positions with valid blocks.
    /// The SSM state stays zero: its step cost does not depend on the values.
    pub fn fill_synthetic(&self, depth: usize) -> Result<(), String> {
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        for b in &self.layers {
            if let BlockKind::Full { deep: Some(d), .. } = &b.kind {
                d.fill_synthetic(depth, seed)?;
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            }
        }
        Ok(())
    }

    /// A state starting at `pos` (used with `fill_synthetic`).
    pub fn state_at(&self, pos: usize) -> GpuState {
        GpuState { pos }
    }

    /// Every SSM layer's carried state, layer by layer: the conv window then
    /// the delta-net matrix. Used by `rows-check` to prove the row-block
    /// recurrence is bit-identical to the single-sequence path.
    pub fn ssm_state_host(&self) -> Result<Vec<f32>, String> {
        let mut out = Vec::new();
        for b in &self.layers {
            if let BlockKind::Ssm(s) = &b.kind {
                out.extend(s.conv_state_buf().to_host()?);
                out.extend(s.s_state_buf().to_host()?);
            }
        }
        Ok(out)
    }
}

/// Quantized-KV decode attention for one layer at `pos`.
fn deep_attention(
    l: &Launcher,
    cfg: &Config,
    attn: &AttnLayer<'_>,
    dk: &DeepKv,
    pos: usize,
) -> Result<(), String> {
    attn_tile::deep_attention_ptr(
        l,
        cfg,
        dk,
        pos,
        attn.q_buf().as_ptr(),
        attn.attn_out().as_ptr(),
    )
}

/// How many blocks to run: all of them, or `HIPML_LAYERS` (a debug bisect hook
/// that both `run_token` and `forward_columns` honour).
pub(crate) fn layer_limit(total: usize) -> usize {
    std::env::var("HIPML_LAYERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .map(|n| n.min(total))
        .unwrap_or(total)
}
