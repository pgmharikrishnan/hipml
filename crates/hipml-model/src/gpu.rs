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
use hipml_hip::{Buffer, ByteBuffer, Event};
use hipml_kernels::launch_kv;
use hipml_kernels::ssm::{SsmDims, SsmGpuLayer, SsmResidentRefs};
use hipml_kernels::{AttnLayer, Launcher};

use crate::gpu_weights::{gemv_entry, mat_kind, Proj};
use crate::Model;

/// Chunked prefill (a prompt tile as batched GEMMs). A child module so it can
/// reach the engine's private scratch and layer list.
#[path = "gpu/prefill.rs"]
mod prefill;
#[path = "gpu/deep_kv.rs"]
mod deep_kv;
#[path = "gpu/layers.rs"]
mod layers;
#[path = "gpu/trace.rs"]
mod trace;

#[path = "gpu/graph.rs"]
mod graph;
pub use graph::{launches_per_token, GraphTiming};
#[path = "gpu/prefix.rs"]
mod prefix;

use layers::{build_attn, build_ssm};
use trace::mark;

pub use prefill::PrefillReport;
pub use prefix::PrefixHit;

pub use deep_kv::DeepKv;
pub use trace::Trace;
use deep_kv::MAX_SPLITS;

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
    /// The embedding tensor's resident payload plus its row stride, so a token
    /// id becomes a device row lookup instead of a host round-trip.
    embed: (&'a ByteBuffer, usize),
    /// Argmax scratch: `parts` partial values, `parts` partial indices, one out.
    arg_v: Buffer,
    arg_i: Buffer,
    arg_out: Buffer,
    /// Chunked-prefill scratch, allocated on first use and reused after.
    chunk: Option<(usize, prefill::Chunk)>,
    /// The captured token graph, built on first use: one `hipGraphLaunch`
    /// submits the whole step instead of ~1,300 kernel launches.
    graph: Option<hipml_hip::Graph>,
    /// Stage timing; `None` unless the caller asked for a trace.
    pub trace: Option<Trace>,
    /// Prefix snapshots (SSM recurrent state + f32 KV rows), when enabled.
    prefix: Option<hipml_core::prefix_cache::PrefixCache<prefix::Snapshot>>,
}

/// Blocks the device argmax reduces over before the one-block fold. 64 blocks
/// is one wave on this card and turns the 993 KB logit pass into a single sweep.
const ARGMAX_PARTS: usize = 64;

/// The f32 prefill mirror is a convenience, not a requirement: it is only
/// allocated while it costs at most this many bytes across the attention
/// layers. At this model's shape the mirror is `2 * 4 B * width * layers` =
/// 128 KiB per position, so 5 GiB covers a 40K context: the 32K bench cell
/// (n_ctx 32832, 4.3 GiB) keeps the mirror, which measured 864 tok/s there
/// against 847 on the q8 chunk path. A 64K context needs 8.6 GiB and 100K
/// 13.4 GiB, so both run the matrix-unit kernel straight out of the q8/q4
/// decode cache and no mirror is ever allocated.
const F32_MIRROR_MAX_BYTES: usize = 5 << 30;

impl<'a> GpuEngine<'a> {
    /// Build every resident layer, borrowing the loader's weight payloads.
    pub fn new(
        model: &'a Model,
        n_ctx: usize,
        kv_mode: KvMode,
    ) -> Result<GpuEngine<'a>, String> {
        let cfg = {
            let mut g = Gguf::open(model.path()).map_err(|e| format!("{e}"))?;
            Config::from_gguf(&mut g).map_err(|e| format!("config: {e}"))?
        };
        let l = model.launcher();
        let n = cfg.n_embd;
        let embed = model
            .tensor(crate::TOKEN_EMBD)
            .ok_or_else(|| "model has no token_embd.weight".to_string())?;
        if embed.ty() != hipml_core::quant::TensorType::Q4K {
            return Err(format!(
                "{}: expected Q4_K, found {}",
                crate::TOKEN_EMBD,
                embed.ty().name()
            ));
        }
        let embed = (embed.buffer(), (embed.nbytes() / embed.rows()) as usize);

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let attn_norm = Buffer::from_host(&model.f32_vec(&format!("blk.{i}.attn_norm.weight"))?)?;
            let post_norm = Buffer::from_host(&model.f32_vec(&format!("blk.{i}.post_attention_norm.weight"))?)?;
            let gate = Proj::of(model, &format!("blk.{i}.ffn_gate.weight"))?;
            let up = Proj::of(model, &format!("blk.{i}.ffn_up.weight"))?;
            let down = Proj::of(model, &format!("blk.{i}.ffn_down.weight"))?;

            let kind = if cfg.is_recr[i] {
                BlockKind::Ssm(build_ssm(model, l, &cfg, i)?)
            } else if let Some(dtype) = kv_mode.dtype() {
                // The mirror is a prefill-only convenience, bounded by a byte
                // budget (see F32_MIRROR_MAX_BYTES) and never allocated at the
                // depths that need the VRAM, where the q8 chunk kernel reads the
                // cache directly.
                let n_attn = cfg.is_recr.iter().filter(|r| !**r).count();
                let mirror_bytes = 2 * n_ctx * (cfg.n_head_kv * cfg.head_dim) * 4 * n_attn;
                let mirror = mirror_bytes <= F32_MIRROR_MAX_BYTES;
                let f32_mirror = if mirror { n_ctx } else { 1 };
                BlockKind::Full {
                    attn: build_attn(model, l, &cfg, i, f32_mirror)?,
                    deep: Some(DeepKv::new(&cfg, n_ctx, dtype, mirror)?),
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
        Ok(GpuEngine {
            model,
            cfg: cfg.clone(),
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
            trace: None,
            prefix: None,
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
        b + self.model.tensor(crate::OUTPUT).map(|t| t.nbytes() as usize).unwrap_or(0)
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
        l.argmax_buf(
            &self.logits,
            &self.arg_v,
            &self.arg_i,
            &self.arg_out,
            self.cfg.n_vocab,
            ARGMAX_PARTS,
        )?;
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
    fn run_token(&mut self, token: u32, st: &mut GpuState, set_step: bool) -> Result<(), String> {
        let l = self.model.launcher();
        let cfg = &self.cfg;
        let pos = st.pos;
        if set_step {
            l.set_step(pos, pos + 1, token as usize)?;
        }

        l.embed_row_q4_k_dev(self.embed.0, self.embed.1, &self.x, cfg.n_embd)?;
        mark(&mut self.trace, l, "embed")?;

        for i in 0..self.layers.len() {
            let block = &self.layers[i];
            let block_out: &Buffer = match &block.kind {
                BlockKind::Ssm(s) => {
                    l.rmsnorm_buf(&self.x, &block.attn_norm, s.input(), cfg.n_embd, cfg.eps)?;
                    mark(&mut self.trace, l, "ssm_in")?;
                    s.gemvs()?;
                    mark(&mut self.trace, l, "ssm_proj")?;
                    s.recurrent()?;
                    mark(&mut self.trace, l, "ssm_rec")?;
                    s.output_gemv()?;
                    mark(&mut self.trace, l, "ssm_step")?;
                    s.output()
                }
                BlockKind::Full { attn, deep: None } => {
                    l.rmsnorm_buf(&self.x, &block.attn_norm, &self.t, cfg.n_embd, cfg.eps)?;
                    mark(&mut self.trace, l, "attn_in")?;
                    attn.forward_buf(l, &self.t, pos)?;
                    mark(&mut self.trace, l, "block")?;
                    attn.output()
                }
                BlockKind::Full { attn, deep: Some(dk) } => {
                    l.rmsnorm_buf(&self.x, &block.attn_norm, &self.t, cfg.n_embd, cfg.eps)?;
                    mark(&mut self.trace, l, "attn_in")?;
                    attn.prepare_qkv(l, &self.t)?;
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
                    deep_attention(l, cfg, attn, dk, pos)?;
                    attn.apply_output(l)?;
                    mark(&mut self.trace, l, "block")?;
                    attn.output()
                }
            };
            l.residual_buf(&self.x, block_out, &self.x1, cfg.n_embd)?;
            l.rmsnorm_buf(&self.x1, &block.post_norm, &self.t, cfg.n_embd, cfg.eps)?;
            mark(&mut self.trace, l, "post_norm")?;

            block.gate.gemv(l, &self.t, &self.gate)?;
            mark(&mut self.trace, l, "gate")?;
            block.up.gemv(l, &self.t, &self.up)?;
            mark(&mut self.trace, l, "up")?;
            l.silu_mul_buf(&self.gate, &self.up, &self.act, cfg.n_ff)?;
            block.down.gemv(l, &self.act, &self.down)?;
            mark(&mut self.trace, l, "down")?;
            l.residual_buf(&self.down, &self.x1, &self.x, cfg.n_embd)?;
        }

        st.pos += 1;
        l.rmsnorm_buf(&self.x, &self.output_norm, &self.final_h, cfg.n_embd, cfg.eps)?;
        mark(&mut self.trace, l, "out_norm")?;
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
        )?;
        mark(&mut self.trace, l, "lm_head")
    }

    pub fn model(&self) -> &Model {
        self.model
    }

    /// Resident attention-cache bytes at the configured depth.
    pub fn kv_bytes(&self, depth: usize) -> usize {
        self.layers
            .iter()
            .map(|b| match &b.kind {
                BlockKind::Full { attn, deep: None } => 2 * depth * attn.dims.width * 4,
                BlockKind::Full { attn, deep: Some(d) } => {
                    d.bytes() + 2 * depth * attn.dims.width * 4 * usize::from(d.mirror)
                }
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
}

/// Quantized-KV decode attention for one layer at `pos`.
fn deep_attention(
    l: &Launcher,
    cfg: &Config,
    attn: &AttnLayer<'_>,
    dk: &DeepKv,
    pos: usize,
) -> Result<(), String> {
    deep_attention_ptr(
        l,
        cfg,
        dk,
        pos,
        attn.q_buf().as_ptr(),
        attn.attn_out().as_ptr(),
    )
}

/// Same decode-attention pair, but reading the query row and writing the
/// output row through raw pointers. Used by the chunked q8 prefill, whose q and
/// attention-out rows live inside the chunk scratch.
fn deep_attention_ptr(
    l: &Launcher,
    cfg: &Config,
    dk: &DeepKv,
    pos: usize,
    q_ptr: *mut std::ffi::c_void,
    out_ptr: *mut std::ffi::c_void,
) -> Result<(), String> {
    if dk.dtype == KvDtype::Q4 && cfg.head_dim > 256 {
        return Err("q4 KV: head_dim above 256 is not covered".to_string());
    }
    let seq = pos + 1;
    if seq > dk.seq {
        return Err(format!("position {pos} beyond quantized cache {}", dk.seq));
    }
    // Chunked prefill is not graph-captured, so size the grid to the positions
    // that exist. The fixed `dk.seq` split count is only needed when the grid
    // becomes part of a captured graph (the decode path).
    let splits = (seq / 64).clamp(1, MAX_SPLITS);
    launch_kv::launch_ptrs(
        l.module(),
        l.stream(),
        l.step_buf(),
        dk.dtype.id(),
        q_ptr,
        &dk.k,
        &dk.v,
        &dk.part_acc,
        &dk.part_ml,
        out_ptr,
        cfg.n_head,
        cfg.n_head_kv,
        cfg.head_dim,
        // The cache's per-head row count, not `seq`: the kernel reads `seq`
        // positions starting at row 0 of a `dk.seq`-stride layout. Passing the
        // position count here made every decode step above position 0 read the
        // wrong rows for every KV head but the first.
        dk.seq,
        1.0f32 / (cfg.head_dim as f32).sqrt(),
        splits,
        0,
        0,
    )
}
