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

/// Chunked prefill (a prompt tile as batched GEMMs). A child module so it can
/// reach the engine's private scratch and layer list.
#[path = "gpu/prefill.rs"]
mod prefill;
#[path = "gpu/deep_kv.rs"]
mod deep_kv;
#[path = "gpu/layers.rs"]
mod layers;

use layers::{build_attn, build_ssm};

pub use deep_kv::DeepKv;
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
}

/// Blocks the device argmax reduces over before the one-block fold. 64 blocks
/// is one wave on this card and turns the 993 KB logit pass into a single sweep.
const ARGMAX_PARTS: usize = 64;

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
                BlockKind::Full {
                    attn: build_attn(model, l, &cfg, i, 1)?,
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
        })
    }

    pub fn new_state(&self) -> GpuState {
        GpuState { pos: 0 }
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
        self.run_token(token, st)?;
        self.model.launcher().synchronize()?;
        self.logits.to_host()
    }

    /// One token, greedy: the argmax runs on the device and only the id comes
    /// back. This is the decode loop; `forward_token` stays for the oracle
    /// comparison, which needs the whole logit vector.
    pub fn forward_token_id(&mut self, token: u32, st: &mut GpuState) -> Result<u32, String> {
        self.run_token(token, st)?;
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
    fn run_token(&mut self, token: u32, st: &mut GpuState) -> Result<(), String> {
        let l = self.model.launcher();
        let cfg = &self.cfg;
        let pos = st.pos;

        l.embed_row_q4_k(self.embed.0, token as usize, self.embed.1, &self.x)?;

        for i in 0..self.layers.len() {
            let block = &self.layers[i];
            let block_out: &Buffer = match &block.kind {
                BlockKind::Ssm(s) => {
                    l.rmsnorm_buf(&self.x, &block.attn_norm, s.input(), cfg.n_embd, cfg.eps)?;
                    s.step()?;
                    s.output()
                }
                BlockKind::Full { attn, deep: None } => {
                    l.rmsnorm_buf(&self.x, &block.attn_norm, &self.t, cfg.n_embd, cfg.eps)?;
                    attn.forward_buf(l, &self.t, pos)?;
                    attn.output()
                }
                BlockKind::Full { attn, deep: Some(dk) } => {
                    l.rmsnorm_buf(&self.x, &block.attn_norm, &self.t, cfg.n_embd, cfg.eps)?;
                    attn.prepare_qkv(l, &self.t, pos)?;
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
                        pos,
                    )?;
                    deep_attention(l, cfg, attn, dk, pos)?;
                    attn.apply_output(l)?;
                    attn.output()
                }
            };
            l.residual_buf(&self.x, block_out, &self.x1, cfg.n_embd)?;
            l.rmsnorm_buf(&self.x1, &block.post_norm, &self.t, cfg.n_embd, cfg.eps)?;

            block.gate.gemv(l, &self.t, &self.gate)?;
            block.up.gemv(l, &self.t, &self.up)?;
            l.silu_mul_buf(&self.gate, &self.up, &self.act, cfg.n_ff)?;
            block.down.gemv(l, &self.act, &self.down)?;
            l.residual_buf(&self.down, &self.x1, &self.x, cfg.n_embd)?;
        }

        st.pos += 1;
        l.rmsnorm_buf(&self.x, &self.output_norm, &self.final_h, cfg.n_embd, cfg.eps)?;
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

    pub fn model(&self) -> &Model {
        self.model
    }

    /// Resident attention-cache bytes at the configured depth.
    pub fn kv_bytes(&self, depth: usize) -> usize {
        self.layers
            .iter()
            .map(|b| match &b.kind {
                BlockKind::Full { attn, deep: None } => 2 * depth * attn.dims.width * 4,
                BlockKind::Full { deep: Some(d), .. } => d.bytes(),
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
    if dk.dtype == KvDtype::Q4 && cfg.head_dim > 256 {
        return Err("q4 KV: head_dim above 256 is not covered".to_string());
    }
    let seq = pos + 1;
    if seq > dk.seq {
        return Err(format!("position {pos} beyond quantized cache {}", dk.seq));
    }
    let splits = (seq / 64).clamp(1, MAX_SPLITS);
    launch_kv::launch(
        l.module(),
        l.stream(),
        dk.dtype.id(),
        attn.q_buf(),
        &dk.k,
        &dk.v,
        &dk.part_acc,
        &dk.part_ml,
        attn.attn_out(),
        cfg.n_head,
        cfg.n_head_kv,
        cfg.head_dim,
        seq,
        1.0f32 / (cfg.head_dim as f32).sqrt(),
        splits,
        0,
        0,
    )
}
