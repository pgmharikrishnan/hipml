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
//! synchronized once, before the logits are read back for argmax.
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

/// One layer's quantized K/V cache plus the split-K scratch.
pub struct DeepKv {
    pub dtype: KvDtype,
    pub seq: usize,
    n_kv: usize,
    head_dim: usize,
    k: Buffer,
    v: Buffer,
    part_acc: Buffer,
    part_ml: Buffer,
}

/// Same bound as `attn_gpu`; the partial buffers are sized once.
const MAX_SPLITS: usize = 64;

impl DeepKv {
    fn new(cfg: &Config, seq: usize, dtype: KvDtype) -> Result<DeepKv, String> {
        let row = dtype.row_bytes(cfg.head_dim);
        let bytes = cfg.n_head_kv * seq * row;
        let f32s = bytes.div_ceil(4);
        Ok(DeepKv {
            dtype,
            seq,
            n_kv: cfg.n_head_kv,
            head_dim: cfg.head_dim,
            k: Buffer::alloc(f32s)?,
            v: Buffer::alloc(f32s)?,
            part_acc: Buffer::alloc(cfg.n_head * MAX_SPLITS * cfg.head_dim)?,
            part_ml: Buffer::alloc(cfg.n_head * MAX_SPLITS * 2)?,
        })
    }

    fn row(&self) -> usize {
        self.dtype.row_bytes(self.head_dim)
    }

    /// `2 * n_kv * seq * row_bytes`.
    pub fn bytes(&self) -> usize {
        2 * self.n_kv * self.seq * self.row()
    }

    /// Fill the first `depth` positions of every head with format-valid blocks
    /// so a 100K decode can be measured without a 100K token prefill.
    fn fill_synthetic(&self, depth: usize, seed: u64) -> Result<(), String> {
        let blocks_per_row = self.head_dim / kv::BLOCK;
        let pattern = kv::synthetic_blocks(self.dtype, depth * blocks_per_row, seed);
        let row = self.row();
        for h in 0..self.n_kv {
            let off = h * self.seq * row;
            self.k.upload_bytes(off, &pattern)?;
            self.v.upload_bytes(off, &pattern)?;
        }
        Ok(())
    }

    /// Quantize one token's K/V rows and append them at `pos`.
    fn append(&self, k: &[f32], v: &[f32], pos: usize) -> Result<(), String> {
        let hd = self.head_dim;
        let row = self.row();
        for h in 0..self.n_kv {
            let kb = kv::quantize(self.dtype, &k[h * hd..(h + 1) * hd]);
            let vb = kv::quantize(self.dtype, &v[h * hd..(h + 1) * hd]);
            let off = (h * self.seq + pos) * row;
            self.k.upload_bytes(off, &kb)?;
            self.v.upload_bytes(off, &vb)?;
        }
        Ok(())
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
}

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
        let l = self.model.launcher();
        let cfg = &self.cfg;
        let pos = st.pos;

        let embed = self.model.embedding_row(token)?;
        self.x.write(&embed)?;

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
                    let kh = attn.k_buf().to_host()?;
                    let vh = attn.v_buf().to_host()?;
                    dk.append(&kh, &vh, pos)?;
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
        )?;
        l.synchronize()?;
        self.logits.to_host()
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

fn build_attn<'a>(
    model: &'a Model,
    l: &Launcher,
    cfg: &Config,
    i: usize,
    n_ctx: usize,
) -> Result<AttnLayer<'a>, String> {
    let q = attn_mat(model, &format!("blk.{i}.attn_q.weight"))?;
    let k = attn_mat(model, &format!("blk.{i}.attn_k.weight"))?;
    let v = attn_mat(model, &format!("blk.{i}.attn_v.weight"))?;
    let o = attn_mat(model, &format!("blk.{i}.attn_output.weight"))?;
    let qn = model.f32_vec(&format!("blk.{i}.attn_q_norm.weight"))?;
    let kn = model.f32_vec(&format!("blk.{i}.attn_k_norm.weight"))?;
    AttnLayer::new_resident(l, q, k, v, o, &qn, &kn, cfg, n_ctx)
}

/// `(resident payload, rows, cols, kind)` for one full-attention projection.
fn attn_mat<'a>(
    model: &'a Model,
    name: &str,
) -> Result<(&'a ByteBuffer, usize, usize, Kind), String> {
    let t = model.tensor(name).ok_or_else(|| format!("model has no tensor {name}"))?;
    let shape = t.shape();
    if shape.len() != 2 {
        return Err(format!("{name}: expected a matrix, got {shape:?}"));
    }
    Ok((
        t.buffer(),
        shape[0] as usize,
        shape[1] as usize,
        mat_kind(t.ty())?,
    ))
}

fn build_ssm<'a>(
    model: &'a Model,
    l: &'a Launcher,
    cfg: &Config,
    i: usize,
) -> Result<SsmGpuLayer<'a>, String> {
    let dims = SsmDims::from_config(cfg);
    let proj = |name: &str| -> Result<(&'a ByteBuffer, &'static str), String> {
        let t = model.tensor(name).ok_or_else(|| format!("model has no tensor {name}"))?;
        Ok((t.buffer(), gemv_entry(t.ty())?))
    };
    let conv = model.f32_vec(&format!("blk.{i}.ssm_conv1d.weight"))?;
    let dt = model.f32_vec(&format!("blk.{i}.ssm_dt.bias"))?;
    let a = model.f32_vec(&format!("blk.{i}.ssm_a"))?;
    let norm = model.f32_vec(&format!("blk.{i}.ssm_norm.weight"))?;
    let refs = SsmResidentRefs {
        wqkv: proj(&format!("blk.{i}.attn_qkv.weight"))?,
        wgate: proj(&format!("blk.{i}.attn_gate.weight"))?,
        alpha: proj(&format!("blk.{i}.ssm_alpha.weight"))?,
        beta: proj(&format!("blk.{i}.ssm_beta.weight"))?,
        wout: proj(&format!("blk.{i}.ssm_out.weight"))?,
        conv_w: &conv,
        dt_bias: &dt,
        a: &a,
        norm: &norm,
    };
    // The small vectors live as long as the layer; `new_resident` uploads them.
    let layer = SsmGpuLayer::new_resident(l.module(), l.stream(), dims, &refs)?;
    let _ = l;
    Ok(layer)
}
