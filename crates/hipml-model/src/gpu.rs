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
use hipml_core::mat::Kind;
use hipml_core::quant::TensorType;
use hipml_hip::{Buffer, ByteBuffer};
use hipml_kernels::ssm::{SsmDims, SsmGpuLayer, SsmResidentRefs};
use hipml_kernels::{AttnLayer, Launcher};

use crate::{DeviceTensor, Model};

/// Map a resident tensor's dtype to the GEMV entry that reads its blocks.
pub fn gemv_entry(ty: TensorType) -> Result<&'static str, String> {
    Ok(match ty {
        TensorType::F32 => "gemv_f32",
        TensorType::Q4K => "gemv_q4_k",
        TensorType::Q6K => "gemv_q6_k",
        TensorType::Q8_0 => "gemv_q8_0",
        other => return Err(format!("{} has no GEMV kernel", other.name())),
    })
}

fn mat_kind(ty: TensorType) -> Result<Kind, String> {
    Ok(match ty {
        TensorType::F32 => Kind::F32,
        TensorType::Q4K => Kind::Q4K,
        TensorType::Q6K => Kind::Q6K,
        TensorType::Q8_0 => Kind::Q8_0,
        other => return Err(format!("{} is not a supported mat kind", other.name())),
    })
}

/// A projection that a launcher can stream straight from VRAM.
#[derive(Clone, Copy)]
pub struct Proj<'a> {
    pub data: &'a ByteBuffer,
    pub entry: &'static str,
    pub rows: usize,
    pub cols: usize,
}

impl<'a> Proj<'a> {
    fn of(model: &'a Model, name: &str) -> Result<Proj<'a>, String> {
        let t: &DeviceTensor = model
            .tensor(name)
            .ok_or_else(|| format!("model has no tensor {name}"))?;
        let shape = t.shape();
        if shape.len() != 2 {
            return Err(format!("{name}: expected a matrix, got {shape:?}"));
        }
        Ok(Proj {
            data: t.buffer(),
            entry: gemv_entry(t.ty())?,
            rows: shape[0] as usize,
            cols: shape[1] as usize,
        })
    }

    fn gemv(
        &self,
        l: &Launcher,
        x: &Buffer,
        y: &Buffer,
    ) -> Result<(), String> {
        l.run_quant_resident(self.entry, self.data, x, y, self.rows, self.cols)
    }
}

struct Block<'a> {
    attn_norm: Buffer,
    post_norm: Buffer,
    gate: Proj<'a>,
    up: Proj<'a>,
    down: Proj<'a>,
    kind: BlockKind<'a>,
}

enum BlockKind<'a> {
    Full(AttnLayer<'a>),
    Ssm(SsmGpuLayer<'a>),
}

/// Per-sequence state: position and the per-layer caches held by the blocks.
pub struct GpuState {
    pub pos: usize,
}

pub struct GpuEngine<'a> {
    model: &'a Model,
    pub cfg: Config,
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
    pub fn new(model: &'a Model, n_ctx: usize) -> Result<GpuEngine<'a>, String> {
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
            } else {
                BlockKind::Full(build_attn(model, l, &cfg, i, n_ctx)?)
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
            b += l.gate.data.len() + l.up.data.len() + l.down.data.len();
            b += match &l.kind {
                BlockKind::Full(a) => a.weight_bytes(),
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
                BlockKind::Full(a) => {
                    l.rmsnorm_buf(&self.x, &block.attn_norm, &self.t, cfg.n_embd, cfg.eps)?;
                    a.forward_buf(l, &self.t, pos)?;
                    a.output()
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

    /// Deep-attention bytes at `depth` (the f32 KV path the short graph uses).
    pub fn kv_bytes(&self, depth: usize) -> usize {
        self.layers
            .iter()
            .map(|b| match &b.kind {
                BlockKind::Full(a) => 2 * depth * a.dims.width * 4,
                BlockKind::Ssm(_) => 0,
            })
            .sum()
    }
}

fn build_attn<'a>(
    model: &'a Model,
    l: &Launcher,
    cfg: &Config,
    i: usize,
    n_ctx: usize,
) -> Result<AttnLayer<'a>, String> {
    let q = Proj::of(model, &format!("blk.{i}.attn_q.weight"))?;
    let k = Proj::of(model, &format!("blk.{i}.attn_k.weight"))?;
    let v = Proj::of(model, &format!("blk.{i}.attn_v.weight"))?;
    let o = Proj::of(model, &format!("blk.{i}.attn_output.weight"))?;
    let tensor = |name: &str| -> Result<&DeviceTensor, String> {
        model.tensor(name).ok_or_else(|| format!("model has no tensor {name}"))
    };
    let qn = model.f32_vec(&format!("blk.{i}.attn_q_norm.weight"))?;
    let kn = model.f32_vec(&format!("blk.{i}.attn_k_norm.weight"))?;
    let kind = |t: &DeviceTensor| mat_kind(t.ty());
    AttnLayer::new_resident(
        l,
        (q.data, q.rows, q.cols, kind(tensor(&format!("blk.{i}.attn_q.weight"))?)?),
        (k.data, k.rows, k.cols, kind(tensor(&format!("blk.{i}.attn_k.weight"))?)?),
        (v.data, v.rows, v.cols, kind(tensor(&format!("blk.{i}.attn_v.weight"))?)?),
        (o.data, o.rows, o.cols, kind(tensor(&format!("blk.{i}.attn_output.weight"))?)?),
        &qn,
        &kn,
        cfg,
        n_ctx,
    )
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
