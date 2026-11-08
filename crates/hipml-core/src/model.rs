//! The decoder: weights, per-sequence state, and the token loop.
//!
//! Lay out the same order as `llama_model_qwen35::graph` in
//! `src/models/qwen35.cpp`:
//!
//! ```text
//!   x = embed(token)
//!   for layer:
//!       x = x + block(rmsnorm(x, attn_norm))        # full attention or GDN
//!       x = x + ffn(rmsnorm(x, post_attention_norm))
//!   logits = output(rmsnorm(x, output_norm))
//! ```
//!
//! Prefill is the same token loop N times: the recurrent state and the KV
//! cache make that exactly the autoregressive path llama.cpp uses when
//! decoding, which is the behaviour this reference has to reproduce.

use std::io::{self, Read, Seek};
use std::path::Path;
use std::sync::Arc;

use crate::attention::{attn_layer_f32, FullAttnW, KvState};
use crate::config::Config;
use crate::delta::{ssm_layer_f32, SsmState, SsmW};
use crate::gguf::Gguf;
use crate::mat::Mat;
use crate::mmap::Map;
use crate::ops::{residual_add_f32, rmsnorm_f32};

pub struct FfnW {
    pub gate: Mat,
    pub up: Mat,
    pub down: Mat,
}

pub struct Layer {
    pub attn_norm: Vec<f32>,
    pub post_norm: Vec<f32>,
    pub ffn: FfnW,
    pub attn: Option<FullAttnW>,
    pub ssm: Option<SsmW>,
}

pub struct Weights {
    pub tok_embd: Mat,
    pub output_norm: Vec<f32>,
    pub output: Mat,
    pub layers: Vec<Layer>,
}

/// Everything that is carried from one token to the next.
pub struct State {
    pub pos: usize,
    pub kv: Vec<Option<KvState>>,
    pub ssm: Vec<Option<SsmState>>,
}

pub struct Model {
    pub cfg: Config,
    pub w: Weights,
}

impl Model {
    /// Load a model by mapping the file: weights stay file-backed and the
    /// kernel can evict them, so a 17 GB model does not fight the page cache.
    pub fn load_file(path: impl AsRef<Path>) -> io::Result<Model> {
        let path = path.as_ref();
        let mut g = Gguf::open(path)?;
        let map = Arc::new(Map::open(path)?);
        Self::load_impl(&mut g, Some(&map))
    }

    /// Load every weight tensor into memory, keeping it quantized. Reads 17 GB
    /// and keeps it resident; prefer [`Model::load_file`] on a shared machine.
    pub fn load<R: Read + Seek>(g: &mut Gguf<R>) -> io::Result<Model> {
        Self::load_impl(g, None)
    }

    fn load_impl<R: Read + Seek>(
        g: &mut Gguf<R>,
        map: Option<&Arc<Map>>,
    ) -> io::Result<Model> {
        // One entry point for weight tensors so the mapped and copied paths
        // share all shape checks.
        let mat = |g: &mut Gguf<R>, name: &str| -> io::Result<Mat> {
            match map {
                Some(m) => Mat::from_map(m, g, name),
                None => Mat::load(g, name),
            }
        };
        let cfg = Config::from_gguf(g)?;
        let tok_embd = mat(g, "token_embd.weight")?;
        let output_norm = read_f32_vec(g, "output_norm.weight")?;
        let output = mat(g, "output.weight")?;

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let attn_norm = read_f32_vec(g, &format!("blk.{i}.attn_norm.weight"))?;
            let post_norm = read_f32_vec(g, &format!("blk.{i}.post_attention_norm.weight"))?;
            let ffn = FfnW {
                gate: mat(g, &format!("blk.{i}.ffn_gate.weight"))?,
                up: mat(g, &format!("blk.{i}.ffn_up.weight"))?,
                down: mat(g, &format!("blk.{i}.ffn_down.weight"))?,
            };
            expect(&ffn.gate, cfg.n_ff, cfg.n_embd, "ffn_gate")?;
            expect(&ffn.up, cfg.n_ff, cfg.n_embd, "ffn_up")?;
            expect(&ffn.down, cfg.n_embd, cfg.n_ff, "ffn_down")?;

            let (attn, ssm) = if cfg.is_recr[i] {
                let w = SsmW {
                    wqkv: mat(g, &format!("blk.{i}.attn_qkv.weight"))?,
                    wgate: mat(g, &format!("blk.{i}.attn_gate.weight"))?,
                    conv1d: read_f32_vec(g, &format!("blk.{i}.ssm_conv1d.weight"))?,
                    alpha: mat(g, &format!("blk.{i}.ssm_alpha.weight"))?,
                    beta: mat(g, &format!("blk.{i}.ssm_beta.weight"))?,
                    dt_bias: read_f32_vec(g, &format!("blk.{i}.ssm_dt.bias"))?,
                    a: read_f32_vec(g, &format!("blk.{i}.ssm_a"))?,
                    norm: read_f32_vec(g, &format!("blk.{i}.ssm_norm.weight"))?,
                    wout: mat(g, &format!("blk.{i}.ssm_out.weight"))?,
                };
                let hd_v = cfg.ssm_head_v();
                expect(&w.wqkv, cfg.ssm_conv_channels(), cfg.n_embd, "attn_qkv")?;
                expect(&w.wgate, cfg.ssm_d_inner, cfg.n_embd, "attn_gate")?;
                expect(&w.alpha, cfg.ssm_dt_rank, cfg.n_embd, "ssm_alpha")?;
                expect(&w.beta, cfg.ssm_dt_rank, cfg.n_embd, "ssm_beta")?;
                expect(&w.wout, cfg.n_embd, cfg.ssm_d_inner, "ssm_out")?;
                check_len(&w.conv1d, cfg.ssm_d_conv * cfg.ssm_conv_channels(), "ssm_conv1d")?;
                check_len(&w.dt_bias, cfg.ssm_dt_rank, "ssm_dt.bias")?;
                check_len(&w.a, cfg.ssm_dt_rank, "ssm_a")?;
                check_len(&w.norm, hd_v, "ssm_norm")?;
                (None, Some(w))
            } else {
                let w = FullAttnW {
                    wq: mat(g, &format!("blk.{i}.attn_q.weight"))?,
                    wk: mat(g, &format!("blk.{i}.attn_k.weight"))?,
                    wv: mat(g, &format!("blk.{i}.attn_v.weight"))?,
                    wo: mat(g, &format!("blk.{i}.attn_output.weight"))?,
                    q_norm: read_f32_vec(g, &format!("blk.{i}.attn_q_norm.weight"))?,
                    k_norm: read_f32_vec(g, &format!("blk.{i}.attn_k_norm.weight"))?,
                };
                expect(&w.wq, cfg.n_head * cfg.head_dim * 2, cfg.n_embd, "attn_q")?;
                expect(&w.wk, cfg.n_head_kv * cfg.head_dim, cfg.n_embd, "attn_k")?;
                expect(&w.wv, cfg.n_head_kv * cfg.head_dim, cfg.n_embd, "attn_v")?;
                expect(&w.wo, cfg.n_embd, cfg.n_head * cfg.head_dim, "attn_output")?;
                check_len(&w.q_norm, cfg.head_dim, "attn_q_norm")?;
                check_len(&w.k_norm, cfg.head_dim, "attn_k_norm")?;
                (Some(w), None)
            };
            layers.push(Layer {
                attn_norm,
                post_norm,
                ffn,
                attn,
                ssm,
            });
        }

        Ok(Model {
            cfg,
            w: Weights {
                tok_embd,
                output_norm,
                output,
                layers,
            },
        })
    }

    /// Fresh KV cache and recurrent state for at most `n_ctx` tokens.
    pub fn new_state(&self, n_ctx: usize) -> State {
        let width = self.cfg.n_head_kv * self.cfg.head_dim;
        let mut kv = Vec::with_capacity(self.cfg.n_layer);
        let mut ssm = Vec::with_capacity(self.cfg.n_layer);
        for i in 0..self.cfg.n_layer {
            if self.cfg.is_recr[i] {
                kv.push(None);
                ssm.push(Some(SsmState::new(
                    self.cfg.ssm_d_conv,
                    self.cfg.ssm_conv_channels(),
                    self.cfg.ssm_dt_rank,
                    self.cfg.ssm_head_v(),
                )));
            } else {
                kv.push(Some(KvState::new(n_ctx, width)));
                ssm.push(None);
            }
        }
        State { pos: 0, kv, ssm }
    }

    /// One token through the whole model; returns `n_vocab` logits.
    ///
    /// The last token of a prompt is fed here too, so the logits after the
    /// prompt are the ones that pick the first generated token.
    pub fn forward_token(&self, token: u32, st: &mut State) -> Vec<f32> {
        let x = self.forward_hidden(token, st);
        let h = rmsnorm_f32(&x, &self.w.output_norm, self.cfg.eps);
        self.w.output.gemv(&h)
    }

    /// [`Model::forward_token`] without the head: the residual stream after
    /// the last block, before `output_norm`. The MTP drafter reads this.
    pub fn forward_hidden(&self, token: u32, st: &mut State) -> Vec<f32> {
        let cfg = &self.cfg;
        let pos = st.pos;
        let mut x = self.w.tok_embd.row_f32(token as usize);

        for i in 0..cfg.n_layer {
            let l = &self.w.layers[i];
            let inp = x;
            let h = rmsnorm_f32(&inp, &l.attn_norm, cfg.eps);

            let block = if let Some(a) = &l.attn {
                let kv = st.kv[i].as_mut().expect("kv cache for full attention");
                attn_layer_f32(a, kv, &h, cfg, pos)
            } else {
                let ssm = st.ssm[i].as_mut().expect("ssm state for recurrent layer");
                ssm_layer_f32(l.ssm.as_ref().unwrap(), ssm, &h, cfg)
            };
            let x1 = residual_add_f32(&block, &inp);

            let t = rmsnorm_f32(&x1, &l.post_norm, cfg.eps);
            let gate = l.ffn.gate.gemv(&t);
            let up = l.ffn.up.gemv(&t);
            let act = crate::ops::silu_mul_f32(&gate, &up);
            let down = l.ffn.down.gemv(&act);
            x = residual_add_f32(&down, &x1);
        }

        st.pos += 1;
        x
    }
}

/// Load one full-attention layer's tensors without touching the other 63
/// blocks. The GPU attention work (WP12) uses this so parity and timing do not
/// need the 17 GB model resident.
pub fn load_attn_layer<R: Read + Seek>(g: &mut Gguf<R>, i: usize) -> io::Result<FullAttnW> {
    let w = FullAttnW {
        wq: Mat::load(g, &format!("blk.{i}.attn_q.weight"))?,
        wk: Mat::load(g, &format!("blk.{i}.attn_k.weight"))?,
        wv: Mat::load(g, &format!("blk.{i}.attn_v.weight"))?,
        wo: Mat::load(g, &format!("blk.{i}.attn_output.weight"))?,
        q_norm: read_f32_vec(g, &format!("blk.{i}.attn_q_norm.weight"))?,
        k_norm: read_f32_vec(g, &format!("blk.{i}.attn_k_norm.weight"))?,
    };
    Ok(w)
}

pub(crate) fn read_f32_vec<R: Read + Seek>(g: &mut Gguf<R>, name: &str) -> io::Result<Vec<f32>> {
    let info = g
        .tensor(name)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("no tensor {name}")))?
        .clone();
    if info.ty != crate::quant::TensorType::F32 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("tensor {name} must be F32, file has {}", info.ty.name()),
        ));
    }
    let raw = g.read_tensor(name)?;
    Ok(raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn expect(m: &Mat, rows: usize, cols: usize, what: &str) -> io::Result<()> {
    if m.rows != rows || m.cols != cols {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "tensor {what}: expected {rows}x{cols}, got {}x{}",
                m.rows, m.cols
            ),
        ));
    }
    Ok(())
}

fn check_len(v: &[f32], n: usize, what: &str) -> io::Result<()> {
    if v.len() != n {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("tensor {what}: expected {n} values, got {}", v.len()),
        ));
    }
    Ok(())
}
