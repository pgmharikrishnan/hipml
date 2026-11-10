// SPDX-License-Identifier: Apache-2.0
//! The MTP (`nextn`) drafter block on the device.
//!
//! ```text
//!   x   = eh_proj [ enorm(embed(t+1)) ; hnorm(h_t) ]      10240 -> 5120
//!   x   = x + full_attention(attn_norm(x))                 own f32 KV cache
//!   x   = x + down(silu(gate(t)) * up(t))
//!   id  = argmax(target_head(rmsnorm(x, shared_head_norm)))
//! ```
//!
//! The block lives in the sidecar GGUF (`blk.64.*`); the embedding and the
//! output head are the target model's, exactly as the CPU reference
//! (`hipml_core::mtp`) does. `h_t` is the target's hidden state *after*
//! `output_norm` — the convention that measured the best acceptance — and the
//! recursive step feeds `shared_head_norm(block output)` back in.
//!
//! Every launch runs on the target model's launcher: one stream orders the
//! verify pass, the draft and the head, and no hidden state round-trips
//! through the host.

use super::*;
use crate::gpu_weights::{gemv_entry, Proj};
use crate::OUTPUT;

/// The MTP sidecar's block index is the base model's decoder depth, i.e.
/// `block_count - nextn_predict_layers`. It was `64` for the 64-layer
/// fine-tune; the stock file also names its inline MTP block `blk.64`, but
/// `block_count` there is 65, so the index has to come from the metadata.
pub fn mtp_blk(cfg: &Config) -> usize {
    cfg.n_layer
}

pub struct MtpGpu<'a> {
    cfg: Config,
    enorm: Buffer,
    hnorm: Buffer,
    attn_norm: Buffer,
    post_norm: Buffer,
    head_norm: Buffer,
    eh_proj: Proj<'a>,
    attn: AttnLayer<'a>,
    gate: Proj<'a>,
    up: Proj<'a>,
    down: Proj<'a>,
    emb: Buffer,
    cat: Buffer,
    x: Buffer,
    x1: Buffer,
    t: Buffer,
    gate_b: Buffer,
    up_b: Buffer,
    act: Buffer,
    down_b: Buffer,
    out: Buffer,
    /// `rmsnorm(out, shared_head_norm)`: the head's input and the next
    /// recursive step's `h`.
    hnext: Buffer,
    logits: Buffer,
    arg_v: Buffer,
    arg_i: Buffer,
    arg_out: Buffer,
    head: &'a ByteBuffer,
    head_ty: hipml_core::quant::TensorType,
    embed: (&'a ByteBuffer, usize),
    launcher: &'a Launcher,
}

impl<'a> MtpGpu<'a> {
    /// Build the drafter from `blk.64` of a sidecar already merged into
    /// `model` (`Model::load_sidecar`), with a KV cache for `n_ctx` positions.
    pub fn new(model: &'a Model, n_ctx: usize) -> Result<MtpGpu<'a>, String> {
        let cfg = {
            let mut g = Gguf::open(model.path()).map_err(|e| format!("{e}"))?;
            Config::from_gguf(&mut g).map_err(|e| format!("config: {e}"))?
        };
        let n = cfg.n_embd;
        let blk = mtp_blk(&cfg);
        let p = |name: &str| Proj::of(model, &format!("blk.{blk}.{name}"));
        let m = |name: &str| -> Result<(&'a ByteBuffer, usize, usize, Kind), String> {
            let t = model
                .tensor(name)
                .ok_or_else(|| format!("model has no tensor {name}"))?;
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
        };
        let v = |name: &str| model.f32_vec(&format!("blk.{blk}.{name}"));
        let l = model.launcher();
        let attn = AttnLayer::new_resident(
            l,
            m(&format!("blk.{blk}.attn_q.weight"))?,
            m(&format!("blk.{blk}.attn_k.weight"))?,
            m(&format!("blk.{blk}.attn_v.weight"))?,
            m(&format!("blk.{blk}.attn_output.weight"))?,
            &v("attn_q_norm.weight")?,
            &v("attn_k_norm.weight")?,
            &cfg,
            n_ctx,
        )?;
        let embed = model
            .tensor(crate::TOKEN_EMBD)
            .ok_or_else(|| "model has no token_embd.weight".to_string())?;
        let embed = (embed.buffer(), (embed.nbytes() / embed.rows()) as usize);
        let head_t = model
            .tensor(OUTPUT)
            .ok_or_else(|| "model has no output.weight".to_string())?;
        let head = head_t.buffer();
        let head_ty = head_t.ty();
        let zeros = |len: usize| Buffer::alloc(len);
        Ok(MtpGpu {
            enorm: Buffer::from_host(&v("nextn.enorm.weight")?)?,
            hnorm: Buffer::from_host(&v("nextn.hnorm.weight")?)?,
            attn_norm: Buffer::from_host(&v("attn_norm.weight")?)?,
            post_norm: Buffer::from_host(&v("post_attention_norm.weight")?)?,
            head_norm: Buffer::from_host(&v("nextn.shared_head_norm.weight")?)?,
            eh_proj: p("nextn.eh_proj.weight")?,
            attn,
            gate: p("ffn_gate.weight")?,
            up: p("ffn_up.weight")?,
            down: p("ffn_down.weight")?,
            emb: zeros(n)?,
            cat: zeros(2 * n)?,
            x: zeros(n)?,
            x1: zeros(n)?,
            t: zeros(n)?,
            gate_b: zeros(cfg.n_ff)?,
            up_b: zeros(cfg.n_ff)?,
            act: zeros(cfg.n_ff)?,
            down_b: zeros(n)?,
            out: zeros(n)?,
            hnext: zeros(n)?,
            logits: zeros(cfg.n_vocab)?,
            arg_v: zeros(ARGMAX_PARTS)?,
            arg_i: zeros(ARGMAX_PARTS)?,
            arg_out: zeros(1)?,
            head,
            head_ty,
            embed,
            launcher: l,
            cfg,
        })
    }

    /// Bytes the drafter's own weights add on top of the target's head.
    pub fn weight_bytes(&self) -> usize {
        self.eh_proj.bytes()
            + self.attn.weight_bytes()
            + self.gate.bytes()
            + self.up.bytes()
            + self.down.bytes()
    }

    /// Bytes of the drafter's f32 KV cache.
    pub fn kv_bytes(&self) -> usize {
        self.attn.kv_bytes()
    }

    /// Take the hidden state from row `row` of a `[rows][n_embd]` buffer.
    pub fn set_hidden_row(&self, src: &Buffer, row: usize) -> Result<(), String> {
        let n = self.cfg.n_embd;
        let l = self.attn_launcher();
        self.hnext
            .copy_from_device_on_at(src, row * n, n, l.stream())
    }

    /// Take a hidden state from host memory (the parity probe).
    pub fn set_hidden_host(&self, h: &[f32]) -> Result<(), String> {
        self.hnext.write(h)
    }

    /// `rmsnorm(x, enorm) ; rmsnorm(h, hnorm)` into the 10240-wide projection
    /// input, then `x = eh_proj cat`.
    fn project(&self, token: u32) -> Result<(), String> {
        let l = self.attn_launcher();
        let cfg = &self.cfg;
        let n = cfg.n_embd;
        l.embed_row_q4_k_at(self.embed.0, token as usize, self.embed.1, &self.emb, 0, n)?;
        l.rmsnorm_buf(&self.emb, &self.enorm, &self.cat, n, cfg.eps)?;
        let dst = unsafe { (self.cat.as_ptr() as *mut u8).add(n * 4) as *mut std::ffi::c_void };
        l.rmsnorm_ptr(self.hnext.as_ptr(), &self.hnorm, dst, n, cfg.eps, 1)?;
        self.eh_proj.gemv(l, &self.cat, &self.x)
    }

    /// One drafter step at absolute position `pos` for `token` (the token
    /// *after* `pos`). Returns the drafted id; leaves `hnext` ready for the
    /// recursive next draft.
    pub fn step(&mut self, token: u32, pos: usize) -> Result<u32, String> {
        let l = self.attn_launcher();
        let cfg = self.cfg.clone();
        let n = cfg.n_embd;
        self.project(token)?;
        l.rmsnorm_buf(&self.x, &self.attn_norm, &self.t, n, cfg.eps)?;
        l.set_step(pos, pos + 1, token as usize)?;
        self.attn.forward_buf(l, &self.t, pos)?;
        l.residual_buf(&self.x, self.attn.output(), &self.x1, n)?;
        l.rmsnorm_buf(&self.x1, &self.post_norm, &self.t, n, cfg.eps)?;
        self.gate.gemv(l, &self.t, &self.gate_b)?;
        self.up.gemv(l, &self.t, &self.up_b)?;
        l.silu_mul_buf(&self.gate_b, &self.up_b, &self.act, cfg.n_ff)?;
        self.down.gemv(l, &self.act, &self.down_b)?;
        l.residual_buf(&self.down_b, &self.x1, &self.out, n)?;
        l.rmsnorm_buf(&self.out, &self.head_norm, &self.hnext, n, cfg.eps)?;
        l.run_quant_resident(
            gemv_entry(self.head_ty)?,
            self.head,
            &self.hnext,
            &self.logits,
            cfg.n_vocab,
            n,
        )?;
        l.argmax_buf(
            &self.logits,
            &self.arg_v,
            &self.arg_i,
            &self.arg_out,
            cfg.n_vocab,
            ARGMAX_PARTS,
        )?;
        l.synchronize()?;
        let b = self.arg_out.download_bytes(0, 4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Build the drafter's KV cache for one prompt position: only `eh_proj`,
    /// `attn_norm`, `wk`/`wv` and the append, which is all a KV row needs.
    /// `h` must already be in `hnext` (`set_hidden_row`).
    pub fn prefill_kv(&mut self, token: u32, pos: usize) -> Result<(), String> {
        let l = self.attn_launcher();
        let cfg = self.cfg.clone();
        let n = cfg.n_embd;
        self.project(token)?;
        l.rmsnorm_buf(&self.x, &self.attn_norm, &self.t, n, cfg.eps)?;
        l.set_step(pos, pos + 1, token as usize)?;
        self.attn.append_kv_only(l, &self.t, pos)
    }

    fn attn_launcher(&self) -> &'a Launcher {
        self.launcher
    }
}
