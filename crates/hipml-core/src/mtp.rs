// SPDX-License-Identifier: Apache-2.0
//! A next-token-prediction (MTP / "nextn") drafter on the CPU reference.
//!
//! Qwen-family hybrids ship one extra decoder block that predicts token
//! `t+2` from the main model's hidden state at `t` and the embedding of token
//! `t+1`:
//!
//! ```text
//! x  = eh_proj [ enorm(embed(t+1)) ; hnorm(h_t) ]     10240 -> 5120
//! x  = full-attention block(x)  (own KV cache)  + FFN, with residuals
//! logits = lm_head(shared_head_norm(x))
//! ```
//!
//! The block lives in a sidecar GGUF (`blk.<n>.nextn.*` plus an ordinary
//! attention/FFN block under the same index). This module only exists to
//! *measure* how often that drafter agrees with greedy decode of a target
//! model; the head is the target's own `output.weight`.
//!
//! `post_norm_hidden` selects which hidden state feeds `hnorm`: the residual
//! stream after the last block (`false`) or after `output_norm` (`true`); the
//! recursive step feeds back the block output under the same convention.

use std::io;
use std::path::Path;
use std::sync::Arc;

use crate::attention::{attn_layer_f32, FullAttnW, KvState};
use crate::gguf::Gguf;
use crate::mat::Mat;
use crate::mmap::Map;
use crate::model::{read_f32_vec, FfnW, Model};
use crate::ops::{residual_add_f32, rmsnorm_f32, silu_mul_f32};

pub struct Mtp {
    pub enorm: Vec<f32>,
    pub hnorm: Vec<f32>,
    pub eh_proj: Mat,
    pub attn_norm: Vec<f32>,
    pub post_norm: Vec<f32>,
    pub attn: FullAttnW,
    pub ffn: FfnW,
    pub head_norm: Vec<f32>,
    pub post_norm_hidden: bool,
}

impl Mtp {
    /// Load block `blk` of the sidecar at `path`.
    pub fn load(path: impl AsRef<Path>, blk: usize, post_norm_hidden: bool) -> io::Result<Mtp> {
        let path = path.as_ref();
        let mut g = Gguf::open(path)?;
        let map = Arc::new(Map::open(path)?);
        let m = |g: &Gguf<_>, n: &str| Mat::from_map(&map, g, &format!("blk.{blk}.{n}"));
        let attn = FullAttnW {
            wq: m(&g, "attn_q.weight")?,
            wk: m(&g, "attn_k.weight")?,
            wv: m(&g, "attn_v.weight")?,
            wo: m(&g, "attn_output.weight")?,
            q_norm: read_f32_vec(&mut g, &format!("blk.{blk}.attn_q_norm.weight"))?,
            k_norm: read_f32_vec(&mut g, &format!("blk.{blk}.attn_k_norm.weight"))?,
        };
        let ffn = FfnW {
            gate: m(&g, "ffn_gate.weight")?,
            up: m(&g, "ffn_up.weight")?,
            down: m(&g, "ffn_down.weight")?,
        };
        let eh_proj = m(&g, "nextn.eh_proj.weight")?;
        let mut v = |n: &str| read_f32_vec(&mut g, &format!("blk.{blk}.{n}"));
        Ok(Mtp {
            enorm: v("nextn.enorm.weight")?,
            hnorm: v("nextn.hnorm.weight")?,
            attn_norm: v("attn_norm.weight")?,
            post_norm: v("post_attention_norm.weight")?,
            head_norm: v("nextn.shared_head_norm.weight")?,
            eh_proj,
            attn,
            ffn,
            post_norm_hidden,
        })
    }

    /// The hidden state to feed for the main model's residual `x`.
    pub fn main_hidden(&self, model: &Model, x: &[f32]) -> Vec<f32> {
        if self.post_norm_hidden {
            rmsnorm_f32(x, &model.w.output_norm, model.cfg.eps)
        } else {
            x.to_vec()
        }
    }

    /// The hidden state to feed back when drafting from this block's output.
    pub fn self_hidden(&self, model: &Model, out: &[f32]) -> Vec<f32> {
        if self.post_norm_hidden {
            rmsnorm_f32(out, &self.head_norm, model.cfg.eps)
        } else {
            out.to_vec()
        }
    }

    /// One drafter step at absolute position `pos`: `h` is the hidden state
    /// that produced `next` (from [`Mtp::main_hidden`], or [`Mtp::self_hidden`]
    /// when drafting recursively). Returns `(block output, logits)`; logits are
    /// skipped (empty) when `want_logits` is false, e.g. over the prompt.
    pub fn step(
        &self,
        model: &Model,
        kv: &mut KvState,
        h: &[f32],
        next: u32,
        pos: usize,
        want_logits: bool,
    ) -> (Vec<f32>, Vec<f32>) {
        let cfg = &model.cfg;
        let e = rmsnorm_f32(
            &model.w.tok_embd.row_f32(next as usize),
            &self.enorm,
            cfg.eps,
        );
        let hh = rmsnorm_f32(h, &self.hnorm, cfg.eps);
        let mut cat = e;
        cat.extend_from_slice(&hh);
        let x = self.eh_proj.gemv(&cat);

        let t = rmsnorm_f32(&x, &self.attn_norm, cfg.eps);
        let x1 = residual_add_f32(&attn_layer_f32(&self.attn, kv, &t, cfg, pos), &x);
        let t = rmsnorm_f32(&x1, &self.post_norm, cfg.eps);
        let act = silu_mul_f32(&self.ffn.gate.gemv(&t), &self.ffn.up.gemv(&t));
        let out = residual_add_f32(&self.ffn.down.gemv(&act), &x1);
        let logits = if want_logits {
            model
                .w
                .output
                .gemv(&rmsnorm_f32(&out, &self.head_norm, cfg.eps))
        } else {
            Vec::new()
        };
        (out, logits)
    }
}
