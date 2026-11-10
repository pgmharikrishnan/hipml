// SPDX-License-Identifier: Apache-2.0
//! The batched decode step. Child of `batch` so it can reach the engine's
//! private per-slot state.
//!
//! One weight pass per projection, then the per-slot work (attention at that
//! slot's position, the SSM recurrence against that slot's state, sampling).
//! The two streams carry alternating slots so the small per-slot kernels
//! overlap instead of serialising.

use super::deep_kv::MAX_SPLITS;
use super::*;
use std::time::Instant;

impl<'a> BatchEngine<'a> {
    /// One decode step for every active slot. `tokens[b]` is the input token
    /// for active slot `b`; returns that slot's greedy next token.
    pub fn forward(&mut self, tokens: &[u32]) -> Result<Vec<u32>, String> {
        let b = tokens.len();
        if b != self.active.len() {
            return Err(format!(
                "{} tokens for {} active slots",
                b,
                self.active.len()
            ));
        }
        // The batched kernels use `grid.y` as both the row index and the
        // physical stateslot, so the active set has to be dense.
        if self.active.iter().enumerate().any(|(i, &s)| i != s) {
            return Err(format!("active set is not dense: {:?}", self.active));
        }
        let l = self.model.launcher();
        let cfg = self.cfg.clone();
        let n = cfg.n_embd;
        let positions: Vec<usize> = self.active.iter().map(|&s| self.positions[s]).collect();
        // `[slots][2]` = (position, seq) for the batched append/rope/attention.
        let mut pos_bytes = vec![0u8; 8 * b];
        for (i, &p) in positions.iter().enumerate() {
            pos_bytes[i * 8..i * 8 + 4].copy_from_slice(&(p as i32).to_le_bytes());
            pos_bytes[i * 8 + 4..i * 8 + 8].copy_from_slice(&((p as i32) + 1).to_le_bytes());
        }
        self.pos_buf.upload_bytes_async(0, &pos_bytes, l.stream())?;

        // Stage timing: `stage!(i)` closes the previous bucket and opens `i`.
        let trace = self.trace;
        let mut stg = self.stages;
        let mut last: Option<Instant> = None;
        let mut prev = 0usize;
        macro_rules! stage {
            ($i:expr) => {
                if trace {
                    l.synchronize()?;
                    if let Some(t) = last.take() {
                        stg[prev] += t.elapsed().as_secs_f64();
                    }
                    last = Some(Instant::now());
                    prev = $i;
                }
            };
        }
        stage!(0);
        for (bi, &tok) in tokens.iter().enumerate() {
            l.embed_row_q4_k_at(self.embed.0, tok as usize, self.embed.1, &self.x, bi * n, n)?;
        }
        stage!(1);

        for i in 0..self.layers.len() {
            let block = &self.layers[i];
            l.rmsnorm_rows(&self.x, &block.attn_norm, &self.t, n, cfg.eps, b)?;
            let block_out: &Buffer = match &block.kind {
                BatchKind::Full(f) => {
                    stage!(1);
                    f.attn.qkv_rows(l, &self.t, &f.qg, &f.k, &f.v, b)?;
                    f.attn.split_rope_rows(
                        l,
                        &f.qg,
                        &f.k,
                        &f.q,
                        &f.gate,
                        &positions,
                        &self.pos_buf,
                    )?;
                    stage!(2);
                    let qw = cfg.n_head * cfg.head_dim;
                    // One append and one split-K attention for every slot; the
                    // per-slot position and contiguous cache are read from the
                    // slot index on grid.y.
                    let dk = &f.kv;
                    let seq_max = positions.iter().max().copied().unwrap_or(0) + 1;
                    let splits = (seq_max / 64).clamp(1, MAX_SPLITS);
                    l.kv_encode_batch(
                        dk.dtype.id(),
                        &f.k,
                        &f.v,
                        &dk.k,
                        &dk.v,
                        cfg.n_head_kv,
                        cfg.head_dim,
                        dk.seq,
                        &self.pos_buf,
                        dk.slot_bytes(),
                        b,
                    )?;
                    l.attn_decode_batch(
                        dk.dtype.id(),
                        &self.pos_buf,
                        &f.q,
                        &dk.k,
                        &dk.v,
                        &dk.part_acc,
                        &dk.part_ml,
                        &f.out,
                        cfg.n_head,
                        cfg.n_head_kv,
                        cfg.head_dim,
                        dk.seq,
                        1.0f32 / (cfg.head_dim as f32).sqrt(),
                        splits,
                        dk.slot_bytes(),
                        dk.acc_slot(),
                        dk.ml_slot(),
                        qw,
                        b,
                    )?;
                    stage!(3);
                    f.attn.out_rows(l, &f.out, &f.gate, &f.y, b)?;
                    stage!(7);
                    &f.y
                }
                BatchKind::Ssm(s) => {
                    stage!(4);
                    s.inst
                        .gemvs_batched(l, &self.t, &s.qkv, &s.z, &s.alpha, &s.beta, b)?;
                    stage!(5);
                    s.inst.recurrent_batched(
                        l,
                        &s.qkv,
                        &s.z,
                        &s.alpha,
                        &s.beta,
                        &s.g,
                        &s.beta_sig,
                        &s.conv,
                        &s.o,
                        &s.gated,
                        &s.conv_state,
                        &s.s_state,
                        b,
                    )?;
                    stage!(6);
                    s.inst.output_gemv_batched(l, &s.gated, &s.out, b)?;
                    stage!(7);
                    &s.out
                }
            };
            l.residual_buf(&self.x, block_out, &self.x1, b * n)?;
            l.rmsnorm_rows(&self.x1, &block.post_norm, &self.t, n, cfg.eps, b)?;
            block.gate.gemm_batched(l, &self.t, &self.gate, b)?;
            block.up.gemm_batched(l, &self.t, &self.up, b)?;
            l.silu_mul_buf(&self.gate, &self.up, &self.act, b * cfg.n_ff)?;
            block.down.gemm_batched(l, &self.act, &self.down, b)?;
            l.residual_buf(&self.down, &self.x1, &self.x, b * n)?;
        }
        for &slot in &self.active {
            self.positions[slot] += 1;
        }

        stage!(8);
        l.rmsnorm_rows(&self.x, &self.output_norm, &self.final_h, n, cfg.eps, b)?;
        let out = self
            .model
            .tensor(crate::OUTPUT)
            .ok_or_else(|| "model has no output.weight".to_string())?;
        let kind = crate::gpu_weights::batch_kind(gemv_entry(out.ty())?)
            .ok_or_else(|| "lm_head has no batched GEMV".to_string())?;
        l.gemv_batched(
            kind,
            out.buffer(),
            &self.final_h,
            &self.logits,
            cfg.n_vocab,
            n,
            b,
        )?;
        l.argmax_batched(
            &self.logits,
            &self.arg_v,
            &self.arg_i,
            &self.arg_out,
            cfg.n_vocab,
            ARGMAX_PARTS,
            b,
        )?;
        l.synchronize()?;
        if trace {
            if let Some(t) = last.take() {
                stg[prev] += t.elapsed().as_secs_f64();
            }
            self.stages = stg;
        }
        let bytes = self.arg_out.download_bytes(0, 4 * b)?;
        Ok((0..b)
            .map(|i| {
                let o = i * 4;
                u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]])
            })
            .collect())
    }
}
