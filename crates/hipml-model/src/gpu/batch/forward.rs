//! The batched decode step. Child of `batch` so it can reach the engine's
//! private per-slot state.
//!
//! One weight pass per projection, then the per-slot work (attention at that
//! slot's position, the SSM recurrence against that slot's state, sampling).
//! The two streams carry alternating slots so the small per-slot kernels
//! overlap instead of serialising.

use super::*;
use std::time::Instant;

impl<'a> BatchEngine<'a> {
    /// One decode step for every active slot. `tokens[b]` is the input token
    /// for active slot `b`; returns that slot's greedy next token.
    pub fn forward(&mut self, tokens: &[u32]) -> Result<Vec<u32>, String> {
        let b = tokens.len();
        if b != self.active.len() {
            return Err(format!("{} tokens for {} active slots", b, self.active.len()));
        }
        let l = self.model.launcher();
        let cfg = self.cfg.clone();
        let n = cfg.n_embd;
        let positions: Vec<usize> = self.active.iter().map(|&s| self.positions[s]).collect();

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
                    f.attn.split_rope_rows(l, &f.qg, &f.k, &f.q, &f.gate, &positions)?;
                    stage!(2);
                    let width = cfg.n_head_kv * cfg.head_dim;
                    let qw = cfg.n_head * cfg.head_dim;
                    // Slot parity picks the stream (see the SSM split): the
                    // side stream has its own step buffer so the KV appends do
                    // not race on one position.
                    let split = b > 1;
                    if split {
                        l.fork()?;
                    }
                    for bi in 0..b {
                        let slot = self.active[bi];
                        let pos = positions[bi];
                        let dk = &f.deep[slot];
                        let side = bi % 2 == 1;
                        if side {
                            l.set_step_side(pos, pos + 1, 0)?;
                            l.kv_encode_ptr_on(
                                l.stream2(),
                                l.step_side_buf(),
                                dk.dtype.id(),
                                row_ptr(&f.k, bi, width),
                                row_ptr(&f.v, bi, width),
                                &dk.k,
                                &dk.v,
                                cfg.n_head_kv,
                                cfg.head_dim,
                                dk.seq,
                            )?;
                        } else {
                            l.set_step(pos, pos + 1, 0)?;
                            l.kv_encode_ptr(
                                dk.dtype.id(),
                                row_ptr(&f.k, bi, width),
                                row_ptr(&f.v, bi, width),
                                &dk.k,
                                &dk.v,
                                cfg.n_head_kv,
                                cfg.head_dim,
                                dk.seq,
                            )?;
                        }
                        super::attn_tile::deep_attention_ptr_on(
                            l,
                            &cfg,
                            dk,
                            pos,
                            row_ptr(&f.q, bi, qw),
                            row_ptr(&f.out, bi, qw),
                            side,
                        )?;
                    }
                    if b > 1 {
                        l.join()?;
                    }
                    stage!(3);
                    f.attn.out_rows(l, &f.out, &f.gate, &f.y, b)?;
                    stage!(7);
                    &f.y
                }
                BatchKind::Ssm(s) => {
                    stage!(4);
                    s.per_slot[0].gemvs_batched(
                        l, &self.t, &s.qkv, &s.z, &s.alpha, &s.beta, b,
                    )?;
                    stage!(5);
                    let d = SsmDims::from_config(&cfg);
                    l.fork()?;
                    for bi in 0..b {
                        let slot = self.active[bi];
                        let inst = &s.per_slot[slot];
                        inst.recurrent_at(
                            row_ptr(&s.qkv, bi, d.conv_ch),
                            row_ptr(&s.z, bi, d.inner()),
                            row_ptr(&s.alpha, bi, d.n_v),
                            row_ptr(&s.beta, bi, d.n_v),
                            row_ptr(&s.g, bi, d.n_v),
                            row_ptr(&s.beta_sig, bi, d.n_v),
                            row_ptr(&s.conv, bi, d.conv_ch),
                            row_ptr(&s.o, bi, d.inner()),
                            row_ptr(&s.gated, bi, d.inner()),
                        )?;
                    }
                    stage!(6);
                    s.per_slot[0].output_gemv_batched(l, &s.gated, &s.out, b)?;
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
