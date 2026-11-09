//! Chunked prefill: a prompt tile as batched GEMMs instead of N decode steps.
//!
//! The decode graph is one token at a time, so a 512-token prompt streams the
//! whole 15.89 GiB of weights 512 times. Here a chunk of `T` tokens streams them
//! once: every projection becomes a tiled GEMM over the chunk, and only the work
//! that is genuinely sequential in time — the gated-delta-net recurrence, the
//! conv window, rope, the KV append and causal attention — stays per token.
//!
//! ```text
//! xc  = embed(tokens)                         T launches (one row each)
//! per layer:
//!   tc   = rmsnorm_rows(xc, attn_norm)
//!   block = chunk projections + per-token recurrence/attention
//!   x1c  = xc + block;  tc = rmsnorm_rows(x1c, post_attention_norm)  (fused)
//!   act  = silu(gemm(gate, tc)) * gemm(up, tc)
//!   xc   = x1c + gemm(down, act)                                     (+ next norm, fused)
//! logits = lm_head(rmsnorm(last row))
//! ```
//!
//! Numerics: the projections use the tiled prefill GEMM, which accumulates a
//! 32-value k-tile at a time rather than the GEMV's 2,048-value shared tile, so
//! a chunk can differ from the token loop in the last bits. Everything else is
//! the same kernel chain, and the decode path is unchanged: `--prefill-chunk`
//! opts in, the token loop remains the default and the parity reference.

use super::*;
use hipml_hip::Buffer;
use hipml_kernels::ssm::SsmChunk;
use std::time::Instant;

use crate::gpu_weights::gemv_entry;
use crate::OUTPUT;

/// Where a chunk's wall time went, measured with one stream sync per stage.
///
/// Only `gpu-bench --prefill-split` asks for it; the ordinary path passes
/// `None` and pays nothing but a branch.
#[derive(Default, Debug)]
pub struct PrefillReport {
    pub embed_ms: f64,
    /// RMS norms, residuals, SwiGLU and the output norm.
    pub elementwise_ms: f64,
    /// Attention q/k/v/o projections.
    pub attn_gemm_ms: f64,
    /// GDN qkv/gate/out projections (alpha/beta are f32 and tiny).
    pub ssm_gemm_ms: f64,
    /// FFN gate/up/down and the SwiGLU between them.
    pub ffn_gemm_ms: f64,
    /// Output norm, `lm_head` and argmax.
    pub final_ms: f64,
    /// The chunked GDN chain (gates, conv, L2, recurrence, gated RMS).
    pub ssm_seq_ms: f64,
    /// Per-token full-attention chain and its output projection.
    pub attn_seq_ms: f64,
    last: Option<Instant>,
}

impl PrefillReport {
    fn start(&mut self) {
        self.last = Some(Instant::now());
    }

    fn lap(&mut self, l: &hipml_kernels::Launcher, slot: fn(&mut PrefillReport) -> &mut f64) {
        let Some(last) = self.last else { return };
        if l.synchronize().is_ok() {
            let now = Instant::now();
            *slot(self) += now.duration_since(last).as_secs_f64() * 1e3;
            self.last = Some(now);
        }
    }

    pub fn total_ms(&self) -> f64 {
        self.embed_ms
            + self.elementwise_ms
            + self.attn_gemm_ms
            + self.ssm_gemm_ms
            + self.ffn_gemm_ms
            + self.final_ms
            + self.ssm_seq_ms
            + self.attn_seq_ms
    }
}

/// Chunk scratch, one set shared by every layer (layers run in sequence).
pub(crate) struct Chunk {
    /// Residual stream and its two normalised copies, `[T][n_embd]`.
    pub x: Buffer,
    pub x1: Buffer,
    pub t: Buffer,
    /// FFN activations, `[T][n_ff]`.
    pub gate: Buffer,
    pub up: Buffer,
    pub act: Buffer,
    pub down: Buffer,
    /// Gated-delta-net chunk.
    pub ssm: SsmChunk,
    /// Attention chunk: q|gate rows, q, k, v, out, and the layer output.
    pub qg: Buffer,
    pub q: Buffer,
    pub aq_gate: Buffer,
    pub k: Buffer,
    pub v: Buffer,
    pub aout: Buffer,
    pub y: Buffer,
    /// fp16 copy of the activation tile, converted once per projection. Holds
    /// `tokens * max(n_embd, n_ff, ssm inner)` halves, i.e. half that many
    /// 4-byte words.
    pub xh: Buffer,
    /// Second k-partial for split-K ffn projections. Only allocated for short
    /// prompts, where the ffn tile is wave-starved; `tokens * max(n_embd,n_ff)`.
    pub sk: Buffer,
    /// Split-K attention partials: `[splits][qslots][n_head][head_dim]` and
    /// `[splits][qslots][n_head][2]`, f32, `qslots` the chunk rounded up to a
    /// 32-query tile. One allocation per engine, reused by every layer.
    pub apart: Buffer,
    pub apart_ml: Buffer,
    /// Key splits the deep chunks use, from `HIPML_ATTN_SPLITS` (default 8,
    /// 1 disables).
    pub attn_splits: usize,
}

impl Chunk {
    /// `deep` asks for the split-attention scratch: a prompt that never passes
    /// the split gate must not pay to allocate it (the 512-token case is 8x
    /// the partials for no work, and its prefill is under half a second).
    fn new(cfg: &hipml_core::config::Config, tokens: usize, deep: bool) -> Result<Chunk, String> {
        let n = cfg.n_embd;
        let ff = cfg.n_ff;
        let qd = cfg.n_head * cfg.head_dim;
        let w = cfg.n_head_kv * cfg.head_dim;
        let zeros = |len: usize| Buffer::alloc(len);
        let sd = hipml_kernels::ssm::SsmDims::from_config(cfg);
        // The fp16 activation copy must hold the widest tile any projection
        // reads: the ffn tiles, the ssm `out` input and the plain residual.
        let wide = n.max(ff).max(sd.inner());
        // Split-K only fires for prompts short enough that the ffn tile is
        // under three waves (see `prefill_should_split`); at the 2048-token
        // chunk used by long prompts the scratch would never be touched.
        let sk = if tokens <= 1024 { tokens * wide } else { 1 };
        // Split-K attention scratch. `splits` is the env knob; the buffer is
        // capped at 1 GiB so a deep prompt cannot turn it into a residency
        // problem (8 splits of a 2048-token chunk is 403 MiB).
        let qslots = (tokens + 31) / 32 * 32;
        let mut attn_splits = if deep { attn_tile::env_splits() } else { 1 };
        let part_len = |splits: usize| splits * qslots * cfg.n_head * cfg.head_dim;
        while attn_splits > 1 && part_len(attn_splits) * 4 > 1 << 30 {
            attn_splits /= 2;
        }
        Ok(Chunk {
            x: zeros(tokens * n)?,
            x1: zeros(tokens * n)?,
            t: zeros(tokens * n)?,
            gate: zeros(tokens * ff)?,
            up: zeros(tokens * ff)?,
            act: zeros(tokens * ff)?,
            down: zeros(tokens * n)?,
            ssm: SsmChunk::new(&sd, tokens)?,
            qg: zeros(tokens * 2 * qd)?,
            q: zeros(tokens * qd)?,
            aq_gate: zeros(tokens * qd)?,
            k: zeros(tokens * w)?,
            v: zeros(tokens * w)?,
            aout: zeros(tokens * qd)?,
            y: zeros(tokens * n)?,
            xh: zeros((tokens * wide + 1) / 2)?,
            sk: zeros(sk)?,
            apart: zeros(part_len(attn_splits).max(1))?,
            apart_ml: zeros((attn_splits * qslots * cfg.n_head * 2).max(2))?,
            attn_splits,
        })
    }

    fn bytes(&self) -> usize {
        (self.x.len()
            + self.x1.len()
            + self.t.len()
            + self.gate.len()
            + self.up.len()
            + self.act.len()
            + self.down.len()
            + self.qg.len()
            + self.q.len()
            + self.aq_gate.len()
            + self.k.len()
            + self.v.len()
            + self.aout.len()
            + self.y.len()
            + self.xh.len()
            + self.sk.len()
            + self.apart.len()
            + self.apart_ml.len()) * 4
            + self.ssm.bytes()
    }
}

impl<'a> GpuEngine<'a> {
    /// Prefill `tokens` in tiles of `chunk` tokens; returns the id for the last
    /// token, i.e. the first sampled token of the reply.
    pub fn prefill_chunked(
        &mut self,
        tokens: &[u32],
        st: &mut GpuState,
        chunk: usize,
    ) -> Result<u32, String> {
        // Scratch is sized by the chunk, so a short prompt never asks for more.
        let chunk = chunk.min(tokens.len()).max(1);
        let deep = attn_tile::split_count(attn_tile::env_splits(), tokens.len()) > 1;
        self.ensure_chunk(chunk, deep)?;
        let mut rep: Option<PrefillReport> = None;
        let mut last = 0u32;
        for tile in tokens.chunks(chunk) {
            last = self.forward_chunk(tile, st, &mut rep)?;
        }
        Ok(last)
    }

    /// Same prefill, with the time per stage measured by synchronizing between
    /// them. Slower than `prefill_chunked`; this is the breakdown command.
    pub fn prefill_chunked_report(
        &mut self,
        tokens: &[u32],
        st: &mut GpuState,
        chunk: usize,
    ) -> Result<(u32, PrefillReport), String> {
        let chunk = chunk.min(tokens.len()).max(1);
        let deep = attn_tile::split_count(attn_tile::env_splits(), tokens.len()) > 1;
        self.ensure_chunk(chunk, deep)?;
        let mut rep = Some(PrefillReport::default());
        let mut last = 0u32;
        for tile in tokens.chunks(chunk) {
            last = self.forward_chunk(tile, st, &mut rep)?;
        }
        Ok((last, rep.unwrap_or_default()))
    }

    /// Same prefill, but also write every token's post-`output_norm` hidden
    /// state into `capture` at its absolute row. The speculative driver seeds
    /// the MTP drafter from these rows.
    pub fn prefill_chunked_capture(
        &mut self,
        tokens: &[u32],
        st: &mut GpuState,
        chunk: usize,
        capture: &Buffer,
    ) -> Result<u32, String> {
        let chunk = chunk.min(tokens.len()).max(1);
        let deep = attn_tile::split_count(attn_tile::env_splits(), tokens.len()) > 1;
        self.ensure_chunk(chunk, deep)?;
        let mut last = 0u32;
        for tile in tokens.chunks(chunk) {
            last = self.forward_chunk_capture(tile, st, &mut None, Some(capture))?;
        }
        Ok(last)
    }

    /// One chunk through the whole graph. `st.pos` advances by the chunk length.
    fn forward_chunk(
        &mut self,
        tokens: &[u32],
        st: &mut GpuState,
        rep: &mut Option<PrefillReport>,
    ) -> Result<u32, String> {
        self.forward_chunk_capture(tokens, st, rep, None)
    }

    fn forward_chunk_capture(
        &mut self,
        tokens: &[u32],
        st: &mut GpuState,
        rep: &mut Option<PrefillReport>,
        capture: Option<&Buffer>,
    ) -> Result<u32, String> {
        let n_tok = tokens.len();
        let pos0 = st.pos;
        if n_tok == 0 {
            return Err("empty chunk".to_string());
        }
        let cfg = self.cfg.clone();
        let n = cfg.n_embd;
        let l = self.model.launcher();

        // Scratch is allocated once per engine and reused by every tile: a
        // 150 MiB scratch is ~20 hipMalloc calls, which is most of a short
        // prompt's wall time if it is rebuilt per chunk.
        let (cap, c) = self
            .chunk
            .as_ref()
            .ok_or_else(|| "chunk scratch not allocated".to_string())?;
        if *cap < n_tok {
            return Err(format!("chunk of {n_tok} tokens exceeds scratch {cap}"));
        }
        // The embedding table and the lm head are the model's, not the engine's.
        let embed = self.embed;
        if let Some(r) = rep.as_mut() {
            r.start();
        }
        for (t, &id) in tokens.iter().enumerate() {
            l.embed_row_q4_k_at(embed.0, id as usize, embed.1, &c.x, t * n, n)?;
        }
        if let Some(r) = rep.as_mut() {
            r.lap(l, |r| &mut r.embed_ms);
        }

        // The first layer's input norm has no residual to fold into it (the
        // embedding wrote `x` directly); every later layer's input norm is
        // fused with the residual that produces its input at the bottom of the
        // loop.
        l.rmsnorm_rows(&c.x, &self.layers[0].attn_norm, &c.t, n, cfg.eps, n_tok)?;
        if let Some(r) = rep.as_mut() {
            r.lap(l, |r| &mut r.elementwise_ms);
        }
        for i in 0..self.layers.len() {
            let block = &self.layers[i];
            let block_out: &Buffer = match &block.kind {
                BlockKind::Ssm(s) => {
                    s.project_chunk(l, &c.t, &c.xh, &c.ssm, n_tok)?;
                    if let Some(r) = rep.as_mut() {
                        r.lap(l, |r| &mut r.ssm_gemm_ms);
                    }
                    if std::env::var_os("HIPML_TOKEN_STEPS").is_some() {
                        for t in 0..n_tok {
                            s.step_chunk_row(&c.ssm, t)?;
                        }
                    } else {
                        s.steps_chunk(&c.ssm, n_tok)?;
                    }
                    s.out_chunk(l, &c.xh, &c.ssm, n_tok)?;
                    if let Some(r) = rep.as_mut() {
                        r.lap(l, |r| &mut r.ssm_seq_ms);
                    }
                    &c.ssm.out
                }
                BlockKind::Full { attn, deep } => {
                    // One batched GEMM per chunk for q/k/v, then the cache-
                    // dependent attention tile (q8/q4 read in place, or the
                    // row-at-a-time fallback).
                    attn.project_chunk(l, &c.t, &c.xh, &c.qg, &c.k, &c.v, n_tok)?;
                    if let Some(r) = rep.as_mut() {
                        r.lap(l, |r| &mut r.attn_gemm_ms);
                    }
                    attn_tile::attention_tile(
                        l,
                        &cfg,
                        attn,
                        deep.as_ref(),
                        c,
                        pos0,
                        n_tok,
                    )?;
                    attn.apply_output_chunk(l, &c.aout, &c.aq_gate, &c.y, n_tok)?;
                    if let Some(r) = rep.as_mut() {
                        r.lap(l, |r| &mut r.attn_seq_ms);
                    }
                    &c.y
                }
            };
            l.residual_rmsnorm_or_pair(
                &c.x,
                block_out,
                &block.post_norm,
                &c.x1,
                &c.t,
                n,
                cfg.eps,
                n_tok,
            )?;
            if let Some(r) = rep.as_mut() {
                r.lap(l, |r| &mut r.elementwise_ms);
            }

            if ::hipml_kernels::prefill_v2() {
                // One fp16 copy of the activation tile per projection: the same
                // tile is staged once per output row tile inside the GEMM, so
                // halving its bytes is the cheapest win in the ffn shapes.
                ::hipml_kernels::prefill_xhalf(
                    l.module(),
                    l.stream(),
                    &c.t,
                    &c.xh,
                    n_tok * cfg.n_embd,
                )?;
                block.gate.gemm_half(l, &c.xh, &c.gate, &c.sk, n_tok)?;
                block.up.gemm_half(l, &c.xh, &c.up, &c.sk, n_tok)?;
                l.silu_mul_buf(&c.gate, &c.up, &c.act, n_tok * cfg.n_ff)?;
                ::hipml_kernels::prefill_xhalf(
                    l.module(),
                    l.stream(),
                    &c.act,
                    &c.xh,
                    n_tok * cfg.n_ff,
                )?;
                block.down.gemm_half(l, &c.xh, &c.down, &c.sk, n_tok)?;
            } else {
                block.gate.gemm(l, &c.t, &c.gate, n_tok)?;
                block.up.gemm(l, &c.t, &c.up, n_tok)?;
                l.silu_mul_buf(&c.gate, &c.up, &c.act, n_tok * cfg.n_ff)?;
                block.down.gemm(l, &c.act, &c.down, n_tok)?;
            }
            if let Some(r) = rep.as_mut() {
                r.lap(l, |r| &mut r.ffn_gemm_ms);
            }
            // Bottom residual: `x = x1 + down`. When another layer follows,
            // fold its input norm into the same pass; the last layer's `x` is
            // only read by the logits projection, so it stays a bare add.
            match self.layers.get(i + 1) {
                Some(next) => l.residual_rmsnorm_or_pair(
                    &c.down,
                    &c.x1,
                    &next.attn_norm,
                    &c.x,
                    &c.t,
                    n,
                    cfg.eps,
                    n_tok,
                )?,
                None => l.residual_buf(&c.down, &c.x1, &c.x, n_tok * n)?,
            }
            if let Some(r) = rep.as_mut() {
                r.lap(l, |r| &mut r.elementwise_ms);
            }
        }

        // Every row's post-norm hidden state, for the speculative drafter.
        if let Some(cap) = capture {
            let dst = unsafe {
                (cap.as_ptr() as *mut u8).add(pos0 * n * 4) as *mut std::ffi::c_void
            };
            l.rmsnorm_ptr(c.x.as_ptr(), &self.output_norm, dst, n, cfg.eps, n_tok)?;
        }

        // The chunk's last row is the only one that needs logits.
        let last_row = unsafe {
            (c.x.as_ptr() as *mut u8).add((n_tok - 1) * n * 4) as *mut std::ffi::c_void
        };
        l.rmsnorm_ptr(last_row, &self.output_norm, self.final_h.as_ptr(), n, cfg.eps, 1)?;
        let out = self
            .model
            .tensor(OUTPUT)
            .ok_or_else(|| "model has no output.weight".to_string())?;
        l.run_quant_resident_ptr(
            gemv_entry(out.ty())?,
            out.buffer(),
            self.final_h.as_ptr(),
            self.logits.as_ptr(),
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
        if let Some(r) = rep.as_mut() {
            r.lap(l, |r| &mut r.final_ms);
        }
        l.synchronize()?;
        let bytes = self.arg_out.download_bytes(0, 4)?;
        st.pos += n_tok;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    /// Allocate (or grow) the chunk scratch. Called by `prefill_chunked`.
    fn ensure_chunk(&mut self, tokens: usize, deep: bool) -> Result<(), String> {
        let have = match &self.chunk {
            Some((cap, c)) => *cap >= tokens && (c.attn_splits > 1) == deep,
            None => false,
        };
        if !have {
            self.chunk = Some((tokens, Chunk::new(&self.cfg, tokens, deep)?));
        }
        Ok(())
    }

    /// Bytes the chunk scratch holds after `prefill_chunked(tokens, ..)`.
    pub fn chunk_bytes(&mut self, tokens: usize) -> usize {
        match self.ensure_chunk(tokens, true) {
            Ok(()) => self.chunk.as_ref().map(|(_, c)| c.bytes()).unwrap_or(0),
            Err(_) => 0,
        }
    }
}
