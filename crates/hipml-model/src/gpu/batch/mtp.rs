//! The MTP drafter, batched over slots.
//!
//! The single-stream drafter ([`crate::gpu::MtpGpu`]) owns one f32 KV cache and
//! one row of scratch. Here the same block runs `B` rows through one weight
//! pass: `eh_proj`, the attention projections and the FFN are batched GEMVs,
//! the per-row q-gate/norm/rope and head are the `_b` kernels, and each slot
//! keeps its own quantized KV cache and hidden state.
//!
//! Draft proposals never affect exactness — the target verifies them — so the
//! batched drafter uses a q8 KV cache (the batch engine's format) instead of the
//! single-stream f32 cache. Accepted/rejected positions are rewritten from the
//! target's verify hidden states, exactly as the single-stream loop does.

use super::*;
use crate::gpu_weights::{gemv_entry, Proj};

/// The drafter block's feed-forward: the dense SwiGLU of a `blk.64` sidecar, or
/// the routed FFN of an inline MTP block on a mixture-of-experts file. The same
/// split, and the same reason, as the decoder's `BatchFfn`: a MoE layer has no
/// `ffn_gate.weight` at all.
enum MtpFfn<'a> {
    Dense {
        gate: Proj<'a>,
        up: Proj<'a>,
        down: Proj<'a>,
    },
    /// Boxed: the expert bank views are larger than the three projections.
    Moe(Box<MoeBlock<'a>>),
}

/// One MTP block's weights and per-slot state.
pub struct MtpBatch<'a> {
    cfg: Config,
    slots: usize,
    enorm: Buffer,
    hnorm: Buffer,
    attn_norm: Buffer,
    post_norm: Buffer,
    head_norm: Buffer,
    eh_proj: Proj<'a>,
    attn: AttnLayer<'a>,
    ffn: MtpFfn<'a>,
    /// The routed FFN's scratch, present only for a mixture of experts: one row
    /// per slot, which is the shape `MoeBlock::forward_batch` runs.
    moe: Option<MoeBatch>,
    kv: kv::BatchKv,
    emb: Buffer,
    en: Buffer,
    hn: Buffer,
    cat: Buffer,
    x: Buffer,
    x1: Buffer,
    t: Buffer,
    g: Buffer,
    u: Buffer,
    act: Buffer,
    d: Buffer,
    out: Buffer,
    /// `[slots][n_embd]`, the drafter's own recursive hidden state.
    hnext: Buffer,
    qg: Buffer,
    q: Buffer,
    gate_b: Buffer,
    k: Buffer,
    v: Buffer,
    aout: Buffer,
    y: Buffer,
    logits: Buffer,
    arg_v: Buffer,
    arg_i: Buffer,
    arg_out: Buffer,
    pos: Buffer,
    launcher: &'a Launcher,
    head: (&'a ByteBuffer, hipml_core::quant::TensorType),
    embed: (&'a ByteBuffer, usize, hipml_core::quant::TensorType),
    weight_bytes: usize,
}

impl<'a> MtpBatch<'a> {
    /// Build the drafter from the sidecar's `blk.64` tensors already merged
    /// into `model` (`Model::load_sidecar`), with one q8 KV cache per slot.
    pub fn new(
        model: &'a Model,
        n_ctx: usize,
        slots: usize,
        cfg: Config,
    ) -> Result<MtpBatch<'a>, String> {
        let n = cfg.n_embd;
        let ff = cfg.n_ff;
        let qw = cfg.n_head * cfg.head_dim;
        let w = cfg.n_head_kv * cfg.head_dim;
        let blk = super::super::mtp::mtp_blk(&cfg);
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
        let embed_t = model
            .tensor(crate::TOKEN_EMBD)
            .ok_or_else(|| "model has no token_embd.weight".to_string())?;
        let embed = (
            embed_t.buffer(),
            (embed_t.nbytes() / embed_t.rows()) as usize,
            embed_t.ty(),
        );
        let head_t = model
            .tensor(crate::OUTPUT)
            .ok_or_else(|| "model has no output.weight".to_string())?;
        let head = (head_t.buffer(), head_t.ty());
        // Everything `write_kv` walks is sized for a full verify tile: one slot
        // can contribute up to 1 + 3 draft columns, and the prompt seed writes
        // 16 rows at a time even when there is one slot.
        let cap = (slots * 4).max(16);
        let z = |len: usize| Buffer::alloc((len * cap).max(1));
        let eh_proj = p("nextn.eh_proj.weight")?;
        // The drafter's FFN is whatever the base's is, exactly as the
        // single-stream drafter builds it: `MoeBlock` reads `blk.<n>.ffn_*`
        // with the names and shapes the decoder's layers use, so an inline MTP
        // block on a MoE file needs no separate reader.
        let ffn = match cfg.n_expert {
            0 => MtpFfn::Dense {
                gate: p("ffn_gate.weight")?,
                up: p("ffn_up.weight")?,
                down: p("ffn_down.weight")?,
            },
            _ => MtpFfn::Moe(Box::new(MoeBlock::new(model, &cfg, blk)?)),
        };
        let moe = match cfg.n_expert {
            0 => None,
            _ => Some(MoeBatch::new(&cfg, slots)?),
        };
        let weight_bytes = eh_proj.bytes()
            + attn.weight_bytes()
            + match &ffn {
                MtpFfn::Dense { gate, up, down } => gate.bytes() + up.bytes() + down.bytes(),
                MtpFfn::Moe(m) => m.bytes(),
            };
        Ok(MtpBatch {
            cfg: cfg.clone(),
            slots,
            enorm: Buffer::from_host(&v("nextn.enorm.weight")?)?,
            hnorm: Buffer::from_host(&v("nextn.hnorm.weight")?)?,
            attn_norm: Buffer::from_host(&v("attn_norm.weight")?)?,
            post_norm: Buffer::from_host(&v("post_attention_norm.weight")?)?,
            head_norm: Buffer::from_host(&v("nextn.shared_head_norm.weight")?)?,
            eh_proj,
            attn,
            ffn,
            moe,
            kv: kv::BatchKv::new(&cfg, n_ctx, KvDtype::Q8, slots)?,
            emb: z(n)?,
            en: z(n)?,
            hn: z(n)?,
            cat: z(2 * n)?,
            x: z(n)?,
            x1: z(n)?,
            t: z(n)?,
            g: z(ff)?,
            u: z(ff)?,
            act: z(ff)?,
            d: z(n)?,
            out: z(n)?,
            hnext: z(n)?,
            qg: z(2 * qw)?,
            q: z(qw)?,
            gate_b: z(qw)?,
            k: z(w)?,
            v: z(w)?,
            aout: z(qw)?,
            y: z(n)?,
            logits: Buffer::alloc(cfg.n_vocab * slots)?,
            arg_v: Buffer::alloc(ARGMAX_PARTS * slots)?,
            arg_i: Buffer::alloc(ARGMAX_PARTS * slots)?,
            arg_out: Buffer::alloc(slots)?,
            pos: Buffer::alloc(2 * cap)?,
            launcher: l,
            head,
            embed,
            weight_bytes,
        })
    }

    pub fn slots(&self) -> usize {
        self.slots
    }

    /// Bytes the drafter's own weights add on top of the target's head.
    pub fn weight_bytes(&self) -> usize {
        self.weight_bytes
    }

    /// Drafter KV bytes for all slots.
    pub fn kv_bytes(&self) -> usize {
        self.kv.bytes()
    }

    /// Copy target hidden rows into the per-slot seed slots.
    /// `rows[i] = (slot, row_of_src)` seeds that slot's next draft.
    pub fn set_seed_hidden(
        &self,
        src: &Buffer,
        rows: &[(usize, usize)],
        width: usize,
    ) -> Result<(), String> {
        let l = self.launcher;
        for &(slot, row) in rows {
            if slot >= self.slots {
                return Err(format!("seed hidden: slot {slot} outside {}", self.slots));
            }
            src.copy_range_on(&self.hnext, row * width, slot * width, width, l.stream())?;
        }
        Ok(())
    }

    /// One draft step for every slot: `tokens[i]` at `positions[i]` produces
    /// slot `i`'s drafted id, and `hnext` becomes the recursive input for the
    /// next step. All `slots` rows run through the block in one weight pass.
    pub fn draft(&mut self, tokens: &[u32], positions: &[usize]) -> Result<Vec<u32>, String> {
        let b = tokens.len();
        if b == 0 || b > self.slots || positions.len() != b {
            return Err(format!("draft: {b} tokens for {} slots", self.slots));
        }
        let cfg = self.cfg.clone();
        let n = cfg.n_embd;
        let ff = cfg.n_ff;
        let l = self.attn_launcher();
        let mut pos_bytes = vec![0u8; 8 * b];
        for (i, &p) in positions.iter().enumerate() {
            pos_bytes[i * 8..i * 8 + 4].copy_from_slice(&(p as i32).to_le_bytes());
            pos_bytes[i * 8 + 4..i * 8 + 8].copy_from_slice(&((p as i32) + 1).to_le_bytes());
        }
        self.pos.upload_bytes_async(0, &pos_bytes, l.stream())?;

        for (i, &tok) in tokens.iter().enumerate() {
            l.embed_row_at(
                self.embed.2,
                self.embed.0,
                tok as usize,
                self.embed.1,
                &self.emb,
                i * n,
                n,
            )?;
        }
        l.rmsnorm_rows(&self.emb, &self.enorm, &self.en, n, cfg.eps, b)?;
        l.rmsnorm_rows(&self.hnext, &self.hnorm, &self.hn, n, cfg.eps, b)?;
        let stream = l.stream();
        for i in 0..b {
            self.en
                .copy_range_on(&self.cat, i * n, i * 2 * n, n, stream)?;
            self.hn
                .copy_range_on(&self.cat, i * n, i * 2 * n + n, n, stream)?;
        }
        self.eh_proj.gemm_batched(l, &self.cat, &self.x, b)?;
        l.rmsnorm_rows(&self.x, &self.attn_norm, &self.t, n, cfg.eps, b)?;
        self.attn
            .qkv_rows(l, &self.t, &self.qg, &self.k, &self.v, b)?;
        self.attn.split_rope_rows(
            l,
            &self.qg,
            &self.k,
            &self.q,
            &self.gate_b,
            positions,
            &self.pos,
        )?;
        // Same split geometry as a plain step: the count is a property of the
        // cache, not of this column's position.
        let split = self.kv.splits;
        l.kv_encode_batch(
            self.kv.dtype.id(),
            &self.k,
            &self.v,
            &self.kv.k,
            &self.kv.v,
            cfg.n_head_kv,
            cfg.head_dim,
            self.kv.seq,
            &self.pos,
            self.kv.slot_bytes(),
            b,
        )?;
        let qw = cfg.n_head * cfg.head_dim;
        l.attn_decode_batch(
            self.kv.dtype.id(),
            &self.pos,
            &self.q,
            &self.kv.k,
            &self.kv.v,
            &self.kv.part_acc,
            &self.kv.part_ml,
            &self.aout,
            cfg.n_head,
            cfg.n_head_kv,
            cfg.head_dim,
            self.kv.seq,
            1.0f32 / (cfg.head_dim as f32).sqrt(),
            split,
            self.kv.slot_bytes(),
            self.kv.acc_slot(),
            self.kv.ml_slot(),
            qw,
            b,
            // The MTP group shares one cache; the staged kernel's split count
            // is the group's, so it keeps the staged path.
            0,
        )?;
        self.attn
            .out_rows(l, &self.aout, &self.gate_b, &self.y, b)?;
        l.residual_buf(&self.x, &self.y, &self.x1, b * n)?;
        l.rmsnorm_rows(&self.x1, &self.post_norm, &self.t, n, cfg.eps, b)?;
        match &self.ffn {
            MtpFfn::Dense { gate, up, down } => {
                gate.gemm_batched(l, &self.t, &self.g, b)?;
                up.gemm_batched(l, &self.t, &self.u, b)?;
                l.silu_mul_buf(&self.g, &self.u, &self.act, b * ff)?;
                down.gemm_batched(l, &self.act, &self.d, b)?;
                l.residual_buf(&self.d, &self.x1, &self.out, b * n)?;
            }
            // The routed step is `MoeBlock::forward_batch`, the same call a
            // plain batched decode makes: slot `i` is row `i` of the tile, so
            // the drafted id is the token that row would have produced.
            MtpFfn::Moe(m) => {
                let sc = self
                    .moe
                    .as_ref()
                    .ok_or_else(|| "moe drafter scratch not allocated".to_string())?;
                m.forward_batch(l, &self.t, b, sc)?;
                l.residual_buf(&sc.out, &self.x1, &self.out, b * n)?;
            }
        }
        l.rmsnorm_rows(&self.out, &self.head_norm, &self.hnext, n, cfg.eps, b)?;
        let kind = crate::gpu_weights::batch_kind(gemv_entry(self.head.1)?)
            .ok_or_else(|| "drafter head has no batched GEMV".to_string())?;
        l.gemv_batched(
            kind,
            self.head.0,
            &self.hnext,
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
        let bytes = self.arg_out.download_bytes(0, 4 * b)?;
        Ok((0..b)
            .map(|i| {
                let o = i * 4;
                u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]])
            })
            .collect())
    }

    /// Build the drafter's KV rows for `tokens[i]` at `positions[i]`, pairing
    /// row `hidden_rows[i]` of `hidden` with each token. Rows must be grouped by
    /// slot (slot 0's rows first, then slot 1's, ...); `counts[b]` gives each
    /// slot's run length, and each run appends to that slot's own cache.
    ///
    /// This is the prompt seed and the accepted-token rewrite: the projection
    /// is one batched pass for every row, then one append per slot.
    #[allow(clippy::too_many_arguments)]
    pub fn write_kv(
        &mut self,
        hidden: &Buffer,
        hidden_rows: &[usize],
        tokens: &[u32],
        positions: &[usize],
        counts: &[usize],
        width: usize,
    ) -> Result<(), String> {
        let total = tokens.len();
        if hidden_rows.len() != total || positions.len() != total || !counts_ok(counts, total) {
            return Err(format!("write_kv: {} rows, {} counts", total, counts.len()));
        }
        if total == 0 {
            return Ok(());
        }
        if total > (self.slots * 4).max(16) {
            return Err(format!("write_kv: {total} rows exceed the seed scratch"));
        }
        let cfg = self.cfg.clone();
        let n = cfg.n_embd;
        let l = self.attn_launcher();
        for (i, &tok) in tokens.iter().enumerate() {
            l.embed_row_at(
                self.embed.2,
                self.embed.0,
                tok as usize,
                self.embed.1,
                &self.emb,
                i * n,
                n,
            )?;
        }
        l.rmsnorm_rows(&self.emb, &self.enorm, &self.en, n, cfg.eps, total)?;
        // Gather the selected hidden rows into a contiguous tile, then norm.
        for (i, &row) in hidden_rows.iter().enumerate() {
            hidden.copy_range_on(&self.hn, row * width, i * n, n, l.stream())?;
        }
        l.rmsnorm_rows(&self.hn, &self.hnorm, &self.t, n, cfg.eps, total)?;
        for i in 0..total {
            self.en
                .copy_range_on(&self.cat, i * n, i * 2 * n, n, l.stream())?;
            self.t
                .copy_range_on(&self.cat, i * n, i * 2 * n + n, n, l.stream())?;
        }
        self.eh_proj.gemm_batched(l, &self.cat, &self.x, total)?;
        l.rmsnorm_rows(&self.x, &self.attn_norm, &self.t, n, cfg.eps, total)?;
        self.attn
            .qkv_rows(l, &self.t, &self.qg, &self.k, &self.v, total)?;
        let mut pos_bytes = vec![0u8; 8 * total];
        for (i, &p) in positions.iter().enumerate() {
            pos_bytes[i * 8..i * 8 + 4].copy_from_slice(&(p as i32).to_le_bytes());
            pos_bytes[i * 8 + 4..i * 8 + 8].copy_from_slice(&((p as i32) + 1).to_le_bytes());
        }
        self.pos.upload_bytes_async(0, &pos_bytes, l.stream())?;
        self.attn.split_rope_rows(
            l,
            &self.qg,
            &self.k,
            &self.q,
            &self.gate_b,
            positions,
            &self.pos,
        )?;
        let kw = cfg.n_head_kv * cfg.head_dim;
        let mut base = 0usize;
        for (b, &count) in counts.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let slot_words = self.kv.slot_bytes() / 4;
            l.kv_encode_rows_ptr(
                self.kv.dtype.id(),
                row_ptr(&self.k, base * kw),
                row_ptr(&self.v, base * kw),
                row_ptr(&self.kv.k, b * slot_words),
                row_ptr(&self.kv.v, b * slot_words),
                cfg.n_head_kv,
                cfg.head_dim,
                self.kv.seq,
                row_ptr(&self.pos, base * 2),
                0,
                count,
            )?;
            base += count;
        }
        Ok(())
    }

    /// Seed the drafter's KV over a prompt for `slot`: row `q` pairs the
    /// target hidden state at `q` with the token at `q+1` (the last prompt
    /// hidden seeds the first draft instead). `capture` holds one hidden row per
    /// prompt token, captured by the chunked prefill.
    pub fn prefill_prompt(
        &mut self,
        slot: usize,
        prompt: &[u32],
        capture: &Buffer,
        width: usize,
    ) -> Result<(), String> {
        if prompt.len() < 2 {
            return Ok(());
        }
        let mut counts = vec![0usize; slot + 1];
        for (q0, ids) in prompt[1..].chunks(16).enumerate() {
            let at = q0 * 16;
            counts[slot] = ids.len();
            let rows: Vec<usize> = (at..at + ids.len()).collect();
            let positions: Vec<usize> = (at..at + ids.len()).collect();
            let tokens: Vec<u32> = ids.to_vec();
            self.write_kv(capture, &rows, &tokens, &positions, &counts, width)?;
        }
        Ok(())
    }

    fn attn_launcher(&self) -> &'a Launcher {
        self.launcher
    }
}

fn counts_ok(counts: &[usize], total: usize) -> bool {
    counts.iter().sum::<usize>() == total
}

fn row_ptr(buf: &Buffer, words: usize) -> *mut std::ffi::c_void {
    unsafe { (buf.as_ptr() as *mut u8).add(words * 4) as *mut std::ffi::c_void }
}
