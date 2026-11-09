//! Batched speculative verify: B slots x R columns through one weight pass.
//!
//! Each slot adds its own draft columns to one flat `[M][..]` activation tile
//! (rows are slot-major, so a slot's columns are contiguous), and the 64 layers
//! run once for the whole tile. The work that is sequential in time stays per
//! slot and is fed a slice of that tile:
//!
//! * attention appends and reads **each slot's own KV** at that slot's own
//!   positions, through the batched split-K kernel with `kv_slot_bytes = 0` and
//!   a pointer advanced to the slot's cache and first column,
//! * the GDN chain walks **each slot's own conv window and delta-net matrix** in
//!   that slot's token order, writing a `[slot][column]` state snapshot so a
//!   rejected prefix restores independently per slot.
//!
//! Every projection and norm is the row-wise kernel the plain batched step
//! already uses, so the same row with the same input produces the same bits.

use crate::gpu::deep_kv::MAX_SPLITS;
use super::*;
use hipml_kernels::ssm::SsmDims;
use std::ffi::c_void;

/// Per-slot, per-column recurrent-state snapshots for a verify pass.
pub struct VerifySnap {
    pub cols: usize,
    conv: Buffer,
    s: Buffer,
}

/// Scratch for the `[M][..]` verify tile, allocated on the first pass.
pub(super) struct VerifyScratch {
    cap: usize,
    x: Buffer,
    x1: Buffer,
    t: Buffer,
    gate: Buffer,
    up: Buffer,
    act: Buffer,
    down: Buffer,
    qg: Buffer,
    q: Buffer,
    agate: Buffer,
    k: Buffer,
    v: Buffer,
    aout: Buffer,
    y: Buffer,
    s_qkv: Buffer,
    s_z: Buffer,
    s_alpha: Buffer,
    s_beta: Buffer,
    s_g: Buffer,
    s_beta_sig: Buffer,
    s_conv: Buffer,
    s_o: Buffer,
    s_gated: Buffer,
    s_out: Buffer,
    final_h: Buffer,
    logits: Buffer,
    arg_v: Buffer,
    arg_i: Buffer,
    arg_out: Buffer,
    pos: Buffer,
    /// Split-K scratch for every row of the tile. One row's stride is
    /// `n_head * splits * head_dim`, where `splits` is the largest count any
    /// column in the current pass needs; a group with fewer splits uses a
    /// prefix of its row.
    part_acc: Buffer,
    part_ml: Buffer,
    parts: usize,
    ml_parts: usize,
}

impl VerifyScratch {
    fn new(cfg: &Config, cap: usize) -> Result<VerifyScratch, String> {
        let n = cfg.n_embd;
        let ff = cfg.n_ff;
        let qd = cfg.n_head * cfg.head_dim;
        let w = cfg.n_head_kv * cfg.head_dim;
        let d = SsmDims::from_config(cfg);
        let inner = d.inner();
        let z = |len: usize| Buffer::alloc(len.max(1));
        Ok(VerifyScratch {
            cap,
            x: z(cap * n)?,
            x1: z(cap * n)?,
            t: z(cap * n)?,
            gate: z(cap * ff)?,
            up: z(cap * ff)?,
            act: z(cap * ff)?,
            down: z(cap * n)?,
            qg: z(cap * 2 * qd)?,
            q: z(cap * qd)?,
            agate: z(cap * qd)?,
            k: z(cap * w)?,
            v: z(cap * w)?,
            aout: z(cap * qd)?,
            y: z(cap * n)?,
            s_qkv: z(cap * d.conv_ch)?,
            s_z: z(cap * inner)?,
            s_alpha: z(cap * d.n_v)?,
            s_beta: z(cap * d.n_v)?,
            s_g: z(cap * d.n_v)?,
            s_beta_sig: z(cap * d.n_v)?,
            s_conv: z(cap * d.conv_ch)?,
            s_o: z(cap * inner)?,
            s_gated: z(cap * inner)?,
            s_out: z(cap * n)?,
            final_h: z(cap * n)?,
            logits: z(cap * cfg.n_vocab)?,
            arg_v: z(cap * ARGMAX_PARTS)?,
            arg_i: z(cap * ARGMAX_PARTS)?,
            arg_out: z(cap)?,
            pos: z(2 * cap)?,
            part_acc: Buffer::alloc(1)?,
            part_ml: Buffer::alloc(1)?,
            parts: 0,
            ml_parts: 0,
        })
    }
}

fn at(buf: &Buffer, words: usize) -> *mut c_void {
    unsafe { (buf.as_ptr() as *mut u8).add(words * 4) as *mut c_void }
}

impl<'a> BatchEngine<'a> {
    /// `(SSM layers, conv floats per layer, delta-net floats per layer)`.
    fn verify_lens(&self) -> (usize, usize, usize) {
        let mut n = 0usize;
        let mut conv = 0usize;
        let mut s = 0usize;
        for b in &self.layers {
            if let BatchKind::Ssm(l) = &b.kind {
                let (cl, sl) = l.inst.state_lens();
                n += 1;
                conv = cl;
                s = sl;
            }
        }
        (n, conv, s)
    }

    /// Allocate snapshots for up to `cols` columns on every slot.
    pub fn alloc_verify_snapshots(&self, cols: usize) -> Result<VerifySnap, String> {
        let (n, cl, sl) = self.verify_lens();
        if n == 0 {
            return Ok(VerifySnap {
                cols,
                conv: Buffer::alloc(1)?,
                s: Buffer::alloc(1)?,
            });
        }
        Ok(VerifySnap {
            cols,
            conv: Buffer::alloc((self.slots * cols * n * cl).max(1))?,
            s: Buffer::alloc((self.slots * cols * n * sl).max(1))?,
        })
    }

    fn ensure_verify(&mut self, rows: usize, splits: usize) -> Result<(), String> {
        let have = matches!(&self.verify, Some(s) if s.cap >= rows);
        if !have {
            let cap = ((rows.max(1) + 15) / 16) * 16;
            self.verify = Some(VerifyScratch::new(&self.cfg, cap)?);
        }
        let parts = self.cfg.n_head * splits * self.cfg.head_dim;
        let ml_parts = self.cfg.n_head * splits * 2;
        let s = self.verify.as_mut().expect("allocated above");
        // One row's split-K scratch is `n_head * splits * head_dim` floats. The
        // cache split count grows with the context, so this is reallocated the
        // few times a run crosses a 64-token boundary.
        if s.parts < parts || s.ml_parts < ml_parts {
            s.part_acc = Buffer::alloc(s.cap * parts)?;
            s.part_ml = Buffer::alloc(s.cap * ml_parts)?;
            s.parts = parts;
            s.ml_parts = ml_parts;
        }
        Ok(())
    }

    /// Run `cols[b]` columns for active slot `b` through every layer in one
    /// weight pass and return each column's greedy id. Positions are not
    /// advanced; the caller accepts a prefix and calls [`Self::accept_verify`].
    ///
    /// `cols[b][0]` must be the slot's next real token; the rest are drafts.
    pub fn forward_verify(
        &mut self,
        cols: &[Vec<u32>],
        snap: &VerifySnap,
    ) -> Result<Vec<Vec<u32>>, String> {
        if cols.len() != self.active.len() {
            return Err(format!("{} column sets for {} slots", cols.len(), self.active.len()));
        }
        if self.active.iter().enumerate().any(|(i, &s)| i != s) {
            return Err("verify: active set is not dense".to_string());
        }
        let mut lens = Vec::with_capacity(cols.len());
        let mut total = 0usize;
        for c in cols {
            if c.is_empty() {
                return Err("verify: empty column set".to_string());
            }
            if c.len() > snap.cols {
                return Err(format!(
                    "verify: {} columns exceed the {}-column snapshot",
                    c.len(),
                    snap.cols
                ));
            }
            lens.push(c.len());
            total += c.len();
        }
        if total == 0 || total > 128 {
            return Err(format!("verify: {total} rows outside 1..=128"));
        }
        // Each column is the plain batched step it would have been: slot `b`'s
        // column `c` sits at that slot's position `p0 + c`, and a plain step at
        // position `p` sizes the cache split count from `(p + 1) / 64`. Slots
        // advance independently, so this is per slot, not per column.
        let mut split_max = 1usize;
        for (bi, &slot) in self.active.iter().enumerate() {
            let p0 = self.positions[slot];
            for c in 0..lens[bi] {
                split_max = split_max.max(((p0 + c + 1) / 64).clamp(1, MAX_SPLITS));
            }
        }
        self.ensure_verify(total, split_max)?;
        let cfg = self.cfg.clone();
        let l = self.model.launcher();

        // Row -> (slot, position). Rows are slot-major so each slot's columns
        // are one contiguous slice for the per-slot launches below.
        let mut base = Vec::with_capacity(cols.len());
        let mut positions = Vec::with_capacity(total);
        let mut row = 0usize;
        for (bi, &slot) in self.active.iter().enumerate() {
            base.push(row);
            let p0 = self.positions[slot];
            for c in 0..lens[bi] {
                positions.push(p0 + c);
            }
            row += lens[bi];
        }
        let mut pos_bytes = vec![0u8; 8 * total];
        for (i, &p) in positions.iter().enumerate() {
            pos_bytes[i * 8..i * 8 + 4].copy_from_slice(&(p as i32).to_le_bytes());
            pos_bytes[i * 8 + 4..i * 8 + 8].copy_from_slice(&((p as i32) + 1).to_le_bytes());
        }
        // SAFETY: the scratch is allocated above, larger than `total`.
        let s = self.verify.take().expect("allocated above");
        s.pos.upload_bytes_async(0, &pos_bytes, l.stream())?;

        let result = self.verify_layers(
            &cfg,
            &s,
            snap,
            cols,
            &lens,
            &base,
            &positions,
            total,
        );
        self.verify = Some(s);
        let ids = result?;
        let mut out = Vec::with_capacity(cols.len());
        let mut off = 0usize;
        for &len in &lens {
            out.push(ids[off..off + len].to_vec());
            off += len;
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn verify_layers(
        &mut self,
        cfg: &Config,
        s: &VerifyScratch,
        snap: &VerifySnap,
        cols: &[Vec<u32>],
        lens: &[usize],
        base: &[usize],
        positions: &[usize],
        m: usize,
    ) -> Result<Vec<u32>, String> {
        let l = self.model.launcher();
        let n = cfg.n_embd;
        let ff = cfg.n_ff;
        let (n_ssm, snap_cl, snap_sl) = self.verify_lens();
        for (r, col) in cols.iter().enumerate() {
            for (c, &id) in col.iter().enumerate() {
                let row = base[r] + c;
                l.embed_row_q4_k_at(self.embed.0, id as usize, self.embed.1, &s.x, row * n, n)?;
            }
        }

        let mut ssm_idx = 0usize;
        for i in 0..self.layers.len() {
            let block = &self.layers[i];
            l.rmsnorm_rows(&s.x, &block.attn_norm, &s.t, n, cfg.eps, m)?;
            let block_out: &Buffer = match &block.kind {
                BatchKind::Full(f) => {
                    f.attn.qkv_rows(l, &s.t, &s.qg, &s.k, &s.v, m)?;
                    f.attn
                        .split_rope_rows(l, &s.qg, &s.k, &s.q, &s.agate, positions, &s.pos)?;
                    let dk = &f.kv;
                    let qw = cfg.n_head * cfg.head_dim;
                    let kw = dk.n_kv * dk.head_dim;
                    let slot_words = dk.slot_bytes() / 4;
                    for b in 0..cols.len() {
                        let slot = self.active[b];
                        let start = self.positions[slot];
                        let len = lens[b];
                        let first = base[b];
                        // Append every column first, then attend: column c sees
                        // positions 0..=start+c, exactly as the single-column
                        // path at that position.
                        l.kv_encode_rows_ptr(
                            dk.dtype.id(),
                            at(&s.k, first * kw),
                            at(&s.v, first * kw),
                            at(&dk.k, slot * slot_words),
                            at(&dk.v, slot * slot_words),
                            dk.n_kv,
                            dk.head_dim,
                            dk.seq,
                            at(&s.pos, first * 2),
                            0,
                            len,
                        )?;
                        // One launch per group of columns whose cache split
                        // count is the same. The count is a function of this
                        // slot's position, so every column is the same
                        // arithmetic a plain step at that position runs.
                        let mut c0 = 0usize;
                        while c0 < len {
                            let splits = ((start + c0 + 1) / 64).clamp(1, MAX_SPLITS);
                            let mut c1 = c0 + 1;
                            while c1 < len
                                && ((start + c1 + 1) / 64).clamp(1, MAX_SPLITS) == splits
                            {
                                c1 += 1;
                            }
                            l.attn_decode_rows_ptr(
                                dk.dtype.id(),
                                at(&s.pos, (first + c0) * 2),
                                at(&s.q, (first + c0) * qw),
                                at(&dk.k, slot * slot_words),
                                at(&dk.v, slot * slot_words),
                                at(&s.part_acc, (first + c0) * s.parts),
                                at(&s.part_ml, (first + c0) * s.ml_parts),
                                at(&s.aout, (first + c0) * qw),
                                cfg.n_head,
                                cfg.n_head_kv,
                                cfg.head_dim,
                                dk.seq,
                                1.0f32 / (cfg.head_dim as f32).sqrt(),
                                splits,
                                0,
                                s.parts,
                                s.ml_parts,
                                qw,
                                c1 - c0,
                            )?;
                            c0 = c1;
                        }
                    }
                    f.attn.out_rows(l, &s.aout, &s.agate, &s.y, m)?;
                    &s.y
                }
                BatchKind::Ssm(ssm) => {
                    ssm.inst
                        .gemvs_batched(l, &s.t, &s.s_qkv, &s.s_z, &s.s_alpha, &s.s_beta, m)?;
                    let conv_stride = ssm.conv_state.len() / self.slots;
                    let s_stride = ssm.s_state.len() / self.slots;
                    for b in 0..cols.len() {
                        let slot = self.active[b];
                        let snap_conv = at(
                            &snap.conv,
                            slot * snap.cols * n_ssm * snap_cl + ssm_idx * snap_cl,
                        );
                        let snap_s =
                            at(&snap.s, slot * snap.cols * n_ssm * snap_sl + ssm_idx * snap_sl);
                        ssm.inst.recurrent_cols_rows(
                            at(&ssm.conv_state, slot * conv_stride),
                            at(&ssm.s_state, slot * s_stride),
                            base[b],
                            lens[b],
                            &s.s_alpha,
                            &s.s_beta,
                            &s.s_g,
                            &s.s_beta_sig,
                            &s.s_qkv,
                            &s.s_conv,
                            &s.s_o,
                            &s.s_z,
                            &s.s_gated,
                            snap_conv,
                            snap_s,
                            n_ssm,
                        )?;
                    }
                    ssm_idx += 1;
                    ssm.inst.output_gemv_batched(l, &s.s_gated, &s.s_out, m)?;
                    &s.s_out
                }
            };
            l.residual_buf(&s.x, block_out, &s.x1, m * n)?;
            l.rmsnorm_rows(&s.x1, &block.post_norm, &s.t, n, cfg.eps, m)?;
            block.gate.gemm_batched(l, &s.t, &s.gate, m)?;
            block.up.gemm_batched(l, &s.t, &s.up, m)?;
            l.silu_mul_buf(&s.gate, &s.up, &s.act, m * ff)?;
            block.down.gemm_batched(l, &s.act, &s.down, m)?;
            l.residual_buf(&s.down, &s.x1, &s.x, m * n)?;
        }

        l.rmsnorm_rows(&s.x, &self.output_norm, &s.final_h, n, cfg.eps, m)?;
        let out = self
            .model
            .tensor(crate::OUTPUT)
            .ok_or_else(|| "model has no output.weight".to_string())?;
        let kind = crate::gpu_weights::batch_kind(gemv_entry(out.ty())?)
            .ok_or_else(|| "lm_head has no batched GEMV".to_string())?;
        l.gemv_batched(kind, out.buffer(), &s.final_h, &s.logits, cfg.n_vocab, n, m)?;
        l.argmax_batched(
            &s.logits,
            &s.arg_v,
            &s.arg_i,
            &s.arg_out,
            cfg.n_vocab,
            ARGMAX_PARTS,
            m,
        )?;
        l.synchronize()?;
        let bytes = s.arg_out.download_bytes(0, 4 * m)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect())
    }

    /// Accept `a[b]` draft columns for active slot `b`: restore that slot's
    /// recurrent state to column `a[b]` and move its position past the seed and
    /// the accepted drafts.
    pub fn accept_verify(
        &mut self,
        cols: &[Vec<u32>],
        accepted: &[usize],
        snap: &VerifySnap,
    ) -> Result<(), String> {
        if cols.len() != accepted.len() || cols.len() != self.active.len() {
            return Err("accept_verify: slot count differs".to_string());
        }
        let l = self.model.launcher();
        let (n_ssm, cl, sl) = self.verify_lens();
        let mut ssm_idx = 0usize;
        for i in 0..self.layers.len() {
            if let BatchKind::Ssm(ssm) = &self.layers[i].kind {
                let conv_stride = ssm.conv_state.len() / self.slots;
                let s_stride = ssm.s_state.len() / self.slots;
                for b in 0..cols.len() {
                    let a = accepted[b];
                    if a >= cols[b].len() {
                        return Err(format!("accept_verify: accepted {a} of {}", cols[b].len()));
                    }
                    if a + 1 == cols[b].len() {
                        continue; // all columns accepted: state is already there
                    }
                    let slot = self.active[b];
                    let src = slot * snap.cols * n_ssm;
                    snap.conv.copy_range_on(
                        &ssm.conv_state,
                        src * cl + a * n_ssm * cl + ssm_idx * cl,
                        slot * conv_stride,
                        cl,
                        l.stream(),
                    )?;
                    snap.s.copy_range_on(
                        &ssm.s_state,
                        src * sl + a * n_ssm * sl + ssm_idx * sl,
                        slot * s_stride,
                        sl,
                        l.stream(),
                    )?;
                }
                ssm_idx += 1;
            }
        }
        for (b, &a) in accepted.iter().enumerate() {
            self.positions[self.active[b]] += a + 1;
        }
        l.synchronize()
    }

    /// The `[M][n_embd]` post-`output_norm` hidden rows of the last verify.
    pub fn verify_hidden(&self) -> Result<&Buffer, String> {
        let s = self.verify.as_ref().ok_or("verify scratch not allocated")?;
        Ok(&s.final_h)
    }

    /// Copy verify row `row`'s post-`output_norm` hidden state into `dst` at
    /// row `dst_row`. The caller uses each slot's last accepted column as the
    /// next draft's seed.
    pub fn copy_verify_hidden(
        &self,
        row: usize,
        dst: &Buffer,
        dst_row: usize,
        width: usize,
    ) -> Result<(), String> {
        let s = self.verify.as_ref().ok_or("verify scratch not allocated")?;
        if row >= s.cap {
            return Err(format!("verify hidden row {row} beyond {}", s.cap));
        }
        let l = self.model.launcher();
        s.final_h.copy_range_on(dst, row * width, dst_row * width, width, l.stream())
    }
}
