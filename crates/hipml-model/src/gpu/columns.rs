//! R-column verify: the speculative decode pass.
//!
//! One weight pass over the 64 layers for `R` tokens at once. Every projection
//! uses the batched GEMV (`gemv_*_b`), which keeps the single-sequence `k`
//! accumulation order per row and is therefore bit-identical to `R` separate
//! decode launches. The work that is genuinely sequential in time — the GDN
//! recurrence, the conv window, the KV append and causal attention — stays per
//! column and runs the same decode kernels, so column `c` produces exactly the
//! logits a plain decode at that position would.
//!
//! The layer-major loop means the recurrent state after column `c` only exists
//! *inside* the layer that is processing it. For rollback, each SSM layer
//! copies its state to `snap[c]` right after column `c`; at the end of the pass
//! `snap[c]` holds every layer's state after column `c`, which is what a
//! rejected draft restores to. KV needs no snapshot: position `t` is appended
//! before it is read, so truncation is just the next step's `pos`.

use super::*;
use crate::gpu_weights::gemv_entry;
use crate::OUTPUT;

/// Per-column scratch, one set shared by every layer.
pub(crate) struct Cols {
    pub cap: usize,
    pub x: Buffer,
    pub x1: Buffer,
    pub t: Buffer,
    pub gate: Buffer,
    pub up: Buffer,
    pub act: Buffer,
    pub down: Buffer,
    /// GDN projections and per-token recurrence outputs, `[cap][..]`.
    pub s_qkv: Buffer,
    pub s_z: Buffer,
    pub s_alpha: Buffer,
    pub s_beta: Buffer,
    pub s_g: Buffer,
    pub s_beta_sig: Buffer,
    pub s_conv: Buffer,
    pub s_o: Buffer,
    pub s_gated: Buffer,
    pub s_out: Buffer,
    /// Attention projections and per-token attention output.
    pub qg: Buffer,
    pub q: Buffer,
    pub agate: Buffer,
    pub k: Buffer,
    pub v: Buffer,
    pub aout: Buffer,
    pub ay: Buffer,
    pub final_h: Buffer,
    pub logits: Buffer,
    pub arg_v: Buffer,
    pub arg_i: Buffer,
    pub arg_out: Buffer,
}

/// The recurrent state of every GDN layer after one verify column.
pub struct Snap {
    pub layers: Vec<(Buffer, Buffer)>,
}

impl Cols {
    pub(crate) fn new(cfg: &Config, cap: usize) -> Result<Cols, String> {
        let n = cfg.n_embd;
        let ff = cfg.n_ff;
        let qd = cfg.n_head * cfg.head_dim;
        let w = cfg.n_head_kv * cfg.head_dim;
        let ssm = SsmDims::from_config(cfg);
        let zeros = |len: usize| Buffer::alloc(len);
        Ok(Cols {
            cap,
            x: zeros(cap * n)?,
            x1: zeros(cap * n)?,
            t: zeros(cap * n)?,
            gate: zeros(cap * ff)?,
            up: zeros(cap * ff)?,
            act: zeros(cap * ff)?,
            down: zeros(cap * n)?,
            s_qkv: zeros(cap * ssm.conv_ch)?,
            s_z: zeros(cap * ssm.inner())?,
            s_alpha: zeros(cap * ssm.n_v)?,
            s_beta: zeros(cap * ssm.n_v)?,
            s_g: zeros(cap * ssm.n_v)?,
            s_beta_sig: zeros(cap * ssm.n_v)?,
            s_conv: zeros(cap * ssm.conv_ch)?,
            s_o: zeros(cap * ssm.inner())?,
            s_gated: zeros(cap * ssm.inner())?,
            s_out: zeros(cap * n)?,
            qg: zeros(cap * 2 * qd)?,
            q: zeros(cap * qd)?,
            agate: zeros(cap * qd)?,
            k: zeros(cap * w)?,
            v: zeros(cap * w)?,
            aout: zeros(cap * qd)?,
            ay: zeros(cap * n)?,
            final_h: zeros(cap * n)?,
            logits: zeros(cap * cfg.n_vocab)?,
            arg_v: zeros(cap * ARGMAX_PARTS)?,
            arg_i: zeros(cap * ARGMAX_PARTS)?,
            arg_out: zeros(cap)?,
        })
    }
}

impl<'a> GpuEngine<'a> {
    /// Allocate `count` recurrent-state snapshots (one per rollback column).
    pub fn alloc_snapshots(&self, count: usize) -> Result<Vec<Snap>, String> {
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let mut layers = Vec::new();
            for b in &self.layers {
                if let BlockKind::Ssm(s) = &b.kind {
                    let (conv, st) = s.state_lens();
                    layers.push((Buffer::alloc(conv)?, Buffer::alloc(st)?));
                }
            }
            out.push(Snap { layers });
        }
        Ok(out)
    }

    /// Copy every GDN layer's live state into `snap`, stream-ordered.
    pub fn snapshot_ssm(&self, snap: &Snap) -> Result<(), String> {
        let l = self.model.launcher();
        let mut i = 0;
        for b in &self.layers {
            if let BlockKind::Ssm(s) = &b.kind {
                let (conv, st) = s.state_lens();
                let (dc, ds) = &snap.layers[i];
                dc.copy_from_device_on(s.conv_state_buf(), conv, l.stream())?;
                ds.copy_from_device_on(s.s_state_buf(), st, l.stream())?;
                i += 1;
            }
        }
        Ok(())
    }

    /// Restore every GDN layer from `snap`, stream-ordered.
    pub fn restore_ssm(&self, snap: &Snap) -> Result<(), String> {
        let l = self.model.launcher();
        let mut i = 0;
        for b in &self.layers {
            if let BlockKind::Ssm(s) = &b.kind {
                let (conv, st) = s.state_lens();
                let (dc, ds) = &snap.layers[i];
                s.conv_state_buf().copy_from_device_on(dc, conv, l.stream())?;
                s.s_state_buf().copy_from_device_on(ds, st, l.stream())?;
                i += 1;
            }
        }
        Ok(())
    }

    /// Run `tokens` columns through the whole graph in one weight pass.
    ///
    /// `snaps[c]` receives every layer's recurrent state after column `c`
    /// (`c < tokens.len() - 1`); the caller restores the one matching the
    /// accepted prefix. Returns one greedy id per column.
    pub fn forward_columns(
        &mut self,
        tokens: &[u32],
        st: &mut GpuState,
        snaps: &mut [Snap],
    ) -> Result<Vec<u32>, String> {
        let rows = tokens.len();
        if rows == 0 {
            return Err("no columns".to_string());
        }
        self.ensure_cols(rows)?;
        if rows > 1 && snaps.len() < rows - 1 {
            return Err(format!("{} snapshots for {rows} columns", snaps.len()));
        }
        let cfg = self.cfg.clone();
        let n = cfg.n_embd;
        let pos0 = st.pos;
        let l = self.model.launcher();
        let embed = self.embed;
        let c = self.cols.as_ref().ok_or("columns scratch not allocated")?;
        if c.cap < rows {
            return Err(format!("{rows} columns exceed scratch {}", c.cap));
        }

        for (i, &id) in tokens.iter().enumerate() {
            l.embed_row_q4_k_at(embed.0, id as usize, embed.1, &c.x, i * n, n)?;
        }
        super::trace::mark(&mut self.trace, l, "embed")?;

        let mut ssm_idx = 0usize;
        for b in 0..self.layers.len() {
            let block = &self.layers[b];
            l.rmsnorm_rows(&c.x, &block.attn_norm, &c.t, n, cfg.eps, rows)?;
            let block_out: &Buffer = match &block.kind {
                BlockKind::Ssm(s) => {
                    super::trace::mark(&mut self.trace, l, "ssm_in")?;
                    s.project_rows(l, &c.t, rows, &c.s_qkv, &c.s_z, &c.s_alpha, &c.s_beta)?;
                    super::trace::mark(&mut self.trace, l, "ssm_proj")?;
                    for r in 0..rows {
                        let alpha = row_ptr(&c.s_alpha, r, s.dims().n_v);
                        let beta = row_ptr(&c.s_beta, r, s.dims().n_v);
                        let g = row_ptr(&c.s_g, r, s.dims().n_v);
                        let bs = row_ptr(&c.s_beta_sig, r, s.dims().n_v);
                        s.gates_at(alpha, beta, g, bs)?;
                        let qkv = row_ptr(&c.s_qkv, r, s.dims().conv_ch);
                        let conv = row_ptr(&c.s_conv, r, s.dims().conv_ch);
                        s.conv_at(qkv, conv)?;
                        s.l2norm_at_ptr(conv, 0, s.dims().n_k, s.dims().head_k)?;
                        s.l2norm_at_ptr(conv, s.dims().key_dim(), s.dims().n_k, s.dims().head_k)?;
                        s.gdn_at(
                            conv,
                            g,
                            bs,
                            row_ptr(&c.s_o, r, s.dims().inner()),
                        )?;
                        s.gate_norm_at(
                            row_ptr(&c.s_o, r, s.dims().inner()),
                            row_ptr(&c.s_z, r, s.dims().inner()),
                            row_ptr(&c.s_gated, r, s.dims().inner()),
                        )?;
                        if r + 1 < rows {
                            let (conv_len, st_len) = s.state_lens();
                            let (dc, ds) = &snaps[r].layers[ssm_idx];
                            dc.copy_from_device_on(s.conv_state_buf(), conv_len, l.stream())?;
                            ds.copy_from_device_on(s.s_state_buf(), st_len, l.stream())?;
                        }
                    }
                    ssm_idx += 1;
                    super::trace::mark(&mut self.trace, l, "ssm_rec")?;
                    s.out_rows(l, &c.s_gated, &c.s_out, rows)?;
                    super::trace::mark(&mut self.trace, l, "ssm_step")?;
                    &c.s_out
                }
                BlockKind::Full { attn, deep } => {
                    super::trace::mark(&mut self.trace, l, "attn_in")?;
                    attn.prepare_rows(
                        l,
                        &c.t,
                        rows,
                        &c.qg,
                        &c.q,
                        &c.agate,
                        &c.k,
                        &c.v,
                        pos0,
                    )?;
                    // Append every column's K/V first, then attend: column `r`
                    // reads positions `0..=pos0 + r`, so a later column's append
                    // is invisible to it and the two loops are one order apart.
                    super::trace::mark(&mut self.trace, l, "attn_append")?;
                    for r in 0..rows {
                        l.set_step(pos0 + r, pos0 + r + 1, 0)?;
                        match deep {
                            None => attn.append_at(
                                l,
                                row_ptr(&c.k, r, attn.dims.width),
                                row_ptr(&c.v, r, attn.dims.width),
                            )?,
                            Some(dk) => l.kv_encode_ptr(
                                dk.dtype.id(),
                                row_ptr(&c.k, r, attn.dims.width),
                                row_ptr(&c.v, r, attn.dims.width),
                                &dk.k,
                                &dk.v,
                                cfg.n_head_kv,
                                cfg.head_dim,
                                dk.seq,
                            )?,
                        }
                    }
                    super::trace::mark(&mut self.trace, l, "attn_mix")?;
                    for r in 0..rows {
                        l.set_step(pos0 + r, pos0 + r + 1, 0)?;
                        match deep {
                            None => attn.attention_at(
                                l,
                                row_ptr(&c.q, r, cfg.n_head * cfg.head_dim),
                                row_ptr(&c.aout, r, cfg.n_head * cfg.head_dim),
                            )?,
                            Some(dk) => super::attn_tile::deep_attention_ptr(
                                l,
                                &cfg,
                                dk,
                                pos0 + r,
                                row_ptr(&c.q, r, cfg.n_head * cfg.head_dim),
                                row_ptr(&c.aout, r, cfg.n_head * cfg.head_dim),
                            )?,
                        }
                    }
                    super::trace::mark(&mut self.trace, l, "attn_wo")?;
                    attn.output_rows(l, &c.aout, &c.agate, &c.ay, rows)?;
                    super::trace::mark(&mut self.trace, l, "block")?;
                    &c.ay
                }
            };
            for r in 0..rows {
                l.residual_ptr(
                    row_ptr(&c.x, r, n),
                    row_ptr(block_out, r, n),
                    row_ptr(&c.x1, r, n),
                    n,
                )?;
            }
            l.rmsnorm_rows(&c.x1, &block.post_norm, &c.t, n, cfg.eps, rows)?;
            super::trace::mark(&mut self.trace, l, "post_norm")?;
            block.gate.gemv_rows(l, &c.t, &c.gate, rows)?;
            super::trace::mark(&mut self.trace, l, "gate")?;
            block.up.gemv_rows(l, &c.t, &c.up, rows)?;
            super::trace::mark(&mut self.trace, l, "up")?;
            for r in 0..rows {
                l.silu_mul_ptr(
                    row_ptr(&c.gate, r, cfg.n_ff),
                    row_ptr(&c.up, r, cfg.n_ff),
                    row_ptr(&c.act, r, cfg.n_ff),
                    cfg.n_ff,
                )?;
            }
            block.down.gemv_rows(l, &c.act, &c.down, rows)?;
            super::trace::mark(&mut self.trace, l, "down")?;
            for r in 0..rows {
                l.residual_ptr(
                    row_ptr(&c.down, r, n),
                    row_ptr(&c.x1, r, n),
                    row_ptr(&c.x, r, n),
                    n,
                )?;
            }
        }

        l.rmsnorm_rows(&c.x, &self.output_norm, &c.final_h, n, cfg.eps, rows)?;
        super::trace::mark(&mut self.trace, l, "out_norm")?;
        let out = self
            .model
            .tensor(OUTPUT)
            .ok_or_else(|| "model has no output.weight".to_string())?;
        let kind = hipml_kernels::batch_kind_entry(gemv_entry(out.ty())?)
            .ok_or_else(|| format!("{} has no batched GEMV", out.ty().name()))?;
        l.gemv_batched(
            kind,
            out.buffer(),
            &c.final_h,
            &c.logits,
            cfg.n_vocab,
            n,
            rows,
        )?;
        super::trace::mark(&mut self.trace, l, "lm_head")?;
        l.argmax_batched(
            &c.logits,
            &c.arg_v,
            &c.arg_i,
            &c.arg_out,
            cfg.n_vocab,
            ARGMAX_PARTS,
            rows,
        )?;
        l.synchronize()?;
        let bytes = c.arg_out.download_bytes(0, 4 * rows)?;
        let ids = bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        st.pos += rows;
        Ok(ids)
    }

    /// The `[rows][n_embd]` post-`output_norm` hidden rows of the last
    /// `forward_columns` call.
    pub fn column_hidden(&self) -> Result<&Buffer, String> {
        let c = self.cols.as_ref().ok_or("columns scratch not allocated")?;
        Ok(&c.final_h)
    }

    /// The post-`output_norm` hidden row of the last `forward_columns` call,
    /// copied into `dst` (one `n_embd` vector).
    pub fn copy_column_hidden(&self, row: usize, dst: &Buffer) -> Result<(), String> {
        let c = self.cols.as_ref().ok_or("columns scratch not allocated")?;
        if row >= c.cap {
            return Err(format!("hidden row {row} beyond scratch {}", c.cap));
        }
        let l = self.model.launcher();
        let n = self.cfg.n_embd;
        dst.copy_from_device_on_at(&c.final_h, row * n, n, l.stream())
    }

    fn ensure_cols(&mut self, rows: usize) -> Result<(), String> {
        let have = matches!(&self.cols, Some(c) if c.cap >= rows);
        if !have {
            let cap = rows.max(4);
            self.cols = Some(Cols::new(&self.cfg, cap)?);
        }
        Ok(())
    }
}

/// Device pointer of row `row` of a `[rows][width]` f32 buffer.
pub(crate) fn row_ptr(buf: &Buffer, row: usize, width: usize) -> *mut std::ffi::c_void {
    unsafe { (buf.as_ptr() as *mut u8).add(row * width * 4) as *mut std::ffi::c_void }
}
