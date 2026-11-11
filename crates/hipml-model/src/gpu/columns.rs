// SPDX-License-Identifier: Apache-2.0
//! R-column verify: the speculative decode pass.
//!
//! One weight pass over the 64 layers for `R` tokens at once. Every projection
//! uses the batched GEMV (`gemv_*_b`), which keeps the single-sequence `k`
//! accumulation order per row and is therefore bit-identical to `R` separate
//! decode launches. The work that is sequential in time — the GDN recurrence,
//! the conv window, the KV append and causal attention — is walked over the
//! columns inside one launch per stage (`*_cols_*`), with the same arithmetic
//! order as the single-column kernels, so column `c` produces exactly the
//! logits a plain decode at that position would.
//!
//! The layer-major loop means the recurrent state after column `c` only exists
//! *inside* the layer that is processing it. Each stage that advances the state
//! writes it to `snaps[c]` as it goes; at the end of the pass `snaps[c]` holds
//! every layer's state after column `c`, which is what a rejected draft
//! restores to. KV needs no snapshot: position `t` is appended before it is
//! read, so truncation is just the next step's `pos`.
//!
//! No `set_step` upload appears in this path: the position, the column index and
//! the row count are kernel arguments.

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
    /// Split-K partials for every column, `[cap][n_heads][splits][..]`.
    pub p_acc: Buffer,
    pub p_ml: Buffer,
    pub final_h: Buffer,
    pub logits: Buffer,
    pub arg_v: Buffer,
    pub arg_i: Buffer,
    pub arg_out: Buffer,
}

/// The recurrent state of every GDN layer after every verify column.
///
/// Layout is `[cols][n_ssm][state_len]`: a stage kernel gets the layer's base
/// pointer (`snap + layer*len`) and one column stride (`n_ssm*len`), so it
/// writes each column's row in place and the verify pass needs no per-column
/// memcpy.
pub struct Snap {
    pub cols: usize,
    pub conv: Buffer,
    pub s: Buffer,
}

impl Cols {
    pub(crate) fn new(cfg: &Config, cap: usize, p_acc: usize, p_ml: usize) -> Result<Cols, String> {
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
            p_acc: zeros(p_acc)?,
            p_ml: zeros(p_ml)?,
            final_h: zeros(cap * n)?,
            logits: zeros(cap * cfg.n_vocab)?,
            arg_v: zeros(cap * ARGMAX_PARTS)?,
            arg_i: zeros(cap * ARGMAX_PARTS)?,
            arg_out: zeros(cap)?,
        })
    }
}

impl<'a> GpuEngine<'a> {
    /// Allocate a snapshot set for up to `count` verify columns.
    pub fn alloc_snapshots(&self, count: usize) -> Result<Snap, String> {
        let (n_ssm, conv_len, st_len) = self.snap_lens()?;
        Ok(Snap {
            cols: count,
            conv: Buffer::alloc((count * n_ssm * conv_len).max(1))?,
            s: Buffer::alloc((count * n_ssm * st_len).max(1))?,
        })
    }

    /// `(SSM layers, conv-window floats per layer, delta-net floats per layer)`.
    fn snap_lens(&self) -> Result<(usize, usize, usize), String> {
        let mut n = 0usize;
        let mut conv = 0usize;
        let mut st = 0usize;
        for b in &self.layers {
            if let BlockKind::Ssm(s) = &b.kind {
                let (cl, sl) = s.state_lens();
                n += 1;
                conv = cl;
                st = sl;
            }
        }
        if n == 0 {
            return Err("no GDN layers to snapshot".to_string());
        }
        Ok((n, conv, st))
    }

    /// Restore every GDN layer's state from column `col` of `snap`,
    /// stream-ordered.
    pub fn restore_ssm(&self, snap: &Snap, col: usize) -> Result<(), String> {
        let (n_ssm, _, _) = self.snap_lens()?;
        if col >= snap.cols {
            return Err(format!("snapshot column {col} beyond {}", snap.cols));
        }
        let l = self.model.launcher();
        let mut i = 0;
        for b in &self.layers {
            if let BlockKind::Ssm(s) = &b.kind {
                let (cl, sl) = s.state_lens();
                s.conv_state_buf().copy_from_device_on_at(
                    &snap.conv,
                    (col * n_ssm + i) * cl,
                    cl,
                    l.stream(),
                )?;
                s.s_state_buf().copy_from_device_on_at(
                    &snap.s,
                    (col * n_ssm + i) * sl,
                    sl,
                    l.stream(),
                )?;
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
        snaps: &mut Snap,
    ) -> Result<Vec<u32>, String> {
        let rows = tokens.len();
        if rows == 0 {
            return Err("no columns".to_string());
        }
        // Exact unless the verify itself opted into the skinny tile
        // (`HIPML_SKINNY`); the server's batch flag must not reach it.
        let _exact = (!hipml_kernels::skinny_on()).then(hipml_kernels::exact_scope);
        self.ensure_cols(rows)?;
        if rows > 1 && snaps.cols < rows - 1 {
            return Err(format!(
                "{} snapshot columns for {rows} columns",
                snaps.cols
            ));
        }
        let cfg = self.cfg.clone();
        let n = cfg.n_embd;
        let pos0 = st.pos;
        let l = self.model.launcher();
        let embed = self.embed;
        let (n_ssm, conv_len, st_len) = self.snap_lens()?;
        let snap_conv = if rows > 1 {
            snaps.conv.as_ptr()
        } else {
            std::ptr::null_mut()
        };
        let snap_s = if rows > 1 {
            snaps.s.as_ptr()
        } else {
            std::ptr::null_mut()
        };
        let c = self.cols.as_ref().ok_or("columns scratch not allocated")?;
        if c.cap < rows {
            return Err(format!("{rows} columns exceed scratch {}", c.cap));
        }

        for (i, &id) in tokens.iter().enumerate() {
            l.embed_row_at(embed.2, embed.0, id as usize, embed.1, &c.x, i * n, n)?;
        }
        super::trace::mark(&mut self.trace, l, "embed")?;

        let mut ssm_idx = 0usize;
        for b in 0..super::layer_limit(self.layers.len()) {
            let block = &self.layers[b];
            l.rmsnorm_rows(&c.x, &block.attn_norm, &c.t, n, cfg.eps, rows)?;
            let block_out: &Buffer = match &block.kind {
                BlockKind::Ssm(s) => {
                    super::trace::mark(&mut self.trace, l, "ssm_in")?;
                    s.project_rows(l, &c.t, rows, &c.s_qkv, &c.s_z, &c.s_alpha, &c.s_beta)?;
                    super::trace::mark(&mut self.trace, l, "ssm_proj")?;
                    let conv_snap = ptr_at(snap_conv, ssm_idx * conv_len);
                    let s_snap = ptr_at(snap_s, ssm_idx * st_len);
                    s.recurrent_cols(
                        rows,
                        &c.s_alpha,
                        &c.s_beta,
                        &c.s_g,
                        &c.s_beta_sig,
                        &c.s_qkv,
                        &c.s_conv,
                        &c.s_o,
                        &c.s_z,
                        &c.s_gated,
                        conv_snap,
                        s_snap,
                        n_ssm,
                    )?;
                    ssm_idx += 1;
                    super::trace::mark(&mut self.trace, l, "ssm_rec")?;
                    s.out_rows(l, &c.s_gated, &c.s_out, rows)?;
                    super::trace::mark(&mut self.trace, l, "ssm_step")?;
                    &c.s_out
                }
                BlockKind::Full { attn, deep } => {
                    super::trace::mark(&mut self.trace, l, "attn_in")?;
                    attn.prepare_rows(l, &c.t, rows, &c.qg, &c.q, &c.agate, &c.k, &c.v, pos0)?;
                    // Append every column's K/V first, then attend: column `r`
                    // reads positions `0..=pos0 + r`, so a later column's append
                    // is invisible to it.
                    super::trace::mark(&mut self.trace, l, "attn_append")?;
                    match deep {
                        None => attn.append_rows(l, &c.k, &c.v, rows, pos0)?,
                        Some(dk) => l.kv_encode_bulk(
                            dk.dtype.id(),
                            &c.k,
                            &c.v,
                            &dk.k,
                            &dk.v,
                            cfg.n_head_kv,
                            cfg.head_dim,
                            dk.seq,
                            pos0,
                            0,
                            rows,
                        )?,
                    }
                    super::trace::mark(&mut self.trace, l, "attn_mix")?;
                    match deep {
                        None => attn.attention_cols(
                            l,
                            c.q.as_ptr(),
                            c.aout.as_ptr(),
                            &c.p_acc,
                            &c.p_ml,
                            pos0,
                            rows,
                        )?,
                        Some(dk) => {
                            // The quantized decode kernel takes the position
                            // count from the step buffer; one upload per column
                            // is left on this path at 8K.
                            for r in 0..rows {
                                l.set_step(pos0 + r, pos0 + r + 1, 0)?;
                                super::attn_tile::deep_attention_ptr(
                                    l,
                                    &cfg,
                                    dk,
                                    pos0 + r,
                                    row_ptr(&c.q, r, cfg.n_head * cfg.head_dim),
                                    row_ptr(&c.aout, r, cfg.n_head * cfg.head_dim),
                                )?;
                            }
                        }
                    }
                    super::trace::mark(&mut self.trace, l, "attn_wo")?;
                    attn.output_rows(l, &c.aout, &c.agate, &c.ay, rows)?;
                    super::trace::mark(&mut self.trace, l, "block")?;
                    &c.ay
                }
            };
            l.residual_buf(&c.x, block_out, &c.x1, rows * n)?;
            l.rmsnorm_rows(&c.x1, &block.post_norm, &c.t, n, cfg.eps, rows)?;
            super::trace::mark(&mut self.trace, l, "post_norm")?;
            block.gate.gemv_rows(l, &c.t, &c.gate, rows)?;
            super::trace::mark(&mut self.trace, l, "gate")?;
            block.up.gemv_rows(l, &c.t, &c.up, rows)?;
            super::trace::mark(&mut self.trace, l, "up")?;
            l.silu_mul_buf(&c.gate, &c.up, &c.act, rows * cfg.n_ff)?;
            block.down.gemv_rows(l, &c.act, &c.down, rows)?;
            super::trace::mark(&mut self.trace, l, "down")?;
            l.residual_buf(&c.down, &c.x1, &c.x, rows * n)?;
        }

        l.rmsnorm_rows(&c.x, &self.output_norm, &c.final_h, n, cfg.eps, rows)?;
        super::trace::mark(&mut self.trace, l, "out_norm")?;
        let out = self
            .model
            .tensor(OUTPUT)
            .ok_or_else(|| "model has no output.weight".to_string())?;
        let entry = gemv_entry(out.ty())?;
        if let Some(name) =
            hipml_kernels::skinny_entry(entry).filter(|_| hipml_kernels::skinny_on())
        {
            l.run_skinny_resident(
                name,
                out.buffer(),
                &c.final_h,
                &c.logits,
                cfg.n_vocab,
                n,
                rows,
            )?;
        } else {
            let kind = hipml_kernels::batch_kind_entry(entry)
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
        }
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

    /// The `[rows][n_embd]` residual stream after the last layer of the last
    /// `forward_columns` call.
    pub fn column_residual(&self) -> Result<&Buffer, String> {
        let c = self.cols.as_ref().ok_or("columns scratch not allocated")?;
        Ok(&c.x)
    }

    /// The `[rows][n_vocab]` logits rows of the last `forward_columns` call.
    pub fn column_logits(&self) -> Result<&Buffer, String> {
        let c = self.cols.as_ref().ok_or("columns scratch not allocated")?;
        Ok(&c.logits)
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
            let (p_acc, p_ml) = self
                .layers
                .iter()
                .find_map(|b| match &b.kind {
                    BlockKind::Full { attn, .. } => Some(attn.split_scratch(cap)),
                    BlockKind::Ssm(_) => None,
                })
                .ok_or_else(|| "model has no full-attention layer".to_string())?;
            self.cols = Some(Cols::new(&self.cfg, cap, p_acc, p_ml)?);
        }
        Ok(())
    }
}

/// Device pointer `base + floats`, or null when `base` is null.
fn ptr_at(base: *mut std::ffi::c_void, floats: usize) -> *mut std::ffi::c_void {
    if base.is_null() {
        base
    } else {
        unsafe { (base as *mut u8).add(floats * 4) as *mut std::ffi::c_void }
    }
}

/// Device pointer of row `row` of a `[rows][width]` f32 buffer.
pub(crate) fn row_ptr(buf: &Buffer, row: usize, width: usize) -> *mut std::ffi::c_void {
    unsafe { (buf.as_ptr() as *mut u8).add(row * width * 4) as *mut std::ffi::c_void }
}
