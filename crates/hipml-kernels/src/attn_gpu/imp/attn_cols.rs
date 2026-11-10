// SPDX-License-Identifier: Apache-2.0
//! R-column attention for the verify pass: one launch for the append and one
//! for the causal mix, plus the front end (q/gate split, head norm, rope) batched
//! over the columns.
//!
//! The split-K partition is fixed in positions (`per = ceil(n_ctx / splits)`),
//! so every column's arithmetic is the same as the one-column launch the plain
//! decode path makes at that position; see `attn_decode_cols_f32.hip`.

use super::*;

impl<'a> AttnLayer<'a> {
    /// `(splits, per)`: the fixed split geometry. Sized from the whole cache, so
    /// one launch covers every position and every column.
    pub(super) fn split_geom(&self) -> (usize, usize) {
        let d = self.dims;
        let splits = d.n_ctx.div_ceil(32).clamp(1, MAX_SPLITS);
        let per = d.n_ctx.div_ceil(splits);
        (splits, per)
    }

    /// Values of split scratch `attention_cols` needs for `rows` columns:
    /// `(part_acc, part_ml)`.
    pub fn split_scratch(&self, rows: usize) -> (usize, usize) {
        let d = self.dims;
        let (splits, _) = self.split_geom();
        (
            rows * d.n_head * splits * d.head_dim,
            rows * d.n_head * splits * 2,
        )
    }

    /// Append row `c`'s K/V at `pos0 + c` for every column, in one launch.
    pub fn append_rows(
        &self,
        l: &Launcher,
        k: &Buffer,
        v: &Buffer,
        rows: usize,
        pos0: usize,
    ) -> Result<(), String> {
        let d = self.dims;
        let append = self.module_kernel(l, "kv_append_cols_f32")?;
        let mut p_kc = self.kv_k.as_ptr();
        let mut p_vc = self.kv_v.as_ptr();
        let mut p_k = k.as_ptr();
        let mut p_v = v.as_ptr();
        let mut v_pos = pos0 as i32;
        let mut v_rows = rows as i32;
        let mut width = d.width as i32;
        let mut params: [*mut c_void; 7] = [
            &mut p_kc as *mut _ as *mut c_void,
            &mut p_vc as *mut _ as *mut c_void,
            &mut p_k as *mut _ as *mut c_void,
            &mut p_v as *mut _ as *mut c_void,
            &mut v_pos as *mut _ as *mut c_void,
            &mut v_rows as *mut _ as *mut c_void,
            &mut width as *mut _ as *mut c_void,
        ];
        append.launch(
            LaunchConfig::linear(2 * d.width * rows, 256),
            &l.stream,
            &mut params,
        )
    }

    /// Causal attention for `rows` columns followed by the column-`r` q/gate
    /// split and rope already applied. `pos0` is column 0's absolute position.
    ///
    /// The partials live in the caller's `[rows][n_heads][splits][..]` scratch;
    /// the one-column decode path passes this layer's own single-row buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_cols(
        &self,
        l: &Launcher,
        q: *mut c_void,
        out: *mut c_void,
        part_acc: &Buffer,
        part_ml: &Buffer,
        pos0: usize,
        rows: usize,
    ) -> Result<(), String> {
        let d = self.dims;
        let group = d.n_head / d.n_kv;
        let dpl = d.head_dim / 32;
        let shared = 2 * SPLIT_TILE * d.head_dim * 4;
        let split_ok =
            d.head_dim % 32 == 0 && dpl <= 8 && group * 32 <= 1024 && shared <= 64 * 1024;

        if !split_ok {
            // Shapes the split kernel does not cover: one launch per column of
            // the single-block-per-head kernel.
            for c in 0..rows {
                l.set_step(pos0 + c, pos0 + c + 1, 0)?;
                self.legacy_attention(
                    l,
                    row_ptr_ptr(q, c, d.n_head * d.head_dim),
                    row_ptr_ptr(out, c, d.n_head * d.head_dim),
                )?;
            }
            return Ok(());
        }

        let (splits, per) = self.split_geom();
        let need = rows * d.n_head * splits * d.head_dim;
        if part_acc.len() < need || part_ml.len() < rows * d.n_head * splits * 2 {
            return Err(format!(
                "attention scratch: {} / {} values for {rows} columns",
                part_acc.len(),
                part_ml.len()
            ));
        }

        let split = self.module_kernel(l, "attn_decode_cols_f32")?;
        let mut p_q = q;
        let mut p_k = self.kv_k.as_ptr();
        let mut p_v = self.kv_v.as_ptr();
        let mut p_acc = part_acc.as_ptr();
        let mut p_ml = part_ml.as_ptr();
        let mut v_heads = d.n_head as i32;
        let mut v_kv = d.n_kv as i32;
        let mut v_hd = d.head_dim as i32;
        let mut v_pos = pos0 as i32;
        let mut v_rows = rows as i32;
        let mut scale = 1.0f32 / (d.head_dim as f32).sqrt();
        let mut v_splits = splits as i32;
        let mut v_per = per as i32;
        let mut width = d.width as i32;
        let mut params: [*mut c_void; 14] = [
            &mut p_q as *mut _ as *mut c_void,
            &mut p_k as *mut _ as *mut c_void,
            &mut p_v as *mut _ as *mut c_void,
            &mut p_acc as *mut _ as *mut c_void,
            &mut p_ml as *mut _ as *mut c_void,
            &mut v_heads as *mut _ as *mut c_void,
            &mut v_kv as *mut _ as *mut c_void,
            &mut v_hd as *mut _ as *mut c_void,
            &mut v_pos as *mut _ as *mut c_void,
            &mut v_rows as *mut _ as *mut c_void,
            &mut scale as *mut _ as *mut c_void,
            &mut v_splits as *mut _ as *mut c_void,
            &mut v_per as *mut _ as *mut c_void,
            &mut width as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: ((d.n_kv * splits) as u32, 1, 1),
            block: ((group * 32) as u32, 1, 1),
            shared_mem_bytes: shared as u32,
        };
        split.launch(cfg, &l.stream, &mut params)?;

        let reduce = self.module_kernel(l, "attn_reduce_cols_f32")?;
        let mut p_acc = part_acc.as_ptr();
        let mut p_ml = part_ml.as_ptr();
        let mut p_out = out;
        let mut v_heads = d.n_head as i32;
        let mut v_hd = d.head_dim as i32;
        let mut v_splits = splits as i32;
        let mut params: [*mut c_void; 6] = [
            &mut p_acc as *mut _ as *mut c_void,
            &mut p_ml as *mut _ as *mut c_void,
            &mut p_out as *mut _ as *mut c_void,
            &mut v_heads as *mut _ as *mut c_void,
            &mut v_hd as *mut _ as *mut c_void,
            &mut v_splits as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (d.n_head as u32, rows as u32, 1),
            block: (d.head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        reduce.launch(cfg, &l.stream, &mut params)
    }

    /// The non-split fallback: one block per query head, at `step[1]`.
    fn legacy_attention(
        &self,
        l: &Launcher,
        q: *mut c_void,
        out: *mut c_void,
    ) -> Result<(), String> {
        let d = self.dims;
        let attn = self.module_kernel(l, "attn_decode_f32")?;
        let mut p_q = q;
        let mut p_k = self.kv_k.as_ptr();
        let mut p_v = self.kv_v.as_ptr();
        let mut p_out = out;
        let mut n_heads = d.n_head as i32;
        let mut n_kv = d.n_kv as i32;
        let mut hd = d.head_dim as i32;
        let mut p_step = l.step_buf().as_ptr();
        let mut scale = 1.0f32 / (d.head_dim as f32).sqrt();
        let mut head_stride = d.head_dim as i32;
        let mut row_stride = d.width as i32;
        let mut params: [*mut c_void; 11] = [
            &mut p_q as *mut _ as *mut c_void,
            &mut p_k as *mut _ as *mut c_void,
            &mut p_v as *mut _ as *mut c_void,
            &mut p_out as *mut _ as *mut c_void,
            &mut n_heads as *mut _ as *mut c_void,
            &mut n_kv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut p_step as *mut _ as *mut c_void,
            &mut scale as *mut _ as *mut c_void,
            &mut head_stride as *mut _ as *mut c_void,
            &mut row_stride as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (d.n_head as u32, 1, 1),
            block: (d.head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        attn.launch(cfg, &l.stream, &mut params)
    }

    /// q/gate split, head RMS norm and M-RoPE for `rows` columns of the
    /// projections `prepare_rows` left in `qg`, `q`, `gate`, `k`, `v`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn split_norm_rope(
        &self,
        l: &Launcher,
        rows: usize,
        qg: &Buffer,
        q: &Buffer,
        gate: &Buffer,
        k: &Buffer,
        pos0: usize,
    ) -> Result<(), String> {
        let d = self.dims;

        let split = self.module_kernel(l, "qg_split_cols_f32")?;
        let mut p_qg = qg.as_ptr();
        let mut p_q = q.as_ptr();
        let mut p_g = gate.as_ptr();
        let mut nh = d.n_head as i32;
        let mut hd = d.head_dim as i32;
        let mut v_rows = rows as i32;
        let mut params: [*mut c_void; 6] = [
            &mut p_qg as *mut _ as *mut c_void,
            &mut p_q as *mut _ as *mut c_void,
            &mut p_g as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut v_rows as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: ((rows * d.n_head) as u32, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        split.launch(cfg, &l.stream, &mut params)?;

        self.head_norm_rope(l, q.as_ptr(), d.n_head, rows, pos0)?;
        self.head_norm_rope(l, k.as_ptr(), d.n_kv, rows, pos0)
    }

    fn head_norm_rope(
        &self,
        l: &Launcher,
        x: *mut c_void,
        heads: usize,
        rows: usize,
        pos0: usize,
    ) -> Result<(), String> {
        let d = self.dims;
        let norm = self.module_kernel(l, "head_rmsnorm_cols_f32")?;
        let w = if heads == d.n_head {
            self.q_norm.as_ptr()
        } else {
            self.k_norm.as_ptr()
        };
        let mut p_x = x;
        let mut p_w = w;
        let mut p_y = x;
        let mut v_h = heads as i32;
        let mut hd = d.head_dim as i32;
        let mut eps = d.eps;
        let mut v_rows = rows as i32;
        let mut params: [*mut c_void; 7] = [
            &mut p_x as *mut _ as *mut c_void,
            &mut p_w as *mut _ as *mut c_void,
            &mut p_y as *mut _ as *mut c_void,
            &mut v_h as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut eps as *mut _ as *mut c_void,
            &mut v_rows as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: ((rows * heads) as u32, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        norm.launch(cfg, &l.stream, &mut params)?;

        let rope = self.module_kernel(l, "mrope_cols_f32")?;
        let mut p_in = x;
        let mut p_out = x;
        let mut v_h = heads as i32;
        let mut hd = d.head_dim as i32;
        let mut nrot = d.n_rot as i32;
        let mut s0 = d.sections[0] as i32;
        let mut s1 = d.sections[1] as i32;
        let mut s2 = d.sections[2] as i32;
        let mut s3 = d.sections[3] as i32;
        let mut v_pos = pos0 as i32;
        let mut base = d.rope_base;
        let mut v_rows = rows as i32;
        let mut params: [*mut c_void; 12] = [
            &mut p_in as *mut _ as *mut c_void,
            &mut p_out as *mut _ as *mut c_void,
            &mut v_h as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nrot as *mut _ as *mut c_void,
            &mut s0 as *mut _ as *mut c_void,
            &mut s1 as *mut _ as *mut c_void,
            &mut s2 as *mut _ as *mut c_void,
            &mut s3 as *mut _ as *mut c_void,
            &mut v_pos as *mut _ as *mut c_void,
            &mut base as *mut _ as *mut c_void,
            &mut v_rows as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: ((rows * heads) as u32, 1, 1),
            block: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        rope.launch(cfg, &l.stream, &mut params)
    }
}

/// Device pointer of row `c` of a `[rows][width]` f32 buffer held as a raw
/// pointer.
fn row_ptr_ptr(base: *mut c_void, c: usize, width: usize) -> *mut c_void {
    unsafe { (base as *mut u8).add(c * width * 4) as *mut c_void }
}
