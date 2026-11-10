// SPDX-License-Identifier: Apache-2.0
//! The full-attention block over a prefill chunk: batched projections, then the
//! per-token rope, cache append and causal attention the decode path already
//! uses.
//!
//! Child module of `imp` in `attn_gpu.rs` so it can reach the layer's private
//! scratch and weight handles without widening the public surface.
//!
//! Numerics: the projections go through the tiled prefill GEMM, which
//! accumulates a 32-value k-tile at a time instead of the GEMV's 2048-value
//! shared tile. Everything after the projection is the identical kernel chain,
//! so a chunk differs from the token loop only in the last bits of the
//! projections.

use super::*;

impl AttnLayer<'_> {
    /// Q/K/V projections for `tokens` rows of `x` into chunk buffers
    /// `[tokens][2*n_head*head_dim]`, `[tokens][width]`, `[tokens][width]`.
    ///
    /// `xh` is scratch for an fp16 copy of `x`; it is only written when the
    /// active prefill path reads converted activations.
    pub fn project_chunk(
        &self,
        l: &Launcher,
        x: &Buffer,
        xh: &Buffer,
        qg: &Buffer,
        k: &Buffer,
        v: &Buffer,
        tokens: usize,
    ) -> Result<(), String> {
        let d = self.dims;
        let plan = [
            (&self.wq, qg, 2 * d.n_head * d.head_dim),
            (&self.wk, k, d.width),
            (&self.wv, v, d.width),
        ];
        let half = crate::prefill_has_xhalf();
        if half {
            crate::prefill_xhalf(l.module(), l.stream(), x, xh, tokens * d.n_embd)?;
        }
        for (w, y, m) in plan {
            if w.rows != m {
                return Err(format!("prefill qkv: {} rows, launch says {m}", w.rows));
            }
            if half {
                crate::prefill_resident_half(
                    l.module(),
                    l.stream(),
                    w.prefill_entry()?,
                    &w.data_bytes(),
                    xh,
                    y,
                    m,
                    w.cols,
                    tokens,
                )?;
            } else {
                crate::prefill_resident(
                    l.module(),
                    l.stream(),
                    w.prefill_entry()?,
                    &w.data_bytes(),
                    x,
                    y,
                    m,
                    w.cols,
                    tokens,
                )?;
            }
        }
        Ok(())
    }

    /// Whether `step_chunk` covers this layer: one thread per head component,
    /// power-of-two head dimension up to the 256-thread block.
    pub fn chunk_supported(&self) -> bool {
        let hd = self.dims.head_dim;
        hd.is_power_of_two() && hd <= 256 && hd >= 32
    }

    /// The front half of the chunk chain: q|gate split, then the per-head RMS
    /// norm and RoPE over the whole tile. Shared by the f32-mirror path and the
    /// quantized-cache path; the caller appends to whichever cache it reads.
    pub fn chunk_front(
        &self,
        l: &Launcher,
        qg: &Buffer,
        k: &Buffer,
        q: &Buffer,
        gate: &Buffer,
        pos0: usize,
        tokens: usize,
    ) -> Result<(), String> {
        if !self.chunk_supported() {
            return Err("chunked attention: unsupported head dimension".to_string());
        }
        let d = self.dims;
        let t = tokens as u32;

        let split = self.module_kernel(l, "qg_split_chunk_f32")?;
        let mut p_qg = qg.as_ptr();
        let mut p_q = q.as_ptr();
        let mut p_g = gate.as_ptr();
        let mut nh = d.n_head as i32;
        let mut hd = d.head_dim as i32;
        let mut nt = tokens as i32;
        let mut params: [*mut c_void; 6] = [
            &mut p_qg as *mut _ as *mut c_void,
            &mut p_q as *mut _ as *mut c_void,
            &mut p_g as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nt as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (d.n_head as u32, t, 1),
            block: (d.head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        split.launch(cfg, &l.stream, &mut params)?;

        let nr = self.module_kernel(l, "norm_rope_chunk_f32")?;
        let mut p_q = q.as_ptr();
        let mut p_k = k.as_ptr();
        let mut p_qw = self.q_norm.as_ptr();
        let mut p_kw = self.k_norm.as_ptr();
        let mut nh = d.n_head as i32;
        let mut nkv = d.n_kv as i32;
        let mut hd = d.head_dim as i32;
        let mut nrot = d.n_rot as i32;
        let mut base = d.rope_base;
        let mut eps = d.eps;
        let mut pos = pos0 as i32;
        let mut nt = tokens as i32;
        let mut params: [*mut c_void; 12] = [
            &mut p_q as *mut _ as *mut c_void,
            &mut p_k as *mut _ as *mut c_void,
            &mut p_qw as *mut _ as *mut c_void,
            &mut p_kw as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nrot as *mut _ as *mut c_void,
            &mut base as *mut _ as *mut c_void,
            &mut eps as *mut _ as *mut c_void,
            &mut pos as *mut _ as *mut c_void,
            &mut nt as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: ((d.n_head + d.n_kv) as u32, t, 1),
            block: (d.head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        nr.launch(cfg, &l.stream, &mut params)
    }

    /// The whole chunk's attention chain in four launches: q-gate split, head
    /// norm + rope, cache append, causal attention. The chunk buffers are
    /// `[tokens][..]`; rows are appended before attention and the kernel's
    /// causal bound hides the future ones.
    #[allow(clippy::too_many_arguments)]
    pub fn step_chunk(
        &self,
        l: &Launcher,
        qg: &Buffer,
        k: &Buffer,
        v: &Buffer,
        q: &Buffer,
        gate: &Buffer,
        out: &Buffer,
        pos0: usize,
        tokens: usize,
    ) -> Result<(), String> {
        self.chunk_front(l, qg, k, q, gate, pos0, tokens)?;

        let d = self.dims;
        let app = self.module_kernel(l, "kv_append_chunk_f32")?;
        let mut p_kc = self.kv_k.as_ptr();
        let mut p_vc = self.kv_v.as_ptr();
        let mut p_k = k.as_ptr();
        let mut p_v = v.as_ptr();
        let mut pos = pos0 as i32;
        let mut width = d.width as i32;
        let mut nt = tokens as i32;
        let mut params: [*mut c_void; 7] = [
            &mut p_kc as *mut _ as *mut c_void,
            &mut p_vc as *mut _ as *mut c_void,
            &mut p_k as *mut _ as *mut c_void,
            &mut p_v as *mut _ as *mut c_void,
            &mut pos as *mut _ as *mut c_void,
            &mut width as *mut _ as *mut c_void,
            &mut nt as *mut _ as *mut c_void,
        ];
        app.launch(
            LaunchConfig::linear(2 * tokens * d.width, 256),
            &l.stream,
            &mut params,
        )?;

        // The tiled kernel needs 8 head components per lane (head_dim 256).
        self.chunk_attention(l, q, &self.kv_k, &self.kv_v, out, pos0, tokens)
    }

    /// One token of a chunk: q-gate split, head norm, rope, cache append and
    /// causal attention, all reading/writing row `t` of the chunk buffers.
    #[allow(clippy::too_many_arguments)]
    /// The row-local front half of `step_chunk_row`: split q|gate, then apply
    /// the per-head RMS norm and RoPE to q and k at `pos`. The quantized-KV
    /// chunk path calls this and then encodes the row on the device.
    pub fn prep_chunk_row(
        &self,
        l: &Launcher,
        qg: &Buffer,
        k: &Buffer,
        q: &Buffer,
        gate: &Buffer,
        t: usize,
        pos: usize,
    ) -> Result<(), String> {
        let d = self.dims;
        // The chunk walks rows, each at its own position: one async 16-byte
        // write per row feeds every position-aware kernel below.
        l.set_step(pos, pos + 1, 0)?;
        let qg_row = row_ptr(qg, t, 2 * d.n_head * d.head_dim);
        let q_row = row_ptr(q, t, d.n_head * d.head_dim);
        let gate_row = row_ptr(gate, t, d.n_head * d.head_dim);
        let k_row = row_ptr(k, t, d.width);

        let split = self.module_kernel(l, "qg_split_f32")?;
        let mut p_qg = qg_row;
        let mut p_q = q_row;
        let mut p_gate = gate_row;
        let mut nh = d.n_head as i32;
        let mut hd = d.head_dim as i32;
        let mut params: [*mut c_void; 5] = [
            &mut p_qg as *mut _ as *mut c_void,
            &mut p_q as *mut _ as *mut c_void,
            &mut p_gate as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (d.n_head as u32, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        split.launch(cfg, &l.stream, &mut params)?;

        self.norm_rope_at(l, q_row, k_row)
    }

    pub fn step_chunk_row(
        &self,
        l: &Launcher,
        qg: &Buffer,
        k: &Buffer,
        v: &Buffer,
        q: &Buffer,
        gate: &Buffer,
        out: &Buffer,
        t: usize,
        pos: usize,
    ) -> Result<(), String> {
        let d = self.dims;
        let q_row = row_ptr(q, t, d.n_head * d.head_dim);
        let k_row = row_ptr(k, t, d.width);
        let v_row = row_ptr(v, t, d.width);
        let out_row = row_ptr(out, t, d.n_head * d.head_dim);

        self.prep_chunk_row(l, qg, k, q, gate, t, pos)?;

        let append = self.module_kernel(l, "kv_append_f32")?;
        let mut p_kc = self.kv_k.as_ptr();
        let mut p_vc = self.kv_v.as_ptr();
        let mut p_k = k_row;
        let mut p_v = v_row;
        let mut p_step = l.step_buf().as_ptr();
        let mut width = d.width as i32;
        let mut params: [*mut c_void; 6] = [
            &mut p_kc as *mut _ as *mut c_void,
            &mut p_vc as *mut _ as *mut c_void,
            &mut p_k as *mut _ as *mut c_void,
            &mut p_v as *mut _ as *mut c_void,
            &mut p_step as *mut _ as *mut c_void,
            &mut width as *mut _ as *mut c_void,
        ];
        append.launch(
            LaunchConfig::linear(2 * d.width, 256),
            &l.stream,
            &mut params,
        )?;

        self.attention_at(l, q_row, out_row, pos)
    }

    /// `y[t, :] = W_o (attn_out[t] * gate[t])` for a whole chunk: one elementwise
    /// pass over `tokens * n_head * head_dim` then one tiled GEMM.
    pub fn apply_output_chunk(
        &self,
        l: &Launcher,
        out: &Buffer,
        gate: &Buffer,
        y: &Buffer,
        tokens: usize,
    ) -> Result<(), String> {
        let d = self.dims;
        let n = d.n_head * d.head_dim * tokens;
        let g = self.module_kernel(l, "q_gate_f32")?;
        let mut p_out = out.as_ptr();
        let mut p_gate = gate.as_ptr();
        let mut p_y = out.as_ptr();
        let mut v_n = n as i32;
        let mut params: [*mut c_void; 4] = [
            &mut p_out as *mut _ as *mut c_void,
            &mut p_gate as *mut _ as *mut c_void,
            &mut p_y as *mut _ as *mut c_void,
            &mut v_n as *mut _ as *mut c_void,
        ];
        g.launch(LaunchConfig::linear(n, 256), &l.stream, &mut params)?;
        crate::prefill_resident(
            l.module(),
            l.stream(),
            self.wo.prefill_entry()?,
            &self.wo.data_bytes(),
            out,
            y,
            d.n_embd,
            d.n_head * d.head_dim,
            tokens,
        )
    }
}

/// Device pointer of row `t` of a `[tokens][width]` buffer.
fn row_ptr(buf: &Buffer, t: usize, width: usize) -> *mut c_void {
    unsafe { (buf.as_ptr() as *mut u8).add(t * width * 4) as *mut c_void }
}
