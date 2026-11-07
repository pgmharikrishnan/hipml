//! The elementwise launches the decode graph chains between its GEMVs.
//!
//! Every method here works on device buffers and does not synchronize; the
//! graph issues a whole token and syncs once. Argument arrays are written out
//! explicitly because they are the kernel ABI.

use super::*;

impl Launcher {
    /// `y = x / sqrt(mean(x^2) + eps) * w`, one block per row (row count 1 here).
    pub fn rmsnorm_buf(
        &self,
        x: &Buffer,
        w: &Buffer,
        y: &Buffer,
        n: usize,
        eps: f32,
    ) -> Result<(), String> {
        self.rmsnorm_rows(x, w, y, n, eps, 1)
    }

    /// `y[r][:] = rmsnorm(x[r][:], w)` for `rows` rows, one block per row. The
    /// prefill path runs a whole prompt tile through the same kernel.
    pub fn rmsnorm_rows(
        &self,
        x: &Buffer,
        w: &Buffer,
        y: &Buffer,
        n: usize,
        eps: f32,
        rows: usize,
    ) -> Result<(), String> {
        self.rmsnorm_ptr(x.as_ptr(), w, y.as_ptr(), n, eps, rows)
    }

    /// Same launch with raw pointers, so a chunk can normalise one row in place
    /// without a copy (`rmsnorm` reads `row * n` from the pointer).
    pub fn rmsnorm_ptr(
        &self,
        x: *mut c_void,
        w: &Buffer,
        y: *mut c_void,
        n: usize,
        eps: f32,
        rows: usize,
    ) -> Result<(), String> {
        let kernel = self.module.kernel("rmsnorm_f32")?;
        let mut p_x = x;
        let mut p_w = w.as_ptr();
        let mut p_y = y;
        let mut v_n = n as i32;
        let mut v_eps = eps;
        let mut params: [*mut c_void; 5] = [
            &mut p_x as *mut _ as *mut c_void,
            &mut p_w as *mut _ as *mut c_void,
            &mut p_y as *mut _ as *mut c_void,
            &mut v_n as *mut _ as *mut c_void,
            &mut v_eps as *mut _ as *mut c_void,
        ];
        kernel.launch(LaunchConfig::per_row(rows, 256), &self.stream, &mut params)
    }

    /// `y = a + b`, one thread per element.
    pub fn residual_buf(&self, a: &Buffer, b: &Buffer, y: &Buffer, n: usize) -> Result<(), String> {
        self.linear3("residual_add_f32", a, b, y, n)
    }

    /// `y = silu(gate) * up`, one thread per element.
    pub fn silu_mul_buf(
        &self,
        gate: &Buffer,
        up: &Buffer,
        y: &Buffer,
        n: usize,
    ) -> Result<(), String> {
        self.linear3("silu_mul_f32", gate, up, y, n)
    }

    /// `out = a * b` elementwise for the two linear kernels above.
    fn linear3(
        &self,
        name: &str,
        a: &Buffer,
        b: &Buffer,
        y: &Buffer,
        n: usize,
    ) -> Result<(), String> {
        let kernel = self.module.kernel(name)?;
        let mut p_a = a.as_ptr();
        let mut p_b = b.as_ptr();
        let mut p_y = y.as_ptr();
        let mut v_n = n as i32;
        let mut params: [*mut c_void; 4] = [
            &mut p_a as *mut _ as *mut c_void,
            &mut p_b as *mut _ as *mut c_void,
            &mut p_y as *mut _ as *mut c_void,
            &mut v_n as *mut _ as *mut c_void,
        ];
        kernel.launch(LaunchConfig::linear(n, 256), &self.stream, &mut params)
    }

    /// Dequantise one embedding row of a resident Q4_K tensor into `out`, on the
    /// device: the graph's `embed` step, with no host round-trip.
    ///
    /// `row_bytes` is the tensor's own row stride, so this works for any
    /// `n_embd` that is a multiple of 256.
    pub fn embed_row_q4_k(
        &self,
        w: &ByteBuffer,
        row: usize,
        row_bytes: usize,
        out: &Buffer,
    ) -> Result<(), String> {
        self.embed_row_q4_k_at(w, row, row_bytes, out, 0, out.len())
    }

    /// Same, writing the dequantized row at float offset `dst` of `out`: the
    /// chunked prefill embeds a whole tile into one `[tokens][n_embd]` buffer.
    pub fn embed_row_q4_k_at(
        &self,
        w: &ByteBuffer,
        row: usize,
        row_bytes: usize,
        out: &Buffer,
        dst: usize,
        n_embd: usize,
    ) -> Result<(), String> {
        let nblocks = n_embd / hipml_core::quant::QK_K;
        let kernel = self.module.kernel("dequant_q4_k_f32")?;
        let mut p_w = unsafe { (w.as_ptr() as *mut u8).add(row * row_bytes) } as *mut c_void;
        let mut p_out = unsafe { (out.as_ptr() as *mut u8).add(dst * 4) } as *mut c_void;
        let mut v_blocks = nblocks as i32;
        let mut params: [*mut c_void; 3] = [
            &mut p_w as *mut _ as *mut c_void,
            &mut p_out as *mut _ as *mut c_void,
            &mut v_blocks as *mut _ as *mut c_void,
        ];
        kernel.launch(
            LaunchConfig::per_row(nblocks, 256),
            &self.stream,
            &mut params,
        )
    }

    /// Device argmax over `n` values, `parts` blocks then one final fold.
    ///
    /// `best_v` holds `parts` f32, `best_i` `parts` i32, `out` one i32; the
    /// caller reads `out` back with `download_bytes(0, 4)`.
    #[allow(clippy::too_many_arguments)]
    pub fn argmax_buf(
        &self,
        x: &Buffer,
        best_v: &Buffer,
        best_i: &Buffer,
        out: &Buffer,
        n: usize,
        parts: usize,
    ) -> Result<(), String> {
        let part = self.module.kernel("argmax_part_f32")?;
        let mut p_x = x.as_ptr();
        let mut p_v = best_v.as_ptr();
        let mut p_i = best_i.as_ptr();
        let mut v_n = n as i32;
        let mut v_parts = parts as i32;
        let mut params: [*mut c_void; 5] = [
            &mut p_x as *mut _ as *mut c_void,
            &mut p_v as *mut _ as *mut c_void,
            &mut p_i as *mut _ as *mut c_void,
            &mut v_n as *mut _ as *mut c_void,
            &mut v_parts as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (parts as u32, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        part.launch(cfg, &self.stream, &mut params)?;

        let fin = self.module.kernel("argmax_final_f32")?;
        let mut p_v = best_v.as_ptr();
        let mut p_i = best_i.as_ptr();
        let mut p_out = out.as_ptr();
        let mut params: [*mut c_void; 4] = [
            &mut p_v as *mut _ as *mut c_void,
            &mut p_i as *mut _ as *mut c_void,
            &mut p_out as *mut _ as *mut c_void,
            &mut v_parts as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (1, 1, 1),
            block: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        fin.launch(cfg, &self.stream, &mut params)
    }

    /// Quantize one token's K and V rows into the cache at `pos` on the device.
    ///
    /// One launch per layer replaces a readback, a host quantize and eight
    /// uploads. `dtype` is `KvDtype::id()`: 0 q8_0, 1 q4_0.
    #[allow(clippy::too_many_arguments)]
    pub fn kv_encode(
        &self,
        dtype: usize,
        k_in: &Buffer,
        v_in: &Buffer,
        k_cache: &Buffer,
        v_cache: &Buffer,
        n_kv: usize,
        head_dim: usize,
        seq: usize,
        pos: usize,
    ) -> Result<(), String> {
        let entry = match dtype {
            0 => "kv_encode_q8",
            1 => "kv_encode_q4",
            other => return Err(format!("unknown KV dtype {other}")),
        };
        let kernel = self.module.kernel(entry)?;
        let mut p_k = k_in.as_ptr();
        let mut p_v = v_in.as_ptr();
        let mut p_kc = k_cache.as_ptr();
        let mut p_vc = v_cache.as_ptr();
        let mut v_hd = head_dim as i32;
        let mut v_seq = seq as i32;
        let mut v_pos = pos as i32;
        let mut params: [*mut c_void; 7] = [
            &mut p_k as *mut _ as *mut c_void,
            &mut p_v as *mut _ as *mut c_void,
            &mut p_kc as *mut _ as *mut c_void,
            &mut p_vc as *mut _ as *mut c_void,
            &mut v_hd as *mut _ as *mut c_void,
            &mut v_seq as *mut _ as *mut c_void,
            &mut v_pos as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (n_kv as u32, 1, 1),
            block: (head_dim.max(32) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        kernel.launch(cfg, &self.stream, &mut params)
    }
}
