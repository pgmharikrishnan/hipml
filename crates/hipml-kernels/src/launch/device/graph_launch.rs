//! The elementwise launches the decode graph chains between its GEMVs.
//!
//! Every method here works on device buffers and does not synchronize; the
//! graph issues a whole token and syncs once. Argument arrays are written out
//! explicitly because they are the kernel ABI.

use super::*;

/// One entry of the `gemv_jobs` device job list. Mirrors `HipmlGemvJob` in
/// `gemv_jobs.hip`; pointers are device addresses.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemvJob {
    pub w: usize,
    pub x: usize,
    pub y: usize,
    pub m: i32,
    pub k: i32,
    pub kind: i32,
    pub pad: i32,
}

/// `kind` values the job kernel dispatches on.
pub const GEMV_JOB_Q4K: i32 = 0;
pub const GEMV_JOB_Q6K: i32 = 1;
pub const GEMV_JOB_Q5K: i32 = 2;
pub const GEMV_JOB_Q80: i32 = 3;

/// Tiles a job occupies (8 rows per tile for the K-quants, one row for Q8_0).
pub fn gemv_job_tiles(kind: i32, m: usize) -> usize {
    if kind == GEMV_JOB_Q80 {
        m
    } else {
        (m + 7) / 8
    }
}

/// Batched-GEMV kind ids: the four resident weight formats whose decode
/// kernels have a `_b` entry (B x rows, one weight pass).
pub const GEMV_B_Q4K: i32 = 0;
pub const GEMV_B_Q6K: i32 = 1;
pub const GEMV_B_Q5K: i32 = 2;
pub const GEMV_B_Q80: i32 = 3;

impl Launcher {
    /// `y[b][row] = W[row,:] . x[b][:]` for `batch` rows, one weight pass.
    ///
    /// `x` is `[batch][k]`, `y` is `[batch][m]`. The `_b` kernels keep each
    /// row's accumulation order identical to the single-sequence GEMV, so this
    /// is a throughput change, not a numerics change.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_batched(
        &self,
        kind: i32,
        w: &ByteBuffer,
        x: &Buffer,
        y: &Buffer,
        m: usize,
        k: usize,
        batch: usize,
    ) -> Result<(), String> {
        if batch == 0 || batch > 8 {
            return Err(format!("batched GEMV: batch {batch} outside 1..=8"));
        }
        let (entry, rows, shared) = match kind {
            GEMV_B_Q4K => ("gemv_q4_k_b", 16usize, batch * 512 * 4),
            GEMV_B_Q6K => ("gemv_q6_k_b", 16, batch * 512 * 4),
            GEMV_B_Q5K => ("gemv_q5_k_b", 16, batch * 512 * 4),
            GEMV_B_Q80 => ("gemv_q8_0_b", 1, 0),
            other => return Err(format!("unknown batched GEMV kind {other}")),
        };
        let kernel = self.module.kernel(entry)?;
        let mut p_w = w.as_ptr();
        let mut p_x = x.as_ptr();
        let mut p_y = y.as_ptr();
        let mut v_m = m as i32;
        let mut v_k = k as i32;
        let mut v_b = batch as i32;
        let mut params: [*mut c_void; 6] = [
            &mut p_w as *mut _ as *mut c_void,
            &mut p_x as *mut _ as *mut c_void,
            &mut p_y as *mut _ as *mut c_void,
            &mut v_m as *mut _ as *mut c_void,
            &mut v_k as *mut _ as *mut c_void,
            &mut v_b as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (m.div_ceil(rows) as u32, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: shared as u32,
        };
        kernel.launch(cfg, &self.stream, &mut params)
    }

    /// Batched greedy argmax over `rows` logit vectors of `n` values each.
    /// `best_v`/`best_i` hold `rows * parts` entries, `out` holds `rows` ids.
    #[allow(clippy::too_many_arguments)]
    pub fn argmax_batched(
        &self,
        x: &Buffer,
        best_v: &Buffer,
        best_i: &Buffer,
        out: &Buffer,
        n: usize,
        parts: usize,
        rows: usize,
    ) -> Result<(), String> {
        let part = self.module.kernel("argmax_part_rows_f32")?;
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
            grid: (parts as u32, rows as u32, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        part.launch(cfg, &self.stream, &mut params)?;

        let fin = self.module.kernel("argmax_final_rows_f32")?;
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
            grid: (rows as u32, 1, 1),
            block: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        fin.launch(cfg, &self.stream, &mut params)
    }

    /// Split-K Q4_K GEMV: row tiles x splits, then the ordered reduce.
    pub fn gemv_q4_k_split(
        &self,
        w: &ByteBuffer,
        x: &Buffer,
        part: &Buffer,
        y: &Buffer,
        m: usize,
        k: usize,
        splits: usize,
    ) -> Result<(), String> {
        let kernel = self.module.kernel("gemv_q4_k_split")?;
        let mut p_w = w.as_ptr();
        let mut p_x = x.as_ptr();
        let mut p_part = part.as_ptr();
        let mut v_m = m as i32;
        let mut v_k = k as i32;
        let mut v_splits = splits as i32;
        let mut params: [*mut c_void; 6] = [
            &mut p_w as *mut _ as *mut c_void,
            &mut p_x as *mut _ as *mut c_void,
            &mut p_part as *mut _ as *mut c_void,
            &mut v_m as *mut _ as *mut c_void,
            &mut v_k as *mut _ as *mut c_void,
            &mut v_splits as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (((m + 7) / 8) as u32, splits as u32, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        kernel.launch(cfg, &self.stream, &mut params)?;

        let reduce = self.module.kernel("gemv_split_reduce")?;
        let mut p_part = part.as_ptr();
        let mut p_y = y.as_ptr();
        let mut params: [*mut c_void; 4] = [
            &mut p_part as *mut _ as *mut c_void,
            &mut p_y as *mut _ as *mut c_void,
            &mut v_m as *mut _ as *mut c_void,
            &mut v_splits as *mut _ as *mut c_void,
        ];
        reduce.launch(LaunchConfig::linear(m, 256), &self.stream, &mut params)
    }

    /// Launch every job in `jobs` with one kernel: grid.y picks the job, grid.x
    /// the row tile inside it. Jobs must be independent of each other.
    pub fn gemv_jobs(&self, jobs: &Buffer, n_jobs: usize, tiles: usize) -> Result<(), String> {
        let kernel = self.module.kernel("gemv_jobs")?;
        let mut p_jobs = jobs.as_ptr();
        let mut v_n = n_jobs as i32;
        let mut params: [*mut c_void; 2] = [
            &mut p_jobs as *mut _ as *mut c_void,
            &mut v_n as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (tiles.max(1) as u32, n_jobs.max(1) as u32, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        kernel.launch(cfg, &self.stream, &mut params)
    }

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

    /// Dequantise the embedding row named by `step[2]` into `out`: the step's
    /// token id, so a captured graph is replayed for a new token without
    /// re-capturing.
    pub fn embed_row_q4_k_dev(
        &self,
        w: &ByteBuffer,
        row_bytes: usize,
        out: &Buffer,
        n_embd: usize,
    ) -> Result<(), String> {
        let nblocks = n_embd / hipml_core::quant::QK_K;
        let kernel = self.module.kernel("embed_q4_k_row")?;
        let mut p_w = w.as_ptr();
        let mut p_out = out.as_ptr();
        let mut v_blocks = nblocks as i32;
        let mut v_row_bytes = row_bytes as i32;
        let mut p_step = self.step.as_ptr();
        let mut params: [*mut c_void; 5] = [
            &mut p_w as *mut _ as *mut c_void,
            &mut p_out as *mut _ as *mut c_void,
            &mut v_blocks as *mut _ as *mut c_void,
            &mut v_row_bytes as *mut _ as *mut c_void,
            &mut p_step as *mut _ as *mut c_void,
        ];
        kernel.launch(
            LaunchConfig::per_row(nblocks, 256),
            &self.stream,
            &mut params,
        )
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

    /// Quantize one token's K and V rows into the cache at `step[0]`.
    ///
    /// One launch per layer replaces a readback, a host quantize and eight
    /// uploads. `dtype` is `KvDtype::id()`: 0 q8_0, 1 q4_0. `stride` is the
    /// cache's rows per KV head and stays fixed; the position comes from the
    /// step buffer so a captured graph replays at every step.
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
        stride: usize,
    ) -> Result<(), String> {
        self.kv_encode_ptr(
            dtype,
            k_in.as_ptr(),
            v_in.as_ptr(),
            k_cache,
            v_cache,
            n_kv,
            head_dim,
            stride,
        )
    }

    /// Quantize and append one K/V row whose source pointer is a row inside a
    /// larger chunk buffer. Used by the chunked q8 prefill. The position comes
    /// from the step buffer, exactly as it does for the whole-buffer entry.
    pub fn kv_encode_ptr(
        &self,
        dtype: usize,
        k_ptr: *mut std::ffi::c_void,
        v_ptr: *mut std::ffi::c_void,
        k_cache: &Buffer,
        v_cache: &Buffer,
        n_kv: usize,
        head_dim: usize,
        stride: usize,
    ) -> Result<(), String> {
        self.kv_encode_ptr_on(&self.stream, &self.step, dtype, k_ptr, v_ptr, k_cache, v_cache,
                              n_kv, head_dim, stride)
    }

    /// Same append on a chosen stream with a chosen step buffer: the two-stream
    /// batch path gives each stream its own position.
    #[allow(clippy::too_many_arguments)]
    pub fn kv_encode_ptr_on(
        &self,
        stream: &Stream,
        step: &Buffer,
        dtype: usize,
        k_ptr: *mut std::ffi::c_void,
        v_ptr: *mut std::ffi::c_void,
        k_cache: &Buffer,
        v_cache: &Buffer,
        n_kv: usize,
        head_dim: usize,
        stride: usize,
    ) -> Result<(), String> {
        let entry = match dtype {
            0 => "kv_encode_q8",
            1 => "kv_encode_q4",
            other => return Err(format!("unknown KV dtype {other}")),
        };
        let kernel = self.module.kernel(entry)?;
        let mut p_k = k_ptr;
        let mut p_v = v_ptr;
        let mut p_kc = k_cache.as_ptr();
        let mut p_vc = v_cache.as_ptr();
        let mut v_hd = head_dim as i32;
        let mut v_stride = stride as i32;
        let mut p_step = step.as_ptr();
        let mut params: [*mut c_void; 7] = [
            &mut p_k as *mut _ as *mut c_void,
            &mut p_v as *mut _ as *mut c_void,
            &mut p_kc as *mut _ as *mut c_void,
            &mut p_vc as *mut _ as *mut c_void,
            &mut v_hd as *mut _ as *mut c_void,
            &mut v_stride as *mut _ as *mut c_void,
            &mut p_step as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (n_kv as u32, 1, 1),
            block: (head_dim.max(32) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        kernel.launch(cfg, stream, &mut params)
    }

    /// Quantize `rows` positions of a position-major f32 buffer into the
    /// head-major q8/q4 cache. `start` is the destination position of the first
    /// row; `src0` is the source row of `k_f32`/`v_f32` that holds it (0 for a
    /// chunk-local tile, `start` for the full f32 mirror).
    #[allow(clippy::too_many_arguments)]
    pub fn kv_encode_bulk(
        &self,
        dtype: usize,
        k_f32: &Buffer,
        v_f32: &Buffer,
        k_cache: &Buffer,
        v_cache: &Buffer,
        n_kv: usize,
        head_dim: usize,
        stride: usize,
        start: usize,
        src0: usize,
        rows: usize,
    ) -> Result<(), String> {
        let entry = match dtype {
            0 => "kv_encode_bulk_q8",
            1 => "kv_encode_bulk_q4",
            other => return Err(format!("unknown KV dtype {other}")),
        };
        let kernel = self.module.kernel(entry)?;
        let mut p_k = k_f32.as_ptr();
        let mut p_v = v_f32.as_ptr();
        let mut p_kc = k_cache.as_ptr();
        let mut p_vc = v_cache.as_ptr();
        let mut v_nkv = n_kv as i32;
        let mut v_hd = head_dim as i32;
        let mut v_stride = stride as i32;
        let mut v_start = start as i32;
        let mut v_src0 = src0 as i32;
        let mut v_rows = rows as i32;
        let mut params: [*mut c_void; 10] = [
            &mut p_k as *mut _ as *mut c_void,
            &mut p_v as *mut _ as *mut c_void,
            &mut p_kc as *mut _ as *mut c_void,
            &mut p_vc as *mut _ as *mut c_void,
            &mut v_nkv as *mut _ as *mut c_void,
            &mut v_hd as *mut _ as *mut c_void,
            &mut v_stride as *mut _ as *mut c_void,
            &mut v_start as *mut _ as *mut c_void,
            &mut v_rows as *mut _ as *mut c_void,
            &mut v_src0 as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: ((n_kv * rows) as u32, 1, 1),
            block: (head_dim.max(32) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        kernel.launch(cfg, &self.stream, &mut params)
    }
}
