// SPDX-License-Identifier: Apache-2.0
//! Pure streaming-read launches, for the bandwidth instrument in `gpu-sweep`.
//!
//! Child module of `device` so `Launcher`'s module and stream stay private.
//! These kernels have no arithmetic purpose; they exist to measure how much of
//! the decode stream's gap to the roofline is per-launch and how much is
//! per-byte.

use super::*;

/// One descriptor in `stream_read_jobs`: a float4 pointer and a count.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReadJob {
    pub ptr: u64,
    pub nvec: u64,
}

impl Launcher {
    /// Read `bytes` bytes of a resident payload once, with `blocks` blocks.
    pub fn stream_read(
        &self,
        w: &ByteBuffer,
        bytes: usize,
        blocks: usize,
        out: &Buffer,
    ) -> Result<(), String> {
        self.stream_read_ptr(w.device_ptr(), bytes, blocks, out)
    }

    /// The same read over a raw device address, for the calibration buffer.
    pub fn stream_read_ptr(
        &self,
        ptr: usize,
        bytes: usize,
        blocks: usize,
        out: &Buffer,
    ) -> Result<(), String> {
        let kernel = self.module.kernel("stream_read")?;
        let mut p = ptr as *const std::ffi::c_void;
        let mut nvec = bytes / 16;
        let mut p_out = out.as_ptr();
        let mut params: [*mut std::ffi::c_void; 3] = [
            &mut p as *mut _ as *mut std::ffi::c_void,
            &mut nvec as *mut _ as *mut std::ffi::c_void,
            &mut p_out as *mut _ as *mut std::ffi::c_void,
        ];
        kernel.launch(LaunchConfig::linear(blocks, 256), &self.stream, &mut params)
    }

    /// Stream a resident payload with the GEMV's own row-tile geometry.
    ///
    /// `entry` is one of `stream_shape_w4`, `stream_shape_w16`,
    /// `stream_shape_w4_r1` or `stream_shape_flat`; the first three take
    /// (w, x, y, m, k, row_bytes) and read `row_bytes = nbytes / m` per row, so
    /// the byte count and the request order match the GEMV that reads the same
    /// tensor. `x` and `k` are unused; `out` doubles as both so the ABI stays
    /// the GEMV's five arguments plus the stride.
    pub fn stream_shape(
        &self,
        entry: &str,
        w: &ByteBuffer,
        m: usize,
        k: usize,
        row_bytes: usize,
        blocks: usize,
        threads: usize,
        out: &Buffer,
    ) -> Result<(), String> {
        let kernel = self.module.kernel(entry)?;
        let mut p_w = w.device_ptr() as *const std::ffi::c_void;
        let mut p_x = out.as_ptr();
        let mut p_y = out.as_ptr();
        let mut v_m = m as i32;
        let mut v_k = k as i32;
        let mut v_rb = row_bytes as i32;
        let mut params: [*mut std::ffi::c_void; 6] = [
            &mut p_w as *mut _ as *mut std::ffi::c_void,
            &mut p_x as *mut _ as *mut std::ffi::c_void,
            &mut p_y as *mut _ as *mut std::ffi::c_void,
            &mut v_m as *mut _ as *mut std::ffi::c_void,
            &mut v_k as *mut _ as *mut std::ffi::c_void,
            &mut v_rb as *mut _ as *mut std::ffi::c_void,
        ];
        let cfg = LaunchConfig {
            grid: (blocks.max(1) as u32, 1, 1),
            block: (threads.max(32) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        kernel.launch(cfg, &self.stream, &mut params)
    }

    /// Read every job in `jobs` in one launch. `jobs` is the device-side copy.
    pub fn stream_read_jobs(
        &self,
        jobs: &Buffer,
        n_jobs: usize,
        blocks: usize,
        out: &Buffer,
    ) -> Result<(), String> {
        let kernel = self.module.kernel("stream_read_jobs")?;
        let mut p_jobs = jobs.as_ptr();
        let mut v_n = n_jobs as i32;
        let mut p_out = out.as_ptr();
        let mut params: [*mut std::ffi::c_void; 3] = [
            &mut p_jobs as *mut _ as *mut std::ffi::c_void,
            &mut v_n as *mut _ as *mut std::ffi::c_void,
            &mut p_out as *mut _ as *mut std::ffi::c_void,
        ];
        kernel.launch(LaunchConfig::linear(blocks, 256), &self.stream, &mut params)
    }
}
