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
        kernel.launch(
            LaunchConfig::linear(blocks, 256),
            &self.stream,
            &mut params,
        )
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
        kernel.launch(
            LaunchConfig::linear(blocks, 256),
            &self.stream,
            &mut params,
        )
    }
}
