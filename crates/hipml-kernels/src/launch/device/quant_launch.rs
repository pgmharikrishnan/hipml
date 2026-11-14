// SPDX-License-Identifier: Apache-2.0
//! Quantized GEMV launches, including the resident-weight path used by the
//! loader.
//!
//! Kept out of `launch.rs` because the argument lists are the kernel ABI and are
//! written out by hand; one file per family keeps both readable.

#![cfg(feature = "hip")]

use std::ffi::c_void;

use hipml_hip::{Buffer, ByteBuffer, LaunchConfig};

use crate::Launcher;

impl Launcher {
    /// Launch a quantized GEMV once. `w` is the raw block payload, row-major.
    pub fn run_quant(
        &self,
        name: &str,
        w: &[u8],
        x: &[f32],
        m: usize,
        k: usize,
    ) -> Result<Vec<f32>, String> {
        self.run_quant_reps(name, w, x, m, k, 1).map(|(y, _)| y)
    }

    /// Launch a quantized GEMV `reps` times over one upload.
    ///
    /// Returns the output and the wall time spent inside the launch loop
    /// (uploads excluded), so the caller can report kernel bandwidth rather
    /// than PCIe bandwidth.
    pub fn run_quant_reps(
        &self,
        name: &str,
        w: &[u8],
        x: &[f32],
        m: usize,
        k: usize,
        reps: usize,
    ) -> Result<(Vec<f32>, f64), String> {
        assert!(reps > 0, "reps must be positive");
        let spec = crate::spec(name).ok_or_else(|| format!("{name} is not in the manifest"))?;
        let wb = ByteBuffer::from_host(w)?;
        let xb = Buffer::from_host(x)?;
        let y = Buffer::alloc(m)?;
        let kernel = self.module.kernel(name)?;

        let mut p_w = wb.as_ptr();
        let mut p_x = xb.as_ptr();
        let mut p_y = y.as_ptr();
        let mut v_m = m as i32;
        let mut v_k = k as i32;
        let mut params: [*mut c_void; 5] = [
            &mut p_w as *mut _ as *mut c_void,
            &mut p_x as *mut _ as *mut c_void,
            &mut p_y as *mut _ as *mut c_void,
            &mut v_m as *mut _ as *mut c_void,
            &mut v_k as *mut _ as *mut c_void,
        ];
        let cfg = match spec.kind {
            crate::KernelKind::RowTile { rows, block } => LaunchConfig::row_tile(m, rows, block),
            crate::KernelKind::PerRow { block } => LaunchConfig::per_row(m, block),
            _ => return Err(format!("{name}: unsupported quantized grid")),
        };
        let start = std::time::Instant::now();
        for _ in 0..reps {
            kernel.launch(cfg, &self.stream, &mut params)?;
        }
        self.stream.synchronize()?;
        let secs = start.elapsed().as_secs_f64();
        Ok((y.to_host()?, secs))
    }

    /// Frozen-weight GEMV with both ends already on the device.
    ///
    /// This is the graph's inner loop: no host copy and no synchronize, so
    /// the caller can chain hundreds of launches before one sync.
    pub fn run_quant_resident(
        &self,
        name: &str,
        w: &ByteBuffer,
        x: &Buffer,
        y: &Buffer,
        m: usize,
        k: usize,
    ) -> Result<(), String> {
        self.run_quant_resident_on("stream", name, w, x, y, m, k)
    }

    /// The same launch on a chosen stream: the side-stream half of a
    /// parallel section (see `Launcher::fork`).
    pub fn run_quant_resident_on(
        &self,
        which: &str,
        name: &str,
        w: &ByteBuffer,
        x: &Buffer,
        y: &Buffer,
        m: usize,
        k: usize,
    ) -> Result<(), String> {
        let stream = if which == "side" {
            &self.side
        } else {
            &self.stream
        };
        assert_eq!(x.len(), k, "x must be k");
        assert_eq!(y.len(), m, "y must be m");
        let spec = crate::spec(name).ok_or_else(|| format!("{name} is not in the manifest"))?;
        let kernel = self.module.kernel(name)?;
        let mut p_w = w.as_ptr();
        let mut p_x = x.as_ptr();
        let mut p_y = y.as_ptr();
        let mut v_m = m as i32;
        let mut v_k = k as i32;
        let mut params: [*mut c_void; 5] = [
            &mut p_w as *mut _ as *mut c_void,
            &mut p_x as *mut _ as *mut c_void,
            &mut p_y as *mut _ as *mut c_void,
            &mut v_m as *mut _ as *mut c_void,
            &mut v_k as *mut _ as *mut c_void,
        ];
        let cfg = match spec.kind {
            crate::KernelKind::RowTile { rows, block } => LaunchConfig::row_tile(m, rows, block),
            crate::KernelKind::PerRow { block } => LaunchConfig::per_row(m, block),
            _ => return Err(format!("{name}: unsupported quantized grid")),
        };
        kernel.launch(cfg, stream, &mut params)
    }

    /// `gemm_f32` with the weight already resident: the multi-row f32
    /// fallback for the prefill chunk (the two Q5_K `ffn_down` tensors).
    pub fn run_gemm_resident(
        &self,
        w: &Buffer,
        x: &Buffer,
        y: &Buffer,
        r: usize,
        m: usize,
        k: usize,
    ) -> Result<(), String> {
        self.run_gemm_ptr(w.as_ptr(), x.as_ptr(), y.as_ptr(), r, m, k)
    }

    /// Same with raw pointers, for an f32 payload that lives in a
    /// `ByteBuffer` (the loader keeps F32 tensors as raw little-endian f32).
    pub fn run_gemm_ptr(
        &self,
        w: *mut c_void,
        x: *mut c_void,
        y: *mut c_void,
        r: usize,
        m: usize,
        k: usize,
    ) -> Result<(), String> {
        let kernel = self.module.kernel("gemm_f32")?;
        let mut p_w = w;
        let mut p_x = x;
        let mut p_y = y;
        let mut v_r = r as i32;
        let mut v_m = m as i32;
        let mut v_k = k as i32;
        let mut params: [*mut c_void; 6] = [
            &mut p_w as *mut _ as *mut c_void,
            &mut p_x as *mut _ as *mut c_void,
            &mut p_y as *mut _ as *mut c_void,
            &mut v_r as *mut _ as *mut c_void,
            &mut v_m as *mut _ as *mut c_void,
            &mut v_k as *mut _ as *mut c_void,
        ];
        kernel.launch(LaunchConfig::per_row(m, 256), &self.stream, &mut params)
    }

    /// `run_quant_resident` with raw x/y pointers (the chunk's last row).
    pub fn run_quant_resident_ptr(
        &self,
        name: &str,
        w: &ByteBuffer,
        x: *mut c_void,
        y: *mut c_void,
        m: usize,
        k: usize,
    ) -> Result<(), String> {
        let spec = crate::spec(name).ok_or_else(|| format!("{name} is not in the manifest"))?;
        let kernel = self.module.kernel(name)?;
        let mut p_w = w.as_ptr();
        let mut p_x = x;
        let mut p_y = y;
        let mut v_m = m as i32;
        let mut v_k = k as i32;
        let mut params: [*mut c_void; 5] = [
            &mut p_w as *mut _ as *mut c_void,
            &mut p_x as *mut _ as *mut c_void,
            &mut p_y as *mut _ as *mut c_void,
            &mut v_m as *mut _ as *mut c_void,
            &mut v_k as *mut _ as *mut c_void,
        ];
        let cfg = match spec.kind {
            crate::KernelKind::RowTile { rows, block } => LaunchConfig::row_tile(m, rows, block),
            crate::KernelKind::PerRow { block } => LaunchConfig::per_row(m, block),
            _ => return Err(format!("{name}: unsupported quantized grid")),
        };
        kernel.launch(cfg, &self.stream, &mut params)
    }

    /// `gemv_f32` with the weight already resident (the Q5_K fallback, which
    /// the loader dequantises once at graph build).
    pub fn run_f32_resident(
        &self,
        w: &Buffer,
        x: &Buffer,
        y: &Buffer,
        m: usize,
        k: usize,
    ) -> Result<(), String> {
        self.run_f32_resident_on("stream", w, x, y, m, k)
    }

    /// `gemv_f32` on a chosen stream.
    pub fn run_f32_resident_on(
        &self,
        which: &str,
        w: &Buffer,
        x: &Buffer,
        y: &Buffer,
        m: usize,
        k: usize,
    ) -> Result<(), String> {
        let stream = if which == "side" {
            &self.side
        } else {
            &self.stream
        };
        let kernel = self.module.kernel("gemv_f32")?;
        let mut p_w = w.as_ptr();
        let mut p_x = x.as_ptr();
        let mut p_y = y.as_ptr();
        let mut v_m = m as i32;
        let mut v_k = k as i32;
        let mut params: [*mut c_void; 5] = [
            &mut p_w as *mut _ as *mut c_void,
            &mut p_x as *mut _ as *mut c_void,
            &mut p_y as *mut _ as *mut c_void,
            &mut v_m as *mut _ as *mut c_void,
            &mut v_k as *mut _ as *mut c_void,
        ];
        kernel.launch(LaunchConfig::per_row(m, 256), stream, &mut params)
    }

    /// Frozen-weight variant of `run_quant_reps`: `w` is already resident in
    /// VRAM, so only x is copied. This is the per-token vocabulary cost.
    pub fn run_quant_resident_reps(
        &self,
        name: &str,
        w: &ByteBuffer,
        x: &[f32],
        m: usize,
        k: usize,
        reps: usize,
    ) -> Result<(Vec<f32>, f64), String> {
        assert!(reps > 0, "reps must be positive");
        assert_eq!(x.len(), k, "x must be k");
        let spec = crate::spec(name).ok_or_else(|| format!("{name} is not in the manifest"))?;
        let xb = Buffer::from_host(x)?;
        let y = Buffer::alloc(m)?;
        let kernel = self.module.kernel(name)?;

        let mut p_w = w.as_ptr();
        let mut p_x = xb.as_ptr();
        let mut p_y = y.as_ptr();
        let mut v_m = m as i32;
        let mut v_k = k as i32;
        let mut params: [*mut c_void; 5] = [
            &mut p_w as *mut _ as *mut c_void,
            &mut p_x as *mut _ as *mut c_void,
            &mut p_y as *mut _ as *mut c_void,
            &mut v_m as *mut _ as *mut c_void,
            &mut v_k as *mut _ as *mut c_void,
        ];
        let cfg = match spec.kind {
            crate::KernelKind::RowTile { rows, block } => LaunchConfig::row_tile(m, rows, block),
            crate::KernelKind::PerRow { block } => LaunchConfig::per_row(m, block),
            _ => return Err(format!("{name}: unsupported quantized grid")),
        };
        let start = std::time::Instant::now();
        for _ in 0..reps {
            kernel.launch(cfg, &self.stream, &mut params)?;
        }
        self.stream.synchronize()?;
        let secs = start.elapsed().as_secs_f64();
        Ok((y.to_host()?, secs))
    }

    /// Dequantise `nblocks` Q4_K super-blocks starting `byte_offset` bytes into
    /// a resident payload. One token embedding row is 20 blocks at k=5120.
    pub fn dequant_q4_k(
        &self,
        w: &ByteBuffer,
        byte_offset: u64,
        nblocks: usize,
    ) -> Result<Vec<f32>, String> {
        self.dequant_row("dequant_q4_k_f32", w, byte_offset, nblocks)
    }

    /// The same for Q6_K blocks (a UD dynamic quant's `token_embd.weight`).
    pub fn dequant_q6_k(
        &self,
        w: &ByteBuffer,
        byte_offset: u64,
        nblocks: usize,
    ) -> Result<Vec<f32>, String> {
        self.dequant_row("dequant_q6_k_f32", w, byte_offset, nblocks)
    }

    /// Dequantise one row of any resident format with a row decoder
    /// (`dequant_rows.hip`): the loader's embedding probe.
    ///
    /// `nblocks` counts 256-value blocks, which is one K-quant super-block,
    /// eight Q8_0/IQ4_NL blocks, or 1 KiB / 512 B of an F32 / BF16 row.
    pub fn dequant_row_typed(
        &self,
        ty: hipml_core::quant::TensorType,
        w: &ByteBuffer,
        byte_offset: u64,
        nblocks: usize,
    ) -> Result<Vec<f32>, String> {
        let out = Buffer::alloc(nblocks * crate::ROW_BLOCK_VALUES)?;
        let kernel = self.module.kernel("dequant_row_any")?;
        let offset = usize::try_from(byte_offset)
            .map_err(|_| "tensor offset does not fit in usize".to_string())?;
        let mut p_w = unsafe { (w.as_ptr() as *mut u8).add(offset) } as *mut c_void;
        let mut p_out = out.as_ptr();
        let mut v_blocks = nblocks as i32;
        let mut v_kind = crate::row_kind(ty)?;
        let mut params: [*mut c_void; 4] = [
            &mut p_w as *mut _ as *mut c_void,
            &mut p_out as *mut _ as *mut c_void,
            &mut v_blocks as *mut _ as *mut c_void,
            &mut v_kind as *mut _ as *mut c_void,
        ];
        kernel.launch(
            LaunchConfig::per_row(nblocks, 256),
            &self.stream,
            &mut params,
        )?;
        self.stream.synchronize()?;
        out.to_host()
    }

    fn dequant_row(
        &self,
        entry: &str,
        w: &ByteBuffer,
        byte_offset: u64,
        nblocks: usize,
    ) -> Result<Vec<f32>, String> {
        let out = Buffer::alloc(nblocks * hipml_core::quant::QK_K)?;
        let kernel = self.module.kernel(entry)?;
        let offset = usize::try_from(byte_offset)
            .map_err(|_| "tensor offset does not fit in usize".to_string())?;
        let mut p_w = unsafe { (w.as_ptr() as *mut u8).add(offset) } as *mut c_void;
        let mut p_out = out.as_ptr();
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
        )?;
        self.stream.synchronize()?;
        out.to_host()
    }
}
