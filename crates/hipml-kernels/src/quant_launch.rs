//! Quantized GEMV launches: the decode path, kept beside the prefill path but
//! out of `launch.rs` so both stay readable.
//!
//! This module is a child of `device`, so it can see `Launcher`'s private
//! module/stream handles; the methods are inherent and called as usual.

use super::*;

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
}
