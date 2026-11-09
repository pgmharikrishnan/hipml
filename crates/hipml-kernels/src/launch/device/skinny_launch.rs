//! Skinny quantized GEMM launches (`R <= 16` column tile).
//!
//! Child module of `device`, like `prefill_launch`: the free functions are what
//! the MTP verify and the batch engine call, and the `Launcher` methods are the
//! bench/parity path.
//!
//! Contract: `y[r, :] = W @ x[r, :]`, `x` is `[r, k]` f32, `y` is `[r, n]` f32,
//! `r <= 16`, `k` a multiple of 256 (Q4_K/Q6_K) or 32 (Q8_0). The 16-row
//! activation tile is padded with zeros when `r < 16`, so the cost is the tile.

use super::*;

/// Resident-weight skinny GEMM: the launch the graph uses.
#[allow(clippy::too_many_arguments)]
pub fn skinny_resident(
    module: &Module,
    stream: &Stream,
    name: &str,
    w: &ByteBuffer,
    x: &Buffer,
    y: &Buffer,
    n: usize,
    k: usize,
    r: usize,
) -> Result<(), String> {
    if r == 0 || r > 16 {
        return Err(format!("{name}: r={r} must be in 1..=16"));
    }
    let spec = crate::spec(name).ok_or_else(|| format!("{name} is not in the manifest"))?;
    let (rows, block) = match spec.kind {
        crate::KernelKind::RowTile { rows, block } => (rows, block),
        _ => return Err(format!("{name}: not a skinny GEMM kernel")),
    };
    if k % 32 != 0 {
        return Err(format!("{name}: k={k} must be a multiple of 32"));
    }
    let kernel = module.kernel(name)?;
    let mut p_w = w.as_ptr();
    let mut p_x = x.as_ptr();
    let mut p_y = y.as_ptr();
    let mut v_n = n as i32;
    let mut v_k = k as i32;
    let mut v_r = r as i32;
    let mut params: [*mut c_void; 6] = [
        &mut p_w as *mut _ as *mut c_void,
        &mut p_x as *mut _ as *mut c_void,
        &mut p_y as *mut _ as *mut c_void,
        &mut v_n as *mut _ as *mut c_void,
        &mut v_k as *mut _ as *mut c_void,
        &mut v_r as *mut _ as *mut c_void,
    ];
    kernel.launch(LaunchConfig::row_tile(n, rows, block), stream, &mut params)
}

impl Launcher {
    /// Skinny GEMM with both ends and the weight already on the device.
    #[allow(clippy::too_many_arguments)]
    pub fn run_skinny_resident(
        &self,
        name: &str,
        w: &ByteBuffer,
        x: &Buffer,
        y: &Buffer,
        n: usize,
        k: usize,
        r: usize,
    ) -> Result<(), String> {
        skinny_resident(&self.module, &self.stream, name, w, x, y, n, k, r)
    }

    /// Time `reps` launches over resident buffers (no upload, no download).
    /// The bench uses this so a 75 MB weight is uploaded once per shape.
    #[allow(clippy::too_many_arguments)]
    pub fn run_skinny_resident_reps(
        &self,
        name: &str,
        w: &ByteBuffer,
        x: &Buffer,
        y: &Buffer,
        n: usize,
        k: usize,
        r: usize,
        reps: usize,
    ) -> Result<f64, String> {
        assert!(reps > 0, "reps must be positive");
        let start = std::time::Instant::now();
        for _ in 0..reps {
            skinny_resident(&self.module, &self.stream, name, w, x, y, n, k, r)?;
        }
        self.stream.synchronize()?;
        Ok(start.elapsed().as_secs_f64())
    }

    /// Upload `w`/`x`, launch `reps` times, return `y` and the launch-loop
    /// seconds (uploads and the one download outside the timer).
    #[allow(clippy::too_many_arguments)]
    pub fn run_skinny_reps(        &self,
        name: &str,
        w: &[u8],
        x: &[f32],
        n: usize,
        k: usize,
        r: usize,
        reps: usize,
    ) -> Result<(Vec<f32>, f64), String> {
        self.skinny_launch(name, w, x, n, k, r, reps, true)
    }

    /// Same launch without the download: the bench's per-shape timing.
    #[allow(clippy::too_many_arguments)]
    pub fn run_skinny_time(
        &self,
        name: &str,
        w: &[u8],
        x: &[f32],
        n: usize,
        k: usize,
        r: usize,
        reps: usize,
    ) -> Result<f64, String> {
        self.skinny_launch(name, w, x, n, k, r, reps, false)
            .map(|(_, secs)| secs)
    }

    #[allow(clippy::too_many_arguments)]
    fn skinny_launch(        &self,
        name: &str,
        w: &[u8],
        x: &[f32],
        n: usize,
        k: usize,
        r: usize,
        reps: usize,
        download: bool,
    ) -> Result<(Vec<f32>, f64), String> {
        assert!(reps > 0, "reps must be positive");
        if x.len() != r * k {
            return Err(format!("x must be [{r}, {k}] = {} floats", r * k));
        }
        let wb = ByteBuffer::from_host(w)?;
        let xb = Buffer::from_host(x)?;
        let y = Buffer::alloc(r * n)?;
        let start = std::time::Instant::now();
        for _ in 0..reps {
            skinny_resident(&self.module, &self.stream, name, &wb, &xb, &y, n, k, r)?;
        }
        self.stream.synchronize()?;
        let secs = start.elapsed().as_secs_f64();
        Ok((if download { y.to_host()? } else { Vec::new() }, secs))
    }
}
