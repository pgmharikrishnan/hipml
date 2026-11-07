//! Tiled prefill GEMM launches.
//!
//! Child module of `device`, so `Launcher`'s private module and stream handles
//! are visible without widening the type's surface.

use super::*;

impl Launcher {
    /// Tiled prefill GEMM with both ends and the weight already on the device:
    /// `y[t, :] = W @ x[t, :]` for `tokens` rows. This is the prefill graph's
    /// projection launch.
    #[allow(clippy::too_many_arguments)]
    pub fn run_prefill_resident(
        &self,
        name: &str,
        w: &ByteBuffer,
        x: &Buffer,
        y: &Buffer,
        m: usize,
        k: usize,
        tokens: usize,
    ) -> Result<(), String> {
        prefill_resident(&self.module, &self.stream, name, w, x, y, m, k, tokens)
    }

    /// Launch a tiled prefill GEMM `reps` times over one upload.
    ///
    /// `x` is `[tokens, k]`, the output is `[tokens, m]`. Returns the output and
    /// the wall time spent inside the launch loop (uploads excluded).
    #[allow(clippy::too_many_arguments)]
    pub fn run_prefill_reps(
        &self,
        name: &str,
        w: &[u8],
        x: &[f32],
        m: usize,
        k: usize,
        tokens: usize,
        reps: usize,
    ) -> Result<(Vec<f32>, f64), String> {
        self.prefill_launch(name, w, x, m, k, tokens, reps, true)
    }

    /// Same launch, but the output is never copied back to the host. Used by the
    /// whole-model pass, where a 35 MB download per 64 MB weight chunk would
    /// dominate wall time and is not part of the kernel measurement.
    #[allow(clippy::too_many_arguments)]
    pub fn run_prefill_time(
        &self,
        name: &str,
        w: &[u8],
        x: &[f32],
        m: usize,
        k: usize,
        tokens: usize,
        reps: usize,
    ) -> Result<f64, String> {
        self.prefill_launch(name, w, x, m, k, tokens, reps, false)
            .map(|(_, secs)| secs)
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill_launch(
        &self,
        name: &str,
        w: &[u8],
        x: &[f32],
        m: usize,
        k: usize,
        tokens: usize,
        reps: usize,
        download: bool,
    ) -> Result<(Vec<f32>, f64), String> {
        assert!(reps > 0, "reps must be positive");
        if k % 64 != 0 {
            return Err(format!(
                "{name}: k={k} must be a multiple of the 64-value k-tile"
            ));
        }
        let spec = crate::spec(name).ok_or_else(|| format!("{name} is not in the manifest"))?;
        let (rpb, tpb, block) = match spec.kind {
            crate::KernelKind::Prefill {
                rows,
                tokens,
                block,
            } => (rows, tokens, block),
            _ => return Err(format!("{name}: not a prefill kernel")),
        };
        let wb = ByteBuffer::from_host(w)?;
        let xb = Buffer::from_host(x)?;
        let y = Buffer::alloc(tokens * m)?;
        let kernel = self.module.kernel(name)?;

        let mut p_w = wb.as_ptr();
        let mut p_x = xb.as_ptr();
        let mut p_y = y.as_ptr();
        let mut v_m = m as i32;
        let mut v_k = k as i32;
        let mut v_tokens = tokens as i32;
        let mut params: [*mut c_void; 6] = [
            &mut p_w as *mut _ as *mut c_void,
            &mut p_x as *mut _ as *mut c_void,
            &mut p_y as *mut _ as *mut c_void,
            &mut v_m as *mut _ as *mut c_void,
            &mut v_k as *mut _ as *mut c_void,
            &mut v_tokens as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig::prefill(m, tokens, rpb, tpb, block);
        let start = std::time::Instant::now();
        for _ in 0..reps {
            kernel.launch(cfg, &self.stream, &mut params)?;
        }
        self.stream.synchronize()?;
        let secs = start.elapsed().as_secs_f64();
        Ok((if download { y.to_host()? } else { Vec::new() }, secs))
    }
}

/// Warp-per-output f32 GEMM for the few wide-and-short projections (the SSM
/// `alpha`/`beta` pair). `gemm_f32`'s row-per-block structure is a decode
/// kernel; on a 512-token chunk it costs more than the weights it reads.
#[allow(clippy::too_many_arguments)]
pub fn gemm_rows_f32(
    module: &Module,
    stream: &Stream,
    w: *mut std::ffi::c_void,
    x: *mut std::ffi::c_void,
    y: *mut std::ffi::c_void,
    tokens: usize,
    m: usize,
    k: usize,
) -> Result<(), String> {
    let kernel = module.kernel("gemm_rows_f32")?;
    let mut p_w = w;
    let mut p_x = x;
    let mut p_y = y;
    let mut v_t = tokens as i32;
    let mut v_m = m as i32;
    let mut v_k = k as i32;
    let mut params: [*mut std::ffi::c_void; 6] = [
        &mut p_w as *mut _ as *mut std::ffi::c_void,
        &mut p_x as *mut _ as *mut std::ffi::c_void,
        &mut p_y as *mut _ as *mut std::ffi::c_void,
        &mut v_t as *mut _ as *mut std::ffi::c_void,
        &mut v_m as *mut _ as *mut std::ffi::c_void,
        &mut v_k as *mut _ as *mut std::ffi::c_void,
    ];
    let warps = 8usize;
    let cfg = LaunchConfig::linear((tokens * m + warps - 1) / warps, (warps * 32) as u32);
    kernel.launch(cfg, stream, &mut params)
}

/// Tiled prefill GEMM over a resident weight payload.
///
/// Free function as well as a `Launcher` method because the gated-delta-net and
/// attention layers hold their own module/stream handles and drive their own
/// projection launches.
#[allow(clippy::too_many_arguments)]
pub fn prefill_resident(
    module: &Module,
    stream: &Stream,
    name: &str,
    w: &ByteBuffer,
    x: &Buffer,
    y: &Buffer,
    m: usize,
    k: usize,
    tokens: usize,
) -> Result<(), String> {
    let name = crate::prefill_entry_for(name).unwrap_or(name);
    let spec = crate::spec(name).ok_or_else(|| format!("{name} is not in the manifest"))?;
    let (rpb, tpb, block) = match spec.kind {
        crate::KernelKind::Prefill {
            rows,
            tokens,
            block,
        } => (rows, tokens, block),
        _ => return Err(format!("{name}: not a prefill kernel")),
    };
    if k % 64 != 0 {
        return Err(format!("{name}: k={k} must be a multiple of the 64-value k-tile"));
    }
    let kernel = module.kernel(name)?;
    let mut p_w = w.as_ptr();
    let mut p_x = x.as_ptr();
    let mut p_y = y.as_ptr();
    let mut v_m = m as i32;
    let mut v_k = k as i32;
    let mut v_tokens = tokens as i32;
    let mut params: [*mut c_void; 6] = [
        &mut p_w as *mut _ as *mut c_void,
        &mut p_x as *mut _ as *mut c_void,
        &mut p_y as *mut _ as *mut c_void,
        &mut v_m as *mut _ as *mut c_void,
        &mut v_k as *mut _ as *mut c_void,
        &mut v_tokens as *mut _ as *mut c_void,
    ];
    let cfg = LaunchConfig::prefill(m, tokens, rpb, tpb, block);
    kernel.launch(cfg, stream, &mut params)
}
