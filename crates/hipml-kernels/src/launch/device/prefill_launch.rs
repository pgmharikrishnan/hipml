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
        // The engine hands these GEMMs an fp16 activation tile. Bench the same
        // path when the v2 shape has a converted-activation entry, so the number
        // is the one the prefill graph actually pays; the one-off conversion is
        // outside the timed loop because the graph amortizes it over the
        // projections that share the tile.
        let half = prefill_entry_for_half(name);
        let name = half.unwrap_or(name);
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
        let xhalf = match half {
            Some(_) => {
                let b = Buffer::alloc((tokens * k + 1) / 2)?;
                prefill_xhalf(&self.module, &self.stream, &xb, &b, tokens * k)?;
                Some(b)
            }
            None => None,
        };
        let y = Buffer::alloc(tokens * m)?;
        let kernel = self.module.kernel(name)?;

        let mut p_w = wb.as_ptr();
        let mut p_x = xhalf.as_ref().map_or(xb.as_ptr(), |b| b.as_ptr());
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

/// Convert a `[tokens, k]` f32 activation tile to fp16, once per projection.
///
/// The tiled GEMM stages the activation tile once per output row tile, which is
/// the widest stream in the ffn shapes (`tokens*k*4 * m/ROW_TILE` bytes). A
/// single fp16 copy halves it; the GEMM's staging does the same
/// `floats2half2_rn` conversion either way, so the result is bit-identical.
pub fn prefill_xhalf(
    module: &Module,
    stream: &Stream,
    x: &Buffer,
    xh: &Buffer,
    n: usize,
) -> Result<(), String> {
    let kernel = module.kernel("prefill_xhalf")?;
    let mut p_x = x.as_ptr();
    let mut p_h = xh.as_ptr();
    let mut v_n = n as i32;
    let mut params: [*mut c_void; 3] = [
        &mut p_x as *mut _ as *mut c_void,
        &mut p_h as *mut _ as *mut c_void,
        &mut v_n as *mut _ as *mut c_void,
    ];
    kernel.launch(LaunchConfig::linear(n, 256), stream, &mut params)
}

/// The fp16-activation variant of the entry the active path would launch.
///
/// `None` means the path has no converted-activation kernel (f32 or fp16 dot2),
/// so callers must keep passing the f32 tile.
pub fn prefill_entry_for_half(name: &str) -> Option<&'static str> {
    let name = crate::prefill_entry_for(name).unwrap_or(name);
    Some(match name {
        "prefill_q4_k_w2" => "prefill_q4_k_w2h",
        "prefill_q6_k_w2" => "prefill_q6_k_w2h",
        "prefill_q8_0_w2" => "prefill_q8_0_w2h",
        "prefill_halfw_w2" => "prefill_halfw_w2h",
        _ => return None,
    })
}

/// Whether the active path stages a converted fp16 activation tile. Lets a
/// caller skip the `prefill_xhalf` pass when there is nothing to read it.
pub fn prefill_has_xhalf() -> bool {
    prefill_entry_for_half("prefill_q4_k").is_some()
}

/// Tiled prefill GEMM reading an fp16 copy of the activation tile that the
/// caller made with [`prefill_xhalf`]. Falls through to the normal entry when
/// the path has no converted-activation kernel.
#[allow(clippy::too_many_arguments)]
pub fn prefill_resident_half(
    module: &Module,
    stream: &Stream,
    name: &str,
    w: &ByteBuffer,
    xh: &Buffer,
    y: &Buffer,
    m: usize,
    k: usize,
    tokens: usize,
) -> Result<(), String> {
    let name = prefill_entry_for_half(name).unwrap_or(name);
    prefill_resident(module, stream, name, w, xh, y, m, k, tokens)
}

/// Split-K entry for a converted-activation kernel, when the shape is
/// wave-starved. `None` means the shape runs the single-pass kernel.
fn split_entry(name: &str) -> Option<&'static str> {
    Some(match name {
        "prefill_q4_k_w2h" => "prefill_q4_k_w2h_sk",
        "prefill_q6_k_w2h" => "prefill_q6_k_w2h_sk",
        "prefill_q8_0_w2h" => "prefill_q8_0_w2h_sk",
        "prefill_halfw_w2h" => "prefill_halfw_w2h_sk",
        _ => return None,
    })
}

/// Whether a shape should split its k range in two. The v2 tile is 128 rows x
/// 256 tokens, so `blocks` tiles are `blocks / 64` waves on this card. Below
/// three waves the matrix units idle at the end of every chunk; at 512 tokens
/// `ffn_gate` (5120x17408) is 80 blocks = 2.5 waves, while the 2048-token
/// chunks used by 8K and deeper prompts are 320 blocks and stay single-pass.
/// The lower bound keeps tiny projections (k/v, 16 blocks) on one launch: two
/// halves of 16 blocks would still be under one wave and only add a pass.
pub fn prefill_should_split(m: usize, k: usize, tokens: usize) -> bool {
    if std::env::var_os("HIPML_SPLITK").map_or(false, |v| v == "0") {
        return false;
    }
    // Both halves must end on a 32-value k-tile boundary and the Q4_K/Q6_K
    // block format needs 256 values per block; 64 is the common multiple.
    if k < 4096 || k % 64 != 0 {
        return false;
    }
    let blocks = m.div_ceil(128) * tokens.div_ceil(256);
    blocks >= 32 && blocks < 3 * 64
}

/// Split-K launch: one grid with `blockIdx.z` selecting the k half, so both
/// halves run concurrently instead of as two serialized launches. Slice 0
/// writes `y`, slice 1 writes `scratch`, then `residual_add_f32` sums them.
/// The split point and the addition order are fixed, so the result is
/// deterministic (but not bit-identical to the single-pass accumulation, whose
/// k-tile sum is one continuous register chain).
#[allow(clippy::too_many_arguments)]
fn prefill_split(
    module: &Module,
    stream: &Stream,
    name: &str,
    w: &ByteBuffer,
    x: &Buffer,
    y: &Buffer,
    scratch: &Buffer,
    m: usize,
    k: usize,
    tokens: usize,
) -> Result<(), String> {
    let spec = crate::spec(name).ok_or_else(|| format!("{name} is not in the manifest"))?;
    let (rpb, tpb, block) = match spec.kind {
        crate::KernelKind::Prefill {
            rows,
            tokens,
            block,
        } => (rows, tokens, block),
        _ => return Err(format!("{name}: not a prefill kernel")),
    };
    // A 256-value split point keeps every k-tile of both halves whole.
    let ks = (k / 2) & !255usize;
    if ks == 0 || ks >= k {
        return Err(format!("{name}: k={k} cannot be split"));
    }
    if scratch.len() < tokens * m {
        return Err(format!(
            "{name}: split scratch has {} values, needs {}",
            scratch.len(),
            tokens * m
        ));
    }
    let kernel = module.kernel(name)?;
    let mut cfg = LaunchConfig::prefill(m, tokens, rpb, tpb, block);
    cfg.grid.2 = 2;
    let mut p_w = w.as_ptr();
    let mut p_x = x.as_ptr();
    let mut p_y = y.as_ptr();
    let mut p_s = scratch.as_ptr();
    let mut v_m = m as i32;
    let mut v_k = k as i32;
    let mut v_tokens = tokens as i32;
    let mut v_ks = ks as i32;
    let mut params: [*mut c_void; 8] = [
        &mut p_w as *mut _ as *mut c_void,
        &mut p_x as *mut _ as *mut c_void,
        &mut p_y as *mut _ as *mut c_void,
        &mut p_s as *mut _ as *mut c_void,
        &mut v_m as *mut _ as *mut c_void,
        &mut v_k as *mut _ as *mut c_void,
        &mut v_tokens as *mut _ as *mut c_void,
        &mut v_ks as *mut _ as *mut c_void,
    ];
    kernel.launch(cfg, stream, &mut params)?;

    // y = y + scratch, the deterministic two-partial sum.
    let reduce = module.kernel("residual_add_f32")?;
    let mut p_a = y.as_ptr();
    let mut p_b = scratch.as_ptr();
    let mut p_o = y.as_ptr();
    let mut v_n = (tokens * m) as i32;
    let mut rparams: [*mut c_void; 4] = [
        &mut p_a as *mut _ as *mut c_void,
        &mut p_b as *mut _ as *mut c_void,
        &mut p_o as *mut _ as *mut c_void,
        &mut v_n as *mut _ as *mut c_void,
    ];
    reduce.launch(LaunchConfig::linear(tokens * m, 256), stream, &mut rparams)
}

/// Split-K variant of [`prefill_resident_half`], for a wave-starved ffn
/// projection. `scratch` must hold `tokens * m` f32.
#[allow(clippy::too_many_arguments)]
pub fn prefill_resident_split_half(
    module: &Module,
    stream: &Stream,
    name: &str,
    w: &ByteBuffer,
    xh: &Buffer,
    y: &Buffer,
    scratch: &Buffer,
    m: usize,
    k: usize,
    tokens: usize,
) -> Result<(), String> {
    let name = prefill_entry_for_half(name).unwrap_or(name);
    let sk = split_entry(name).ok_or_else(|| format!("{name}: no split-K entry"))?;
    prefill_split(module, stream, sk, w, xh, y, scratch, m, k, tokens)
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
