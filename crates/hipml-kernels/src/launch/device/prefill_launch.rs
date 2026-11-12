// SPDX-License-Identifier: Apache-2.0
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
        // Follow the graph's own split-K decision. Parity used to exercise the
        // single-pass launch only, so a split-K defect had no gate at all: the
        // 17408x5120 case here is exactly a shape the engine splits.
        if let Some(half_name) = half {
            if prefill_should_split(m, k, tokens) {
                let scratch = Buffer::alloc(tokens * m)?;
                let entry = split_entry(half_name)
                    .ok_or_else(|| format!("{half_name}: no split-K entry"))?;
                let start = std::time::Instant::now();
                // `reps` must be honoured here too: callers divide the elapsed
                // time by it, so a single launch at reps=5 reported 5x the
                // TFLOPS. The split path is two launches (the GEMM pair and the
                // residual add), and both repeat.
                for _ in 0..reps {
                    prefill_split(
                        &self.module,
                        &self.stream,
                        entry,
                        &wb,
                        xhalf.as_ref().expect("half entry implies a converted tile"),
                        &y,
                        &scratch,
                        m,
                        k,
                        tokens,
                    )?;
                }
                self.stream.synchronize()?;
                let secs = start.elapsed().as_secs_f64();
                return Ok((if download { y.to_host()? } else { Vec::new() }, secs));
            }
        }
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

impl Launcher {
    /// Launch the fused gate+up pass `reps` times over one upload and return the
    /// activation tile, its fp16 copy and the time inside the launch loop.
    ///
    /// This exists for parity: the fused epilogue has to produce the same logits as
    /// the two launches it replaces, so the case compares both outputs of one fused
    /// launch against the host reference of the two GEMMs plus `silu_mul_f32`.
    #[allow(clippy::too_many_arguments)]
    pub fn run_prefill_gu_reps(
        &self,
        gate: &str,
        up: &str,
        wg: &[u8],
        wu: &[u8],
        x: &[f32],
        m: usize,
        k: usize,
        tokens: usize,
        reps: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, f64), String> {
        assert!(reps > 0, "reps must be positive");
        if k % 64 != 0 {
            return Err(format!("k={k} must be a multiple of the 64-value k-tile"));
        }
        let wgb = ByteBuffer::from_host(wg)?;
        let wub = ByteBuffer::from_host(wu)?;
        let xb = Buffer::from_host(x)?;
        let xh = Buffer::alloc((tokens * k + 1) / 2)?;
        prefill_xhalf(&self.module, &self.stream, &xb, &xh, tokens * k)?;
        let act = Buffer::alloc(tokens * m)?;
        let acth = Buffer::alloc((tokens * m + 1) / 2)?;
        let start = std::time::Instant::now();
        for _ in 0..reps {
            prefill_resident_gu(
                &self.module,
                &self.stream,
                gate,
                up,
                &wgb,
                &wub,
                &xh,
                &act,
                &acth,
                m,
                k,
                tokens,
            )?;
        }
        self.stream.synchronize()?;
        let secs = start.elapsed().as_secs_f64();
        // The fp16 copy comes back packed two halves per f32 word: value `2i` in
        // the low half of word `i`, `2i + 1` in the high half.
        let halves: Vec<f32> = acth
            .to_host()?
            .iter()
            .flat_map(|&w| {
                let bits = w.to_bits();
                [
                    hipml_core::quant::f16_to_f32((bits & 0xFFFF) as u16),
                    hipml_core::quant::f16_to_f32((bits >> 16) as u16),
                ]
            })
            .take(tokens * m)
            .collect();
        Ok((act.to_host()?, halves, secs))
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
    // `linear(n, block)` derives the grid from a *thread count*: it launches
    // `ceil(n / block)` blocks. The kernel wants one warp per output element,
    // so `ceil(tokens * m / warps)` blocks of `warps * 32` threads. Passing the
    // block count itself under-launched by `warps * 32` and left every alpha and
    // beta row past the first tile unwritten (the WP42 defect: 40 of 48 rows
    // zero on a 1-token chunk, so the GDN gates ran on zeros).
    let (blocks, threads) = crate::launch::gemm_rows_grid(tokens, m);
    let cfg = LaunchConfig {
        grid: (blocks as u32, 1, 1),
        block: (threads as u32, 1, 1),
        shared_mem_bytes: 0,
    };
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
        // The IQ4 family has no `_w2` shape of its own: the base entry is the
        // 64-row f32-tile kernel and `_w2h` is the 128-row fp16-tile one, so
        // the mapping only exists on the v2 wmma path that stages `xh`.
        "prefill_iq4_xs"
            if crate::prefill_path() == crate::PrefillPath::Wmma && crate::prefill_v2() =>
        {
            "prefill_iq4_xs_w2h"
        }
        "prefill_iq4_nl"
            if crate::prefill_path() == crate::PrefillPath::Wmma && crate::prefill_v2() =>
        {
            "prefill_iq4_nl_w2h"
        }
        "prefill_q3_k"
            if crate::prefill_path() == crate::PrefillPath::Wmma && crate::prefill_v2() =>
        {
            "prefill_q3_k_w2h"
        }
        "prefill_iq3_s"
            if crate::prefill_path() == crate::PrefillPath::Wmma && crate::prefill_v2() =>
        {
            "prefill_iq3_s_w2h"
        }
        "prefill_q4_k_w2" => "prefill_q4_k_w2h",
        "prefill_q5_k_w2" => "prefill_q5_k_w2h",
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

/// The SwiGLU-epilogue variant of a converted-activation entry: same tile and
/// same decoder, and its store loop reads the already-written gate tile and
/// writes both the f32 activation and its fp16 copy. `None` for an entry that
/// cannot carry it.
fn silu_entry(name: &str) -> Option<&'static str> {
    Some(match name {
        "prefill_q4_k_w2h" => "prefill_q4_k_w2h_silu",
        "prefill_q5_k_w2h" => "prefill_q5_k_w2h_silu",
        "prefill_q6_k_w2h" => "prefill_q6_k_w2h_silu",
        "prefill_q8_0_w2h" => "prefill_q8_0_w2h_silu",
        "prefill_iq4_xs_w2h" => "prefill_iq4_xs_w2h_silu",
        "prefill_iq4_nl_w2h" => "prefill_iq4_nl_w2h_silu",
        "prefill_q3_k_w2h" => "prefill_q3_k_w2h_silu",
        "prefill_iq3_s_w2h" => "prefill_iq3_s_w2h_silu",
        _ => return None,
    })
}

/// Whether `name` (a `_w2h` entry) has a fused-swiGLU launch.
pub fn prefill_has_silu(name: &str) -> bool {
    silu_entry(name).is_some()
}

/// Tiled prefill GEMM whose epilogue applies the ffn's swiGLU:
/// `y = silu(gate) * (W @ x)` and `yh` the fp16 copy of it.
///
/// The graph used to run three launches between the `up` and `down`
/// projections (`silu_mul`, then the fp16 copy of its output, then `down`).
/// Doing it in the `up` epilogue reads the gate tile and writes the two outputs
/// in the launch that already owns those rows, which removes a 1.7 GB
/// read-pass pair per layer-chunk. The arithmetic is the elementwise kernel's
/// expression verbatim, so the logits are unchanged.
#[allow(clippy::too_many_arguments)]
pub fn prefill_resident_silu(
    module: &Module,
    stream: &Stream,
    name: &str,
    w: &ByteBuffer,
    xh: &Buffer,
    gate: &Buffer,
    y: &Buffer,
    yh: &Buffer,
    m: usize,
    k: usize,
    tokens: usize,
) -> Result<(), String> {
    let entry = silu_entry(name).ok_or_else(|| format!("{name}: no swiGLU epilogue"))?;
    let spec = crate::spec(name).ok_or_else(|| format!("{name} is not in the manifest"))?;
    let (rpb, tpb, block) = match spec.kind {
        crate::KernelKind::Prefill {
            rows,
            tokens,
            block,
        } => (rows, tokens, block),
        _ => return Err(format!("{name}: not a prefill kernel")),
    };
    let kernel = module.kernel(entry)?;
    let mut p_w = w.as_ptr();
    let mut p_x = xh.as_ptr();
    let mut p_y = y.as_ptr();
    let mut p_g = gate.as_ptr();
    let mut p_h = yh.as_ptr();
    let mut v_m = m as i32;
    let mut v_k = k as i32;
    let mut v_tokens = tokens as i32;
    let mut params: [*mut c_void; 8] = [
        &mut p_w as *mut _ as *mut c_void,
        &mut p_x as *mut _ as *mut c_void,
        &mut p_y as *mut _ as *mut c_void,
        &mut p_g as *mut _ as *mut c_void,
        &mut p_h as *mut _ as *mut c_void,
        &mut v_m as *mut _ as *mut c_void,
        &mut v_k as *mut _ as *mut c_void,
        &mut v_tokens as *mut _ as *mut c_void,
    ];
    kernel.launch(
        LaunchConfig::prefill(m, tokens, rpb, tpb, block),
        stream,
        &mut params,
    )
}

/// Format tag of a converted-activation entry, as `prefill_fuse.inc` decodes it.
fn fuse_kind(name: &str) -> Option<i32> {
    Some(match name {
        "prefill_q4_k_w2h" => 0,
        "prefill_q6_k_w2h" => 1,
        "prefill_q8_0_w2h" => 2,
        "prefill_halfw_w2h" => 3,
        "prefill_q5_k_w2h" => 4,
        "prefill_iq4_xs_w2h" => 5,
        "prefill_iq4_nl_w2h" => 6,
        "prefill_q3_k_w2h" => 7,
        "prefill_iq3_s_w2h" => 8,
        _ => return None,
    })
}

/// The fused gate+up entry for a gate/up format pair, or `None` when the pair
/// has no instantiation (the caller then runs the two separate launches).
///
/// The pair is a template argument because a runtime format tag in the decoder
/// spills (317 slots); the instantiated list is the pairs the shipped UD mixes
/// use. A new mix only loses the fusion, never correctness.
pub fn prefill_gu_entry(gate: &str, up: &str) -> Option<&'static str> {
    let g = fuse_kind(prefill_entry_for_half(gate).unwrap_or(gate))?;
    let u = fuse_kind(prefill_entry_for_half(up).unwrap_or(up))?;
    Some(match (g, u) {
        (0, 0) => "prefill_gu_q4_k_q4_k",
        (1, 1) => "prefill_gu_q6_k_q6_k",
        (2, 2) => "prefill_gu_q8_0_q8_0",
        (3, 3) => "prefill_gu_halfw_halfw",
        (4, 4) => "prefill_gu_q5_k_q5_k",
        (5, 5) => "prefill_gu_iq4_xs_iq4_xs",
        (6, 6) => "prefill_gu_iq4_nl_iq4_nl",
        (7, 7) => "prefill_gu_q3_k_q3_k",
        (8, 8) => "prefill_gu_iq3_s_iq3_s",
        (0, 4) => "prefill_gu_q4_k_q5_k",
        (0, 5) => "prefill_gu_q4_k_iq4_xs",
        (0, 1) => "prefill_gu_q4_k_q6_k",
        (4, 0) => "prefill_gu_q5_k_q4_k",
        (4, 1) => "prefill_gu_q5_k_q6_k",
        (5, 0) => "prefill_gu_iq4_xs_q4_k",
        (5, 4) => "prefill_gu_iq4_xs_q5_k",
        (5, 7) => "prefill_gu_iq4_xs_q3_k",
        (5, 8) => "prefill_gu_iq4_xs_iq3_s",
        (6, 4) => "prefill_gu_iq4_nl_q5_k",
        (6, 5) => "prefill_gu_iq4_nl_iq4_xs",
        (7, 5) => "prefill_gu_q3_k_iq4_xs",
        (1, 2) => "prefill_gu_q6_k_q8_0",
        (1, 4) => "prefill_gu_q6_k_q5_k",
        _ => return None,
    })
}

/// Whether a gate/up pair can run as one fused launch.
pub fn prefill_has_gu(gate: &str, up: &str) -> bool {
    prefill_gu_entry(gate, up).is_some()
}

/// Fused ffn `gate`+`up`: one launch reads the fp16 activation tile once, streams
/// both weight matrices, and applies the SwiGLU in its epilogue, so the gate tile
/// is never written.
///
/// The tile is 256 tokens x 64 rows per matrix (256 threads), i.e. one shipped
/// block's accumulator budget shared by the pair; `prefill_gu_w2h` carries the
/// geometry.
#[allow(clippy::too_many_arguments)]
pub fn prefill_resident_gu(
    module: &Module,
    stream: &Stream,
    gate: &str,
    up: &str,
    wg: &ByteBuffer,
    wu: &ByteBuffer,
    xh: &Buffer,
    act: &Buffer,
    acth: &Buffer,
    m: usize,
    k: usize,
    tokens: usize,
) -> Result<(), String> {
    let entry =
        prefill_gu_entry(gate, up).ok_or_else(|| format!("{gate}+{up}: no fused gate/up entry"))?;
    let spec = crate::spec("prefill_gu_w2h").ok_or_else(|| "prefill_gu_w2h missing".to_string())?;
    let (rpb, tpb, block) = match spec.kind {
        crate::KernelKind::Prefill {
            rows,
            tokens,
            block,
        } => (rows, tokens, block),
        _ => return Err("prefill_gu_w2h: not a prefill kernel".to_string()),
    };
    let kernel = module.kernel(entry)?;
    let mut p_wg = wg.as_ptr();
    let mut p_wu = wu.as_ptr();
    let mut p_x = xh.as_ptr();
    let mut p_a = act.as_ptr();
    let mut p_h = acth.as_ptr();
    let mut v_m = m as i32;
    let mut v_k = k as i32;
    let mut v_tokens = tokens as i32;
    let mut params: [*mut c_void; 8] = [
        &mut p_wg as *mut _ as *mut c_void,
        &mut p_wu as *mut _ as *mut c_void,
        &mut p_x as *mut _ as *mut c_void,
        &mut p_a as *mut _ as *mut c_void,
        &mut p_h as *mut _ as *mut c_void,
        &mut v_m as *mut _ as *mut c_void,
        &mut v_k as *mut _ as *mut c_void,
        &mut v_tokens as *mut _ as *mut c_void,
    ];
    kernel.launch(
        LaunchConfig::prefill(m, tokens, rpb, tpb, block),
        stream,
        &mut params,
    )
}

/// Split-K entry for a converted-activation kernel, when the shape is
/// wave-starved. `None` means the shape runs the single-pass kernel.
fn split_entry(name: &str) -> Option<&'static str> {
    Some(match name {
        "prefill_q4_k_w2h" => "prefill_q4_k_w2h_sk",
        "prefill_q5_k_w2h" => "prefill_q5_k_w2h_sk",
        "prefill_q6_k_w2h" => "prefill_q6_k_w2h_sk",
        "prefill_q8_0_w2h" => "prefill_q8_0_w2h_sk",
        "prefill_halfw_w2h" => "prefill_halfw_w2h_sk",
        "prefill_iq4_xs_w2h" => "prefill_iq4_xs_w2h_sk",
        "prefill_iq4_nl_w2h" => "prefill_iq4_nl_w2h_sk",
        "prefill_q3_k_w2h" => "prefill_q3_k_w2h_sk",
        "prefill_iq3_s_w2h" => "prefill_iq3_s_w2h_sk",
        _ => return None,
    })
}

/// n-slice entry for a converted-activation kernel. Only the raw-fp16 weights
/// have one: the SSM alpha/beta projections are the shape that needs it.
fn slice_entry(name: &str) -> Option<&'static str> {
    Some(match name {
        "prefill_halfw_w2h" => "prefill_halfw_w2h_sn",
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

/// Largest split-K scratch (f32 values) that a projection of at most
/// `max_rows` rows can need for any tile of at most `max_tokens` tokens.
///
/// This is the allocation side of [`prefill_should_split`]. A tile of `t`
/// tokens can select the split only while
/// `ceil(m / 128) * ceil(t / 256) < 3 * 64`, so the widest `m` it can select at
/// that length is `128 * ((3 * 64 - 1) / ceil(t / 256))`. Taking the maximum of
/// `t * m` over every tile length gives the one size at which the launch check
/// in [`prefill_split`] (`scratch.len() < tokens * m`) can never fail.
///
/// Sizing by the chunk instead is what crashed an 8,193-token prompt: the last
/// tile was one token, the predicate split the ffn shapes, and the buffer had
/// been allocated as a single value because the chunk was 2,048.
pub fn prefill_split_scratch_cap(max_tokens: usize, max_rows: usize) -> usize {
    let mut need = 1;
    for tokens in 1..=max_tokens.max(1) {
        let tiles = tokens.div_ceil(256);
        // `m.div_ceil(128) <= (3 * 64 - 1) / tiles` is the widest shape that
        // still satisfies `blocks < 3 * 64`.
        let rows = max_rows.min(128 * ((3 * 64 - 1) / tiles));
        need = need.max(tokens * rows);
    }
    need
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

/// n-slice k split of the fp16-activation prefill GEMM, for a shape whose row
/// tile is only a few blocks (the 64-row SSM alpha/beta projections).
///
/// Slice 0 writes `y`, slices 1..n write `scratch` at `(z - 1) * tokens * m`,
/// and one `reduce_slices_f32` launch sums them in slice order, so the result
/// is deterministic. `n == 1` falls through to the single-pass kernel.
#[allow(clippy::too_many_arguments)]
pub fn prefill_resident_slices(
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
    let n = crate::prefill_slices(m, k, tokens);
    let sn = match slice_entry(name) {
        Some(sn) if n > 1 => sn,
        _ => return prefill_resident(module, stream, name, w, xh, y, m, k, tokens),
    };
    let spec = crate::spec(sn).ok_or_else(|| format!("{sn} is not in the manifest"))?;
    let (rpb, tpb, block) = match spec.kind {
        crate::KernelKind::Prefill {
            rows,
            tokens,
            block,
        } => (rows, tokens, block),
        _ => return Err(format!("{sn}: not a prefill kernel")),
    };
    let ks = k / n;
    if ks % 64 != 0 || ks == 0 {
        return prefill_resident(module, stream, name, w, xh, y, m, k, tokens);
    }
    if scratch.len() < (n - 1) * tokens * m {
        return Err(format!(
            "{sn}: slice scratch has {} values, needs {}",
            scratch.len(),
            (n - 1) * tokens * m
        ));
    }
    let kernel = module.kernel(sn)?;
    let mut cfg = LaunchConfig::prefill(m, tokens, rpb, tpb, block);
    cfg.grid.2 = n as u32;
    let mut p_w = w.as_ptr();
    let mut p_x = xh.as_ptr();
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

    // y = y + s0 + s1 + ..., in slice order.
    let reduce = module.kernel("reduce_slices_f32")?;
    let mut r_y = y.as_ptr();
    let mut r_s = scratch.as_ptr();
    let mut r_n = (n - 1) as i32;
    let mut r_len = (tokens * m) as i32;
    let mut rparams: [*mut c_void; 4] = [
        &mut r_y as *mut _ as *mut c_void,
        &mut r_s as *mut _ as *mut c_void,
        &mut r_n as *mut _ as *mut c_void,
        &mut r_len as *mut _ as *mut c_void,
    ];
    reduce.launch(LaunchConfig::linear(tokens * m, 256), stream, &mut rparams)
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
        return Err(format!(
            "{name}: k={k} must be a multiple of the 64-value k-tile"
        ));
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Every `(tokens, rows)` pair the predicate can split must fit the
    /// allocation the same predicate produced. This is the invariant the
    /// 8,193-token crash broke.
    #[test]
    fn split_scratch_cap_covers_every_selectable_tile() {
        std::env::remove_var("HIPML_SPLITK");
        let cap = 2048;
        let max_rows = 24_448;
        let scratch = prefill_split_scratch_cap(cap, max_rows);
        for tokens in 1..=cap {
            for rows in (1..=max_rows).step_by(31) {
                if prefill_should_split(rows, 8192, tokens) {
                    assert!(
                        tokens * rows <= scratch,
                        "tokens {tokens} x rows {rows} = {} exceeds cap {scratch}",
                        tokens * rows
                    );
                }
            }
        }
    }
}
