//! Quantized-KV decode attention bench: depth x dtype x (1 head | 24 heads x 16
//! layers), at the Qwen3.8-27B geometry.
//!
//! This cannot go through the parity `OpCase` table: the cache is a byte buffer
//! rather than an f32 tensor, the largest case is 3.5 GB, and the interesting
//! dimension is `seq` from 1K to 100K. It reuses the same launch function the
//! parity arm uses, so the ABI is still written once.
//!
//! Cache bytes are synthetic but format-valid (see `kv::synthetic_blocks`): every
//! block has a real f16 scale and a payload, so the kernel's dequantize path runs
//! at full cost. Correctness at 100K is `hipml parity --deep`, not this.

use std::time::Instant;

use hipml_core::kv::{self, KvDtype};
use hipml_core::parity::bench_cases;
use hipml_core::rng::Rng;
use hipml_hip::{Buffer, Device, Module, Stream};

use crate::launch_kv::launch;

/// Qwen3.8-27B Q4_K_M weight bytes: the other term in a decode step.
pub const WEIGHTS_BYTES: f64 = 17.19e9;

/// One line of the report.
pub struct Row {
    pub dtype: KvDtype,
    /// Kernel entry that produced the row (`q8`, `q8s`, `q8g`, `q4`).
    pub variant: &'static str,
    pub heads: usize,
    pub layers: usize,
    pub seq: usize,
    pub splits: usize,
    /// Microseconds for one full pass over `layers` cache copies.
    pub us: f64,
    /// K+V bytes read in one pass over all layers.
    pub bytes: usize,
    pub gbs: f64,
}

pub struct Report {
    pub device: String,
    pub rows: Vec<Row>,
    /// Measured gemv bandwidth and the weight-stream time it implies.
    pub weights_gbs: f64,
    pub weights_us: f64,
    pub vram_total: usize,
    pub vram_free_before: usize,
    pub vram_free_after: usize,
}

/// `depths` in tokens; `reps` is the minimum number of timed passes.
/// `splits_override` forces a split count (the full-stack path clamps to 64).
pub fn run(depths: &[usize], reps_min: usize, splits_override: Option<usize>) -> Result<Report, String> {
    let head_dim = 256usize;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let dir = crate::kernel_dir()
        .ok_or_else(|| "kernels were not compiled: set HIP_PATH and rebuild".to_string())?;
    let device = Device::first()?;
    let module = Module::load(&dir.join("hipml.co"))?;
    let stream = Stream::new()?;
    let (free0, total) = device.mem_info()?;

    // Measure the weight stream first: a second of 838 MB gemv also brings the
    // card to its sustained clock, so the short 1-head cases that follow are not
    // measured at a ramping clock (it showed as 8K reading slower than 32K).
    let (weights_gbs, weights_us) = gemv_bandwidth(&module, &stream)?;

    let mut rows = Vec::new();
    for (dtype, variant, entry) in [
        (KvDtype::Q8, "q8", 0usize),
        (KvDtype::Q8, "q8s", 2),
        (KvDtype::Q8, "q8g", 3),
        (KvDtype::Q4, "q4", 1),
    ] {
        let row_bytes = dtype.row_bytes(head_dim);
        for (heads, n_kv, layers) in [(1usize, 1usize, 1usize), (24, 4, 16)] {
            for &seq in depths {
                // Enough blocks to fill the card, but at least 64 rows per split
                // so the fp32 partials stay small against the cache read.
                let splits = splits_override
                    .unwrap_or_else(|| (seq / 64).clamp(1, (256 / n_kv).max(1)));
                let layer_bytes = n_kv * seq * row_bytes;
                let k = Buffer::alloc((layer_bytes * layers).div_ceil(4))?;
                let v = Buffer::alloc((layer_bytes * layers).div_ceil(4))?;
                fill(&k, dtype, layer_bytes * layers, 0x51 + seq as u64)?;
                fill(&v, dtype, layer_bytes * layers, 0x91 + seq as u64)?;

                let mut rng = Rng::new(0x0B1);
                let qhost: Vec<f32> = (0..heads * head_dim).map(|_| rng.normal()).collect();
                let q = Buffer::from_host(&qhost)?;
                let out = Buffer::alloc(heads * head_dim)?;
                let part_acc = Buffer::alloc(heads * splits * head_dim)?;
                let part_ml = Buffer::alloc(heads * splits * 2)?;

                let pass = |n: usize| -> Result<f64, String> {
                    let t0 = Instant::now();
                    for _ in 0..n {
                        for l in 0..layers {
                            launch(
                                &module,
                                &stream,
                                entry,
                                &q,
                                &k,
                                &v,
                                &part_acc,
                                &part_ml,
                                &out,
                                heads,
                                n_kv,
                                head_dim,
                                seq,
                                scale,
                                splits,
                                l * layer_bytes,
                                l * layer_bytes,
                            )?;
                        }
                    }
                    stream.synchronize()?;
                    Ok(t0.elapsed().as_secs_f64())
                };

                pass(2)?; // warm: module resident, caches primed
                let est = pass(1)?;
                let reps = ((0.2 / est.max(1e-9)) as usize).clamp(reps_min, 50);
                let secs = pass(reps)? / reps as f64;
                let bytes = 2 * layer_bytes * layers;
                rows.push(Row {
                    dtype,
                    variant,
                    heads,
                    layers,
                    seq,
                    splits,
                    us: secs * 1e6,
                    bytes,
                    gbs: bytes as f64 / secs / 1e9,
                });
            }
        }
    }

    let (free1, _) = device.mem_info()?;
    Ok(Report {
        device: device.name().to_string(),
        rows,
        weights_gbs,
        weights_us,
        vram_total: total,
        vram_free_before: free0,
        vram_free_after: free1,
    })
}

/// Upload `bytes` of format-valid blocks from a 16 MB repeating pattern, so the
/// host never holds a copy of a 3.5 GB cache.
fn fill(buf: &Buffer, dtype: KvDtype, bytes: usize, seed: u64) -> Result<(), String> {
    const CHUNK: usize = 16 << 20;
    let bb = dtype.block_bytes();
    let pattern = kv::synthetic_blocks(dtype, (CHUNK / bb).max(1), seed);
    let mut off = 0;
    while off < bytes {
        let n = (bytes - off).min(pattern.len());
        buf.upload_bytes(off, &pattern[..n])?;
        off += n;
    }
    Ok(())
}

/// The 838 MB gemv bench case, measured the same way `hipml bench` measures it:
/// the difference between one repetition and many is the launch, not the copy.
fn gemv_bandwidth(module: &Module, stream: &Stream) -> Result<(f64, f64), String> {
    let case = bench_cases()
        .into_iter()
        .find(|c| c.name == "gemv_f32")
        .ok_or_else(|| "no gemv bench case".to_string())?;
    let (inputs, _) = case.reference();
    let (m, k) = (case.scalar("m") as usize, case.scalar("k") as usize);

    let w = Buffer::from_host(inputs[0].data())?;
    let x = Buffer::from_host(inputs[1].data())?;
    let y = Buffer::alloc(m)?;
    let kernel = module.kernel("gemv_f32")?;
    let mut p_w = w.as_ptr();
    let mut p_x = x.as_ptr();
    let mut p_y = y.as_ptr();
    let mut v_m = m as i32;
    let mut v_k = k as i32;
    let mut params: [*mut std::ffi::c_void; 5] = [
        &mut p_w as *mut _ as *mut std::ffi::c_void,
        &mut p_x as *mut _ as *mut std::ffi::c_void,
        &mut p_y as *mut _ as *mut std::ffi::c_void,
        &mut v_m as *mut _ as *mut std::ffi::c_void,
        &mut v_k as *mut _ as *mut std::ffi::c_void,
    ];
    let cfg = hipml_hip::LaunchConfig::per_row(m, 256);
    let mut time = |reps: usize| -> Result<f64, String> {
        let t0 = Instant::now();
        for _ in 0..reps {
            kernel.launch(cfg, stream, &mut params)?;
        }
        stream.synchronize()?;
        Ok(t0.elapsed().as_secs_f64())
    };
    time(2)?;
    let t1 = time(1)?;
    let tn = time(51)?;
    let launch_s = (tn - t1) / 50.0;
    let bytes = (m * k + k + m) * 4;
    let gbs = bytes as f64 / launch_s / 1e9;
    Ok((gbs, WEIGHTS_BYTES / (gbs * 1e9) * 1e6))
}
