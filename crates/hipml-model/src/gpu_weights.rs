// SPDX-License-Identifier: Apache-2.0
//! Resident weight views for the GPU graph.
//!
//! A `Proj` points at a loader `ByteBuffer` in its GGUF block form and carries
//! the GEMV entry that reads it. Every format the model uses has a kernel that
//! reads its blocks directly: Q4_K, Q6_K, Q8_0, Q5_K and the IQ4 family.
//! Q5_K keeps a half copy as well: the GEMV reads the blocks, but the tiled
//! prefill GEMM needs an fp16 operand and there is no Q5_K dequant-in-shared
//! tile kernel.

use hipml_core::mat::Kind;
use hipml_core::quant::TensorType;
use hipml_hip::{Buffer, ByteBuffer};
use hipml_kernels::Launcher;

use crate::{DeviceTensor, Model};

/// Map a resident tensor's dtype to the GEMV entry that reads its blocks.
pub fn gemv_entry(ty: TensorType) -> Result<&'static str, String> {
    Ok(match ty {
        TensorType::F32 => "gemv_f32",
        TensorType::Q4K => "gemv_q4_k",
        TensorType::Q6K => "gemv_q6_k",
        TensorType::Q8_0 => "gemv_q8_0",
        TensorType::Q5K => "gemv_q5_k",
        TensorType::IQ4XS => "gemv_iq4_xs",
        TensorType::IQ4NL => "gemv_iq4_nl",
        other => return Err(format!("{} has no GEMV kernel", other.name())),
    })
}

/// The batched-GEMV kind for a resident GEMV entry. Only the block formats
/// with a `_b` kernel are batchable; F32 projections are small and stay
/// per-slot.
pub(crate) fn batch_kind(entry: &str) -> Option<i32> {
    use hipml_kernels::{
        GEMV_B_IQ4NL, GEMV_B_IQ4XS, GEMV_B_Q4K, GEMV_B_Q5K, GEMV_B_Q6K, GEMV_B_Q80,
    };
    Some(match entry {
        "gemv_q4_k" => GEMV_B_Q4K,
        "gemv_q6_k" => GEMV_B_Q6K,
        "gemv_q5_k" => GEMV_B_Q5K,
        "gemv_q8_0" => GEMV_B_Q80,
        "gemv_iq4_xs" => GEMV_B_IQ4XS,
        "gemv_iq4_nl" => GEMV_B_IQ4NL,
        _ => return None,
    })
}

/// The batched-GEMV kind for a tensor type, for kernel probes.
pub fn batch_kind_tensor(ty: TensorType) -> Option<i32> {
    batch_kind(gemv_entry(ty).ok()?)
}

/// The tiled GEMM entry for a quantized GEMV entry; the two families share the
/// block format.
fn prefill_entry(gemv: &str) -> Result<&'static str, String> {
    Ok(match gemv {
        "gemv_q4_k" => "prefill_q4_k",
        "gemv_q6_k" => "prefill_q6_k",
        "gemv_q8_0" => "prefill_q8_0",
        "gemv_iq4_xs" => "prefill_iq4_xs",
        "gemv_iq4_nl" => "prefill_iq4_nl",
        // A tensor the loader expanded to F32 (Q3_K / IQ3_S) prefill through
        // the raw-f32 tile; there is no fp16-activation sibling for it.
        "gemv_f32" => "prefill_f32",
        other => return Err(format!("{other}: no tiled prefill kernel")),
    })
}

/// The fp16-activation variant of the same entry (reads a converted copy of the
/// activation tile, half the stream).
fn prefill_entry_half(gemv: &str) -> Result<&'static str, String> {
    Ok(match gemv {
        "gemv_q4_k" => "prefill_q4_k_w2h",
        "gemv_q6_k" => "prefill_q6_k_w2h",
        "gemv_q8_0" => "prefill_q8_0_w2h",
        "gemv_iq4_xs" => "prefill_iq4_xs_w2h",
        "gemv_iq4_nl" => "prefill_iq4_nl_w2h",
        other => return Err(format!("{other}: no fp16-activation prefill kernel")),
    })
}

pub(crate) fn mat_kind(ty: TensorType) -> Result<Kind, String> {
    Ok(match ty {
        TensorType::F32 => Kind::F32,
        TensorType::Q4K => Kind::Q4K,
        TensorType::Q5K => Kind::Q5K,
        TensorType::Q6K => Kind::Q6K,
        TensorType::Q8_0 => Kind::Q8_0,
        TensorType::IQ4XS => Kind::IQ4XS,
        TensorType::IQ4NL => Kind::IQ4NL,
        TensorType::Q3K | TensorType::IQ3S => {
            return Err(format!(
                "{} is expanded to F32 at load and has no block kind",
                ty.name()
            ))
        }
        other => return Err(format!("{} is not a supported mat kind", other.name())),
    })
}

/// A projection that a launcher can stream straight from VRAM: the loader's
/// block payload plus the GEMV entry that reads it.
pub struct Proj<'a> {
    w: &'a ByteBuffer,
    entry: &'static str,
    /// Q5_K only: the fp16 copy the tiled prefill GEMM streams. One per tensor,
    /// 0.12 GB for Qwen3.8-27B's two `ffn_down` tensors; the decode GEMV
    /// still reads the original blocks.
    q5_half: Option<ByteBuffer>,
    pub rows: usize,
    pub cols: usize,
}

impl<'a> Proj<'a> {
    pub(crate) fn of(model: &'a Model, name: &str) -> Result<Proj<'a>, String> {
        let t: &DeviceTensor = model
            .tensor(name)
            .ok_or_else(|| format!("model has no tensor {name}"))?;
        let shape = t.shape();
        if shape.len() != 2 {
            return Err(format!("{name}: expected a matrix, got {shape:?}"));
        }
        let (rows, cols) = (shape[0] as usize, shape[1] as usize);
        let entry = gemv_entry(t.ty())?;
        let q5_half = if entry == "gemv_q5_k" {
            let v = dequant_q5_k(t, rows, cols)?;
            let mut half = Vec::with_capacity(v.len() * 2);
            for &x in &v {
                half.extend_from_slice(&hipml_core::kv::f32_to_f16(x).to_le_bytes());
            }
            Some(ByteBuffer::from_host(&half)?)
        } else {
            None
        };
        Ok(Proj {
            w: t.buffer(),
            entry,
            q5_half,
            rows,
            cols,
        })
    }

    pub(crate) fn gemv(&self, l: &Launcher, x: &Buffer, y: &Buffer) -> Result<(), String> {
        self.gemv_on(l, "stream", x, y)
    }

    /// The same matvec on a chosen stream ("stream" or "side"), so two
    /// independent projections can share the machine.
    pub(crate) fn gemv_on(
        &self,
        l: &Launcher,
        which: &str,
        x: &Buffer,
        y: &Buffer,
    ) -> Result<(), String> {
        l.run_quant_resident_on(
            which,
            hipml_kernels::stream_entry(self.entry),
            self.w,
            x,
            y,
            self.rows,
            self.cols,
        )
    }

    /// The same projection over `rows` x-vectors (`[rows][cols]`) with one
    /// pass over the weights: the batched decode step.
    pub(crate) fn gemm_batched(
        &self,
        l: &Launcher,
        x: &Buffer,
        y: &Buffer,
        rows: usize,
    ) -> Result<(), String> {
        let kind = batch_kind(self.entry).ok_or_else(|| {
            format!(
                "{} ({}x{}): no batched GEMV (F32 weights stay per-slot)",
                self.entry, self.rows, self.cols
            )
        })?;
        l.gemv_batched(kind, self.w, x, y, self.rows, self.cols, rows)
    }

    /// The same projection over `rows` x-rows (`[rows][cols]`) with one weight
    /// pass: the speculative verify's projection. Falls back to a GEMV per row
    /// for formats with no `_b` kernel.
    pub(crate) fn gemv_rows(
        &self,
        l: &Launcher,
        x: &Buffer,
        y: &Buffer,
        rows: usize,
    ) -> Result<(), String> {
        if hipml_kernels::skinny_on() {
            if let Some(name) = hipml_kernels::skinny_entry(self.entry) {
                return l.run_skinny_resident(name, self.w, x, y, self.rows, self.cols, rows);
            }
        }
        if let Some(kind) = hipml_kernels::batch_kind_entry(self.entry) {
            return l.gemv_batched(kind, self.w, x, y, self.rows, self.cols, rows);
        }
        for r in 0..rows {
            let p_x = unsafe { (x.as_ptr() as *mut u8).add(r * self.cols * 4) };
            let p_y = unsafe { (y.as_ptr() as *mut u8).add(r * self.rows * 4) };
            l.run_quant_resident_ptr(
                hipml_kernels::stream_entry(self.entry),
                self.w,
                p_x as *mut std::ffi::c_void,
                p_y as *mut std::ffi::c_void,
                self.rows,
                self.cols,
            )?;
        }
        Ok(())
    }

    /// The same projection over `tokens` rows of `x` (`[tokens][cols]`), one
    /// tiled GEMM instead of `tokens` GEMVs. The prefill chunk's projection.
    pub(crate) fn gemm(
        &self,
        l: &Launcher,
        x: &Buffer,
        y: &Buffer,
        tokens: usize,
    ) -> Result<(), String> {
        // Q5_K has no dequant-in-shared tile kernel, so its prefill runs on the
        // fp16 copy made at load. Running the decode GEMV once per row instead
        // cost ~22% of a 512-token prompt after the graphs merge.
        if let Some(half) = &self.q5_half {
            return hipml_kernels::prefill_resident(
                l.module(),
                l.stream(),
                "prefill_halfw_h",
                half,
                x,
                y,
                self.rows,
                self.cols,
                tokens,
            );
        }
        let e = prefill_entry(self.entry)?;
        l.run_prefill_resident(e, self.w, x, y, self.rows, self.cols, tokens)
    }

    /// The same projection, reading an fp16 copy of the activation tile that the
    /// caller has already converted (`prefill_xhalf`). Only the `_w2h` entries
    /// exist for it, so this requires the v2 prefill kernels.
    ///
    /// `scratch` holds the worst split-K tile the ffn shapes can select for any
    /// tile up to the chunk (see `prefill_split_scratch_cap`); only a
    /// wave-starved tile touches it, and a full 2048-token chunk does not.
    pub(crate) fn gemm_half(
        &self,
        l: &Launcher,
        xh: &Buffer,
        y: &Buffer,
        scratch: &Buffer,
        tokens: usize,
    ) -> Result<(), String> {
        if let Some(half) = &self.q5_half {
            let e = "prefill_halfw_w2h";
            if hipml_kernels::prefill_should_split(self.rows, self.cols, tokens) {
                return hipml_kernels::prefill_resident_split_half(
                    l.module(),
                    l.stream(),
                    e,
                    half,
                    xh,
                    y,
                    scratch,
                    self.rows,
                    self.cols,
                    tokens,
                );
            }
            return hipml_kernels::prefill_resident(
                l.module(),
                l.stream(),
                e,
                half,
                xh,
                y,
                self.rows,
                self.cols,
                tokens,
            );
        }
        let e = prefill_entry_half(self.entry)?;
        if hipml_kernels::prefill_should_split(self.rows, self.cols, tokens) {
            return hipml_kernels::prefill_resident_split_half(
                l.module(),
                l.stream(),
                e,
                self.w,
                xh,
                y,
                scratch,
                self.rows,
                self.cols,
                tokens,
            );
        }
        l.run_prefill_resident(e, self.w, xh, y, self.rows, self.cols, tokens)
    }

    pub(crate) fn bytes(&self) -> usize {
        self.w.len() + self.q5_half.as_ref().map_or(0, |b| b.len())
    }

    /// Whether [`Proj::gemm_half`] can serve this format. F32 weights (the
    /// loader's expansion of Q3_K / IQ3_S) have no fp16-activation tile, so
    /// the caller passes the f32 activation to [`Proj::gemm`] instead.
    pub(crate) fn has_half(&self) -> bool {
        self.q5_half.is_some() || prefill_entry_half(self.entry).is_ok()
    }
}

/// Compare the batched GEMV against `rows` single launches on a resident
/// tensor, bit for bit.
///
/// This is the exactness premise of the speculative verify: if `_b` keeps the
/// single-sequence accumulation order, then row `r` of the batched result is
/// the same 32 bits as a one-column decode. Returns
/// `(values, differing, max |d|)`.
pub fn check_batched_gemv(
    model: &Model,
    name: &str,
    rows: usize,
    seed: u64,
) -> Result<(usize, usize, f32), String> {
    let proj = Proj::of(model, name)?;
    let kind = hipml_kernels::batch_kind_entry(proj.entry)
        .ok_or_else(|| format!("{name}: no batched GEMV"))?;
    let (m, k) = (proj.rows, proj.cols);
    let mut s = seed | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 11) as f32 / (1u64 << 53) as f32) - 0.5
    };
    let x: Vec<f32> = (0..rows * k).map(|_| next()).collect();
    let xb = Buffer::from_host(&x)?;
    let yb = Buffer::alloc(rows * m)?;
    let ys = Buffer::alloc(rows * m)?;
    let l = model.launcher();
    l.gemv_batched(kind, proj.w, &xb, &yb, m, k, rows)?;
    for r in 0..rows {
        let px = unsafe { (xb.as_ptr() as *mut u8).add(r * k * 4) };
        let py = unsafe { (ys.as_ptr() as *mut u8).add(r * m * 4) };
        l.run_quant_resident_ptr(
            // The row-tile entry the plain decode path launches, not the `_r1`
            // streaming variant: exactness is against what `run_token` runs.
            proj.entry,
            proj.w,
            px as *mut std::ffi::c_void,
            py as *mut std::ffi::c_void,
            m,
            k,
        )?;
    }
    l.synchronize()?;
    let a = yb.to_host()?;
    let b = ys.to_host()?;
    let mut diff = 0usize;
    let mut worst = 0.0f32;
    for i in 0..rows * m {
        if a[i].to_bits() != b[i].to_bits() {
            diff += 1;
            worst = worst.max((a[i] - b[i]).abs());
        }
    }
    Ok((rows * m, diff, worst))
}

/// One timing sample for the batched GEMV.
pub struct GemvTiming {
    pub name: String,
    pub kind: &'static str,
    pub bytes: usize,
    pub rows: usize,
    pub single_ms: f64,
    pub batched_ms: f64,
}

impl GemvTiming {
    /// Weight bytes the batched launch streams per millisecond, in GB/s.
    pub fn batched_gbs(&self) -> f64 {
        self.bytes as f64 / (self.batched_ms * 1e6)
    }

    /// Weight bytes a `rows`-launch fallback streams per millisecond.
    pub fn single_gbs(&self) -> f64 {
        self.bytes as f64 / (self.single_ms * 1e6)
    }

    /// Batched cost relative to `rows` single launches.
    pub fn ratio(&self) -> f64 {
        self.batched_ms / self.single_ms.max(1e-9)
    }
}

/// Time the batched GEMV against `rows` single launches on a resident tensor,
/// `reps` times each, and report the fastest sample. This is the cost side of
/// the verify pass: a `_b` launch that costs more than one single GEMV spends
/// the extra column's budget on kernel overhead instead of weights.
pub fn bench_batched_gemv(
    model: &Model,
    name: &str,
    max_rows: usize,
    reps: usize,
) -> Result<Vec<GemvTiming>, String> {
    use std::time::Instant;
    let proj = Proj::of(model, name)?;
    let bkind = hipml_kernels::batch_kind_entry(proj.entry)
        .ok_or_else(|| format!("{name}: no batched GEMV"))?;
    let (m, k) = (proj.rows, proj.cols);
    let x = Buffer::from_host(&vec![0.125f32; max_rows * k])?;
    let y = Buffer::alloc(max_rows * m)?;
    let l = model.launcher();
    let mut out = Vec::new();
    for rows in 1..=max_rows {
        let mut single = f64::MAX;
        let mut batched = f64::MAX;
        for _ in 0..reps {
            let t = Instant::now();
            for r in 0..rows {
                let px = unsafe { (x.as_ptr() as *mut u8).add(r * k * 4) };
                let py = unsafe { (y.as_ptr() as *mut u8).add(r * m * 4) };
                l.run_quant_resident_ptr(
                    hipml_kernels::stream_entry(proj.entry),
                    proj.w,
                    px as *mut std::ffi::c_void,
                    py as *mut std::ffi::c_void,
                    m,
                    k,
                )?;
            }
            l.synchronize()?;
            single = single.min(t.elapsed().as_secs_f64() * 1e3);

            let t = Instant::now();
            l.gemv_batched(bkind, proj.w, &x, &y, m, k, rows)?;
            l.synchronize()?;
            batched = batched.min(t.elapsed().as_secs_f64() * 1e3);
        }
        out.push(GemvTiming {
            name: name.to_string(),
            kind: proj.entry,
            bytes: proj.w.len(),
            rows,
            single_ms: single,
            batched_ms: batched,
        });
    }
    Ok(out)
}

/// Dequantize a resident Q5_K matrix once, for the fp16 prefill copy.
fn dequant_q5_k(t: &DeviceTensor, rows: usize, cols: usize) -> Result<Vec<f32>, String> {
    use hipml_core::quant::{dequant_q5_k, Q5_K_BYTES, QK_K};
    let raw = t.read_back()?;
    let blocks = cols / QK_K;
    let row_bytes = blocks * Q5_K_BYTES;
    if raw.len() != rows * row_bytes {
        return Err(format!(
            "{}: {} bytes, expected {} rows x {} bytes",
            t.name(),
            raw.len(),
            rows,
            row_bytes
        ));
    }
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for b in 0..blocks {
            let src = &raw[r * row_bytes + b * Q5_K_BYTES..r * row_bytes + (b + 1) * Q5_K_BYTES];
            dequant_q5_k(
                src,
                &mut out[r * cols + b * QK_K..r * cols + (b + 1) * QK_K],
            );
        }
    }
    Ok(out)
}
