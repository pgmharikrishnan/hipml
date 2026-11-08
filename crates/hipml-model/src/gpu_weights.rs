//! Resident weight views for the GPU graph.
//!
//! A `Proj` points at a loader `ByteBuffer` in its GGUF block form and carries
//! the GEMV entry that reads it. Every format the model uses has a kernel that
//! reads its blocks directly: Q4_K, Q6_K, Q8_0 and Q5_K. Q5_K keeps a half copy
//! as well: the GEMV reads the blocks, but the tiled prefill GEMM needs an
//! fp16 operand and there is no Q5_K dequant-in-shared tile kernel.

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
        other => return Err(format!("{} has no GEMV kernel", other.name())),
    })
}

/// The batched-GEMV kind for a resident GEMV entry. Only the four block
/// formats with a `_b` kernel are batchable; F32 projections are small and stay
/// per-slot.
pub(crate) fn batch_kind(entry: &str) -> Option<i32> {
    use hipml_kernels::{GEMV_B_Q4K, GEMV_B_Q5K, GEMV_B_Q6K, GEMV_B_Q80};
    Some(match entry {
        "gemv_q4_k" => GEMV_B_Q4K,
        "gemv_q6_k" => GEMV_B_Q6K,
        "gemv_q5_k" => GEMV_B_Q5K,
        "gemv_q8_0" => GEMV_B_Q80,
        _ => return None,
    })
}

/// The tiled GEMM entry for a quantized GEMV entry; the two families share the
/// block format.
fn prefill_entry(gemv: &str) -> Result<&'static str, String> {
    Ok(match gemv {
        "gemv_q4_k" => "prefill_q4_k",
        "gemv_q6_k" => "prefill_q6_k",
        "gemv_q8_0" => "prefill_q8_0",
        other => return Err(format!("{other}: no tiled prefill kernel")),
    })
}

pub(crate) fn mat_kind(ty: TensorType) -> Result<Kind, String> {
    Ok(match ty {
        TensorType::F32 => Kind::F32,
        TensorType::Q4K => Kind::Q4K,
        TensorType::Q6K => Kind::Q6K,
        TensorType::Q8_0 => Kind::Q8_0,
        other => return Err(format!("{} is not a supported mat kind", other.name())),
    })
}

/// A projection that a launcher can stream straight from VRAM: the loader's
/// block payload plus the GEMV entry that reads it.
pub struct Proj<'a> {
    w: &'a ByteBuffer,
    entry: &'static str,
    /// Q5_K only: the fp16 copy the tiled prefill GEMM streams. One per tensor,
    /// 0.12 GB for Qwen3.8-27B's two `ffn_down` tensors; the decode GEMV still reads
    /// the original blocks.
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

    pub(crate) fn bytes(&self) -> usize {
        self.w.len() + self.q5_half.as_ref().map_or(0, |b| b.len())
    }
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
            dequant_q5_k(src, &mut out[r * cols + b * QK_K..r * cols + (b + 1) * QK_K]);
        }
    }
    Ok(out)
}
