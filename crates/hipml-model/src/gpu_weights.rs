//! Resident weight views for the GPU graph.
//!
//! A `Proj` points at a loader `ByteBuffer` in its GGUF block form and carries
//! the GEMV entry that reads it. The one exception is Q5_K: the loader has the
//! bytes but no GEMV kernel, so those two Qwen3.8-27B tensors are dequantized once at
//! build time into an f32 buffer and served by `gemv_f32`.

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
        other => return Err(format!("{} has no GEMV kernel", other.name())),
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

enum ProjData<'a> {
    /// A GGUF block payload streamed by the quantized GEMV.
    Resident(&'a ByteBuffer, &'static str),
    /// An f32 matrix dequantized once at build time (Q5_K has no GEMV kernel).
    F32(Buffer),
}

/// A projection that a launcher can stream straight from VRAM.
pub struct Proj<'a> {
    data: ProjData<'a>,
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
        let data = match t.ty() {
            TensorType::Q5K => ProjData::F32(Buffer::from_host(&dequant_q5_k(t, rows, cols)?)?),
            ty => ProjData::Resident(t.buffer(), gemv_entry(ty)?),
        };
        Ok(Proj { data, rows, cols })
    }

    pub(crate) fn gemv(&self, l: &Launcher, x: &Buffer, y: &Buffer) -> Result<(), String> {
        match &self.data {
            ProjData::Resident(w, entry) => {
                l.run_quant_resident(entry, w, x, y, self.rows, self.cols)
            }
            ProjData::F32(w) => l.run_f32_resident(w, x, y, self.rows, self.cols),
        }
    }

    pub(crate) fn bytes(&self) -> usize {
        match &self.data {
            ProjData::Resident(w, _) => w.len(),
            ProjData::F32(w) => w.len() * 4,
        }
    }
}

/// Dequantize a resident Q5_K matrix once. Qwen3.8-27B stores exactly two of these
/// (`blk.51/55.ffn_down`), and the loader keeps their bytes resident.
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

