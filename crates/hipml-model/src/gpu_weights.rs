//! Resident weight views for the GPU graph.
//!
//! A `Proj` points at a loader `ByteBuffer` in its GGUF block form and carries
//! the GEMV entry that reads it. Every format the model uses has a kernel that
//! reads its blocks directly: Q4_K, Q6_K, Q8_0 and Q5_K.

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
        Ok(Proj {
            w: t.buffer(),
            entry: gemv_entry(t.ty())?,
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

    /// The same projection over `tokens` rows of `x` (`[tokens][cols]`), one
    /// tiled GEMM instead of `tokens` GEMVs. The prefill chunk's projection.
    pub(crate) fn gemm(
        &self,
        l: &Launcher,
        x: &Buffer,
        y: &Buffer,
        tokens: usize,
    ) -> Result<(), String> {
        // Q5_K has no tiled prefill GEMM; it is two tensors and prefill is not
        // the acceptance path, so run the exact GEMV per row instead of
        // introducing a second accumulation order for them.
        if self.entry == "gemv_q5_k" {
            let x0 = x.as_ptr() as usize;
            let y0 = y.as_ptr() as usize;
            for t in 0..tokens {
                l.run_quant_resident_ptr(
                    self.entry,
                    self.w,
                    (x0 + t * self.cols * 4) as *mut _,
                    (y0 + t * self.rows * 4) as *mut _,
                    self.rows,
                    self.cols,
                )?;
            }
            return Ok(());
        }
        let e = prefill_entry(self.entry)?;
        l.run_prefill_resident(e, self.w, x, y, self.rows, self.cols, tokens)
    }

    pub(crate) fn bytes(&self) -> usize {
        self.w.len()
    }
}
