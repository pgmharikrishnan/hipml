// SPDX-License-Identifier: Apache-2.0
//! Weight matrices in their on-disk quantized form, and the reference GEMV.
//!
//! A 27B Q4_K_M model is 17 GB of bytes but 68 GB of f32, so the forward pass
//! keeps every weight tensor quantized exactly as GGUF stores it and
//! dequantises a row at a time while multiplying. That is also the shape the
//! GPU kernel will have: stream the row, accumulate the dot.
//!
//! Tensor layout follows GGML: `ne0` is the contraction axis, so a `[ne0,
//! ne1]` tensor in the file is `ne1` rows of `ne0` values here.

use std::io::{self, Read, Seek};
use std::sync::Arc;

use crate::gguf::Gguf;
use crate::mmap::Map;
use crate::quant::{
    dequant_q4_k, dequant_q5_k, dequant_q6_k, dequant_q8_0, TensorType, Q4_K_BYTES, Q5_K_BYTES,
    Q6_K_BYTES, Q8_0_BYTES, QK_K,
};

/// Storage of one matrix, matching the GGUF tensor type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    F32,
    Q4K,
    Q5K,
    Q6K,
    Q8_0,
}

impl Kind {
    fn of(ty: TensorType) -> io::Result<Kind> {
        Ok(match ty {
            TensorType::F32 => Kind::F32,
            TensorType::Q4K => Kind::Q4K,
            // Q5_K was out of scope, but Qwen3.8-27B stores blk.51/55.ffn_down
            // as Q5_K, so the reference has to read it.
            TensorType::Q5K => Kind::Q5K,
            TensorType::Q6K => Kind::Q6K,
            TensorType::Q8_0 => Kind::Q8_0,
            TensorType::Unsupported(id) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unsupported tensor type {id}"),
                ))
            }
        })
    }

    fn row_bytes(self, cols: usize) -> usize {
        match self {
            Kind::F32 => cols * 4,
            Kind::Q8_0 => cols / 32 * Q8_0_BYTES,
            Kind::Q4K => cols / QK_K * Q4_K_BYTES,
            Kind::Q5K => cols / QK_K * Q5_K_BYTES,
            Kind::Q6K => cols / QK_K * Q6_K_BYTES,
        }
    }
}

/// Where the matrix bytes live: read into memory, or a window into the
/// read-only file map.
pub enum Bytes {
    Owned(Vec<u8>),
    /// `(map, start, len)`; `len` is exactly the tensor payload.
    Mapped(Arc<Map>, usize, usize),
}

impl Bytes {
    pub fn as_slice(&self) -> &[u8] {
        match self {
            Bytes::Owned(v) => v,
            Bytes::Mapped(m, start, len) => m.slice(*start, *len),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Bytes::Owned(v) => v.len(),
            Bytes::Mapped(_, _, len) => *len,
        }
    }
}

/// A `rows x cols` matrix, row-major, in `kind` blocks.
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub kind: Kind,
    pub data: Bytes,
}

impl Mat {
    pub fn load<R: Read + Seek>(g: &mut Gguf<R>, name: &str) -> io::Result<Mat> {
        let info = g
            .tensor(name)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("no tensor {name}")))?
            .clone();
        let cols = info.ne0() as usize;
        let rows = info.rows() as usize;
        let kind = Kind::of(info.ty)
            .map_err(|e| io::Error::new(e.kind(), format!("tensor {name}: {e}")))?;
        let expect = rows * kind.row_bytes(cols);
        if expect as u64 != info.nbytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "tensor {name}: {}x{} {} needs {expect} bytes, file has {}",
                    rows,
                    cols,
                    info.ty.name(),
                    info.nbytes
                ),
            ));
        }
        Ok(Mat {
            rows,
            cols,
            kind,
            data: Bytes::Owned(g.read_tensor(name)?),
        })
    }

    /// Map a tensor directly from the file, without copying it.
    pub fn from_map<R: Read + Seek>(map: &Arc<Map>, g: &Gguf<R>, name: &str) -> io::Result<Mat> {
        let info = g
            .tensor(name)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("no tensor {name}")))?
            .clone();
        let cols = info.ne0() as usize;
        let rows = info.rows() as usize;
        let kind = Kind::of(info.ty)
            .map_err(|e| io::Error::new(e.kind(), format!("tensor {name}: {e}")))?;
        let expect = rows * kind.row_bytes(cols);
        if expect as u64 != info.nbytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "tensor {name}: {}x{} {} needs {expect} bytes, file has {}",
                    rows,
                    cols,
                    info.ty.name(),
                    info.nbytes
                ),
            ));
        }
        let start = usize::try_from(g.data_offset + info.offset)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "tensor offset too large"))?;
        let len = usize::try_from(info.nbytes)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "tensor size too large"))?;
        Ok(Mat {
            rows,
            cols,
            kind,
            data: Bytes::Mapped(map.clone(), start, len),
        })
    }

    /// Dequantise `row` into f32.
    ///
    /// Kernel: `dequant_row_f32`, one block per K-quant super-block.
    pub fn row_f32(&self, row: usize) -> Vec<f32> {
        assert!(row < self.rows, "row {row} out of {}", self.rows);
        let rb = self.kind.row_bytes(self.cols);
        let data = self.data.as_slice();
        let bytes = &data[row * rb..(row + 1) * rb];
        let mut out = vec![0.0f32; self.cols];
        dequant_row(self.kind, bytes, &mut out);
        out
    }

    /// `y[rows] = W[rows, cols] · x[cols]`.
    ///
    /// Kernel: `gemv_quant_f32`, one block per output row; the host version
    /// splits rows across threads because a full model pass is 17 GB.
    pub fn gemv(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.cols, "gemv: x must be cols");
        let threads = match std::env::var("HIPML_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
        {
            Some(n) => n,
            None => std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
        }
        .max(1)
        .min(self.rows.max(1));
        let chunk = self.rows.div_ceil(threads);
        let mut y = vec![0.0f32; self.rows];
        std::thread::scope(|scope| {
            let mut start = 0usize;
            for out in y.chunks_mut(chunk) {
                let s = start;
                let len = out.len();
                scope.spawn(move || self.gemv_chunk(s, out, x));
                start += len;
            }
        });
        y
    }

    fn gemv_chunk(&self, start: usize, out: &mut [f32], x: &[f32]) {
        let rb = self.kind.row_bytes(self.cols);
        // One bounds check per chunk instead of per row; the map is huge.
        let data = self.data.as_slice();
        for (i, o) in out.iter_mut().enumerate() {
            let row = start + i;
            let bytes = &data[row * rb..(row + 1) * rb];
            *o = dot_row(self.kind, bytes, self.cols, x);
        }
    }
}

/// One row of a matrix as f32, used for embedding lookup and tests.
pub fn dequant_row(kind: Kind, bytes: &[u8], out: &mut [f32]) {
    match kind {
        Kind::F32 => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = f32::from_le_bytes([
                    bytes[i * 4],
                    bytes[i * 4 + 1],
                    bytes[i * 4 + 2],
                    bytes[i * 4 + 3],
                ]);
            }
        }
        Kind::Q8_0 => {
            for (b, chunk) in out.chunks_mut(32).enumerate() {
                let blk = &bytes[b * Q8_0_BYTES..(b + 1) * Q8_0_BYTES];
                let mut tmp = [0.0f32; 32];
                dequant_q8_0(blk, &mut tmp);
                chunk.copy_from_slice(&tmp);
            }
        }
        Kind::Q4K | Kind::Q5K | Kind::Q6K => {
            let bb = match kind {
                Kind::Q4K => Q4_K_BYTES,
                Kind::Q5K => Q5_K_BYTES,
                _ => Q6_K_BYTES,
            };
            let mut tmp = [0.0f32; QK_K];
            for (b, chunk) in out.chunks_mut(QK_K).enumerate() {
                let blk = &bytes[b * bb..(b + 1) * bb];
                match kind {
                    Kind::Q4K => dequant_q4_k(blk, &mut tmp),
                    Kind::Q5K => dequant_q5_k(blk, &mut tmp),
                    _ => dequant_q6_k(blk, &mut tmp),
                }
                chunk.copy_from_slice(&tmp);
            }
        }
    }
}

/// Dot product of one quantized row against `x`, dequantising block by block so
/// the full row never exists in f32.
fn dot_row(kind: Kind, bytes: &[u8], cols: usize, x: &[f32]) -> f32 {
    match kind {
        Kind::F32 => {
            let mut acc = 0.0f32;
            for (i, v) in x.iter().enumerate() {
                let w = f32::from_le_bytes([
                    bytes[i * 4],
                    bytes[i * 4 + 1],
                    bytes[i * 4 + 2],
                    bytes[i * 4 + 3],
                ]);
                acc += w * v;
            }
            acc
        }
        Kind::Q8_0 => {
            let mut acc = 0.0f32;
            let mut tmp = [0.0f32; 32];
            for b in 0..cols / 32 {
                let blk = &bytes[b * Q8_0_BYTES..(b + 1) * Q8_0_BYTES];
                dequant_q8_0(blk, &mut tmp);
                for (l, w) in tmp.iter().enumerate() {
                    acc += w * x[b * 32 + l];
                }
            }
            acc
        }
        Kind::Q4K | Kind::Q5K | Kind::Q6K => {
            let bb = match kind {
                Kind::Q4K => Q4_K_BYTES,
                Kind::Q5K => Q5_K_BYTES,
                _ => Q6_K_BYTES,
            };
            let mut acc = 0.0f32;
            let mut tmp = [0.0f32; QK_K];
            for b in 0..cols / QK_K {
                let blk = &bytes[b * bb..(b + 1) * bb];
                match kind {
                    Kind::Q4K => dequant_q4_k(blk, &mut tmp),
                    Kind::Q5K => dequant_q5_k(blk, &mut tmp),
                    _ => dequant_q6_k(blk, &mut tmp),
                }
                let xb = &x[b * QK_K..(b + 1) * QK_K];
                for (l, w) in tmp.iter().enumerate() {
                    acc += w * xb[l];
                }
            }
            acc
        }
    }
}
