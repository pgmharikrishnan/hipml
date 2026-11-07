//! A read-only GGUF v2/v3 reader: header, metadata KV table, tensor-info table,
//! and 32-byte-aligned tensor data.
//!
//! Deliberately streaming: an Qwen3.8-27B Q4_K_M file is 17.2 GB, so nothing here
//! reads the tensor data until `read_tensor` (or `read_tensor_range`) is called,
//! and even then only the requested bytes are pulled. The metadata and tensor
//! tables are tiny and live in memory.
//!
//! No crate does this: the format is a handful of little-endian integers and
//! length-prefixed strings, and the whole point is to be able to read the real
//! model without adding a dependency.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};

use crate::quant::TensorType;

/// GGUF magic as a little-endian u32: the bytes are `G G U F`.
pub const GGUF_MAGIC: u32 = 0x4655_4747;

/// One tensor-info record.
#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    /// Dimensions in GGML order: `dims[0]` is the fastest-varying axis (ne0,
    /// the row length / contraction axis for a matrix).
    pub dims: Vec<u64>,
    pub ty: TensorType,
    /// Byte offset relative to the start of the tensor-data section.
    pub offset: u64,
    /// Exact byte length of the tensor payload.
    pub nbytes: u64,
}

impl TensorInfo {
    /// Number of values in the tensor.
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }

    /// Row length (contraction axis) for a matrix-like tensor.
    pub fn ne0(&self) -> u64 {
        self.dims.first().copied().unwrap_or(0)
    }

    /// Number of rows if the tensor is treated as `[ne0, m, ...]`.
    pub fn rows(&self) -> u64 {
        let ne0 = self.ne0();
        if ne0 == 0 {
            0
        } else {
            self.n_elements() / ne0
        }
    }

    /// Shape in the order a human writes it (slowest-varying first), i.e. the
    /// GGML dims reversed.
    pub fn shape(&self) -> Vec<u64> {
        let mut d = self.dims.clone();
        d.reverse();
        d
    }
}

/// A GGUF file positioned at its tensor-data section.
pub struct Gguf<R: Read + Seek> {
    reader: R,
    pub version: u32,
    pub tensor_count: u64,
    pub kv_count: u64,
    pub alignment: u64,
    pub data_offset: u64,
    pub tensors: Vec<TensorInfo>,
}

impl Gguf<File> {
    /// Open a GGUF file from disk.
    pub fn open(path: impl AsRef<std::path::Path>) -> io::Result<Gguf<File>> {
        Gguf::from_reader(File::open(path)?)
    }
}

impl<R: Read + Seek> Gguf<R> {
    /// Parse the header, metadata and tensor-info table. The reader is left at
    /// the start of the tensor-data section.
    pub fn from_reader(mut reader: R) -> io::Result<Gguf<R>> {
        let magic = read_u32(&mut reader)?;
        if magic != GGUF_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("bad GGUF magic 0x{magic:08X}, expected 0x{GGUF_MAGIC:08X}"),
            ));
        }
        let version = read_u32(&mut reader)?;
        if version != 2 && version != 3 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported GGUF version {version} (expected 2 or 3)"),
            ));
        }
        let tensor_count = read_u64(&mut reader)?;
        let kv_count = read_u64(&mut reader)?;

        let mut alignment = 32u64;
        for _ in 0..kv_count {
            let key = read_string(&mut reader)?;
            let value_type = read_u32(&mut reader)?;
            if (key == "general.alignment" || key == "alignment") && value_type == VT_UINT32 {
                alignment = read_u32(&mut reader)? as u64;
            } else if (key == "general.alignment" || key == "alignment") && value_type == VT_UINT64
            {
                alignment = read_u64(&mut reader)?;
            } else {
                skip_value(&mut reader, value_type, 0)?;
            }
        }
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid GGUF alignment {alignment}"),
            ));
        }

        let mut tensors = Vec::with_capacity(tensor_count.min(1 << 20) as usize);
        for _ in 0..tensor_count {
            let name = read_string(&mut reader)?;
            let n_dims = read_u32(&mut reader)?;
            if n_dims == 0 || n_dims > 8 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("tensor {name}: invalid dim count {n_dims}"),
                ));
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(read_u64(&mut reader)?);
            }
            let ty = TensorType::from_id(read_u32(&mut reader)?);
            let offset = read_u64(&mut reader)?;
            let n_elements: u64 = dims.iter().product();
            let nbytes = ty.nbytes(n_elements).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "tensor {name}: cannot size type {} over {n_elements} values",
                        ty.name()
                    ),
                )
            })?;
            tensors.push(TensorInfo {
                name,
                dims,
                ty,
                offset,
                nbytes,
            });
        }

        let pos = reader.stream_position()?;
        let data_offset = align_up(pos, alignment);
        reader.seek(SeekFrom::Start(data_offset))?;

        Ok(Gguf {
            reader,
            version,
            tensor_count,
            kv_count,
            alignment,
            data_offset,
            tensors,
        })
    }

    /// The tensor-info record for `name`.
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Read a whole tensor payload by name.
    pub fn read_tensor(&mut self, name: &str) -> io::Result<Vec<u8>> {
        let info = self
            .tensor(name)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("no tensor {name}")))?;
        let nbytes = usize::try_from(info.nbytes)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "tensor too large"))?;
        let mut out = vec![0u8; nbytes];
        self.read_tensor_range(name, 0, nbytes, &mut out)?;
        Ok(out)
    }

    /// Read `out.len()` bytes starting `byte_offset` bytes into the tensor
    /// payload. This is the chunked path the decode bench uses.
    pub fn read_tensor_range(
        &mut self,
        name: &str,
        byte_offset: u64,
        len: usize,
        out: &mut [u8],
    ) -> io::Result<()> {
        assert_eq!(out.len(), len, "out must be exactly len bytes");
        let info = self
            .tensor(name)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("no tensor {name}")))?;
        let end = byte_offset
            .checked_add(len as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "range overflow"))?;
        if end > info.nbytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "tensor {name}: range {byte_offset}+{len} exceeds {} bytes",
                    info.nbytes
                ),
            ));
        }
        let absolute = self
            .data_offset
            .checked_add(info.offset)
            .and_then(|v| v.checked_add(byte_offset))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "offset overflow"))?;
        self.reader.seek(SeekFrom::Start(absolute))?;
        self.reader.read_exact(out)
    }
}

/// Round `v` up to the next multiple of `align` (a power of two).
pub fn align_up(v: u64, align: u64) -> u64 {
    (v + align - 1) & !(align - 1)
}

// GGUF metadata value types.
const VT_UINT8: u32 = 0;
const VT_INT8: u32 = 1;
const VT_UINT16: u32 = 2;
const VT_INT16: u32 = 3;
const VT_UINT32: u32 = 4;
const VT_INT32: u32 = 5;
const VT_FLOAT32: u32 = 6;
const VT_BOOL: u32 = 7;
const VT_STRING: u32 = 8;
const VT_ARRAY: u32 = 9;
const VT_UINT64: u32 = 10;
const VT_INT64: u32 = 11;
const VT_FLOAT64: u32 = 12;

fn read_exact<const N: usize, R: Read>(r: &mut R) -> io::Result<[u8; N]> {
    let mut buf = [0u8; N];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    Ok(u32::from_le_bytes(read_exact::<4, _>(r)?))
}

fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    Ok(u64::from_le_bytes(read_exact::<8, _>(r)?))
}

/// A GGUF string: u64 length followed by that many UTF-8 bytes (v2/v3).
fn read_string<R: Read>(r: &mut R) -> io::Result<String> {
    let len = read_u64(r)?;
    // A metadata key or tensor name is never gigabytes; reject corruption early.
    if len > 1 << 20 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("string length {len} is implausible"),
        ));
    }
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("invalid UTF-8: {e}")))
}

/// Skip one metadata value of `ty`, including arrays (recursively, with a depth
/// bound so a corrupt file cannot recurse forever).
fn skip_value<R: Read + Seek>(r: &mut R, ty: u32, depth: u32) -> io::Result<()> {
    if depth > 4 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "metadata arrays nested too deeply",
        ));
    }
    match ty {
        VT_UINT8 | VT_INT8 | VT_BOOL => skip_bytes(r, 1),
        VT_UINT16 | VT_INT16 => skip_bytes(r, 2),
        VT_UINT32 | VT_INT32 | VT_FLOAT32 => skip_bytes(r, 4),
        VT_UINT64 | VT_INT64 | VT_FLOAT64 => skip_bytes(r, 8),
        VT_STRING => {
            let len = read_u64(r)?;
            skip_bytes(r, len as i64)
        }
        VT_ARRAY => {
            let elem = read_u32(r)?;
            let count = read_u64(r)?;
            for _ in 0..count {
                skip_value(r, elem, depth + 1)?;
            }
            Ok(())
        }
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown GGUF metadata value type {other}"),
        )),
    }
}

fn skip_bytes<R: Seek>(r: &mut R, n: i64) -> io::Result<()> {
    if n < 0 || n > 1 << 40 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("implausible metadata length {n}"),
        ));
    }
    r.seek(SeekFrom::Current(n))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Build a minimal GGUF v3 with one F32 tensor of 4 values.
    fn tiny_gguf() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes()); // tensor count
        b.extend_from_slice(&0u64.to_le_bytes()); // kv count
                                                  // tensor info: name, n_dims, dims, type, offset
        b.extend_from_slice(&1u64.to_le_bytes());
        b.extend_from_slice(b"w");
        b.extend_from_slice(&2u32.to_le_bytes());
        b.extend_from_slice(&2u64.to_le_bytes());
        b.extend_from_slice(&2u64.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes()); // F32
        b.extend_from_slice(&0u64.to_le_bytes());
        // pad to 32
        while b.len() % 32 != 0 {
            b.push(0);
        }
        for v in [1.0f32, 2.0, 3.0, 4.0] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b
    }

    #[test]
    fn parses_a_minimal_file_and_reads_the_tensor() {
        let bytes = tiny_gguf();
        let mut g = Gguf::from_reader(Cursor::new(bytes)).expect("parse");
        assert_eq!(g.version, 3);
        assert_eq!(g.tensor_count, 1);
        assert_eq!(g.alignment, 32);
        assert_eq!(g.data_offset, 96);
        let info = g.tensor("w").expect("w");
        assert_eq!(info.dims, vec![2, 2]);
        assert_eq!(info.shape(), vec![2, 2]);
        assert_eq!(info.ty, TensorType::F32);
        assert_eq!(info.nbytes, 16);
        let raw = g.read_tensor("w").expect("read");
        let vals: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0]);

        let mut chunk = [0u8; 8];
        g.read_tensor_range("w", 8, 8, &mut chunk).expect("range");
        assert_eq!(&chunk[0..4], &3.0f32.to_le_bytes());
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = tiny_gguf();
        bytes[0] ^= 0xFF;
        assert!(Gguf::from_reader(Cursor::new(bytes)).is_err());
    }

    #[test]
    fn alignment_rounds_up() {
        assert_eq!(align_up(0, 32), 0);
        assert_eq!(align_up(1, 32), 32);
        assert_eq!(align_up(32, 32), 32);
        assert_eq!(align_up(33, 32), 64);
    }
}
