// SPDX-License-Identifier: Apache-2.0
//! GGUF metadata values.
//!
//! The tensor table is enough to read weights; the tokenizer and the
//! hyperparameters live in the metadata KV table, so the reader has to
//! materialise values rather than skip them. The type ids are the on-disk
//! GGUF enum, little-endian, with strings as `u64 len` + UTF-8 bytes and
//! arrays as `u32 elem type` + `u64 count` + values.
//!
//! Split out of `gguf.rs` so that file stays about the file layout.

use std::io::{self, Read, Seek, SeekFrom};

pub const VT_UINT8: u32 = 0;
pub const VT_INT8: u32 = 1;
pub const VT_UINT16: u32 = 2;
pub const VT_INT16: u32 = 3;
pub const VT_UINT32: u32 = 4;
pub const VT_INT32: u32 = 5;
pub const VT_FLOAT32: u32 = 6;
pub const VT_BOOL: u32 = 7;
pub const VT_STRING: u32 = 8;
pub const VT_ARRAY: u32 = 9;
pub const VT_UINT64: u32 = 10;
pub const VT_INT64: u32 = 11;
pub const VT_FLOAT64: u32 = 12;

/// One metadata value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    pub fn as_u32(&self) -> Option<u32> {
        match *self {
            Value::U8(v) => Some(v as u32),
            Value::U16(v) => Some(v as u32),
            Value::U32(v) => Some(v),
            Value::U64(v) => u32::try_from(v).ok(),
            Value::I32(v) => u32::try_from(v).ok(),
            _ => None,
        }
    }

    pub fn as_usize(&self) -> Option<usize> {
        self.as_u32().map(|v| v as usize)
    }

    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Value::F32(v) => Some(v),
            Value::F64(v) => Some(v as f32),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Value::Bool(v) => Some(v),
            Value::U8(v) => Some(v != 0),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(v) => Some(v),
            _ => None,
        }
    }

    /// `[true, false, ...]` as a bool list (recurrent layer flags).
    pub fn as_bools(&self) -> Option<Vec<bool>> {
        self.as_array()?.iter().map(Value::as_bool).collect()
    }

    /// `[7, 7, 10, 0]` as a usize list (rope sections).
    pub fn as_usizes(&self) -> Option<Vec<usize>> {
        self.as_array()?.iter().map(Value::as_usize).collect()
    }

    /// `[1, 1, 3, ...]` as a u32 list (`tokenizer.ggml.token_type`).
    pub fn as_u32s(&self) -> Option<Vec<u32>> {
        self.as_array()?.iter().map(Value::as_u32).collect()
    }

    /// Token list, e.g. `tokenizer.ggml.tokens`.
    pub fn as_strings(&self) -> Option<Vec<&str>> {
        self.as_array()?.iter().map(Value::as_str).collect()
    }
}

fn read_exact<const N: usize, R: Read>(r: &mut R) -> io::Result<[u8; N]> {
    let mut buf = [0u8; N];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

pub fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    Ok(u32::from_le_bytes(read_exact::<4, _>(r)?))
}

pub fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    Ok(u64::from_le_bytes(read_exact::<8, _>(r)?))
}

pub fn read_string<R: Read>(r: &mut R) -> io::Result<String> {
    let len = read_u64(r)?;
    if len > 1 << 22 {
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

/// Read one value of metadata type `ty`.
pub fn read_value<R: Read + Seek>(r: &mut R, ty: u32) -> io::Result<Value> {
    Ok(match ty {
        VT_UINT8 => Value::U8(read_exact::<1, _>(r)?[0]),
        VT_INT8 => Value::I8(read_exact::<1, _>(r)?[0] as i8),
        VT_UINT16 => Value::U16(u16::from_le_bytes(read_exact::<2, _>(r)?)),
        VT_INT16 => Value::I16(i16::from_le_bytes(read_exact::<2, _>(r)?)),
        VT_UINT32 => Value::U32(read_u32(r)?),
        VT_INT32 => Value::I32(read_u32(r)? as i32),
        VT_FLOAT32 => Value::F32(f32::from_bits(read_u32(r)?)),
        VT_BOOL => Value::Bool(read_exact::<1, _>(r)?[0] != 0),
        VT_STRING => Value::String(read_string(r)?),
        VT_ARRAY => {
            let elem = read_u32(r)?;
            let count = read_u64(r)?;
            // 248320 tokenizer entries is the largest real array; anything past
            // this is corruption, not a model.
            if count > 1 << 24 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("metadata array of {count} values is implausible"),
                ));
            }
            // Tokenizer arrays are ~250k small values; reserve exactly.
            let mut vals = Vec::with_capacity(count as usize);
            for _ in 0..count {
                vals.push(read_value(r, elem)?);
            }
            Value::Array(vals)
        }
        VT_UINT64 => Value::U64(read_u64(r)?),
        VT_INT64 => Value::I64(read_u64(r)? as i64),
        VT_FLOAT64 => Value::F64(f64::from_bits(read_u64(r)?)),
        other => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown GGUF metadata value type {other}"),
            ))
        }
    })
}

/// Skip a value. Kept for callers that only need the tensor table.
pub fn skip_value<R: Read + Seek>(r: &mut R, ty: u32) -> io::Result<()> {
    match ty {
        VT_UINT8 | VT_INT8 | VT_BOOL => {
            r.seek(SeekFrom::Current(1))?;
        }
        VT_UINT16 | VT_INT16 => {
            r.seek(SeekFrom::Current(2))?;
        }
        VT_UINT32 | VT_INT32 | VT_FLOAT32 => {
            r.seek(SeekFrom::Current(4))?;
        }
        VT_UINT64 | VT_INT64 | VT_FLOAT64 => {
            r.seek(SeekFrom::Current(8))?;
        }
        VT_STRING => {
            let len = read_u64(r)?;
            if len > 1 << 40 {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "implausible string"));
            }
            r.seek(SeekFrom::Current(len as i64))?;
        }
        VT_ARRAY => {
            let elem = read_u32(r)?;
            let count = read_u64(r)?;
            for _ in 0..count {
                skip_value(r, elem)?;
            }
        }
        other => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown GGUF metadata value type {other}"),
            ))
        }
    }
    Ok(())
}
