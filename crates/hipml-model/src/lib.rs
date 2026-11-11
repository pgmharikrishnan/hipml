// SPDX-License-Identifier: Apache-2.0
//! Load every GGUF tensor into VRAM once, then expose the model by name.
//!
//! # Allocation policy
//!
//! One `hipMalloc` per tensor, in GGUF tensor-info order, with the tensor's own
//! handle kept alive for the model's lifetime. No arena: 851 small allocations
//! cost a few milliseconds at load and make ownership, readback and reporting
//! trivial. Uploads happen once, synchronously, before `load` returns, so the
//! residency figure is a settled number and not a queue depth.
//!
//! Quantized payloads are uploaded byte-for-byte (Q4_K, Q5_K, Q6_K, Q8_0) and
//! stay in their on-disk block layout; `F32` is uploaded as raw little-endian
//! f32. Q5_K has no numeric dequant yet, so its bytes are loaded and the tensor
//! is marked `supported() == false` rather than failing the load.

#[cfg(feature = "hip")]
pub mod gpu;
#[cfg(feature = "hip")]
pub mod gpu_weights;
pub mod verify;

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use hipml_core::gguf::Gguf;
use hipml_core::quant::{TensorType, QK_K};
use hipml_hip::ByteBuffer;
use hipml_kernels::Launcher;

/// Embedding matrix: `[n_vocab, n_embd]` as a human shape, Q4_K in this model.
pub const TOKEN_EMBD: &str = "token_embd.weight";
/// Vocabulary projection: `[n_vocab, n_embd]`, Q6_K in this model.
pub const OUTPUT: &str = "output.weight";

/// One tensor resident on the device.
pub struct DeviceTensor {
    name: String,
    dims: Vec<u64>,
    ty: TensorType,
    nbytes: u64,
    supported: bool,
    buffer: ByteBuffer,
}

impl DeviceTensor {
    pub fn name(&self) -> &str {
        &self.name
    }

    /// GGML dimension order, `dims[0]` fastest-varying.
    pub fn dims(&self) -> &[u64] {
        &self.dims
    }

    /// Human shape: `dims` reversed, slowest-varying first.
    pub fn shape(&self) -> Vec<u64> {
        self.dims.iter().rev().copied().collect()
    }

    pub fn ty(&self) -> TensorType {
        self.ty
    }

    pub fn nbytes(&self) -> u64 {
        self.nbytes
    }

    /// False for formats whose bytes load but have no numeric dequant yet.
    pub fn supported(&self) -> bool {
        self.supported
    }

    pub fn device_ptr(&self) -> usize {
        self.buffer.device_ptr()
    }

    /// The resident payload, for a launcher's frozen-weight entry points.
    pub fn buffer(&self) -> &ByteBuffer {
        &self.buffer
    }

    pub fn ne0(&self) -> u64 {
        self.dims.first().copied().unwrap_or(0)
    }

    pub fn rows(&self) -> u64 {
        let ne0 = self.ne0();
        if ne0 == 0 {
            0
        } else {
            self.dims.iter().product::<u64>() / ne0
        }
    }

    /// Copy the resident payload back to host memory.
    pub fn read_back(&self) -> Result<Vec<u8>, String> {
        self.buffer
            .to_host()
            .map_err(|e| format!("{}: readback failed: {e}", self.name))
    }
}

/// Per-dtype residency total.
pub struct TypeTotal {
    pub ty: String,
    pub tensors: usize,
    pub bytes: u64,
}

/// What the load did, with the numbers the acceptance gate asks for.
pub struct LoadReport {
    pub file_bytes: u64,
    pub tensor_count: usize,
    pub tensor_bytes: u64,
    pub free_before: usize,
    pub free_after: usize,
    pub total_device_bytes: usize,
    pub upload_secs: f64,
    pub per_type: Vec<TypeTotal>,
    /// `(name, dtype, bytes)` for tensors loaded but not numerically supported.
    pub unsupported: Vec<(String, String, u64)>,
}

impl LoadReport {
    /// Bytes the model actually took from the device free pool.
    pub fn resident_delta(&self) -> i64 {
        self.free_before as i64 - self.free_after as i64
    }
}

/// A loaded model: tensor table plus the launcher its accessors use.
pub struct Model {
    launcher: Launcher,
    tensors: Vec<DeviceTensor>,
    index: HashMap<String, usize>,
    total_bytes: u64,
    path: std::path::PathBuf,
}

fn supported(ty: TensorType) -> bool {
    // Q5_K has a decode GEMV (`gemv_q5_k`), a batched GEMV and an fp16-copy
    // prefill path; it is a fine-tune tensor type too (`blk.51/55.ffn_down`),
    // so treating it as raw-only was wrong.
    matches!(
        ty,
        TensorType::F32
            | TensorType::Q4K
            | TensorType::Q5K
            | TensorType::Q6K
            | TensorType::Q8_0
            | TensorType::IQ4XS
            | TensorType::IQ4NL
    )
}

fn mem_info() -> Result<(usize, usize), String> {
    #[cfg(feature = "hip")]
    {
        hipml_hip::Device::first()?.mem_info()
    }
    #[cfg(not(feature = "hip"))]
    {
        Err("built without the `hip` feature".to_string())
    }
}

impl Model {
    /// Read, allocate and upload every tensor in `path`, once.
    pub fn load(path: impl AsRef<Path>) -> Result<(Model, LoadReport), String> {
        let path = path.as_ref();
        let mut g = Gguf::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let file_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);

        // Load the code object before allocating 17 GB: a kernel/ABI problem
        // must not leave the card full.
        let launcher = Launcher::new()?;
        let (free_before, _total) = mem_info()?;

        let infos = g.tensors.clone();
        let mut tensors = Vec::with_capacity(infos.len());
        let mut unsupported = Vec::new();
        let mut total_bytes = 0u64;
        let start = Instant::now();

        for info in &infos {
            let raw = g
                .read_tensor(&info.name)
                .map_err(|e| format!("{}: {e}", info.name))?;
            if raw.len() as u64 != info.nbytes {
                return Err(format!(
                    "{}: read {} bytes, tensor info says {}",
                    info.name,
                    raw.len(),
                    info.nbytes
                ));
            }
            let buffer = ByteBuffer::from_host(&raw).map_err(|e| {
                format!("{}: upload of {} bytes failed: {e}", info.name, info.nbytes)
            })?;
            let ok = supported(info.ty);
            if !ok {
                unsupported.push((info.name.clone(), info.ty.name(), info.nbytes));
            }
            total_bytes += info.nbytes;
            tensors.push(DeviceTensor {
                name: info.name.clone(),
                dims: info.dims.clone(),
                ty: info.ty,
                nbytes: info.nbytes,
                supported: ok,
                buffer,
            });
        }
        let upload_secs = start.elapsed().as_secs_f64();
        let (free_after, _total) = mem_info()?;

        let mut index = HashMap::with_capacity(tensors.len());
        for (i, t) in tensors.iter().enumerate() {
            index.insert(t.name.clone(), i);
        }

        // Fixed order so two runs print identical tables.
        let mut per_type = Vec::new();
        for name in [
            "F32", "F16", "Q8_0", "Q6_K", "Q5_K", "Q4_K", "Q3_K", "Q2_K", "Q4_0", "IQ4_XS",
            "IQ4_NL", "IQ3_S",
        ] {
            let mut n = 0usize;
            let mut bytes = 0u64;
            for t in &tensors {
                if t.ty.name() == name {
                    n += 1;
                    bytes += t.nbytes;
                }
            }
            if n > 0 {
                per_type.push(TypeTotal {
                    ty: name.to_string(),
                    tensors: n,
                    bytes,
                });
            }
        }

        let report = LoadReport {
            file_bytes,
            tensor_count: tensors.len(),
            tensor_bytes: total_bytes,
            free_before,
            free_after,
            total_device_bytes: total_bytes as usize,
            upload_secs,
            per_type,
            unsupported,
        };
        Ok((
            Model {
                launcher,
                tensors,
                index,
                total_bytes,
                path: path.to_path_buf(),
            },
            report,
        ))
    }

    /// Upload every tensor of a sidecar GGUF into this model's resident table.
    ///
    /// The sidecar (an MTP / `nextn` block) shares this model's launcher, so
    /// main-model and drafter launches are ordered on one stream and a hidden
    /// state never round-trips through the host. Names already resident (the
    /// sidecar's own `token_embd`, `output` and `output_norm` copies) are
    /// skipped: the drafter must read the target model's embedding and head.
    ///
    /// Returns the bytes uploaded.
    pub fn load_sidecar(&mut self, path: impl AsRef<Path>) -> Result<u64, String> {
        let path = path.as_ref();
        let mut g = Gguf::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let infos = g.tensors.clone();
        let mut added = 0u64;
        for info in &infos {
            if self.index.contains_key(&info.name) {
                continue;
            }
            let raw = g
                .read_tensor(&info.name)
                .map_err(|e| format!("{}: {e}", info.name))?;
            if raw.len() as u64 != info.nbytes {
                return Err(format!(
                    "{}: read {} bytes, tensor info says {}",
                    info.name,
                    raw.len(),
                    info.nbytes
                ));
            }
            let buffer = ByteBuffer::from_host(&raw).map_err(|e| {
                format!("{}: upload of {} bytes failed: {e}", info.name, info.nbytes)
            })?;
            let ok = supported(info.ty);
            self.total_bytes += info.nbytes;
            added += info.nbytes;
            self.index.insert(info.name.clone(), self.tensors.len());
            self.tensors.push(DeviceTensor {
                name: info.name.clone(),
                dims: info.dims.clone(),
                ty: info.ty,
                nbytes: info.nbytes,
                supported: ok,
                buffer,
            });
        }
        Ok(added)
    }

    pub fn tensor(&self, name: &str) -> Option<&DeviceTensor> {
        self.index.get(name).map(|&i| &self.tensors[i])
    }

    pub fn tensors(&self) -> &[DeviceTensor] {
        &self.tensors
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn total_resident_bytes(&self) -> u64 {
        self.total_bytes
    }

    pub fn device_name(&self) -> &str {
        &self.launcher.device_name
    }

    /// The GGUF this model was loaded from (config metadata is re-read lazily).
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    /// The launcher that owns the code object and stream these tensors live on.
    pub fn launcher(&self) -> &Launcher {
        &self.launcher
    }

    /// Read an F32 tensor back as a host vector (norms, conv taps, biases).
    pub fn f32_vec(&self, name: &str) -> Result<Vec<f32>, String> {
        let t = self
            .tensor(name)
            .ok_or_else(|| format!("model has no tensor {name}"))?;
        if t.ty() != TensorType::F32 {
            return Err(format!("{name}: expected F32, found {}", t.ty().name()));
        }
        let raw = t.read_back()?;
        Ok(raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    pub fn device_mem_info(&self) -> Result<(usize, usize), String> {
        mem_info()
    }

    fn require(&self, name: &str) -> Result<&DeviceTensor, String> {
        let t = self
            .tensor(name)
            .ok_or_else(|| format!("model has no tensor {name}"))?;
        if !t.supported() {
            return Err(format!("{name}: {} has no numeric dequant", t.ty().name()));
        }
        Ok(t)
    }

    /// Dequantise one vocabulary row (`token_embd.weight[token, :]`) on device.
    ///
    /// One Q4_K super-block is 144 bytes / 256 values, so a k=5120 row is 20
    /// blocks and 2880 bytes; only those bytes are read from VRAM.
    pub fn embedding_row(&self, token: u32) -> Result<Vec<f32>, String> {
        let t = self.require(TOKEN_EMBD)?;
        if t.ty() != TensorType::Q4K {
            return Err(format!(
                "{TOKEN_EMBD}: expected Q4_K, found {}",
                t.ty().name()
            ));
        }
        let k = t.ne0();
        let rows = t.rows();
        if k as usize % QK_K != 0 {
            return Err(format!("{TOKEN_EMBD}: ne0 {k} is not a multiple of {QK_K}"));
        }
        if token as u64 >= rows {
            return Err(format!(
                "{TOKEN_EMBD}: token {token} out of range (rows {rows})"
            ));
        }
        let blocks = k as usize / QK_K;
        let row_bytes = t.nbytes / rows;
        self.launcher
            .dequant_q4_k(&t.buffer, token as u64 * row_bytes, blocks)
    }

    /// `logits = output.weight @ x` on device, with the weights already resident.
    pub fn lm_head(&self, x: &[f32]) -> Result<(Vec<f32>, f64), String> {
        self.lm_head_reps(x, 1)
    }

    pub fn lm_head_reps(&self, x: &[f32], reps: usize) -> Result<(Vec<f32>, f64), String> {
        let t = self.require(OUTPUT)?;
        let k = t.ne0() as usize;
        if x.len() != k {
            return Err(format!(
                "{OUTPUT}: vector is {} values, row length is {k}",
                x.len()
            ));
        }
        // The head's type is whatever the file stores (Q6_K on the fine-tune,
        // Q8_0 on the stock Q6_K build); pick the matching GEMV entry.
        let entry = match t.ty() {
            TensorType::F32 => "gemv_f32",
            TensorType::Q4K => "gemv_q4_k",
            TensorType::Q5K => "gemv_q5_k",
            TensorType::Q6K => "gemv_q6_k",
            TensorType::Q8_0 => "gemv_q8_0",
            other => return Err(format!("{OUTPUT}: {} has no GEMV entry", other.name())),
        };
        self.launcher
            .run_quant_resident_reps(entry, &t.buffer, x, t.rows() as usize, k, reps)
    }
}
