// SPDX-License-Identifier: Apache-2.0
//! One gated-delta-net layer on the device, end to end.
//!
//! Weights and state stay resident; `step` launches the seven kernels that make
//! a layer in the same order as `delta::ssm_layer_f32`:
//!
//! ```text
//!   qkv,z      = W_qkv x, W_gate x
//!   alpha,beta = W_alpha x, W_beta x
//!   g,beta     = softplus(alpha+dt_bias)*a, sigmoid(beta)
//!   conv       = conv1d(qkv, state) -> silu
//!   q,k        = l2norm(conv[...])
//!   o          = gdn(s, q, k, v, g, beta)
//!   gated      = rmsnorm(o, ssm_norm) * silu(z)
//!   out        = W_out gated
//! ```
//!
//! The launches live in `ssm_ops.rs`; this file owns the buffers and the
//! public surface. Weights here are f32 — the `feat/loader` branch keeps
//! tensors quantized, so the GEMV calls are the one thing that changes when
//! this is wired to it.

use hipml_core::config::Config;
use hipml_core::parity::OpCase;
use hipml_hip::{ffi, Buffer, ByteBuffer, Kernel, Module, Stream};
use std::ffi::c_void;

/// Shapes of one GDN layer, in the units the kernels take.
#[derive(Clone, Copy, Debug)]
pub struct SsmDims {
    pub n_embd: usize,
    /// q | k | v channels fed to the conv1d (2048 + 2048 + 6144 for Qwen3.8-27B).
    pub conv_ch: usize,
    pub d_conv: usize,
    /// Value heads (`ssm_dt_rank`).
    pub n_v: usize,
    /// Key heads (`ssm.group_count`).
    pub n_k: usize,
    pub head_k: usize,
    pub head_v: usize,
    pub eps: f32,
}

impl SsmDims {
    pub fn from_config(cfg: &Config) -> SsmDims {
        SsmDims {
            n_embd: cfg.n_embd,
            conv_ch: cfg.ssm_conv_channels(),
            d_conv: cfg.ssm_d_conv,
            n_v: cfg.ssm_dt_rank,
            n_k: cfg.ssm_n_group,
            head_k: cfg.ssm_head_k(),
            head_v: cfg.ssm_head_v(),
            eps: cfg.eps,
        }
    }

    /// Read the scalars the SSM parity cases declare.
    pub fn from_case(case: &OpCase) -> SsmDims {
        let u = |k: &str| case.scalar(k) as usize;
        SsmDims {
            n_embd: u("n_embd"),
            conv_ch: u("conv_ch"),
            d_conv: u("d_conv"),
            n_v: u("n_v"),
            n_k: u("n_k"),
            head_k: u("head_k"),
            head_v: u("head_v"),
            eps: case.scalar("eps"),
        }
    }

    pub fn from_parity(d: &hipml_core::parity::SsmParityDims) -> SsmDims {
        SsmDims {
            n_embd: d.n_embd,
            conv_ch: d.conv_ch,
            d_conv: d.d_conv,
            n_v: d.n_v,
            n_k: d.n_k,
            head_k: d.head_k,
            head_v: d.head_v,
            eps: d.eps,
        }
    }

    pub fn key_dim(&self) -> usize {
        self.n_k * self.head_k
    }

    pub fn inner(&self) -> usize {
        self.n_v * self.head_v
    }

    /// Floats carried between tokens: conv window + delta-net state.
    pub fn state_floats(&self) -> usize {
        (self.d_conv - 1) * self.conv_ch + self.n_v * self.head_v * self.head_v
    }

    fn validate(&self) -> Result<(), String> {
        if self.conv_ch != 2 * self.key_dim() + self.inner() {
            return Err(format!(
                "conv_ch {} != 2*{} + {}",
                self.conv_ch,
                self.key_dim(),
                self.inner()
            ));
        }
        if self.d_conv < 1 {
            return Err("d_conv must be positive".to_string());
        }
        if self.head_v > 128 || self.head_k > 128 {
            // gdn_ar_f32 holds sd/warpSize values per lane, GDN_MAXV = 4.
            return Err("gdn_ar_f32 supports head dims up to 128".to_string());
        }
        Ok(())
    }
}

/// Chunk-sized scratch for the prefill path: one row per prompt token, so the
/// projections run as one tiled GEMM over the whole chunk and only the
/// recurrence stays per token.
pub struct SsmChunk {
    pub qkv: Buffer,
    pub z: Buffer,
    pub alpha: Buffer,
    pub beta: Buffer,
    pub conv: Buffer,
    pub o: Buffer,
    pub gated: Buffer,
    pub out: Buffer,
    /// Per-token gate and sigmoid(beta), `[tokens][n_v]`; the chunk's
    /// replacement for the layer's one-token scratch.
    pub g: Buffer,
    pub beta_sig: Buffer,
    /// The carried window after this chunk. Written by the conv kernel and
    /// copied into the layer's `conv_state` (the conv reads and writes the
    /// window, so it cannot alias).
    pub conv_next: Buffer,
    /// k-slice partials for the wave-starved SSM alpha/beta projection, at most
    /// seven of `tokens * n_v` (see [`crate::prefill_slices`]).
    pub asplit: Buffer,
    pub tokens: usize,
}

impl SsmChunk {
    pub fn new(d: &SsmDims, tokens: usize) -> Result<SsmChunk, String> {
        let zeros = |n: usize| Buffer::alloc(n);
        Ok(SsmChunk {
            qkv: zeros(tokens * d.conv_ch)?,
            z: zeros(tokens * d.inner())?,
            alpha: zeros(tokens * d.n_v)?,
            beta: zeros(tokens * d.n_v)?,
            conv: zeros(tokens * d.conv_ch)?,
            o: zeros(tokens * d.inner())?,
            gated: zeros(tokens * d.inner())?,
            out: zeros(tokens * d.n_embd)?,
            g: zeros(tokens * d.n_v)?,
            beta_sig: zeros(tokens * d.n_v)?,
            conv_next: zeros((d.d_conv - 1) * d.conv_ch)?,
            asplit: zeros(7 * tokens * d.n_v)?,
            tokens,
        })
    }

    /// Bytes the chunk scratch holds.
    pub fn bytes(&self) -> usize {
        (self.qkv.len() + self.z.len() + self.alpha.len() + self.beta.len() + self.conv.len()
            + self.o.len()
            + self.gated.len()
            + self.out.len()
            + self.g.len()
            + self.beta_sig.len()
            + self.conv_next.len()
            + self.asplit.len())
            * 4
    }
}

/// Borrowed host-side f32 weights, in `SsmW` order.
pub struct SsmLayerRefs<'a> {
    pub wqkv: &'a [f32],
    pub wgate: &'a [f32],
    pub conv_w: &'a [f32],
    pub alpha: &'a [f32],
    pub beta: &'a [f32],
    pub dt_bias: &'a [f32],
    pub a: &'a [f32],
    pub norm: &'a [f32],
    pub wout: &'a [f32],
}

/// A projection weight: f32 (the standalone parity path) or a borrowed GGUF
/// block payload from the resident loader (`feat/loader`).
pub enum SsmWeight<'a> {
    F32(Buffer),
    Quant(&'a ByteBuffer, &'static str),
}

impl SsmWeight<'_> {
    fn bytes(&self) -> usize {
        match self {
            SsmWeight::F32(b) => b.len() * 4,
            SsmWeight::Quant(b, _) => b.len(),
        }
    }
}

/// Borrowed resident weights for one GDN layer.
pub struct SsmResidentRefs<'a, 'b> {
    pub wqkv: (&'a ByteBuffer, &'static str),
    pub wgate: (&'a ByteBuffer, &'static str),
    pub alpha: (&'a ByteBuffer, &'static str),
    pub beta: (&'a ByteBuffer, &'static str),
    pub wout: (&'a ByteBuffer, &'static str),
    pub conv_w: &'b [f32],
    pub dt_bias: &'b [f32],
    pub a: &'b [f32],
    pub norm: &'b [f32],
}

/// A resident layer: weights, scratch, and the two state buffers.
pub struct SsmGpuLayer<'a> {
    pub(crate) dims: SsmDims,
    pub(crate) stream: &'a Stream,
    pub(crate) module: &'a Module,

    pub(crate) k_gemv: Kernel<'a>,
    pub(crate) k_conv: Kernel<'a>,
    pub(crate) k_silu: Kernel<'a>,
    pub(crate) k_l2norm: Kernel<'a>,
    pub(crate) k_gdn: Kernel<'a>,
    pub(crate) k_gate_norm: Kernel<'a>,
    pub(crate) k_gates: Kernel<'a>,

    pub(crate) wqkv: SsmWeight<'a>,
    pub(crate) wgate: SsmWeight<'a>,
    pub(crate) alpha_w: SsmWeight<'a>,
    pub(crate) beta_w: SsmWeight<'a>,
    /// fp16 row-major copies of the f32 alpha/beta weights. The prefill chunk
    /// otherwise has no tensor-core path for these two projections, and the
    /// f32 fallback costs ~3.4 s on a 100K prompt. Only the resident
    /// constructor fills them; the token loop keeps using the f32 weights.
    pub(crate) alpha_half: Option<ByteBuffer>,
    pub(crate) beta_half: Option<ByteBuffer>,
    pub(crate) conv_w: Buffer,
    pub(crate) dt_bias: Buffer,
    pub(crate) a: Buffer,
    pub(crate) norm: Buffer,
    pub(crate) wout: SsmWeight<'a>,

    pub(crate) x: Buffer,
    pub(crate) qkv: Buffer,
    pub(crate) z: Buffer,
    pub(crate) alpha_raw: Buffer,
    pub(crate) beta_raw: Buffer,
    pub(crate) g: Buffer,
    pub(crate) beta_sig: Buffer,
    pub(crate) conv_out: Buffer,
    pub(crate) o: Buffer,
    pub(crate) gated: Buffer,
    pub(crate) out: Buffer,
    pub(crate) conv_state: Buffer,
    pub(crate) s_state: Buffer,
}

impl<'a> SsmGpuLayer<'a> {
    pub fn new(
        module: &'a Module,
        stream: &'a Stream,
        dims: SsmDims,
        w: &SsmLayerRefs<'_>,
    ) -> Result<SsmGpuLayer<'a>, String> {
        dims.validate()?;
        let n_embd = dims.n_embd;
        let inner = dims.inner();
        let check = |name: &str, got: usize, want: usize| {
            if got != want {
                Err(format!("{name}: {} values, expected {want}", got))
            } else {
                Ok(())
            }
        };
        check("wqkv", w.wqkv.len(), dims.conv_ch * n_embd)?;
        check("wgate", w.wgate.len(), inner * n_embd)?;
        check("conv_w", w.conv_w.len(), dims.d_conv * dims.conv_ch)?;
        check("alpha", w.alpha.len(), dims.n_v * n_embd)?;
        check("beta", w.beta.len(), dims.n_v * n_embd)?;
        check("dt_bias", w.dt_bias.len(), dims.n_v)?;
        check("a", w.a.len(), dims.n_v)?;
        check("norm", w.norm.len(), dims.head_v)?;
        check("wout", w.wout.len(), n_embd * inner)?;

        let zeros = |n: usize| Buffer::from_host(&vec![0.0f32; n]);
        Ok(SsmGpuLayer {
            dims,
            stream,
            module,
            k_gemv: module.kernel("gemv_f32")?,
            k_conv: module.kernel("ssm_conv1d_f32")?,
            k_silu: module.kernel("ssm_silu_inplace_f32")?,
            k_l2norm: module.kernel("l2norm_f32")?,
            k_gdn: module.kernel("gdn_ar_f32")?,
            k_gate_norm: module.kernel("ssm_gate_norm_f32")?,
            k_gates: module.kernel("ssm_gates_f32")?,

            wqkv: SsmWeight::F32(Buffer::from_host(w.wqkv)?),
            wgate: SsmWeight::F32(Buffer::from_host(w.wgate)?),
            alpha_w: SsmWeight::F32(Buffer::from_host(w.alpha)?),
            beta_w: SsmWeight::F32(Buffer::from_host(w.beta)?),
            alpha_half: None,
            beta_half: None,
            conv_w: Buffer::from_host(w.conv_w)?,
            dt_bias: Buffer::from_host(w.dt_bias)?,
            a: Buffer::from_host(w.a)?,
            norm: Buffer::from_host(w.norm)?,
            wout: SsmWeight::F32(Buffer::from_host(w.wout)?),

            x: zeros(n_embd)?,
            qkv: zeros(dims.conv_ch)?,
            z: zeros(inner)?,
            alpha_raw: zeros(dims.n_v)?,
            beta_raw: zeros(dims.n_v)?,
            g: zeros(dims.n_v)?,
            beta_sig: zeros(dims.n_v)?,
            conv_out: zeros(dims.conv_ch)?,
            o: zeros(inner)?,
            gated: zeros(inner)?,
            out: zeros(n_embd)?,
            conv_state: zeros((dims.d_conv - 1) * dims.conv_ch)?,
            s_state: zeros(dims.n_v * dims.head_v * dims.head_v)?,
        })
    }

    /// A layer whose five projection matrices are the resident GGUF tensors.
    ///
    /// Only the small per-channel vectors (conv taps, dt bias, A, norm) are
    /// uploaded here; the projections stay in their Q4_K/Q6_K block form and go
    /// through the quantized GEMV.
    pub fn new_resident(
        module: &'a Module,
        stream: &'a Stream,
        dims: SsmDims,
        w: &SsmResidentRefs<'a, '_>,
    ) -> Result<SsmGpuLayer<'a>, String> {
        dims.validate()?;
        let n_embd = dims.n_embd;
        let inner = dims.inner();
        let check = |name: &str, got: usize, want: usize| {
            if got != want {
                Err(format!("{name}: {} values, expected {want}", got))
            } else {
                Ok(())
            }
        };
        check("conv_w", w.conv_w.len(), dims.d_conv * dims.conv_ch)?;
        check("dt_bias", w.dt_bias.len(), dims.n_v)?;
        check("a", w.a.len(), dims.n_v)?;
        check("norm", w.norm.len(), dims.head_v)?;

        let zeros = |n: usize| Buffer::from_host(&vec![0.0f32; n]);
        Ok(SsmGpuLayer {
            dims,
            stream,
            module,
            k_gemv: module.kernel("gemv_f32")?,
            k_conv: module.kernel("ssm_conv1d_f32")?,
            k_silu: module.kernel("ssm_silu_inplace_f32")?,
            k_l2norm: module.kernel("l2norm_f32")?,
            k_gdn: module.kernel("gdn_ar_f32")?,
            k_gate_norm: module.kernel("ssm_gate_norm_f32")?,
            k_gates: module.kernel("ssm_gates_f32")?,

            wqkv: SsmWeight::Quant(w.wqkv.0, w.wqkv.1),
            wgate: SsmWeight::Quant(w.wgate.0, w.wgate.1),
            alpha_w: SsmWeight::Quant(w.alpha.0, w.alpha.1),
            beta_w: SsmWeight::Quant(w.beta.0, w.beta.1),
            alpha_half: f16_copy(w.alpha.0, w.alpha.1)?,
            beta_half: f16_copy(w.beta.0, w.beta.1)?,
            conv_w: Buffer::from_host(w.conv_w)?,
            dt_bias: Buffer::from_host(w.dt_bias)?,
            a: Buffer::from_host(w.a)?,
            norm: Buffer::from_host(w.norm)?,
            wout: SsmWeight::Quant(w.wout.0, w.wout.1),

            x: zeros(n_embd)?,
            qkv: zeros(dims.conv_ch)?,
            z: zeros(inner)?,
            alpha_raw: zeros(dims.n_v)?,
            beta_raw: zeros(dims.n_v)?,
            g: zeros(dims.n_v)?,
            beta_sig: zeros(dims.n_v)?,
            conv_out: zeros(dims.conv_ch)?,
            o: zeros(inner)?,
            gated: zeros(inner)?,
            out: zeros(n_embd)?,
            conv_state: zeros((dims.d_conv - 1) * dims.conv_ch)?,
            s_state: zeros(dims.n_v * dims.head_v * dims.head_v)?,
        })
    }

    pub fn dims(&self) -> SsmDims {
        self.dims
    }

    /// Bytes of per-sequence recurrent state held in VRAM.
    pub fn state_bytes(&self) -> usize {
        self.dims.state_floats() * 4
    }

    /// The four input projections, which the decode step launches first.
    pub fn proj_bytes(&self) -> usize {
        self.wqkv.bytes() + self.wgate.bytes() + self.alpha_w.bytes() + self.beta_w.bytes()
    }

    /// The output projection, launched after the recurrence.
    pub fn step_bytes(&self) -> usize {
        self.wout.bytes()
    }

    /// Bytes of resident projection weights plus the small f32 vectors.
    pub fn weight_bytes(&self) -> usize {
        let d = &self.dims;
        self.wqkv.bytes()
            + self.wgate.bytes()
            + self.wout.bytes()
            + self.alpha_w.bytes()
            + self.beta_w.bytes()
            + (d.d_conv * d.conv_ch + 2 * d.n_v + d.head_v) * 4
    }

    pub fn write_x(&self, x: &[f32]) -> Result<(), String> {
        upload(&self.x, x)
    }

    /// Load the carried state (conv window and delta-net matrix).
    pub fn write_state(&self, conv: &[f32], s: &[f32]) -> Result<(), String> {
        upload(&self.conv_state, conv)?;
        upload(&self.s_state, s)
    }

    pub fn read_out(&self) -> Result<Vec<f32>, String> {
        self.out.to_host()
    }

    pub fn read_state(&self) -> Result<(Vec<f32>, Vec<f32>), String> {
        Ok((self.conv_state.to_host()?, self.s_state.to_host()?))
    }

    /// Lengths of the two carried buffers: `(conv window, delta-net matrix)`.
    /// A prefix snapshot copies exactly these many values.
    pub fn state_lens(&self) -> (usize, usize) {
        let d = &self.dims;
        (
            (d.d_conv - 1) * d.conv_ch,
            d.n_v * d.head_v * d.head_v,
        )
    }

    /// The live conv window buffer (destination for a restore).
    pub fn conv_state_buf(&self) -> &Buffer {
        &self.conv_state
    }

    /// The live delta-net state buffer (destination for a restore).
    pub fn s_state_buf(&self) -> &Buffer {
        &self.s_state
    }

    pub fn write_x_slice(&self, x: &[f32]) -> Result<(), String> {
        upload(&self.x, x)
    }

    /// Scratch the graph normalises into, and the projection output.
    pub fn input(&self) -> &Buffer {
        &self.x
    }

    pub fn output(&self) -> &Buffer {
        &self.out
    }

    /// One token through the layer. State buffers are updated in place.
    pub fn step(&self) -> Result<(), String> {
        self.gemvs()?;
        self.recurrent()?;
        self.output_gemv()
    }

    pub fn synchronize(&self) -> Result<(), String> {
        self.stream.synchronize()
    }
}

/// Synchronous host -> device copy into an existing buffer.
pub(crate) fn upload(buf: &Buffer, data: &[f32]) -> Result<(), String> {
    if data.len() > buf.len() {
        return Err(format!(
            "upload: {} values into a {} element buffer",
            data.len(),
            buf.len()
        ));
    }
    let code = unsafe {
        ffi::hipMemcpy(
            buf.as_ptr(),
            data.as_ptr() as *const c_void,
            data.len() * 4,
            ffi::HIP_MEMCPY_HOST_TO_DEVICE,
        )
    };
    if code != ffi::HIP_SUCCESS {
        return Err(format!("hipMemcpy host->device failed: {code}"));
    }
    let code = unsafe { ffi::hipDeviceSynchronize() };
    if code != ffi::HIP_SUCCESS {
        return Err(format!("hipDeviceSynchronize failed: {code}"));
    }
    Ok(())
}

/// fp16 row-major copy of a raw-f32 resident tensor, for the tensor-core
/// prefill path. `None` for any other entry, so a quantized or absent tensor
/// keeps its normal kernel.
fn f16_copy(w: &ByteBuffer, entry: &str) -> Result<Option<ByteBuffer>, String> {
    if entry != "gemv_f32" {
        return Ok(None);
    }
    let raw = w.to_host()?;
    let mut half = Vec::with_capacity(raw.len() / 2);
    for c in raw.chunks_exact(4) {
        let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        half.extend_from_slice(&hipml_core::kv::f32_to_f16(v).to_le_bytes());
    }
    Ok(Some(ByteBuffer::from_host(&half)?))
}
