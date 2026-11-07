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
use hipml_hip::{ffi, Buffer, Kernel, Module, Stream};
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

/// A resident layer: weights, scratch, and the two state buffers.
pub struct SsmGpuLayer<'a> {
    pub(crate) dims: SsmDims,
    pub(crate) stream: &'a Stream,

    pub(crate) k_gemv: Kernel<'a>,
    pub(crate) k_conv: Kernel<'a>,
    pub(crate) k_silu: Kernel<'a>,
    pub(crate) k_l2norm: Kernel<'a>,
    pub(crate) k_gdn: Kernel<'a>,
    pub(crate) k_gate_norm: Kernel<'a>,
    pub(crate) k_gates: Kernel<'a>,

    pub(crate) wqkv: Buffer,
    pub(crate) wgate: Buffer,
    pub(crate) alpha_w: Buffer,
    pub(crate) beta_w: Buffer,
    pub(crate) conv_w: Buffer,
    pub(crate) dt_bias: Buffer,
    pub(crate) a: Buffer,
    pub(crate) norm: Buffer,
    pub(crate) wout: Buffer,

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
            k_gemv: module.kernel("gemv_f32")?,
            k_conv: module.kernel("ssm_conv1d_f32")?,
            k_silu: module.kernel("ssm_silu_inplace_f32")?,
            k_l2norm: module.kernel("l2norm_f32")?,
            k_gdn: module.kernel("gdn_ar_f32")?,
            k_gate_norm: module.kernel("ssm_gate_norm_f32")?,
            k_gates: module.kernel("ssm_gates_f32")?,

            wqkv: Buffer::from_host(w.wqkv)?,
            wgate: Buffer::from_host(w.wgate)?,
            alpha_w: Buffer::from_host(w.alpha)?,
            beta_w: Buffer::from_host(w.beta)?,
            conv_w: Buffer::from_host(w.conv_w)?,
            dt_bias: Buffer::from_host(w.dt_bias)?,
            a: Buffer::from_host(w.a)?,
            norm: Buffer::from_host(w.norm)?,
            wout: Buffer::from_host(w.wout)?,

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

    /// Bytes of resident f32 weights (the quantized loader would be smaller).
    pub fn weight_bytes(&self) -> usize {
        let d = &self.dims;
        // wqkv + (wgate + wout) + conv tap + alpha/beta + dt_bias/a + norm.
        (d.conv_ch * d.n_embd
            + 2 * d.inner() * d.n_embd
            + d.d_conv * d.conv_ch
            + 2 * d.n_v * d.n_embd
            + 2 * d.n_v
            + d.head_v)
            * 4
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
