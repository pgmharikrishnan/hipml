//! The gated-delta-net layer over a prefill chunk.
//!
//! The five projections are linear in x, so a chunk runs them as tiled GEMMs
//! (one weight stream per chunk instead of per token). Everything after them is
//! a recurrence in time: the conv window and the delta-net state both advance
//! one token at a time, so those launches stay per token, reading row `t` of the
//! chunk buffers. The maths is the same chain `ssm_ops` runs for one token.

use crate::ssm::{SsmChunk, SsmGpuLayer, SsmWeight};
use crate::Launcher;
use hipml_hip::Buffer;
use std::ffi::c_void;

impl<'a> SsmGpuLayer<'a> {
    /// `qkv, z, alpha, beta = W x` for every row of the chunk.
    pub fn project_chunk(
        &self,
        l: &Launcher,
        x: &Buffer,
        c: &SsmChunk,
        tokens: usize,
    ) -> Result<(), String> {
        let d = &self.dims;
        let plan = [
            (&self.wqkv, &c.qkv, d.conv_ch),
            (&self.wgate, &c.z, d.inner()),
            (&self.alpha_w, &c.alpha, d.n_v),
            (&self.beta_w, &c.beta, d.n_v),
        ];
        for (w, y, m) in plan {
            // The loader keeps F32 tensors (ssm_alpha, ssm_beta in Qwen3.8-27B) as raw
            // little-endian f32, so those go through the f32 multi-row GEMM.
            match w {
                SsmWeight::Quant(b, "gemv_f32") => l.run_gemm_ptr(
                    b.as_ptr(),
                    x.as_ptr(),
                    y.as_ptr(),
                    tokens,
                    m,
                    d.n_embd,
                )?,
                SsmWeight::Quant(b, e) => crate::prefill_resident(
                    l.module(),
                    l.stream(),
                    prefill_entry(e)?,
                    b,
                    x,
                    y,
                    m,
                    d.n_embd,
                    tokens,
                )?,
                SsmWeight::F32(_) => {
                    return Err("prefill chunk needs resident weights".to_string())
                }
            }
        }
        Ok(())
    }

    /// One token of the chunk: gates, conv, L2 norms, recurrence, gated RMS.
    pub fn step_chunk_row(&self, c: &SsmChunk, t: usize) -> Result<(), String> {
        let d = &self.dims;
        let qkv = row_ptr(&c.qkv, t, d.conv_ch);
        let alpha = row_ptr(&c.alpha, t, d.n_v);
        let beta = row_ptr(&c.beta, t, d.n_v);
        let z = row_ptr(&c.z, t, d.inner());
        let conv = row_ptr(&c.conv, t, d.conv_ch);
        let o = row_ptr(&c.o, t, d.inner());
        let gated = row_ptr(&c.gated, t, d.inner());

        // `g` and `beta_sig` are the layer's 64-value per-token scratch.
        self.gates_at(alpha, beta, self.g.as_ptr(), self.beta_sig.as_ptr())?;
        self.conv_at(qkv, conv)?;
        self.l2norm_at_ptr(conv, 0, d.n_k, d.head_k)?;
        self.l2norm_at_ptr(conv, d.key_dim(), d.n_k, d.head_k)?;
        self.gdn_at(conv, self.g.as_ptr(), self.beta_sig.as_ptr(), o)?;
        self.gate_norm_at(o, z, gated)
    }

    /// `out[t, :] = W_out gated[t, :]` for the whole chunk.
    pub fn out_chunk(
        &self,
        l: &Launcher,
        c: &SsmChunk,
        tokens: usize,
    ) -> Result<(), String> {
        let d = &self.dims;
        match &self.wout {
            SsmWeight::Quant(b, "gemv_f32") => l.run_gemm_ptr(
                b.as_ptr(),
                c.gated.as_ptr(),
                c.out.as_ptr(),
                tokens,
                d.n_embd,
                d.inner(),
            ),
            SsmWeight::Quant(b, e) => crate::prefill_resident(
                l.module(),
                l.stream(),
                prefill_entry(e)?,
                b,
                &c.gated,
                &c.out,
                d.n_embd,
                d.inner(),
                tokens,
            ),
            SsmWeight::F32(_) => Err("prefill chunk needs resident weights".to_string()),
        }
    }
}

/// The tiled GEMM entry for a quantized GEMV entry (`gemv_q4_k` ->
/// `prefill_q4_k`); the two families share the block format.
fn prefill_entry(gemv: &str) -> Result<&'static str, String> {
    match gemv {
        "gemv_q4_k" => Ok("prefill_q4_k"),
        "gemv_q6_k" => Ok("prefill_q6_k"),
        "gemv_q8_0" => Ok("prefill_q8_0"),
        other => Err(format!("{other}: no tiled prefill kernel")),
    }
}

fn row_ptr(buf: &Buffer, t: usize, width: usize) -> *mut c_void {
    unsafe { (buf.as_ptr() as *mut u8).add(t * width * 4) as *mut c_void }
}
