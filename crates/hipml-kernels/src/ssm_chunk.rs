// SPDX-License-Identifier: Apache-2.0
//! The gated-delta-net layer over a prefill chunk.
//!
//! The five projections are linear in x, so a chunk runs them as tiled GEMMs
//! (one weight stream per chunk instead of per token). The stage after them —
//! gates, conv, L2, recurrence, gated RMS — also runs once for the whole chunk:
//! `steps_chunk` launches five kernels, and the recurrence holds each head's
//! state matrix in registers for every token of the tile. `step_chunk_row`
//! remains as the per-token reference the chunk kernels mirror.

use crate::ssm::{SsmChunk, SsmGpuLayer, SsmWeight};
use crate::Launcher;
use hipml_hip::{Buffer, LaunchConfig};
use std::ffi::c_void;

impl<'a> SsmGpuLayer<'a> {
    /// `qkv, z, alpha, beta = W x` for every row of the chunk.
    ///
    /// `xh` is scratch for an fp16 copy of the activation tile (see
    /// [`crate::prefill_xhalf`]); it is only touched when the active prefill
    /// path stages converted activations.
    pub fn project_chunk(
        &self,
        l: &Launcher,
        x: &Buffer,
        xh: &Buffer,
        c: &SsmChunk,
        tokens: usize,
    ) -> Result<(), String> {
        let d = &self.dims;
        let half = crate::prefill_has_xhalf();
        if half {
            crate::prefill_xhalf(l.module(), l.stream(), x, xh, tokens * d.n_embd)?;
        }
        let plan = [
            (&self.wqkv, &c.qkv, d.conv_ch, None),
            (&self.wgate, &c.z, d.inner(), None),
            (&self.alpha_w, &c.alpha, d.n_v, self.alpha_half.as_ref()),
            (&self.beta_w, &c.beta, d.n_v, self.beta_half.as_ref()),
        ];
        for (w, y, m, half_w) in plan {
            // The loader keeps F32 tensors (ssm_alpha, ssm_beta in Qwen3.8-27B)
            // as raw
            // little-endian f32, so those go through the f32 multi-row GEMM. The
            // resident loader also leaves an fp16 copy, which rides the
            // tensor-core path with the converted activation tile, k-sliced
            // because 64 rows is only 8 blocks of a 64-block wave.
            match w {
                SsmWeight::Quant(b, "gemv_f32") => match (half, half_w) {
                    (true, Some(hw)) => crate::prefill_resident_slices(
                        l.module(),
                        l.stream(),
                        "prefill_halfw_w2h",
                        hw,
                        xh,
                        y,
                        &c.asplit,
                        m,
                        d.n_embd,
                        tokens,
                    )?,
                    _ => crate::prefill_resident(
                        l.module(),
                        l.stream(),
                        "prefill_f32",
                        b,
                        x,
                        y,
                        m,
                        d.n_embd,
                        tokens,
                    )?,
                },
                SsmWeight::Quant(b, e) => {
                    let entry = prefill_entry(e)?;
                    if half {
                        crate::prefill_resident_half(
                            l.module(),
                            l.stream(),
                            entry,
                            b,
                            xh,
                            y,
                            m,
                            d.n_embd,
                            tokens,
                        )?
                    } else {
                        crate::prefill_resident(
                            l.module(),
                            l.stream(),
                            entry,
                            b,
                            x,
                            y,
                            m,
                            d.n_embd,
                            tokens,
                        )?
                    }
                }
                SsmWeight::F32(_) => return Err("prefill chunk needs resident weights".to_string()),
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

    /// The whole recurrent chain for the chunk: six launches for the layer
    /// (gates, conv, the carried window copy, L2, recurrence, gated RMS)
    /// instead of `tokens * 6` of them.
    pub fn steps_chunk(&self, c: &SsmChunk, tokens: usize) -> Result<(), String> {
        let d = &self.dims;
        let s = self.stream;

        // gates
        {
            let mut p_alpha = c.alpha.as_ptr();
            let mut p_beta = c.beta.as_ptr();
            let mut p_dt = self.dt_bias.as_ptr();
            let mut p_a = self.a.as_ptr();
            let mut p_g = c.g.as_ptr();
            let mut p_bs = c.beta_sig.as_ptr();
            let mut v_nv = d.n_v as i32;
            let mut v_t = tokens as i32;
            let mut params: [*mut c_void; 8] = [
                &mut p_alpha as *mut _ as *mut c_void,
                &mut p_beta as *mut _ as *mut c_void,
                &mut p_dt as *mut _ as *mut c_void,
                &mut p_a as *mut _ as *mut c_void,
                &mut p_g as *mut _ as *mut c_void,
                &mut p_bs as *mut _ as *mut c_void,
                &mut v_nv as *mut _ as *mut c_void,
                &mut v_t as *mut _ as *mut c_void,
            ];
            self.module.kernel("ssm_gates_chunk_f32")?.launch(
                LaunchConfig::linear(tokens * d.n_v, 256),
                s,
                &mut params,
            )?;
        }

        // conv + silu, carrying the window in a separate buffer
        {
            let mut p_x = c.qkv.as_ptr();
            let mut p_w = self.conv_w.as_ptr();
            let mut p_si = self.conv_state.as_ptr();
            let mut p_so = c.conv_next.as_ptr();
            let mut p_y = c.conv.as_ptr();
            let mut v_c = d.conv_ch as i32;
            let mut v_k = d.d_conv as i32;
            let mut v_t = tokens as i32;
            let mut params: [*mut c_void; 8] = [
                &mut p_x as *mut _ as *mut c_void,
                &mut p_w as *mut _ as *mut c_void,
                &mut p_si as *mut _ as *mut c_void,
                &mut p_so as *mut _ as *mut c_void,
                &mut p_y as *mut _ as *mut c_void,
                &mut v_c as *mut _ as *mut c_void,
                &mut v_k as *mut _ as *mut c_void,
                &mut v_t as *mut _ as *mut c_void,
            ];
            self.module.kernel("ssm_conv1d_chunk_f32")?.launch(
                LaunchConfig::linear(tokens * d.conv_ch, 256),
                s,
                &mut params,
            )?;
        }
        {
            let n = (d.d_conv - 1) * d.conv_ch;
            let mut p_src = c.conv_next.as_ptr();
            let mut p_dst = self.conv_state.as_ptr();
            let mut v_n = n as i32;
            let mut params: [*mut c_void; 3] = [
                &mut p_src as *mut _ as *mut c_void,
                &mut p_dst as *mut _ as *mut c_void,
                &mut v_n as *mut _ as *mut c_void,
            ];
            self.module
                .kernel("copy_f32")?
                .launch(LaunchConfig::linear(n, 256), s, &mut params)?;
        }

        // per-head L2 of q and k, both halves in one launch
        {
            let mut p_x = c.conv.as_ptr();
            let mut p_y = c.conv.as_ptr();
            let mut v_nk = d.n_k as i32;
            let mut v_hk = d.head_k as i32;
            let mut v_kd = d.key_dim() as i32;
            let mut v_cc = d.conv_ch as i32;
            let mut v_eps = d.eps;
            let mut v_t = tokens as i32;
            let mut params: [*mut c_void; 8] = [
                &mut p_x as *mut _ as *mut c_void,
                &mut p_y as *mut _ as *mut c_void,
                &mut v_nk as *mut _ as *mut c_void,
                &mut v_hk as *mut _ as *mut c_void,
                &mut v_kd as *mut _ as *mut c_void,
                &mut v_cc as *mut _ as *mut c_void,
                &mut v_eps as *mut _ as *mut c_void,
                &mut v_t as *mut _ as *mut c_void,
            ];
            let cfg = LaunchConfig {
                grid: ((2 * d.n_k) as u32, tokens as u32, 1),
                block: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            self.module
                .kernel("l2norm_chunk_f32")?
                .launch(cfg, s, &mut params)?;
        }

        // the recurrence, state in registers: 32-row blocks with the operands
        // staged in shared when the head is 128 wide, else one block per head
        let rows = d.head_v == 128 && d.head_k == 128;
        if rows && std::env::var_os("HIPML_GDN_HEAD").is_none() {
            self.gdn_rows(c, tokens)?;
        } else {
            let mut p_s = self.s_state.as_ptr();
            let mut p_qkv = c.conv.as_ptr();
            let mut p_g = c.g.as_ptr();
            let mut p_b = c.beta_sig.as_ptr();
            let mut p_o = c.o.as_ptr();
            let mut v_nv = d.n_v as i32;
            let mut v_nk = d.n_k as i32;
            let mut v_sd = d.head_v as i32;
            let mut v_cc = d.conv_ch as i32;
            let mut v_kd = d.key_dim() as i32;
            let mut v_t = tokens as i32;
            let mut params: [*mut c_void; 11] = [
                &mut p_s as *mut _ as *mut c_void,
                &mut p_qkv as *mut _ as *mut c_void,
                &mut p_g as *mut _ as *mut c_void,
                &mut p_b as *mut _ as *mut c_void,
                &mut p_o as *mut _ as *mut c_void,
                &mut v_nv as *mut _ as *mut c_void,
                &mut v_nk as *mut _ as *mut c_void,
                &mut v_sd as *mut _ as *mut c_void,
                &mut v_cc as *mut _ as *mut c_void,
                &mut v_kd as *mut _ as *mut c_void,
                &mut v_t as *mut _ as *mut c_void,
            ];
            let cfg = LaunchConfig {
                grid: (d.n_v as u32, 1, 1),
                block: (1024, 1, 1),
                shared_mem_bytes: 0,
            };
            self.module
                .kernel("gdn_chunk_f32")?
                .launch(cfg, s, &mut params)?;
        }

        // gated RMS norm
        {
            let mut p_o = c.o.as_ptr();
            let mut p_n = self.norm.as_ptr();
            let mut p_z = c.z.as_ptr();
            let mut p_y = c.gated.as_ptr();
            let mut v_nv = d.n_v as i32;
            let mut v_d = d.head_v as i32;
            let mut v_eps = d.eps;
            let mut params: [*mut c_void; 7] = [
                &mut p_o as *mut _ as *mut c_void,
                &mut p_n as *mut _ as *mut c_void,
                &mut p_z as *mut _ as *mut c_void,
                &mut p_y as *mut _ as *mut c_void,
                &mut v_nv as *mut _ as *mut c_void,
                &mut v_d as *mut _ as *mut c_void,
                &mut v_eps as *mut _ as *mut c_void,
            ];
            let cfg = LaunchConfig {
                grid: (d.n_v as u32, tokens as u32, 1),
                block: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            self.module
                .kernel("gate_norm_chunk_f32")?
                .launch(cfg, s, &mut params)?;
        }
        Ok(())
    }

    /// `gdn_rows_f32`: the chunk recurrence split into 32-row blocks of a
    /// head, bit-identical to `gdn_chunk_f32` (`HIPML_GDN_HEAD=1` selects that).
    fn gdn_rows(&self, c: &SsmChunk, tokens: usize) -> Result<(), String> {
        let d = &self.dims;
        let mut p_s = self.s_state.as_ptr();
        let mut p_qkv = c.conv.as_ptr();
        let mut p_g = c.g.as_ptr();
        let mut p_b = c.beta_sig.as_ptr();
        let mut p_o = c.o.as_ptr();
        let mut v_nv = d.n_v as i32;
        let mut v_nk = d.n_k as i32;
        let mut v_cc = d.conv_ch as i32;
        let mut v_kd = d.key_dim() as i32;
        let mut v_t = tokens as i32;
        let mut params: [*mut c_void; 10] = [
            &mut p_s as *mut _ as *mut c_void,
            &mut p_qkv as *mut _ as *mut c_void,
            &mut p_g as *mut _ as *mut c_void,
            &mut p_b as *mut _ as *mut c_void,
            &mut p_o as *mut _ as *mut c_void,
            &mut v_nv as *mut _ as *mut c_void,
            &mut v_nk as *mut _ as *mut c_void,
            &mut v_cc as *mut _ as *mut c_void,
            &mut v_kd as *mut _ as *mut c_void,
            &mut v_t as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (d.n_v as u32, (d.head_v / 32) as u32, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        self.module
            .kernel("gdn_rows_f32")?
            .launch(cfg, self.stream, &mut params)
    }

    /// `out[t, :] = W_out gated[t, :]` for the whole chunk.
    pub fn out_chunk(
        &self,
        l: &Launcher,
        xh: &Buffer,
        c: &SsmChunk,
        tokens: usize,
    ) -> Result<(), String> {
        let d = &self.dims;
        let entry = match &self.wout {
            SsmWeight::Quant(_, "gemv_f32") => None,
            SsmWeight::Quant(_, e) => Some(prefill_entry(e)?),
            SsmWeight::F32(_) => None,
        };
        let half = entry.is_some() && crate::prefill_has_xhalf();
        if half {
            crate::prefill_xhalf(l.module(), l.stream(), &c.gated, xh, tokens * d.inner())?;
        }
        match &self.wout {
            SsmWeight::Quant(b, "gemv_f32") => crate::prefill_resident(
                l.module(),
                l.stream(),
                "prefill_f32",
                b,
                &c.gated,
                &c.out,
                d.n_embd,
                d.inner(),
                tokens,
            ),
            SsmWeight::Quant(b, _) => {
                let e = entry.expect("checked above");
                if half {
                    crate::prefill_resident_half(
                        l.module(),
                        l.stream(),
                        e,
                        b,
                        xh,
                        &c.out,
                        d.n_embd,
                        d.inner(),
                        tokens,
                    )
                } else {
                    crate::prefill_resident(
                        l.module(),
                        l.stream(),
                        e,
                        b,
                        &c.gated,
                        &c.out,
                        d.n_embd,
                        d.inner(),
                        tokens,
                    )
                }
            }
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
