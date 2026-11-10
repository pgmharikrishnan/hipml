// SPDX-License-Identifier: Apache-2.0
//! Batched per-slot SSM launches (WP37).
//!
//! One launch per stage for all `B` slots: the slot index is `grid.y` and every
//! per-slot buffer is contiguous (`slot * stride`), so the per-slot recurrence
//! stops being `B` serial launches of the same kernel. The arithmetic is the
//! single-slot kernel's, unchanged.

use crate::ssm::SsmGpuLayer;
use crate::Launcher;
use hipml_hip::{Buffer, LaunchConfig};
use std::ffi::c_void;

impl<'a> SsmGpuLayer<'a> {
    /// Gates, conv+SiLU, the q|k L2 norms, the delta-net recurrence and the
    /// gated RMS norm for `slots` rows of the shared scratch. `conv_state` and
    /// `s_state` hold `[slots][state]`; this layer's own state buffers are not
    /// touched.
    #[allow(clippy::too_many_arguments)]
    pub fn recurrent_batched(
        &self,
        l: &Launcher,
        qkv: &Buffer,
        z: &Buffer,
        alpha: &Buffer,
        beta: &Buffer,
        g: &Buffer,
        beta_sig: &Buffer,
        conv: &Buffer,
        o: &Buffer,
        gated: &Buffer,
        conv_state: &Buffer,
        s_state: &Buffer,
        slots: usize,
    ) -> Result<(), String> {
        let d = self.dims;
        let stream = l.stream();
        let n_v = d.n_v as i32;
        let vec_stride = d.n_v as i32;
        let inner = d.inner();
        let state_conv = (d.d_conv - 1) * d.conv_ch;
        let state_s = d.n_v * d.head_v * d.head_v;

        // gates
        {
            let k = self.module.kernel("ssm_gates_b")?;
            let mut p_a = alpha.as_ptr();
            let mut p_b = beta.as_ptr();
            let mut p_dt = self.dt_bias.as_ptr();
            let mut p_av = self.a.as_ptr();
            let mut p_g = g.as_ptr();
            let mut p_out = beta_sig.as_ptr();
            let mut v_n = n_v;
            let mut v_stride = vec_stride;
            let mut params: [*mut c_void; 8] = [
                &mut p_a as *mut _ as *mut c_void,
                &mut p_b as *mut _ as *mut c_void,
                &mut p_dt as *mut _ as *mut c_void,
                &mut p_av as *mut _ as *mut c_void,
                &mut p_g as *mut _ as *mut c_void,
                &mut p_out as *mut _ as *mut c_void,
                &mut v_n as *mut _ as *mut c_void,
                &mut v_stride as *mut _ as *mut c_void,
            ];
            let cfg = LaunchConfig {
                grid: ((d.n_v as u32).div_ceil(64), slots as u32, 1),
                block: (64, 1, 1),
                shared_mem_bytes: 0,
            };
            k.launch(cfg, stream, &mut params)?;
        }

        // conv + silu
        {
            let k = self.module.kernel("ssm_conv1d_b")?;
            let mut p_x = qkv.as_ptr();
            let mut p_w = self.conv_w.as_ptr();
            let mut p_state = conv_state.as_ptr();
            let mut p_y = conv.as_ptr();
            let mut v_c = d.conv_ch as i32;
            let mut v_k = d.d_conv as i32;
            let mut v_xs = d.conv_ch as i32;
            let mut v_ss = state_conv as i32;
            let mut params: [*mut c_void; 8] = [
                &mut p_x as *mut _ as *mut c_void,
                &mut p_w as *mut _ as *mut c_void,
                &mut p_state as *mut _ as *mut c_void,
                &mut p_y as *mut _ as *mut c_void,
                &mut v_c as *mut _ as *mut c_void,
                &mut v_k as *mut _ as *mut c_void,
                &mut v_xs as *mut _ as *mut c_void,
                &mut v_ss as *mut _ as *mut c_void,
            ];
            let cfg = LaunchConfig {
                grid: ((d.conv_ch as u32).div_ceil(256), slots as u32, 1),
                block: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            k.launch(cfg, stream, &mut params)?;
        }

        // q|k L2 norm (the two slices are adjacent in the conv buffer)
        {
            let k = self.module.kernel("l2norm_b")?;
            let mut p_x = conv.as_ptr();
            let mut p_y = conv.as_ptr();
            let mut v_h = (2 * d.n_k) as i32;
            let mut v_d = d.head_k as i32;
            let mut v_eps = d.eps;
            let mut v_stride = d.conv_ch as i32;
            let mut params: [*mut c_void; 6] = [
                &mut p_x as *mut _ as *mut c_void,
                &mut p_y as *mut _ as *mut c_void,
                &mut v_h as *mut _ as *mut c_void,
                &mut v_d as *mut _ as *mut c_void,
                &mut v_eps as *mut _ as *mut c_void,
                &mut v_stride as *mut _ as *mut c_void,
            ];
            let cfg = LaunchConfig {
                grid: ((2 * d.n_k) as u32, slots as u32, 1),
                block: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            k.launch(cfg, stream, &mut params)?;
        }

        // delta-net recurrence
        {
            let k = self.module.kernel("gdn_ar_b")?;
            let mut p_s = s_state.as_ptr();
            let mut p_q = conv.as_ptr();
            let mut p_k = unsafe {
                (conv.as_ptr() as *mut u8).add(d.key_dim() * 4) as *mut c_void
            };
            let mut p_v = unsafe {
                (conv.as_ptr() as *mut u8).add(2 * d.key_dim() * 4) as *mut c_void
            };
            let mut p_g = g.as_ptr();
            let mut p_beta = beta_sig.as_ptr();
            let mut p_out = o.as_ptr();
            let mut v_nv = n_v;
            let mut v_nk = d.n_k as i32;
            let mut v_sd = d.head_v as i32;
            let mut v_qkv = d.conv_ch as i32;
            let mut v_vec = vec_stride;
            let mut v_out = inner as i32;
            let mut v_state = state_s as i32;
            let mut params: [*mut c_void; 14] = [
                &mut p_s as *mut _ as *mut c_void,
                &mut p_q as *mut _ as *mut c_void,
                &mut p_k as *mut _ as *mut c_void,
                &mut p_v as *mut _ as *mut c_void,
                &mut p_g as *mut _ as *mut c_void,
                &mut p_beta as *mut _ as *mut c_void,
                &mut p_out as *mut _ as *mut c_void,
                &mut v_nv as *mut _ as *mut c_void,
                &mut v_nk as *mut _ as *mut c_void,
                &mut v_sd as *mut _ as *mut c_void,
                &mut v_qkv as *mut _ as *mut c_void,
                &mut v_vec as *mut _ as *mut c_void,
                &mut v_out as *mut _ as *mut c_void,
                &mut v_state as *mut _ as *mut c_void,
            ];
            let cfg = LaunchConfig {
                grid: (d.n_v as u32, slots as u32, 1),
                block: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            k.launch(cfg, stream, &mut params)?;
        }

        // gated RMS norm
        {
            let k = self.module.kernel("ssm_gate_norm_b")?;
            let mut p_o = o.as_ptr();
            let mut p_norm = self.norm.as_ptr();
            let mut p_z = z.as_ptr();
            let mut p_y = gated.as_ptr();
            let mut v_h = n_v;
            let mut v_d = d.head_v as i32;
            let mut v_eps = d.eps;
            let mut v_stride = inner as i32;
            let mut params: [*mut c_void; 8] = [
                &mut p_o as *mut _ as *mut c_void,
                &mut p_norm as *mut _ as *mut c_void,
                &mut p_z as *mut _ as *mut c_void,
                &mut p_y as *mut _ as *mut c_void,
                &mut v_h as *mut _ as *mut c_void,
                &mut v_d as *mut _ as *mut c_void,
                &mut v_eps as *mut _ as *mut c_void,
                &mut v_stride as *mut _ as *mut c_void,
            ];
            let cfg = LaunchConfig {
                grid: (d.n_v as u32, slots as u32, 1),
                block: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            k.launch(cfg, stream, &mut params)?;
        }
        Ok(())
    }
}
