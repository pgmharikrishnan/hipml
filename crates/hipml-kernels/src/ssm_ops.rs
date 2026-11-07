//! Kernel launches for one GDN layer, split from `ssm.rs` for size.
//!
//! Every launch writes its arguments out explicitly, matching the existing
//! launchers: `hipModuleLaunchKernel` takes pointers to the arguments, so each
//! value needs a stable home in a local.

use crate::ssm::SsmGpuLayer;
use hipml_hip::{Buffer, LaunchConfig};
use std::ffi::c_void;
use std::time::Instant;

impl<'a> SsmGpuLayer<'a> {
    /// The four input projections.
    pub fn gemvs(&self) -> Result<(), String> {
        let d = &self.dims;
        self.gemv(&self.wqkv, &self.x, &self.qkv, d.conv_ch, d.n_embd)?;
        self.gemv(&self.wgate, &self.x, &self.z, d.inner(), d.n_embd)?;
        self.gemv(&self.alpha_w, &self.x, &self.alpha_raw, d.n_v, d.n_embd)?;
        self.gemv(&self.beta_w, &self.x, &self.beta_raw, d.n_v, d.n_embd)?;
        Ok(())
    }

    /// Everything except the projections: gates, conv, norm, recurrence,
    /// gated RMS norm. This is the compute the recurrent block adds on top of
    /// streaming the weights.
    pub fn recurrent(&self) -> Result<(), String> {
        self.gates()?;
        self.conv()?;
        self.l2norms()?;
        self.recurrence()?;
        self.gate_norm()
    }

    pub fn step_reps(&self, reps: usize) -> Result<f64, String> {
        self.time_reps(reps, || self.step())
    }

    pub fn gemv_reps(&self, reps: usize) -> Result<f64, String> {
        self.time_reps(reps, || self.gemvs())
    }

    pub fn recurrent_reps(&self, reps: usize) -> Result<f64, String> {
        self.time_reps(reps, || self.recurrent())
    }

    pub fn output_gemv_reps(&self, reps: usize) -> Result<f64, String> {
        self.time_reps(reps, || self.output_gemv())
    }

    fn time_reps(&self, reps: usize, f: impl Fn() -> Result<(), String>) -> Result<f64, String> {
        if reps == 0 {
            return Err("reps must be positive".to_string());
        }
        let start = Instant::now();
        for _ in 0..reps {
            f()?;
        }
        self.stream.synchronize()?;
        Ok(start.elapsed().as_secs_f64() / reps as f64)
    }

    fn gemv(&self, w: &Buffer, x: &Buffer, y: &Buffer, m: usize, k: usize) -> Result<(), String> {
        let mut p_w = w.as_ptr();
        let mut p_x = x.as_ptr();
        let mut p_y = y.as_ptr();
        let mut v_m = m as i32;
        let mut v_k = k as i32;
        let mut params: [*mut c_void; 5] = [
            &mut p_w as *mut _ as *mut c_void,
            &mut p_x as *mut _ as *mut c_void,
            &mut p_y as *mut _ as *mut c_void,
            &mut v_m as *mut _ as *mut c_void,
            &mut v_k as *mut _ as *mut c_void,
        ];
        self.k_gemv
            .launch(LaunchConfig::per_row(m, 256), self.stream, &mut params)
    }

    pub(crate) fn output_gemv(&self) -> Result<(), String> {
        let d = &self.dims;
        self.gemv(&self.wout, &self.gated, &self.out, d.n_embd, d.inner())
    }

    fn gates(&self) -> Result<(), String> {
        let d = &self.dims;
        let mut p_alpha = self.alpha_raw.as_ptr();
        let mut p_beta = self.beta_raw.as_ptr();
        let mut p_dt = self.dt_bias.as_ptr();
        let mut p_a = self.a.as_ptr();
        let mut p_g = self.g.as_ptr();
        let mut p_out = self.beta_sig.as_ptr();
        let mut v_n = d.n_v as i32;
        let mut params: [*mut c_void; 7] = [
            &mut p_alpha as *mut _ as *mut c_void,
            &mut p_beta as *mut _ as *mut c_void,
            &mut p_dt as *mut _ as *mut c_void,
            &mut p_a as *mut _ as *mut c_void,
            &mut p_g as *mut _ as *mut c_void,
            &mut p_out as *mut _ as *mut c_void,
            &mut v_n as *mut _ as *mut c_void,
        ];
        self.k_gates
            .launch(LaunchConfig::linear(d.n_v, 64), self.stream, &mut params)
    }

    fn conv(&self) -> Result<(), String> {
        let d = &self.dims;
        // conv writes raw output then silu overwrites it, mirroring the
        // reference's conv-then-activate pair.
        let mut p_x = self.qkv.as_ptr();
        let mut p_w = self.conv_w.as_ptr();
        let mut p_state = self.conv_state.as_ptr();
        let mut p_y = self.conv_out.as_ptr();
        let mut v_c = d.conv_ch as i32;
        let mut v_k = d.d_conv as i32;
        let mut params: [*mut c_void; 6] = [
            &mut p_x as *mut _ as *mut c_void,
            &mut p_w as *mut _ as *mut c_void,
            &mut p_state as *mut _ as *mut c_void,
            &mut p_y as *mut _ as *mut c_void,
            &mut v_c as *mut _ as *mut c_void,
            &mut v_k as *mut _ as *mut c_void,
        ];
        self.k_conv.launch(
            LaunchConfig::linear(d.conv_ch, 256),
            self.stream,
            &mut params,
        )?;

        let mut p_x = self.conv_out.as_ptr();
        let mut v_n = d.conv_ch as i32;
        let mut params: [*mut c_void; 2] = [
            &mut p_x as *mut _ as *mut c_void,
            &mut v_n as *mut _ as *mut c_void,
        ];
        self.k_silu
            .launch(LaunchConfig::linear(d.conv_ch, 256), self.stream, &mut params)
    }

    fn l2norms(&self) -> Result<(), String> {
        let d = &self.dims;
        // In place: the kernel drops __restrict__ on its two pointers.
        self.l2norm_at(0, d.n_k, d.head_k)?;
        self.l2norm_at(d.key_dim(), d.n_k, d.head_k)
    }

    fn l2norm_at(&self, off: usize, heads: usize, dim: usize) -> Result<(), String> {
        let mut p_x = offset(self.conv_out.as_ptr(), off);
        let mut p_y = p_x;
        let mut v_h = heads as i32;
        let mut v_d = dim as i32;
        let mut v_eps = self.dims.eps;
        let mut params: [*mut c_void; 5] = [
            &mut p_x as *mut _ as *mut c_void,
            &mut p_y as *mut _ as *mut c_void,
            &mut v_h as *mut _ as *mut c_void,
            &mut v_d as *mut _ as *mut c_void,
            &mut v_eps as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (heads as u32, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        self.k_l2norm.launch(cfg, self.stream, &mut params)
    }

    fn recurrence(&self) -> Result<(), String> {
        let d = &self.dims;
        let mut p_s = self.s_state.as_ptr();
        let mut p_q = self.conv_out.as_ptr();
        let mut p_k = offset(self.conv_out.as_ptr(), d.key_dim());
        let mut p_v = offset(self.conv_out.as_ptr(), 2 * d.key_dim());
        let mut p_g = self.g.as_ptr();
        let mut p_beta = self.beta_sig.as_ptr();
        let mut p_out = self.o.as_ptr();
        let mut v_nv = d.n_v as i32;
        let mut v_nk = d.n_k as i32;
        let mut v_sd = d.head_v as i32;
        let mut params: [*mut c_void; 10] = [
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
        ];
        let cfg = LaunchConfig {
            grid: (d.n_v as u32, 1, 1),
            block: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        self.k_gdn.launch(cfg, self.stream, &mut params)
    }

    fn gate_norm(&self) -> Result<(), String> {
        let d = &self.dims;
        let mut p_o = self.o.as_ptr();
        let mut p_norm = self.norm.as_ptr();
        let mut p_z = self.z.as_ptr();
        let mut p_y = self.gated.as_ptr();
        let mut v_h = d.n_v as i32;
        let mut v_d = d.head_v as i32;
        let mut v_eps = d.eps;
        let mut params: [*mut c_void; 7] = [
            &mut p_o as *mut _ as *mut c_void,
            &mut p_norm as *mut _ as *mut c_void,
            &mut p_z as *mut _ as *mut c_void,
            &mut p_y as *mut _ as *mut c_void,
            &mut v_h as *mut _ as *mut c_void,
            &mut v_d as *mut _ as *mut c_void,
            &mut v_eps as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (d.n_v as u32, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        self.k_gate_norm.launch(cfg, self.stream, &mut params)
    }
}

/// Device pointer advanced by `floats` f32s, for the q|k|v slices.
fn offset(p: *mut c_void, floats: usize) -> *mut c_void {
    unsafe { (p as *mut u8).add(floats * 4) as *mut c_void }
}
