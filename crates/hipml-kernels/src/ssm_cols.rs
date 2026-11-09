//! The GDN per-token chain over a chunk of verify columns, in one launch per
//! stage. Split from `ssm_ops.rs` for size; the arithmetic is the single-token
//! kernels' and the row/column index is folded into the block index.

use crate::ssm::SsmGpuLayer;
use hipml_hip::{Buffer, LaunchConfig};
use std::ffi::c_void;

impl<'a> SsmGpuLayer<'a> {
    /// Gates, conv (with SiLU folded in), both L2 norms, the recurrence and the
    /// gated RMS norm over `rows` columns.
    ///
    /// `conv_snap` / `s_snap` receive the carried state after each column but
    /// the last (`[rows-1][state_len]`), or may be null when `rows == 1`. They
    /// are written by the kernels that produce the state, so the verify pass
    /// needs no per-column memcpy.
    #[allow(clippy::too_many_arguments)]
    pub fn recurrent_cols(
        &self,
        rows: usize,
        alpha: &Buffer,
        beta: &Buffer,
        g: &Buffer,
        beta_sig: &Buffer,
        qkv: &Buffer,
        conv: &Buffer,
        o: &Buffer,
        z: &Buffer,
        gated: &Buffer,
        conv_snap: *mut c_void,
        s_snap: *mut c_void,
    ) -> Result<(), String> {
        let d = &self.dims;
        self.gates_cols(rows, alpha, beta, g, beta_sig)?;
        self.conv_cols(rows, qkv, conv, conv_snap)?;
        let base = conv.as_ptr();
        self.l2norm_cols(base, 0, rows)?;
        self.l2norm_cols(base, d.key_dim(), rows)?;
        self.gdn_cols(rows, conv, g, beta_sig, o, s_snap)?;
        self.gate_norm_cols(rows, o, z, gated)
    }

    fn gates_cols(
        &self,
        rows: usize,
        alpha: &Buffer,
        beta: &Buffer,
        g: &Buffer,
        beta_sig: &Buffer,
    ) -> Result<(), String> {
        let d = &self.dims;
        let kernel = self.module.kernel("ssm_gates_cols_f32")?;
        let mut p_alpha = alpha.as_ptr();
        let mut p_beta = beta.as_ptr();
        let mut p_dt = self.dt_bias.as_ptr();
        let mut p_a = self.a.as_ptr();
        let mut p_g = g.as_ptr();
        let mut p_out = beta_sig.as_ptr();
        let mut v_n = d.n_v as i32;
        let mut v_rows = rows as i32;
        let mut params: [*mut c_void; 8] = [
            &mut p_alpha as *mut _ as *mut c_void,
            &mut p_beta as *mut _ as *mut c_void,
            &mut p_dt as *mut _ as *mut c_void,
            &mut p_a as *mut _ as *mut c_void,
            &mut p_g as *mut _ as *mut c_void,
            &mut p_out as *mut _ as *mut c_void,
            &mut v_n as *mut _ as *mut c_void,
            &mut v_rows as *mut _ as *mut c_void,
        ];
        kernel.launch(
            LaunchConfig::linear(rows * d.n_v, 64),
            self.stream,
            &mut params,
        )
    }

    fn conv_cols(
        &self,
        rows: usize,
        qkv: &Buffer,
        conv: &Buffer,
        snap: *mut c_void,
    ) -> Result<(), String> {
        let d = &self.dims;
        let kernel = self.module.kernel("ssm_conv1d_cols_f32")?;
        let mut p_x = qkv.as_ptr();
        let mut p_w = self.conv_w.as_ptr();
        let mut p_state = self.conv_state.as_ptr();
        let mut p_y = conv.as_ptr();
        let mut p_snap = snap;
        let mut v_c = d.conv_ch as i32;
        let mut v_k = d.d_conv as i32;
        let mut v_rows = rows as i32;
        let mut params: [*mut c_void; 8] = [
            &mut p_x as *mut _ as *mut c_void,
            &mut p_w as *mut _ as *mut c_void,
            &mut p_state as *mut _ as *mut c_void,
            &mut p_y as *mut _ as *mut c_void,
            &mut p_snap as *mut _ as *mut c_void,
            &mut v_c as *mut _ as *mut c_void,
            &mut v_k as *mut _ as *mut c_void,
            &mut v_rows as *mut _ as *mut c_void,
        ];
        kernel.launch(
            LaunchConfig::linear(d.conv_ch, 256),
            self.stream,
            &mut params,
        )
    }

    fn l2norm_cols(
        &self,
        base: *mut c_void,
        off: usize,
        rows: usize,
    ) -> Result<(), String> {
        let d = &self.dims;
        let kernel = self.module.kernel("l2norm_cols_f32")?;
        let mut p_x = offset(base, off);
        let mut p_y = p_x;
        let mut v_h = d.n_k as i32;
        let mut v_d = d.head_k as i32;
        let mut v_eps = d.eps;
        let mut v_rows = rows as i32;
        let mut params: [*mut c_void; 6] = [
            &mut p_x as *mut _ as *mut c_void,
            &mut p_y as *mut _ as *mut c_void,
            &mut v_h as *mut _ as *mut c_void,
            &mut v_d as *mut _ as *mut c_void,
            &mut v_eps as *mut _ as *mut c_void,
            &mut v_rows as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: ((rows * d.n_k) as u32, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        kernel.launch(cfg, self.stream, &mut params)
    }

    fn gdn_cols(
        &self,
        rows: usize,
        conv: &Buffer,
        g: &Buffer,
        beta: &Buffer,
        o: &Buffer,
        snap: *mut c_void,
    ) -> Result<(), String> {
        let d = &self.dims;
        let kernel = self.module.kernel("gdn_cols_f32")?;
        let mut p_s = self.s_state.as_ptr();
        let mut p_qkv = conv.as_ptr();
        let mut p_g = g.as_ptr();
        let mut p_beta = beta.as_ptr();
        let mut p_out = o.as_ptr();
        let mut p_snap = snap;
        let mut v_nv = d.n_v as i32;
        let mut v_nk = d.n_k as i32;
        let mut v_stride = d.conv_ch as i32;
        let mut v_key = d.key_dim() as i32;
        let mut v_sd = d.head_v as i32;
        let mut v_rows = rows as i32;
        let mut params: [*mut c_void; 12] = [
            &mut p_s as *mut _ as *mut c_void,
            &mut p_qkv as *mut _ as *mut c_void,
            &mut p_g as *mut _ as *mut c_void,
            &mut p_beta as *mut _ as *mut c_void,
            &mut p_out as *mut _ as *mut c_void,
            &mut p_snap as *mut _ as *mut c_void,
            &mut v_nv as *mut _ as *mut c_void,
            &mut v_nk as *mut _ as *mut c_void,
            &mut v_stride as *mut _ as *mut c_void,
            &mut v_key as *mut _ as *mut c_void,
            &mut v_sd as *mut _ as *mut c_void,
            &mut v_rows as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (d.n_v as u32, 1, 1),
            block: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        kernel.launch(cfg, self.stream, &mut params)
    }

    fn gate_norm_cols(
        &self,
        rows: usize,
        o: &Buffer,
        z: &Buffer,
        gated: &Buffer,
    ) -> Result<(), String> {
        let d = &self.dims;
        let kernel = self.module.kernel("ssm_gate_norm_cols_f32")?;
        let mut p_o = o.as_ptr();
        let mut p_norm = self.norm.as_ptr();
        let mut p_z = z.as_ptr();
        let mut p_y = gated.as_ptr();
        let mut v_h = d.n_v as i32;
        let mut v_d = d.head_v as i32;
        let mut v_eps = d.eps;
        let mut v_rows = rows as i32;
        let mut params: [*mut c_void; 8] = [
            &mut p_o as *mut _ as *mut c_void,
            &mut p_norm as *mut _ as *mut c_void,
            &mut p_z as *mut _ as *mut c_void,
            &mut p_y as *mut _ as *mut c_void,
            &mut v_h as *mut _ as *mut c_void,
            &mut v_d as *mut _ as *mut c_void,
            &mut v_eps as *mut _ as *mut c_void,
            &mut v_rows as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: ((rows * d.n_v) as u32, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        kernel.launch(cfg, self.stream, &mut params)
    }
}

/// Device pointer advanced by `floats` f32s.
fn offset(p: *mut c_void, floats: usize) -> *mut c_void {
    unsafe { (p as *mut u8).add(floats * 4) as *mut c_void }
}
