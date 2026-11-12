//! Windowed form of the GDN column chain: rows `row0 .. row0+rows` of a shared
//! `[M][..]` scratch against an explicit recurrent state.
//!
//! Split from `ssm_cols.rs` for size. This is the batched speculative verify's
//! shape: many slots share one flat activation tile and each slot walks its own
//! contiguous slice in token order, with its own conv window and delta-net
//! matrix. The arithmetic is the single-column chain's, unchanged.

use crate::ssm::SsmGpuLayer;
use hipml_hip::{Buffer, LaunchConfig};
use std::ffi::c_void;

impl<'a> SsmGpuLayer<'a> {
    /// Gates, conv+SiLU, the q|k L2 norms, the delta-net recurrence and the
    /// gated RMS norm over rows `row0 .. row0+rows` of the scratch buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn recurrent_cols_rows(
        &self,
        conv_state: *mut c_void,
        s_state: *mut c_void,
        row0: usize,
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
        snap_layers: usize,
    ) -> Result<(), String> {
        let d = &self.dims;
        let nv = d.n_v;
        let inner = d.inner();
        let cc = d.conv_ch;
        let conv_stride = snap_layers * (d.d_conv - 1) * cc;
        let s_stride = snap_layers * nv * d.head_v * d.head_v;
        let p_alpha = offset(alpha.as_ptr(), row0 * nv);
        let p_beta = offset(beta.as_ptr(), row0 * nv);
        let p_g = offset(g.as_ptr(), row0 * nv);
        let p_bsig = offset(beta_sig.as_ptr(), row0 * nv);
        let p_qkv = offset(qkv.as_ptr(), row0 * cc);
        let p_conv = offset(conv.as_ptr(), row0 * cc);
        let p_o = offset(o.as_ptr(), row0 * inner);
        let p_z = offset(z.as_ptr(), row0 * inner);
        let p_gated = offset(gated.as_ptr(), row0 * inner);

        // gates
        {
            let k = self.module.kernel("ssm_gates_cols_f32")?;
            let mut p_a = p_alpha;
            let mut p_b = p_beta;
            let mut p_dt = self.dt_bias.as_ptr();
            let mut p_av = self.a.as_ptr();
            let mut p_gg = p_g;
            let mut p_out = p_bsig;
            let mut v_n = nv as i32;
            let mut v_rows = rows as i32;
            let mut params: [*mut c_void; 8] = [
                &mut p_a as *mut _ as *mut c_void,
                &mut p_b as *mut _ as *mut c_void,
                &mut p_dt as *mut _ as *mut c_void,
                &mut p_av as *mut _ as *mut c_void,
                &mut p_gg as *mut _ as *mut c_void,
                &mut p_out as *mut _ as *mut c_void,
                &mut v_n as *mut _ as *mut c_void,
                &mut v_rows as *mut _ as *mut c_void,
            ];
            k.launch(
                LaunchConfig::linear(rows * nv, 64),
                self.stream,
                &mut params,
            )?;
        }
        // conv + silu
        {
            let k = self.module.kernel("ssm_conv1d_cols_f32")?;
            let mut p_x = p_qkv;
            let mut p_w = self.conv_w.as_ptr();
            let mut p_state = conv_state;
            let mut p_y = p_conv;
            let mut p_snap = conv_snap;
            let mut v_stride = conv_stride as i32;
            let mut v_c = cc as i32;
            let mut v_kk = d.d_conv as i32;
            let mut v_rows = rows as i32;
            let mut params: [*mut c_void; 9] = [
                &mut p_x as *mut _ as *mut c_void,
                &mut p_w as *mut _ as *mut c_void,
                &mut p_state as *mut _ as *mut c_void,
                &mut p_y as *mut _ as *mut c_void,
                &mut p_snap as *mut _ as *mut c_void,
                &mut v_stride as *mut _ as *mut c_void,
                &mut v_c as *mut _ as *mut c_void,
                &mut v_kk as *mut _ as *mut c_void,
                &mut v_rows as *mut _ as *mut c_void,
            ];
            k.launch(LaunchConfig::linear(cc, 256), self.stream, &mut params)?;
        }
        // q|k L2 norms, in the conv slice's own stride
        for off in [0usize, d.key_dim()] {
            let k = self.module.kernel("l2norm_cols_f32")?;
            let mut p_x = offset(p_conv, off);
            let mut p_y = p_x;
            let mut v_h = d.n_k as i32;
            let mut v_d = d.head_k as i32;
            let mut v_eps = d.eps;
            let mut v_rows = rows as i32;
            let mut v_stride = cc as i32;
            let mut params: [*mut c_void; 7] = [
                &mut p_x as *mut _ as *mut c_void,
                &mut p_y as *mut _ as *mut c_void,
                &mut v_h as *mut _ as *mut c_void,
                &mut v_d as *mut _ as *mut c_void,
                &mut v_eps as *mut _ as *mut c_void,
                &mut v_rows as *mut _ as *mut c_void,
                &mut v_stride as *mut _ as *mut c_void,
            ];
            let cfg = LaunchConfig {
                grid: ((rows * d.n_k) as u32, 1, 1),
                block: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            k.launch(cfg, self.stream, &mut params)?;
        }
        // delta-net recurrence
        {
            let k = self.module.kernel("gdn_cols_f32")?;
            let mut p_s = s_state;
            let mut p_qkv = p_conv;
            let mut p_gg = p_g;
            let mut p_beta = p_bsig;
            let mut p_out = p_o;
            let mut p_snap = s_snap;
            let mut v_stride = s_stride as i32;
            let mut v_nv = nv as i32;
            let mut v_nk = d.n_k as i32;
            let mut v_qkv = cc as i32;
            let mut v_key = d.key_dim() as i32;
            let mut v_sd = d.head_v as i32;
            let mut v_rows = rows as i32;
            let mut params: [*mut c_void; 13] = [
                &mut p_s as *mut _ as *mut c_void,
                &mut p_qkv as *mut _ as *mut c_void,
                &mut p_gg as *mut _ as *mut c_void,
                &mut p_beta as *mut _ as *mut c_void,
                &mut p_out as *mut _ as *mut c_void,
                &mut p_snap as *mut _ as *mut c_void,
                &mut v_stride as *mut _ as *mut c_void,
                &mut v_nv as *mut _ as *mut c_void,
                &mut v_nk as *mut _ as *mut c_void,
                &mut v_qkv as *mut _ as *mut c_void,
                &mut v_key as *mut _ as *mut c_void,
                &mut v_sd as *mut _ as *mut c_void,
                &mut v_rows as *mut _ as *mut c_void,
            ];
            let cfg = LaunchConfig {
                grid: (nv as u32, super::ssm_cols::j_split(d.head_v) as u32, 1),
                block: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            k.launch(cfg, self.stream, &mut params)?;
        }
        self.gate_norm_cols_at(p_o, p_z, p_gated, rows)
    }
}

/// Device pointer advanced by `floats` f32s.
fn offset(p: *mut c_void, floats: usize) -> *mut c_void {
    unsafe { (p as *mut u8).add(floats * 4) as *mut c_void }
}
