//! Row-block SSM stages (WP47).
//!
//! A row block is a concatenation of independent sequences' tails: each tail
//! is a contiguous run of rows with its own slot and position base. The
//! stateless stages (gates, L2 norm, gated RMS) are per row and
//! reuse the chunk kernels over `rows` tokens.
//!
//! The two stages that carry state across tokens — the causal conv and the
//! delta-net recurrence — are launched **once per tail** against that slot's
//! own window/matrix. The batch's rows for a slot are contiguous, so a tail
//! is a pointer base and a token count into the shared chunk buffers; the
//! kernels are the exact single-sequence ones (`ssm_conv1d_chunk_f32`,
//! `gdn_rows_f32`), which keeps every tail bit-identical to `prefill_tile`.
//! Per-tail launches do not duplicate work: the kernels are block-parallel in
//! the token axis, so the total block count matches one wider launch.

use crate::ssm::{SsmChunk, SsmGpuLayer};
use crate::Launcher;
use hipml_hip::LaunchConfig;
use std::ffi::c_void;

impl<'a> SsmGpuLayer<'a> {
    /// Whether the per-tail recurrence is covered: `gdn_rows_f32` keeps 32
    /// state rows of a 128-wide head in registers.
    pub fn rows_supported(&self) -> bool {
        self.dims.head_v == 128 && self.dims.head_k == 128
    }

    /// `g, beta = softplus(alpha + dt) * a, sigmoid(beta)` for every row.
    pub fn gates_rows(&self, l: &Launcher, c: &SsmChunk, rows: usize) -> Result<(), String> {
        let d = &self.dims;
        let _ = l;
        let mut p_alpha = c.alpha.as_ptr();
        let mut p_beta = c.beta.as_ptr();
        let mut p_dt = self.dt_bias.as_ptr();
        let mut p_a = self.a.as_ptr();
        let mut p_g = c.g.as_ptr();
        let mut p_bs = c.beta_sig.as_ptr();
        let mut v_nv = d.n_v as i32;
        let mut v_t = rows as i32;
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
        self.module
            .kernel("ssm_gates_chunk_f32")?
            .launch(LaunchConfig::linear(rows * d.n_v, 256), self.stream, &mut params)
    }

    /// `ssm_conv1d_chunk_f32` on one tail: `tokens` rows at `x`, this slot's
    /// carried window as input, a scratch window as output. The caller copies
    /// the scratch window into the slot's state after the launch (the kernel
    /// reads and writes the window, so the two may not alias).
    pub fn conv_chunk_at(
        &self,
        l: &Launcher,
        x: *mut c_void,
        state_in: *mut c_void,
        state_out: *mut c_void,
        y: *mut c_void,
        tokens: usize,
    ) -> Result<(), String> {
        let d = &self.dims;
        let mut p_x = x;
        let mut p_w = self.conv_w.as_ptr();
        let mut p_si = state_in;
        let mut p_so = state_out;
        let mut p_y = y;
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
        let _ = l;
        self.module
            .kernel("ssm_conv1d_chunk_f32")?
            .launch(LaunchConfig::linear(tokens * d.conv_ch, 256), self.stream, &mut params)
    }

    /// Per-head L2 norm of the q and k halves of every row's conv output.
    pub fn l2norm_rows(&self, l: &Launcher, c: &SsmChunk, rows: usize) -> Result<(), String> {
        let d = &self.dims;
        let mut p_x = c.conv.as_ptr();
        let mut p_y = c.conv.as_ptr();
        let mut v_nk = d.n_k as i32;
        let mut v_hk = d.head_k as i32;
        let mut v_kd = d.key_dim() as i32;
        let mut v_cc = d.conv_ch as i32;
        let mut v_eps = d.eps;
        let mut v_t = rows as i32;
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
            grid: ((2 * d.n_k) as u32, rows as u32, 1),
            block: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let _ = l;
        self.module.kernel("l2norm_chunk_f32")?.launch(cfg, self.stream, &mut params)
    }

    /// `gdn_rows_f32` on one tail: block grid `(n_v, head_v/32)`, the state
    /// rows at `s`, and the tail's rows at the pointer bases. The caller
    /// offsets `s` by the slot and `qkv`/`g`/`beta`/`out` by the tail's first
    /// row.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_rows_at(
        &self,
        l: &Launcher,
        s: *mut c_void,
        qkv: *mut c_void,
        g: *mut c_void,
        beta: *mut c_void,
        out: *mut c_void,
        tokens: usize,
    ) -> Result<(), String> {
        let d = &self.dims;
        let mut p_s = s;
        let mut p_qkv = qkv;
        let mut p_g = g;
        let mut p_b = beta;
        let mut p_o = out;
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
        let _ = l;
        self.module.kernel("gdn_rows_f32")?.launch(cfg, self.stream, &mut params)
    }

    /// Gated RMS norm over every row's recurrence output.
    pub fn gate_norm_rows(&self, l: &Launcher, c: &SsmChunk, rows: usize) -> Result<(), String> {
        let d = &self.dims;
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
            grid: (d.n_v as u32, rows as u32, 1),
            block: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let _ = l;
        self.module
            .kernel("gate_norm_chunk_f32")?
            .launch(cfg, self.stream, &mut params)
    }
}
