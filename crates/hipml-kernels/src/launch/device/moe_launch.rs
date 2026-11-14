// SPDX-License-Identifier: Apache-2.0
//! The mixture-of-experts launches: router top-k, the expert gather, the
//! weighted combine and the shared-expert scalar.
//!
//! Every one of these stays on the device. The token graph is captured once and
//! replayed (`gpu/graph.rs`), so a host readback of the selected expert ids per
//! layer would break capture — the ids and weights are device buffers the
//! expert GEMV reads, never host values.

use super::*;

/// Upper bound on the expert count `moe_topk_f32` can hold in shared memory.
/// Mirrors `MOE_TOP_K_MAX` in `moe.hip`; the router refuses a wider bank rather
/// than reading past the array.
pub const MOE_TOP_K_MAX: usize = 1024;

impl Launcher {
    /// Softmax over every expert, top-k, renormalise: `ids` and `weights` are
    /// written on the device and nothing comes back.
    ///
    /// `ids` holds `k` u32 and `weights` `k` f32, both resident across steps.
    pub fn moe_topk(
        &self,
        logits: &Buffer,
        n_expert: usize,
        k: usize,
        ids: &Buffer,
        weights: &Buffer,
    ) -> Result<(), String> {
        if k == 0 || k > n_expert {
            return Err(format!("moe_topk: {k} of {n_expert} experts"));
        }
        if n_expert > MOE_TOP_K_MAX {
            return Err(format!(
                "moe_topk: {n_expert} experts exceeds the kernel's {MOE_TOP_K_MAX} \
                 (MOE_TOP_K_MAX in moe.hip)"
            ));
        }
        let kernel = self.module.kernel("moe_topk_f32")?;
        let mut p_logits = logits.as_ptr();
        let mut p_ids = ids.as_ptr();
        let mut p_w = weights.as_ptr();
        let mut v_n = n_expert as i32;
        let mut v_k = k as i32;
        let mut params: [*mut c_void; 5] = [
            &mut p_logits as *mut _ as *mut c_void,
            &mut v_n as *mut _ as *mut c_void,
            &mut v_k as *mut _ as *mut c_void,
            &mut p_ids as *mut _ as *mut c_void,
            &mut p_w as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (1, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        kernel.launch(cfg, &self.stream, &mut params)
    }

    /// One GEMV per selected expert in one launch: `grid.y` is the top-k rank,
    /// `grid.x` the row tile inside that expert's matrix.
    ///
    /// `w` is the expert bank's device base and `stride` one expert's byte
    /// length, so rank `j` reads `w + ids[j] * stride` — resolved inside the
    /// kernel, because `ids` is only known on the device. `x_stride` is 0 when
    /// every rank reads the same activation row (gate, up) and `k` when each
    /// rank reads its own (down).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_gemv(
        &self,
        w: usize,
        ids: &Buffer,
        stride: usize,
        x: &Buffer,
        x_stride: usize,
        y: &Buffer,
        m: usize,
        k: usize,
        kind: i32,
        n_used: usize,
    ) -> Result<(), String> {
        let kernel = self.module.kernel("moe_gemv")?;
        let mut p_w = w as *mut c_void;
        let mut p_ids = ids.as_ptr();
        let mut v_stride = stride as i64;
        let mut p_x = x.as_ptr();
        let mut v_x_stride = x_stride as i64;
        let mut p_y = y.as_ptr();
        let mut v_m = m as i32;
        let mut v_k = k as i32;
        let mut v_kind = kind;
        let mut params: [*mut c_void; 9] = [
            &mut p_w as *mut _ as *mut c_void,
            &mut p_ids as *mut _ as *mut c_void,
            &mut v_stride as *mut _ as *mut c_void,
            &mut p_x as *mut _ as *mut c_void,
            &mut v_x_stride as *mut _ as *mut c_void,
            &mut p_y as *mut _ as *mut c_void,
            &mut v_m as *mut _ as *mut c_void,
            &mut v_k as *mut _ as *mut c_void,
            &mut v_kind as *mut _ as *mut c_void,
        ];
        let tiles = gemv_job_tiles(kind, m);
        let cfg = LaunchConfig {
            grid: (tiles.max(1) as u32, n_used.max(1) as u32, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        kernel.launch(cfg, &self.stream, &mut params)
    }

    /// `y[i] = ((w[0]*d[0][i] + w[1]*d[1][i]) + …)`, left fold in rank order.
    pub fn moe_combine(
        &self,
        y: &Buffer,
        down: &Buffer,
        weights: &Buffer,
        n: usize,
        n_used: usize,
    ) -> Result<(), String> {
        let kernel = self.module.kernel("moe_combine_f32")?;
        let mut p_y = y.as_ptr();
        let mut p_d = down.as_ptr();
        let mut p_w = weights.as_ptr();
        let mut v_n = n as i32;
        let mut v_u = n_used as i32;
        let mut params: [*mut c_void; 5] = [
            &mut p_y as *mut _ as *mut c_void,
            &mut p_d as *mut _ as *mut c_void,
            &mut p_w as *mut _ as *mut c_void,
            &mut v_n as *mut _ as *mut c_void,
            &mut v_u as *mut _ as *mut c_void,
        ];
        kernel.launch(LaunchConfig::linear(n, 256), &self.stream, &mut params)
    }

    /// `y += shared * scale[0]`: the shared expert's sigmoid, applied once to
    /// the whole vector and added outside the routed sum.
    pub fn moe_shared_add(
        &self,
        y: &Buffer,
        shared: &Buffer,
        scale: &Buffer,
        n: usize,
    ) -> Result<(), String> {
        let kernel = self.module.kernel("moe_shared_add_f32")?;
        let mut p_y = y.as_ptr();
        let mut p_s = shared.as_ptr();
        let mut p_k = scale.as_ptr();
        let mut v_n = n as i32;
        let mut params: [*mut c_void; 4] = [
            &mut p_y as *mut _ as *mut c_void,
            &mut p_s as *mut _ as *mut c_void,
            &mut p_k as *mut _ as *mut c_void,
            &mut v_n as *mut _ as *mut c_void,
        ];
        kernel.launch(LaunchConfig::linear(n, 256), &self.stream, &mut params)
    }

    /// `out = 1 / (1 + exp(-x))`, elementwise. The shared expert's gate is one
    /// dot product, so this runs over one value; it is a kernel so the token
    /// stays inside the captured graph.
    pub fn sigmoid_buf(&self, x: &Buffer, out: &Buffer, n: usize) -> Result<(), String> {
        let kernel = self.module.kernel("sigmoid_f32")?;
        let mut p_x = x.as_ptr();
        let mut p_y = out.as_ptr();
        let mut v_n = n as i32;
        let mut params: [*mut c_void; 3] = [
            &mut p_x as *mut _ as *mut c_void,
            &mut p_y as *mut _ as *mut c_void,
            &mut v_n as *mut _ as *mut c_void,
        ];
        kernel.launch(LaunchConfig::linear(n, 256), &self.stream, &mut params)
    }
}
