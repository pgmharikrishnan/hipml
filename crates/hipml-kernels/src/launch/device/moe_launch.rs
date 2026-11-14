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

/// Output rows per block of the grouped expert GEMM. Mirrors `MG_RT` in
/// `prefill_moe.hip`.
pub const MOE_PREFILL_ROWS: usize = 128;

/// Prompt tokens (slots) per block of the grouped expert GEMM. Mirrors `MG_TT`.
/// An expert's run is padded up to a multiple of this, so it is also the number
/// the bucketing diagnostic uses to price the padding.
///
/// 64 is measured, not assumed: the sweep in `docs/notes/moe.md` §8.6 runs the
/// same chunk at 16, 32, 64 and 128 slots a tile, and 64 is the fastest at both
/// 511 and 8,184 tokens. A shorter tile cuts the padding — 3.68x to 1.48x at
/// 511 tokens — and is 28% slower, because the cost of a tile is the 128 rows
/// of weights it stages and dequantises, not the slots it computes.
pub const MOE_PREFILL_TOKENS: usize = 64;

/// The slot tiles the padding table prices a run-length distribution at. The
/// shipped height is `MOE_PREFILL_TOKENS`; the others are there so a reader can
/// see what the choice was between.
pub const MOE_PREFILL_TILES: [usize; 3] = [16, 32, 64];

/// The padding a run-length table pays at a given tile height:
/// `(padded slots, tiles)`.
pub fn moe_prefill_padding(runs: &[u32], tile: usize) -> (usize, usize) {
    let padded = runs
        .iter()
        .map(|&r| (r as usize).div_ceil(tile) * tile)
        .sum();
    let tiles = runs.iter().map(|&r| (r as usize).div_ceil(tile)).sum();
    (padded, tiles)
}

/// Threads per block of the grouped expert GEMM. Mirrors `MG_THREADS`.
pub const MOE_PREFILL_THREADS: usize = 256;

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

    /// The same gather for every slot at once: `grid.z` is the slot, so one
    /// launch serves the batched decode step instead of one per slot.
    ///
    /// `ids` is `[slots][n_used]`, `x` is `[slots][x_slot]` with rank `j`
    /// reading `x + slot*x_slot + j*x_stride`, and `y` is `[slots][y_slot]`.
    /// The tile body is [`Launcher::moe_gemv`]'s, so slot `s` is bit-identical
    /// to the single-sequence launch on that slot's row.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_gemv_batch(
        &self,
        w: usize,
        ids: &Buffer,
        stride: usize,
        x: &Buffer,
        x_stride: usize,
        x_slot: usize,
        y: &Buffer,
        y_slot: usize,
        m: usize,
        k: usize,
        kind: i32,
        n_used: usize,
        slots: usize,
    ) -> Result<(), String> {
        let kernel = self.module.kernel("moe_gemv_batch")?;
        let mut p_w = w as *mut c_void;
        let mut p_ids = ids.as_ptr();
        let mut v_stride = stride as i64;
        let mut p_x = x.as_ptr();
        let mut v_x_stride = x_stride as i64;
        let mut v_x_slot = x_slot as i64;
        let mut p_y = y.as_ptr();
        let mut v_y_slot = y_slot as i64;
        let mut v_m = m as i32;
        let mut v_k = k as i32;
        let mut v_kind = kind;
        let mut params: [*mut c_void; 11] = [
            &mut p_w as *mut _ as *mut c_void,
            &mut p_ids as *mut _ as *mut c_void,
            &mut v_stride as *mut _ as *mut c_void,
            &mut p_x as *mut _ as *mut c_void,
            &mut v_x_stride as *mut _ as *mut c_void,
            &mut v_x_slot as *mut _ as *mut c_void,
            &mut p_y as *mut _ as *mut c_void,
            &mut v_y_slot as *mut _ as *mut c_void,
            &mut v_m as *mut _ as *mut c_void,
            &mut v_k as *mut _ as *mut c_void,
            &mut v_kind as *mut _ as *mut c_void,
        ];
        let tiles = gemv_job_tiles(kind, m);
        let cfg = LaunchConfig {
            grid: (
                tiles.max(1) as u32,
                n_used.max(1) as u32,
                slots.max(1) as u32,
            ),
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

    /// Softmax over every expert, top-k, renormalise, for a whole prompt tile:
    /// one block per token, `logits` is `[n_tokens][n_expert]`.
    ///
    /// Same body as [`Launcher::moe_topk`], so the chunked path and the decode
    /// path cannot disagree on a convention.
    pub fn moe_topk_rows(
        &self,
        logits: &Buffer,
        n_expert: usize,
        k: usize,
        n_tokens: usize,
        ids: &Buffer,
        weights: &Buffer,
    ) -> Result<(), String> {
        if k == 0 || k > n_expert {
            return Err(format!("moe_topk_rows: {k} of {n_expert} experts"));
        }
        if n_expert > MOE_TOP_K_MAX {
            return Err(format!(
                "moe_topk_rows: {n_expert} experts exceeds the kernel's {MOE_TOP_K_MAX}"
            ));
        }
        let kernel = self.module.kernel("moe_topk_rows_f32")?;
        let mut p_logits = logits.as_ptr();
        let mut v_n = n_expert as i32;
        let mut v_k = k as i32;
        let mut p_ids = ids.as_ptr();
        let mut p_w = weights.as_ptr();
        let mut params: [*mut c_void; 5] = [
            &mut p_logits as *mut _ as *mut c_void,
            &mut v_n as *mut _ as *mut c_void,
            &mut v_k as *mut _ as *mut c_void,
            &mut p_ids as *mut _ as *mut c_void,
            &mut p_w as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (n_tokens.max(1) as u32, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        kernel.launch(cfg, &self.stream, &mut params)
    }

    /// Counting sort of a tile's routing decisions into per-expert runs.
    ///
    /// `off` holds `n_expert + 1` u32 offsets, `xtok` and `yslot` hold
    /// `n_tokens * n_used` entries each. One block: the bank is at most
    /// `MOE_TOP_K_MAX` wide and the slot list is at most a chunk's worth.
    pub fn moe_bucket(
        &self,
        ids: &Buffer,
        n_tokens: usize,
        n_used: usize,
        n_expert: usize,
        off: &Buffer,
        xtok: &Buffer,
        yslot: &Buffer,
    ) -> Result<(), String> {
        if n_expert > MOE_TOP_K_MAX {
            return Err(format!(
                "moe_bucket: {n_expert} experts exceeds the kernel's {MOE_TOP_K_MAX}"
            ));
        }
        let kernel = self.module.kernel("moe_bucket_f32")?;
        let mut p_ids = ids.as_ptr();
        let mut v_tok = n_tokens as i32;
        let mut v_used = n_used as i32;
        let mut v_exp = n_expert as i32;
        let mut p_off = off.as_ptr();
        let mut p_xtok = xtok.as_ptr();
        let mut p_yslot = yslot.as_ptr();
        let mut params: [*mut c_void; 7] = [
            &mut p_ids as *mut _ as *mut c_void,
            &mut v_tok as *mut _ as *mut c_void,
            &mut v_used as *mut _ as *mut c_void,
            &mut v_exp as *mut _ as *mut c_void,
            &mut p_off as *mut _ as *mut c_void,
            &mut p_xtok as *mut _ as *mut c_void,
            &mut p_yslot as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (1, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        kernel.launch(cfg, &self.stream, &mut params)
    }

    /// One launch for every expert of a bank: the grouped GEMM of a chunk's
    /// routed FFN. `blockIdx.y` is the expert, `off[e]..off[e+1]` its run of
    /// slots, `xmap`/`ymap` the activation and output rows of a slot.
    ///
    /// `silu` selects the `up` projection's SwiGLU epilogue (`gate` and `yh`
    /// are read/written only then).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_prefill_gemm(
        &self,
        w: usize,
        stride: usize,
        off: &Buffer,
        x: &Buffer,
        xmap: &Buffer,
        y: &Buffer,
        ymap: &Buffer,
        gate: &Buffer,
        yh: &Buffer,
        m: usize,
        k: usize,
        n_expert: usize,
        kind: i32,
        silu: bool,
    ) -> Result<(), String> {
        let kernel = self.module.kernel(if silu {
            "moe_prefill_gemm_silu"
        } else {
            "moe_prefill_gemm"
        })?;
        let mut p_w = w as *mut c_void;
        let mut v_stride = stride as i64;
        let mut p_off = off.as_ptr();
        let mut p_x = x.as_ptr();
        let mut p_xmap = xmap.as_ptr();
        let mut p_y = y.as_ptr();
        let mut p_ymap = ymap.as_ptr();
        let mut p_gate = gate.as_ptr();
        let mut p_yh = yh.as_ptr();
        let mut v_m = m as i32;
        let mut v_k = k as i32;
        let mut v_kind = kind;
        let mut params: [*mut c_void; 12] = [
            &mut p_w as *mut _ as *mut c_void,
            &mut v_stride as *mut _ as *mut c_void,
            &mut p_off as *mut _ as *mut c_void,
            &mut p_x as *mut _ as *mut c_void,
            &mut p_xmap as *mut _ as *mut c_void,
            &mut p_y as *mut _ as *mut c_void,
            &mut p_ymap as *mut _ as *mut c_void,
            &mut p_gate as *mut _ as *mut c_void,
            &mut p_yh as *mut _ as *mut c_void,
            &mut v_m as *mut _ as *mut c_void,
            &mut v_k as *mut _ as *mut c_void,
            &mut v_kind as *mut _ as *mut c_void,
        ];
        let rows = m.div_ceil(MOE_PREFILL_ROWS);
        let cfg = LaunchConfig {
            grid: (rows.max(1) as u32, n_expert.max(1) as u32, 1),
            block: (MOE_PREFILL_THREADS as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        kernel.launch(cfg, &self.stream, &mut params)
    }

    /// `y[t] = fold_rank(w[t][j] * down[t][j])` for a chunk.
    pub fn moe_combine_rows(
        &self,
        y: &Buffer,
        down: &Buffer,
        weights: &Buffer,
        n: usize,
        n_used: usize,
        n_tokens: usize,
    ) -> Result<(), String> {
        let kernel = self.module.kernel("moe_combine_rows_f32")?;
        let mut p_y = y.as_ptr();
        let mut p_d = down.as_ptr();
        let mut p_w = weights.as_ptr();
        let mut v_n = n as i32;
        let mut v_u = n_used as i32;
        let mut v_t = n_tokens as i32;
        let mut params: [*mut c_void; 6] = [
            &mut p_y as *mut _ as *mut c_void,
            &mut p_d as *mut _ as *mut c_void,
            &mut p_w as *mut _ as *mut c_void,
            &mut v_n as *mut _ as *mut c_void,
            &mut v_u as *mut _ as *mut c_void,
            &mut v_t as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (n.div_ceil(256) as u32, n_tokens.max(1) as u32, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        kernel.launch(cfg, &self.stream, &mut params)
    }

    /// `y[t] += shared[t] * scale[t]` for a chunk.
    pub fn moe_shared_add_rows(
        &self,
        y: &Buffer,
        shared: &Buffer,
        scale: &Buffer,
        n: usize,
        n_tokens: usize,
    ) -> Result<(), String> {
        let kernel = self.module.kernel("moe_shared_add_rows_f32")?;
        let mut p_y = y.as_ptr();
        let mut p_s = shared.as_ptr();
        let mut p_k = scale.as_ptr();
        let mut v_n = n as i32;
        let mut v_t = n_tokens as i32;
        let mut params: [*mut c_void; 5] = [
            &mut p_y as *mut _ as *mut c_void,
            &mut p_s as *mut _ as *mut c_void,
            &mut p_k as *mut _ as *mut c_void,
            &mut v_n as *mut _ as *mut c_void,
            &mut v_t as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (n.div_ceil(256) as u32, n_tokens.max(1) as u32, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        kernel.launch(cfg, &self.stream, &mut params)
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
