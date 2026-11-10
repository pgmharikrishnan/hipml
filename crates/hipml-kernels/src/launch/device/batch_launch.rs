// SPDX-License-Identifier: Apache-2.0
//! Batched cache append + attention launches (WP37).
//!
//! One launch for all slots: `grid.y` is the slot, the per-slot cache and
//! partial buffers are contiguous, and the position comes from a `[slots][2]`
//! device array `(position, seq)` written once per step.

use super::*;

impl Launcher {
    /// Quantize `slots` K/V rows straight into `slots` contiguous caches.
    #[allow(clippy::too_many_arguments)]
    pub fn kv_encode_batch(
        &self,
        dtype: usize,
        k_f32: &Buffer,
        v_f32: &Buffer,
        k_cache: &Buffer,
        v_cache: &Buffer,
        n_kv: usize,
        head_dim: usize,
        stride: usize,
        pos: &Buffer,
        slot_bytes: usize,
        slots: usize,
    ) -> Result<(), String> {
        self.kv_encode_rows_ptr(
            dtype,
            k_f32.as_ptr(),
            v_f32.as_ptr(),
            k_cache.as_ptr(),
            v_cache.as_ptr(),
            n_kv,
            head_dim,
            stride,
            pos.as_ptr(),
            slot_bytes,
            slots,
        )
    }

    /// Same append from explicit device pointers.
    ///
    /// Used by the batched speculative verify, which addresses a group of
    /// columns at one slot's own cache and position array: pass the slot's cache
    /// base as `k_cache`/`v_cache`, its row block as `k_f32`/`v_f32`, and
    /// `slot_bytes = 0` so every column of the group lands in that cache while
    /// `pos[i]` carries column `i`'s absolute position.
    #[allow(clippy::too_many_arguments)]
    pub fn kv_encode_rows_ptr(
        &self,
        dtype: usize,
        k_f32: *mut c_void,
        v_f32: *mut c_void,
        k_cache: *mut c_void,
        v_cache: *mut c_void,
        n_kv: usize,
        head_dim: usize,
        stride: usize,
        pos: *mut c_void,
        slot_bytes: usize,
        slots: usize,
    ) -> Result<(), String> {
        let entry = match dtype {
            0 => "kv_encode_q8_b",
            1 => "kv_encode_q4_b",
            other => return Err(format!("unknown KV dtype {other}")),
        };
        let kernel = self.module.kernel(entry)?;
        let mut p_k = k_f32;
        let mut p_v = v_f32;
        let mut p_kc = k_cache;
        let mut p_vc = v_cache;
        let mut v_nkv = n_kv as i32;
        let mut v_hd = head_dim as i32;
        let mut v_stride = stride as i32;
        let mut p_pos = pos;
        let mut v_slotb = slot_bytes as i32;
        let mut params: [*mut c_void; 9] = [
            &mut p_k as *mut _ as *mut c_void,
            &mut p_v as *mut _ as *mut c_void,
            &mut p_kc as *mut _ as *mut c_void,
            &mut p_vc as *mut _ as *mut c_void,
            &mut v_nkv as *mut _ as *mut c_void,
            &mut v_hd as *mut _ as *mut c_void,
            &mut v_stride as *mut _ as *mut c_void,
            &mut p_pos as *mut _ as *mut c_void,
            &mut v_slotb as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (n_kv as u32, slots as u32, 1),
            block: (head_dim.max(32) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        kernel.launch(cfg, &self.stream, &mut params)
    }

    /// Split-K decode attention for `slots` rows, then the split reduce.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode_batch(
        &self,
        dtype: usize,
        pos: &Buffer,
        q: &Buffer,
        k_cache: &Buffer,
        v_cache: &Buffer,
        part_acc: &Buffer,
        part_ml: &Buffer,
        out: &Buffer,
        n_heads: usize,
        n_kv: usize,
        head_dim: usize,
        stride: usize,
        scale: f32,
        splits: usize,
        kv_slot_bytes: usize,
        acc_slot: usize,
        ml_slot: usize,
        q_slot: usize,
        slots: usize,
    ) -> Result<(), String> {
        self.attn_decode_rows_ptr(
            dtype,
            pos.as_ptr(),
            q.as_ptr(),
            k_cache.as_ptr(),
            v_cache.as_ptr(),
            part_acc.as_ptr(),
            part_ml.as_ptr(),
            out.as_ptr(),
            n_heads,
            n_kv,
            head_dim,
            stride,
            scale,
            splits,
            kv_slot_bytes,
            acc_slot,
            ml_slot,
            q_slot,
            slots,
        )
    }

    /// Same attention from explicit device pointers, for the speculative
    /// verify: a group of columns at one slot's cache uses `kv_slot_bytes = 0`
    /// and advances the q/pos/scratch/out pointers to the group's first row.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode_rows_ptr(
        &self,
        dtype: usize,
        pos: *mut c_void,
        q: *mut c_void,
        k_cache: *mut c_void,
        v_cache: *mut c_void,
        part_acc: *mut c_void,
        part_ml: *mut c_void,
        out: *mut c_void,
        n_heads: usize,
        n_kv: usize,
        head_dim: usize,
        stride: usize,
        scale: f32,
        splits: usize,
        kv_slot_bytes: usize,
        acc_slot: usize,
        ml_slot: usize,
        q_slot: usize,
        slots: usize,
    ) -> Result<(), String> {
        if n_kv == 0 || n_heads % n_kv != 0 {
            return Err(format!(
                "n_heads {n_heads} is not a multiple of n_kv {n_kv}"
            ));
        }
        let group = n_heads / n_kv;
        let (entry, tile) = match dtype {
            0 => ("attn_decode_kv_q8_b", 4usize),
            1 => ("attn_decode_kv_q4_b", 8),
            other => return Err(format!("unknown KV dtype {other}")),
        };
        let shared = 2 * tile * head_dim * 4;
        let kernel = self.module.kernel(entry)?;
        let mut p_q = q;
        let mut p_k = k_cache;
        let mut p_v = v_cache;
        let mut p_acc = part_acc;
        let mut p_ml = part_ml;
        let mut v_heads = n_heads as i32;
        let mut v_kv = n_kv as i32;
        let mut v_dim = head_dim as i32;
        let mut p_pos = pos;
        let mut v_stride = stride as i32;
        let mut v_scale = scale;
        let mut v_splits = splits as i32;
        let mut v_kvsb = kv_slot_bytes as i32;
        let mut v_acc = acc_slot as i32;
        let mut v_ml = ml_slot as i32;
        let mut v_qs = q_slot as i32;
        let mut params: [*mut c_void; 16] = [
            &mut p_q as *mut _ as *mut c_void,
            &mut p_k as *mut _ as *mut c_void,
            &mut p_v as *mut _ as *mut c_void,
            &mut p_acc as *mut _ as *mut c_void,
            &mut p_ml as *mut _ as *mut c_void,
            &mut v_heads as *mut _ as *mut c_void,
            &mut v_kv as *mut _ as *mut c_void,
            &mut v_dim as *mut _ as *mut c_void,
            &mut p_pos as *mut _ as *mut c_void,
            &mut v_stride as *mut _ as *mut c_void,
            &mut v_scale as *mut _ as *mut c_void,
            &mut v_splits as *mut _ as *mut c_void,
            &mut v_kvsb as *mut _ as *mut c_void,
            &mut v_acc as *mut _ as *mut c_void,
            &mut v_ml as *mut _ as *mut c_void,
            &mut v_qs as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: ((n_kv * splits) as u32, slots as u32, 1),
            block: ((32 * group) as u32, 1, 1),
            shared_mem_bytes: shared as u32,
        };
        kernel.launch(cfg, &self.stream, &mut params)?;

        let red = self.module.kernel("attn_decode_kv_reduce_b")?;
        let mut p_acc = part_acc;
        let mut p_ml = part_ml;
        let mut p_out = out;
        let mut v_dim = head_dim as i32;
        let mut v_splits = splits as i32;
        let mut v_acc = acc_slot as i32;
        let mut v_ml = ml_slot as i32;
        let mut v_out = q_slot as i32;
        let mut params: [*mut c_void; 8] = [
            &mut p_acc as *mut _ as *mut c_void,
            &mut p_ml as *mut _ as *mut c_void,
            &mut p_out as *mut _ as *mut c_void,
            &mut v_dim as *mut _ as *mut c_void,
            &mut v_splits as *mut _ as *mut c_void,
            &mut v_acc as *mut _ as *mut c_void,
            &mut v_ml as *mut _ as *mut c_void,
            &mut v_out as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (n_heads as u32, slots as u32, 1),
            block: (head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        red.launch(cfg, &self.stream, &mut params)
    }
}
