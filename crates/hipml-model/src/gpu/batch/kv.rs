// SPDX-License-Identifier: Apache-2.0
//! Per-layer quantized KV caches for the batched engine.
//!
//! The caches are contiguous across slots: slot `b`'s K cache starts at
//! `b * slot_bytes` inside one allocation. That is what lets the batched append
//! and attention kernels address a slot from `blockIdx.y` instead of needing a
//! per-slot pointer table.

use super::*;

pub(super) struct BatchKv {
    pub dtype: KvDtype,
    pub seq: usize,
    /// Key splits one decode launch uses: [`attn_splits`].
    pub splits: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub slots: usize,
    pub k: Buffer,
    pub v: Buffer,
    pub part_acc: Buffer,
    pub part_ml: Buffer,
}

/// Key splits for the batched decode attention.
///
/// One run per 64 cache rows is the granularity the staged warp kernel was
/// tuned at (`deep_kv::splits`). The run grows with the slot count instead of
/// the run count, because a wider batch already has `slots` blocks per run:
/// that keeps the block count around 150 at every B and leaves the split-reduce
/// traffic where it was.
///
/// The count is derived from the cache's row count, never from the live
/// position. The kernel partitions the cache into `splits` runs of
/// `ceil(stride / splits)` rows fixed in cache positions, and only the runs that
/// start below the live sequence do any work. A position-derived count on a 32K
/// cache at a 2.4K sequence gives 37 runs of 886 rows: three live runs, 12
/// blocks on a 64-CU device, and the cost of the whole sequence's KV again.
pub(super) fn attn_splits(stride: usize, slots: usize) -> usize {
    let per = 64 * slots.div_ceil(4).max(1);
    stride.div_ceil(per).clamp(1, SPLITS_CAP)
}

/// Hard ceiling on the split count. It binds at a 32K cache with one slot, and
/// it is a measurement, not a guess: 512 runs of 64 rows there measured 0.5 ms
/// *slower* than 256 runs of 128, because the split-reduce reads every run's
/// partial and the extra blocks did not buy back the traffic. Scratch is
/// allocated for the count in use, not for this cap: at 16 slots in an 8K cache
/// that is 32 runs, 1.4 GiB below the old fixed 256.
const SPLITS_CAP: usize = 256;

impl BatchKv {
    pub(super) fn new(
        cfg: &Config,
        seq: usize,
        dtype: KvDtype,
        slots: usize,
    ) -> Result<BatchKv, String> {
        let row = dtype.row_bytes(cfg.head_dim);
        let slot_bytes = cfg.n_head_kv * seq * row;
        let cache_f32 = (slots * slot_bytes).div_ceil(4);
        let splits = attn_splits(seq, slots);
        Ok(BatchKv {
            dtype,
            seq,
            splits,
            n_kv: cfg.n_head_kv,
            head_dim: cfg.head_dim,
            slots,
            k: Buffer::alloc(cache_f32)?,
            v: Buffer::alloc(cache_f32)?,
            part_acc: Buffer::alloc(slots * cfg.n_head * splits * cfg.head_dim)?,
            part_ml: Buffer::alloc(slots * cfg.n_head * splits * 2)?,
        })
    }

    pub(super) fn row(&self) -> usize {
        self.dtype.row_bytes(self.head_dim)
    }

    /// Bytes of one slot's cache (K, and the same for V).
    pub(super) fn slot_bytes(&self) -> usize {
        self.n_kv * self.seq * self.row()
    }

    /// Floats of split-K scratch per slot.
    pub(super) fn acc_slot(&self) -> usize {
        self.part_acc.len() / self.slots
    }

    pub(super) fn ml_slot(&self) -> usize {
        self.part_ml.len() / self.slots
    }

    /// `2 * slots * slot_bytes`.
    pub(super) fn bytes(&self) -> usize {
        2 * self.slots * self.slot_bytes()
    }

    /// Fill the first `depths[b]` positions of slot `b` with format-valid
    /// blocks, so a deep decode can be measured without a long prefill.
    pub(super) fn fill_synthetic(&self, depths: &[usize], mut seed: u64) -> Result<(), String> {
        let blocks_per_row = self.head_dim / hipml_core::kv::BLOCK;
        let row = self.row();
        let slot_bytes = self.slot_bytes();
        for (b, &depth) in depths.iter().enumerate() {
            let pattern =
                hipml_core::kv::synthetic_blocks(self.dtype, depth * blocks_per_row, seed);
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            for h in 0..self.n_kv {
                let off = b * slot_bytes + h * self.seq * row;
                self.k.upload_bytes(off, &pattern)?;
                self.v.upload_bytes(off, &pattern)?;
            }
        }
        Ok(())
    }
}
