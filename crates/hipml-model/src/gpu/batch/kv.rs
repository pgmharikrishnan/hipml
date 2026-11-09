//! Per-layer quantized KV caches for the batched engine.
//!
//! The caches are contiguous across slots: slot `b`'s K cache starts at
//! `b * slot_bytes` inside one allocation. That is what lets the batched append
//! and attention kernels address a slot from `blockIdx.y` instead of needing a
//! per-slot pointer table.

use super::*;
use super::deep_kv::MAX_SPLITS;

pub(super) struct BatchKv {
    pub dtype: KvDtype,
    pub seq: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub slots: usize,
    pub k: Buffer,
    pub v: Buffer,
    pub part_acc: Buffer,
    pub part_ml: Buffer,
}

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
        Ok(BatchKv {
            dtype,
            seq,
            n_kv: cfg.n_head_kv,
            head_dim: cfg.head_dim,
            slots,
            k: Buffer::alloc(cache_f32)?,
            v: Buffer::alloc(cache_f32)?,
            part_acc: Buffer::alloc(slots * cfg.n_head * MAX_SPLITS * cfg.head_dim)?,
            part_ml: Buffer::alloc(slots * cfg.n_head * MAX_SPLITS * 2)?,
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
            let pattern = hipml_core::kv::synthetic_blocks(self.dtype, depth * blocks_per_row, seed);
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
