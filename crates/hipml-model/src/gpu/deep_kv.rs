//! The quantized K/V cache for one full-attention layer, and its split-K
//! scratch.
//!
//! Child module of `gpu` so the engine can reach the cache buffers it hands to
//! `kv_encode` and the decode pair.

use super::*;

/// One layer's quantized K/V cache plus the split-K scratch.
pub struct DeepKv {
    pub dtype: KvDtype,
    pub seq: usize,
    n_kv: usize,
    head_dim: usize,
    pub(super) k: Buffer,
    pub(super) v: Buffer,
    pub(super) part_acc: Buffer,
    pub(super) part_ml: Buffer,
}

/// The split count sets the grid: `n_kv * splits` blocks of
/// `32 * n_head/n_kv` threads.
///
/// 64 is measured best at 100K (12.34 ms for 3.24 GiB, 282 GB/s). The grid is
/// 256 blocks of 192 threads, which looks under-occupied, but more blocks do not
/// help: 32 splits gives 14.3 ms, 128 gives 12.98, 256 gives 12.49 and 512 gives
/// 12.58 (`bench-kv --splits N`). The kernel is limited by its own per-row work,
/// not by block count.
pub(super) const MAX_SPLITS: usize = 64;

impl DeepKv {
    pub(super) fn new(cfg: &Config, seq: usize, dtype: KvDtype) -> Result<DeepKv, String> {
        let row = dtype.row_bytes(cfg.head_dim);
        let bytes = cfg.n_head_kv * seq * row;
        let f32s = bytes.div_ceil(4);
        Ok(DeepKv {
            dtype,
            seq,
            n_kv: cfg.n_head_kv,
            head_dim: cfg.head_dim,
            k: Buffer::alloc(f32s)?,
            v: Buffer::alloc(f32s)?,
            part_acc: Buffer::alloc(cfg.n_head * MAX_SPLITS * cfg.head_dim)?,
            part_ml: Buffer::alloc(cfg.n_head * MAX_SPLITS * 2)?,
        })
    }

    fn row(&self) -> usize {
        self.dtype.row_bytes(self.head_dim)
    }

    /// `2 * n_kv * seq * row_bytes`.
    pub fn bytes(&self) -> usize {
        2 * self.n_kv * self.seq * self.row()
    }

    /// Fill the first `depth` positions of every head with format-valid blocks
    /// so a 100K decode can be measured without a 100K token prefill.
    pub(super) fn fill_synthetic(&self, depth: usize, seed: u64) -> Result<(), String> {
        let blocks_per_row = self.head_dim / kv::BLOCK;
        let pattern = kv::synthetic_blocks(self.dtype, depth * blocks_per_row, seed);
        let row = self.row();
        for h in 0..self.n_kv {
            let off = h * self.seq * row;
            self.k.upload_bytes(off, &pattern)?;
            self.v.upload_bytes(off, &pattern)?;
        }
        Ok(())
    }

    /// Quantize one token's K/V rows and append them at `pos`.
    ///
    /// Kept as the host definition of the format for tests and for the parity
    /// harness; the graph itself appends with `kv_encode` on the device.
    #[allow(dead_code)]
    fn append(&self, k: &[f32], v: &[f32], pos: usize) -> Result<(), String> {
        let hd = self.head_dim;
        let row = self.row();
        for h in 0..self.n_kv {
            let kb = kv::quantize(self.dtype, &k[h * hd..(h + 1) * hd]);
            let vb = kv::quantize(self.dtype, &v[h * hd..(h + 1) * hd]);
            let off = (h * self.seq + pos) * row;
            self.k.upload_bytes(off, &kb)?;
            self.v.upload_bytes(off, &vb)?;
        }
        Ok(())
    }
}

