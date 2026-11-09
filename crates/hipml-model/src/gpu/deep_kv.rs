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
    /// Decode-attention entry id: 4/5 are the warp-per-KV-head variants
    /// (`attn_decode_kv_w2`/`w4`), 7/13/14 the three-heads-per-warp variants
    /// (`s2`/`s4`/`s1`), 0/1 the staged kernels.
    pub kernel: usize,
    n_kv: usize,
    head_dim: usize,
    pub(super) k: Buffer,
    pub(super) v: Buffer,
    pub(super) part_acc: Buffer,
    pub(super) part_ml: Buffer,
}

/// Split-count ceiling. The scratch is sized for the largest of the two
/// policies below, so one allocation serves both.
///
/// The staged kernel is measured best at 64 splits (12.34 ms for 3.24 GiB,
/// 282 GB/s); more blocks do not help it because it is limited by its own
/// per-row work. The warp kernel is one warp per (KV head, split), so it wants
/// four times as many tasks to put 16 warps on each CU, and its partials are
/// small against the cache read (128 splits add ~0.16 ms of reduce traffic).
pub(super) const MAX_SPLITS: usize = 256;
const STAGED_SPLITS: usize = 64;
const WARP_SPLITS: usize = 256;

impl DeepKv {
    pub(super) fn new(cfg: &Config, seq: usize, dtype: KvDtype) -> Result<DeepKv, String> {
        let row = dtype.row_bytes(cfg.head_dim);
        let bytes = cfg.n_head_kv * seq * row;
        let f32s = bytes.div_ceil(4);
        // head_dim 256 and a GQA group of at most 8 is exactly the shape the
        // warp kernel's lane map covers; every other case keeps the staged one.
        let group = cfg.n_head / cfg.n_head_kv;
        let kernel = if dtype == KvDtype::Q8 && cfg.head_dim == 256 && group <= 8 {
            13
        } else {
            dtype.id()
        };
        Ok(DeepKv {
            dtype,
            seq,
            kernel,
            n_kv: cfg.n_head_kv,
            head_dim: cfg.head_dim,
            k: Buffer::alloc(f32s)?,
            v: Buffer::alloc(f32s)?,
            part_acc: Buffer::alloc(cfg.n_head * MAX_SPLITS * cfg.head_dim)?,
            part_ml: Buffer::alloc(cfg.n_head * MAX_SPLITS * 2)?,
        })
    }

    /// Split count for one decode at `seq` positions. Fixed at graph build.
    pub(super) fn splits(&self, seq: usize) -> usize {
        if self.kernel >= 4 {
            (seq / 256).clamp(1, WARP_SPLITS)
        } else {
            (seq / 64).clamp(1, STAGED_SPLITS)
        }
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

