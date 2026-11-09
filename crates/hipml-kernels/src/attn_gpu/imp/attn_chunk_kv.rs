//! The chunk-attention launch: one matrix-unit kernel over an f32 mirror, a
//! q8_0 cache or a q4_0 cache.
//!
//! Split from `attn_chunk.rs` only by size. The kernel stages each KV tile
//! through `aw_ld4`, which dequantizes a 4-value group of a quantized row into
//! the same f16 shared tile the f32 mirror fills, so the matrix-unit math and
//! the causal mask are unchanged by the cache format.
//!
//! Child module of `imp` in `attn_gpu.rs`, like `attn_chunk.rs`, so it can
//! reach the layer's private scratch and kernel handles.

use super::*;

/// Query rows per block in `attn_chunk_wmma_f16` (two warps, 16 each) and the
/// warps per query head.
const ATTN_WMMA_QUERIES: u32 = 32;
const ATTN_WMMA_WARPS: usize = 2;

/// Query tokens per block in `attn_chunk_tile_f32`. Must match ACQ_TOKENS in
/// `attn_chunk_f32.hip`, which is also the KV re-read stride.
const ATTN_QUERY_TILE: u32 = 64;

impl AttnLayer<'_> {
    /// Whether the matrix-unit chunk kernel can read a quantized cache: it is
    /// built for head_dim 256 and stages one KV head for up to 8 GQA query
    /// heads.
    pub fn q8_chunk_supported(&self) -> bool {
        let d = self.dims;
        d.head_dim == 256
            && d.n_kv > 0
            && d.n_head % d.n_kv == 0
            && d.n_head / d.n_kv <= 8
    }

    /// Causal chunk attention over the quantized decode cache, dequantizing
    /// each KV tile inside the matrix-unit kernel. `dtype` is 1 for q8_0 and 2
    /// for q4_0; `stride` is the cache's per-head row count. No f32 mirror.
    #[allow(clippy::too_many_arguments)]
    pub fn chunk_attention_q8(
        &self,
        l: &Launcher,
        q: &Buffer,
        kc: &Buffer,
        vc: &Buffer,
        out: &Buffer,
        pos0: usize,
        tokens: usize,
        dtype: usize,
        stride: usize,
    ) -> Result<(), String> {
        if !self.q8_chunk_supported() || !(1..=2).contains(&dtype) {
            return Err(
                "quantized chunk attention: only head_dim 256 with a q8/q4 cache".to_string(),
            );
        }
        self.chunk_attention_kv(l, q, kc, vc, out, pos0, tokens, dtype, stride, None)
    }

    /// The same attention with the key axis cut across `splits` blocks:
    /// `attn_chunk_split_f16` writes `part_acc`/`part_ml` and
    /// `attn_chunk_combine_f32` folds them into `out`.
    ///
    /// Scratch is `[splits][tokens rounded up to 32][n_head][head_dim]` f32 and
    /// `[splits][same rows][n_head][2]` f32. `splits` of 1 keeps the mono
    /// kernel, so the two paths can be A/B'd with one env knob.
    #[allow(clippy::too_many_arguments)]
    pub fn chunk_attention_q8_split(
        &self,
        l: &Launcher,
        q: &Buffer,
        kc: &Buffer,
        vc: &Buffer,
        out: &Buffer,
        pos0: usize,
        tokens: usize,
        dtype: usize,
        stride: usize,
        part_acc: &Buffer,
        part_ml: &Buffer,
        splits: usize,
    ) -> Result<(), String> {
        if !self.q8_chunk_supported() || !(1..=2).contains(&dtype) {
            return Err(
                "quantized chunk attention: only head_dim 256 with a q8/q4 cache".to_string(),
            );
        }
        if splits < 2 {
            return self.chunk_attention_kv(l, q, kc, vc, out, pos0, tokens, dtype, stride, None);
        }
        self.chunk_attention_kv(
            l,
            q,
            kc,
            vc,
            out,
            pos0,
            tokens,
            dtype,
            stride,
            Some((part_acc, part_ml, splits)),
        )
    }

    /// Causal chunk attention over explicit q and cache buffers. Split out so
    /// the parity harness can run it against the CPU reference without going
    /// through the projection/append chain.
    pub fn chunk_attention(
        &self,
        l: &Launcher,
        q: &Buffer,
        kc: &Buffer,
        vc: &Buffer,
        out: &Buffer,
        pos0: usize,
        tokens: usize,
    ) -> Result<(), String> {
        self.chunk_attention_kv(l, q, kc, vc, out, pos0, tokens, 0, 0, None)
    }

    /// The one launch both cache formats share. `dtype` is 0 for the f32
    /// mirror, 1 for q8_0 and 2 for q4_0; `stride` is the quantized cache's
    /// per-head row count (unused for f32); `part` is the split-K scratch.
    #[allow(clippy::too_many_arguments)]
    fn chunk_attention_kv(
        &self,
        l: &Launcher,
        q: &Buffer,
        kc: &Buffer,
        vc: &Buffer,
        out: &Buffer,
        pos0: usize,
        tokens: usize,
        dtype: usize,
        stride: usize,
        part: Option<(&Buffer, &Buffer, usize)>,
    ) -> Result<(), String> {
        let d = self.dims;
        let t = tokens as u32;
        let tiled = d.head_dim == 256;
        let warp_heads = d.n_head / d.n_kv;
        if dtype != 0 && !(tiled && warp_heads <= 8) {
            return Err("quantized chunk attention: unsupported shape".to_string());
        }
        let att = self.module_kernel(
            l,
            if tiled { "attn_chunk_tile_f32" } else { "attn_chunk_f32" },
        )?;
        let mut p_q = q.as_ptr();
        let mut p_kc = kc.as_ptr();
        let mut p_vc = vc.as_ptr();
        let mut p_out = out.as_ptr();
        let mut nh = d.n_head as i32;
        let mut nkv = d.n_kv as i32;
        let mut hd = d.head_dim as i32;
        let mut width = d.width as i32;
        let mut pos = pos0 as i32;
        let mut nt = tokens as i32;
        let mut scale = 1.0f32 / (d.head_dim as f32).sqrt();
        let mut dt = dtype as i32;
        let mut st = stride as i32;
        let mut params: [*mut c_void; 13] = [
            &mut p_q as *mut _ as *mut c_void,
            &mut p_kc as *mut _ as *mut c_void,
            &mut p_vc as *mut _ as *mut c_void,
            &mut p_out as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut width as *mut _ as *mut c_void,
            &mut pos as *mut _ as *mut c_void,
            &mut nt as *mut _ as *mut c_void,
            &mut scale as *mut _ as *mut c_void,
            &mut dt as *mut _ as *mut c_void,
            &mut st as *mut _ as *mut c_void,
        ];
        let cfg = if tiled
            && warp_heads <= 8
            && (dtype != 0 || std::env::var_os("HIPML_ATTN_F32").is_none())
        {
            // Split-K path: every (KV head, 32-query tile) block streams one
            // slice of the key axis, then a combine pass folds the slices.
            if let Some((pa, pm, splits)) = part {
                let qslots = (tokens + 31) / 32 * 32;
                let split = self.module_kernel(l, "attn_chunk_split_f16")?;
                let mut p_pa = pa.as_ptr();
                let mut p_pm = pm.as_ptr();
                let mut nsp = splits as i32;
                let mut qs = qslots as i32;
                let mut params: [*mut c_void; 16] = [
                    &mut p_q as *mut _ as *mut c_void,
                    &mut p_kc as *mut _ as *mut c_void,
                    &mut p_vc as *mut _ as *mut c_void,
                    &mut p_pa as *mut _ as *mut c_void,
                    &mut p_pm as *mut _ as *mut c_void,
                    &mut nh as *mut _ as *mut c_void,
                    &mut nkv as *mut _ as *mut c_void,
                    &mut hd as *mut _ as *mut c_void,
                    &mut width as *mut _ as *mut c_void,
                    &mut pos as *mut _ as *mut c_void,
                    &mut nt as *mut _ as *mut c_void,
                    &mut scale as *mut _ as *mut c_void,
                    &mut dt as *mut _ as *mut c_void,
                    &mut st as *mut _ as *mut c_void,
                    &mut nsp as *mut _ as *mut c_void,
                    &mut qs as *mut _ as *mut c_void,
                ];
                let cfg = LaunchConfig {
                    // blockIdx.z is split * 2 + d half.
                    grid: (
                        d.n_kv as u32,
                        (t + ATTN_WMMA_QUERIES - 1) / ATTN_WMMA_QUERIES,
                        2 * splits as u32,
                    ),
                    block: ((warp_heads * ATTN_WMMA_WARPS * 32) as u32, 1, 1),
                    shared_mem_bytes: 0,
                };
                split.launch(cfg, &l.stream, &mut params)?;

                let comb = self.module_kernel(l, "attn_chunk_combine_f32")?;
                let mut p_pa = pa.as_ptr();
                let mut p_pm = pm.as_ptr();
                let mut p_out = out.as_ptr();
                let mut nh = d.n_head as i32;
                let mut hd = d.head_dim as i32;
                let mut nt = tokens as i32;
                let mut qs = qslots as i32;
                let mut nsp = splits as i32;
                let mut params: [*mut c_void; 7] = [
                    &mut p_pa as *mut _ as *mut c_void,
                    &mut p_pm as *mut _ as *mut c_void,
                    &mut p_out as *mut _ as *mut c_void,
                    &mut nh as *mut _ as *mut c_void,
                    &mut hd as *mut _ as *mut c_void,
                    &mut nt as *mut _ as *mut c_void,
                    &mut qs as *mut _ as *mut c_void,
                ];
                let cfg = LaunchConfig {
                    grid: (qslots as u32, d.n_head as u32, 1),
                    block: (d.head_dim as u32, 1, 1),
                    shared_mem_bytes: 0,
                };
                return comb.launch(cfg, &l.stream, &mut params);
            }
            // Matrix-unit flash attention: one block per (KV head, 16-query
            // tile); every query head of the GQA group reads the staged K/V
            // tile once, and each head runs a score warp beside a value warp.
            // Must match AW_QUERIES / AW_WARPS_PER_HEAD in attn_chunk_wmma.hip.
            let kernel = self.module_kernel(l, "attn_chunk_wmma_f16")?;
            let cfg = LaunchConfig {
                // blockIdx.z picks half of the head width: 8 output
                // fragments per warp instead of 16 (no register spill).
                grid: (
                    d.n_kv as u32,
                    (t + ATTN_WMMA_QUERIES - 1) / ATTN_WMMA_QUERIES,
                    2,
                ),
                block: ((warp_heads * ATTN_WMMA_WARPS * 32) as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            return kernel.launch(cfg, &l.stream, &mut params);
        } else if tiled {
            // Must match ACQ_TOKENS / ACQ_PER_WARP in attn_chunk_f32.hip.
            LaunchConfig {
                grid: (
                    d.n_head as u32,
                    (tokens as u32 + ATTN_QUERY_TILE - 1) / ATTN_QUERY_TILE,
                    1,
                ),
                block: ((ATTN_QUERY_TILE / 4) * 32, 1, 1),
                shared_mem_bytes: 0,
            }
        } else {
            LaunchConfig {
                grid: (d.n_head as u32, t, 1),
                block: (d.head_dim as u32, 1, 1),
                shared_mem_bytes: 0,
            }
        };
        att.launch(cfg, &l.stream, &mut params)
    }
}
