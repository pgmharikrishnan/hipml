//! Row-block full attention (WP47).
//!
//! A row block concatenates several slots' tails. The projections run over all
//! rows at once, but the cache append and the causal attention are per slot:
//! row `r` of slot `s` must be written at that slot's own position and must
//! read only that slot's key/value rows. Each slot's rows are contiguous in the
//! block, so these launches take the block's row pointers and a slot-local
//! position base — no per-row map is needed.
//!
//! The arithmetic is the chunk chain (`chunk_front` + bulk encode + matrix-unit
//! attention) the single-sequence tail prefill already uses, so a tail's KV,
//! logits and state are unchanged by being batched with other tails.

use super::*;

impl AttnLayer<'_> {
    /// `chunk_front` on explicit row pointers: q|gate split, then per-head RMS
    /// norm and M-RoPE for `tokens` consecutive positions starting at `pos0`.
    #[allow(clippy::too_many_arguments)]
    pub fn chunk_front_rows(
        &self,
        l: &Launcher,
        qg: *mut c_void,
        k: *mut c_void,
        q: *mut c_void,
        gate: *mut c_void,
        pos0: usize,
        tokens: usize,
    ) -> Result<(), String> {
        if !self.chunk_supported() {
            return Err("row attention: unsupported head dimension".to_string());
        }
        let d = self.dims;
        let t = tokens as u32;

        let split = self.module_kernel(l, "qg_split_chunk_f32")?;
        let mut p_qg = qg;
        let mut p_q = q;
        let mut p_g = gate;
        let mut nh = d.n_head as i32;
        let mut hd = d.head_dim as i32;
        let mut nt = tokens as i32;
        let mut params: [*mut c_void; 6] = [
            &mut p_qg as *mut _ as *mut c_void,
            &mut p_q as *mut _ as *mut c_void,
            &mut p_g as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nt as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (d.n_head as u32, t, 1),
            block: (d.head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        split.launch(cfg, &l.stream, &mut params)?;

        let nr = self.module_kernel(l, "norm_rope_chunk_f32")?;
        let mut p_q = q;
        let mut p_k = k;
        let mut p_qw = self.q_norm.as_ptr();
        let mut p_kw = self.k_norm.as_ptr();
        let mut nh = d.n_head as i32;
        let mut nkv = d.n_kv as i32;
        let mut hd = d.head_dim as i32;
        let mut nrot = d.n_rot as i32;
        let mut base = d.rope_base;
        let mut eps = d.eps;
        let mut pos = pos0 as i32;
        let mut nt = tokens as i32;
        let mut params: [*mut c_void; 12] = [
            &mut p_q as *mut _ as *mut c_void,
            &mut p_k as *mut _ as *mut c_void,
            &mut p_qw as *mut _ as *mut c_void,
            &mut p_kw as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nrot as *mut _ as *mut c_void,
            &mut base as *mut _ as *mut c_void,
            &mut eps as *mut _ as *mut c_void,
            &mut pos as *mut _ as *mut c_void,
            &mut nt as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: ((d.n_head + d.n_kv) as u32, t, 1),
            block: (d.head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        nr.launch(cfg, &l.stream, &mut params)
    }

    /// Quantize `rows` consecutive rows of `k`/`v` straight into one slot's
    /// quantized cache, starting at that slot's position `start`.
    #[allow(clippy::too_many_arguments)]
    pub fn kv_encode_bulk_rows(
        &self,
        l: &Launcher,
        k: *mut c_void,
        v: *mut c_void,
        kc: *mut c_void,
        vc: *mut c_void,
        dtype: usize,
        n_kv: usize,
        head_dim: usize,
        stride: usize,
        start: usize,
        rows: usize,
    ) -> Result<(), String> {
        let entry = match dtype {
            0 => "kv_encode_bulk_q8",
            1 => "kv_encode_bulk_q4",
            other => return Err(format!("unknown KV dtype {other}")),
        };
        let kernel = self.module_kernel(l, entry)?;
        let mut p_k = k;
        let mut p_v = v;
        let mut p_kc = kc;
        let mut p_vc = vc;
        let mut v_nkv = n_kv as i32;
        let mut v_hd = head_dim as i32;
        let mut v_stride = stride as i32;
        let mut v_start = start as i32;
        let mut v_rows = rows as i32;
        let mut v_src0 = 0i32;
        let mut params: [*mut c_void; 10] = [
            &mut p_k as *mut _ as *mut c_void,
            &mut p_v as *mut _ as *mut c_void,
            &mut p_kc as *mut _ as *mut c_void,
            &mut p_vc as *mut _ as *mut c_void,
            &mut v_nkv as *mut _ as *mut c_void,
            &mut v_hd as *mut _ as *mut c_void,
            &mut v_stride as *mut _ as *mut c_void,
            &mut v_start as *mut _ as *mut c_void,
            &mut v_rows as *mut _ as *mut c_void,
            &mut v_src0 as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: ((n_kv * rows) as u32, 1, 1),
            block: (head_dim.max(32) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        kernel.launch(cfg, &l.stream, &mut params)
    }

    /// The mono (no split-K) matrix-unit chunk attention on explicit pointers:
    /// `tokens` query rows starting at `pos0`, one slot's quantized cache.
    #[allow(clippy::too_many_arguments)]
    pub fn chunk_attention_rows(
        &self,
        l: &Launcher,
        q: *mut c_void,
        kc: *mut c_void,
        vc: *mut c_void,
        out: *mut c_void,
        pos0: usize,
        tokens: usize,
        dtype: usize,
        stride: usize,
    ) -> Result<(), String> {
        if !self.q8_chunk_supported() || !(1..=2).contains(&dtype) {
            return Err(
                "row attention: only head_dim 256 with a q8/q4 cache".to_string(),
            );
        }
        let d = self.dims;
        let t = tokens as u32;
        let warp_heads = d.n_head / d.n_kv;
        let kernel = self.module_kernel(l, "attn_chunk_wmma_f16")?;
        let mut p_q = q;
        let mut p_kc = kc;
        let mut p_vc = vc;
        let mut p_out = out;
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
        let cfg = LaunchConfig {
            grid: (d.n_kv as u32, (t + 31) / 32, 2),
            block: ((warp_heads * 2 * 32) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        kernel.launch(cfg, &l.stream, &mut params)
    }
}
