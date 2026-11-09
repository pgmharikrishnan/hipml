//! One layer's chunk attention, dispatched by cache: the q8/q4 decode cache
//! read in place, the f32 tile, or the row-at-a-time fallbacks.
//!
//! Child module of `gpu` so it can reach the engine's chunk scratch, the
//! quantized cache buffers and `deep_attention_ptr`. `prefill.rs` keeps the
//! projections, the norms and the FFN; everything between `c.qg` and `c.aout`
//! lives here because it is the part that depends on which KV cache the engine
//! carries.

use super::*;
use super::deep_kv::MAX_SPLITS;

/// Fill `c.aout` for `n_tok` rows of the chunk starting at `pos0`.
///
/// `deep` is the layer's quantized cache, absent on the f32-only path. The
/// chunk buffers are laid out `[tokens][..]`; rows are appended to the cache
/// before attention and the kernel's causal bound hides the future ones.
pub(super) fn attention_tile(
    l: &Launcher,
    cfg: &Config,
    attn: &AttnLayer<'_>,
    deep: Option<&DeepKv>,
    c: &prefill::Chunk,
    pos0: usize,
    n_tok: usize,
) -> Result<(), String> {
    let steps = std::env::var_os("HIPML_TOKEN_STEPS").is_some();
    match deep {
        None => {
            if steps || !attn.chunk_supported() {
                for t in 0..n_tok {
                    attn.step_chunk_row(
                        l, &c.qg, &c.k, &c.v, &c.q, &c.aq_gate, &c.aout, t, pos0 + t,
                    )?;
                }
            } else {
                attn.step_chunk(
                    l, &c.qg, &c.k, &c.v, &c.q, &c.aq_gate, &c.aout, pos0, n_tok,
                )?;
            }
        }
        Some(dk) => {
            // Escape hatches HIPML_TOKEN_STEPS and HIPML_Q8_ROW both select the
            // row-at-a-time quantized path; this engine carries no f32 mirror
            // for `step_chunk_row` to append to.
            if attn.q8_chunk_supported()
                && !steps
                && std::env::var_os("HIPML_Q8_ROW").is_none()
            {
                // Quantize the tile straight into the decode cache, then run the
                // same matrix-unit attention over it.
                attn.chunk_front(l, &c.qg, &c.k, &c.q, &c.aq_gate, pos0, n_tok)?;
                l.kv_encode_bulk(
                    dk.dtype.id(),
                    &c.k,
                    &c.v,
                    &dk.k,
                    &dk.v,
                    cfg.n_head_kv,
                    cfg.head_dim,
                    dk.seq,
                    pos0,
                    0,
                    n_tok,
                )?;
                attn.chunk_attention_q8(
                    l,
                    &c.q,
                    &dk.k,
                    &dk.v,
                    &c.aout,
                    pos0,
                    n_tok,
                    dk.dtype.id() + 1,
                    dk.seq,
                )?;
            } else {
                // Shapes the matrix-unit kernel does not cover: one row at a
                // time through the quantized cache and split-K decode attention.
                let width = cfg.n_head_kv * cfg.head_dim;
                let q_width = cfg.n_head * cfg.head_dim;
                for t in 0..n_tok {
                    let pos = pos0 + t;
                    attn.prep_chunk_row(l, &c.qg, &c.k, &c.q, &c.aq_gate, t, pos)?;
                    l.kv_encode_ptr(
                        dk.dtype.id(),
                        row_ptr(&c.k, t, width),
                        row_ptr(&c.v, t, width),
                        &dk.k,
                        &dk.v,
                        cfg.n_head_kv,
                        cfg.head_dim,
                        dk.seq,
                    )?;
                    deep_attention_ptr(
                        l,
                        cfg,
                        dk,
                        pos,
                        row_ptr(&c.q, t, q_width),
                        row_ptr(&c.aout, t, q_width),
                    )?;
                }
            }
        }
    }
    Ok(())
}

/// Device pointer of row `t` of a `[tokens][width]` f32 buffer.
fn row_ptr(buf: &Buffer, t: usize, width: usize) -> *mut std::ffi::c_void {
    unsafe { (buf.as_ptr() as *mut u8).add(t * width * 4) as *mut std::ffi::c_void }
}

/// Quantized-KV decode attention for one layer at `pos`, reading the query row
/// and writing the output row through raw pointers. Used by the row-at-a-time
/// quantized prefill, whose rows live inside the chunk scratch.
pub(super) fn deep_attention_ptr(
    l: &Launcher,
    cfg: &Config,
    dk: &DeepKv,
    pos: usize,
    q_ptr: *mut std::ffi::c_void,
    out_ptr: *mut std::ffi::c_void,
) -> Result<(), String> {
    if dk.dtype == KvDtype::Q4 && cfg.head_dim > 256 {
        return Err("q4 KV: head_dim above 256 is not covered".to_string());
    }
    let seq = pos + 1;
    if seq > dk.seq {
        return Err(format!("position {pos} beyond quantized cache {}", dk.seq));
    }
    // Chunked prefill is not graph-captured, so size the grid to the positions
    // that exist. The split count is fixed from the cache's row count (not the
    // sequence length) so every verify column and the plain decode launch share
    // one partition; see the comment in attn_decode_kv.hip.
    let splits = (dk.seq / 64).clamp(1, MAX_SPLITS);
    launch_kv::launch_ptrs(
        l.module(),
        l.stream(),
        l.step_buf(),
        dk.dtype.id(),
        q_ptr,
        &dk.k,
        &dk.v,
        &dk.part_acc,
        &dk.part_ml,
        out_ptr,
        cfg.n_head,
        cfg.n_head_kv,
        cfg.head_dim,
        // The cache's per-head row count, not `seq`: the kernel reads `seq`
        // positions starting at row 0 of a `dk.seq`-stride layout. Passing the
        // position count here made every decode step above position 0 read the
        // wrong rows for every KV head but the first.
        dk.seq,
        1.0f32 / (cfg.head_dim as f32).sqrt(),
        splits,
        0,
        0,
    )
}
