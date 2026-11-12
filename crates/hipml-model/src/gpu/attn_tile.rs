// SPDX-License-Identifier: Apache-2.0
//! One layer's chunk attention, dispatched by cache: the q8/q4 decode cache
//! read in place, the f32 tile, or the row-at-a-time fallbacks.
//!
//! Child module of `gpu` so it can reach the engine's chunk scratch, the
//! quantized cache buffers and `deep_attention_ptr`. `prefill.rs` keeps the
//! projections, the norms and the FFN; everything between `c.qg` and `c.aout`
//! lives here because it is the part that depends on which KV cache the engine
//! carries.

use super::*;

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
                        l,
                        &c.qg,
                        &c.k,
                        &c.v,
                        &c.q,
                        &c.aq_gate,
                        &c.aout,
                        t,
                        pos0 + t,
                    )?;
                }
            } else {
                attn.step_chunk(l, &c.qg, &c.k, &c.v, &c.q, &c.aq_gate, &c.aout, pos0, n_tok)?;
            }
        }
        Some(dk) => {
            // Escape hatches HIPML_TOKEN_STEPS and HIPML_Q8_ROW both select the
            // row-at-a-time quantized path; this engine carries no f32 mirror
            // for `step_chunk_row` to append to.
            if attn.q8_chunk_supported() && !steps && std::env::var_os("HIPML_Q8_ROW").is_none() {
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
                attn.chunk_attention_q8_split(
                    l,
                    &c.q,
                    &dk.k,
                    &dk.v,
                    &c.aout,
                    pos0,
                    n_tok,
                    dk.dtype.id() + 1,
                    dk.seq,
                    &c.apart,
                    &c.apart_ml,
                    split_count(c.attn_splits, pos0),
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

/// `HIPML_ATTN_SPLITS`: key splits for a deep chunk (default 8, 1 disables).
pub(super) fn env_splits() -> usize {
    std::env::var("HIPML_ATTN_SPLITS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(8)
        .clamp(1, 32)
}

/// Key splits for a chunk starting at `pos0`.
///
/// Short chunks gain nothing (the combine pass costs more than the loop it
/// shortens), so the mono kernel keeps everything below
/// `HIPML_ATTN_SPLIT_MIN` positions (default 8K). That also makes an 8K prompt
/// and anything shorter bit-identical to the mono path: its chunks all sit
/// below the gate. `HIPML_ATTN_SPLITS` sets the count; 1 keeps the mono kernel
/// everywhere.
pub(super) fn split_count(splits: usize, pos0: usize) -> usize {
    if splits <= 1 {
        return 1;
    }
    let min_pos = std::env::var("HIPML_ATTN_SPLIT_MIN")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(8_192);
    if pos0 < min_pos {
        1
    } else {
        splits
    }
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
    deep_attention_ptr_on(l, cfg, dk, pos, q_ptr, out_ptr, false)
}

/// Same launch on the main or side stream: the batch path splits its per-slot
/// attention across both so the small kernels overlap.
#[allow(clippy::too_many_arguments)]
pub(super) fn deep_attention_ptr_on(
    l: &Launcher,
    cfg: &Config,
    dk: &DeepKv,
    pos: usize,
    q_ptr: *mut std::ffi::c_void,
    out_ptr: *mut std::ffi::c_void,
    side: bool,
) -> Result<(), String> {
    if dk.dtype == KvDtype::Q4 && cfg.head_dim > 256 {
        return Err("q4 KV: head_dim above 256 is not covered".to_string());
    }
    let seq = pos + 1;
    if seq > dk.seq {
        return Err(format!("position {pos} beyond quantized cache {}", dk.seq));
    }
    // Chunked prefill is not graph-captured, so size the grid to the positions
    // that exist. The fixed `dk.seq` split count is only needed when the grid
    // becomes part of a captured graph (the decode path). `DeepKv::splits`
    // issues one split per 64 positions, which is the measured optimum for
    // the warp kernel at 8K through 100K.
    let splits = dk.splits(seq);
    launch_kv::launch_ptrs(
        l.module(),
        if side { l.stream2() } else { l.stream() },
        // Each stream carries its own position, so the side stream must read
        // the side step buffer (the KV append above already used it).
        if side {
            l.step_side_buf()
        } else {
            l.step_buf()
        },
        // Decode-attention entry id (warp-per-KV-head variants included); the
        // batch path shares the same selector as the single-stream path.
        dk.kernel,
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

/// Quantized-KV decode attention for `cols` consecutive columns of one
/// sequence: one launch per group of equal cache split count, instead of one
/// pair per column. The columns share the cache, so only the query row, the
/// partial scratch slice and the (position, seq) pair move.
///
/// Returns false when the selector has no batched warp entry; the caller then
/// keeps its per-column loop. The batched entry is the same warp body, so every
/// column's partials are the ones the one-column launch writes.
#[allow(clippy::too_many_arguments)]
pub(super) fn deep_attention_cols(
    l: &Launcher,
    cfg: &Config,
    dk: &DeepKv,
    pos0: usize,
    cols: usize,
    q_ptr: *mut std::ffi::c_void,
    out_ptr: *mut std::ffi::c_void,
    pos_buf: &Buffer,
    part_acc: *mut std::ffi::c_void,
    part_ml: *mut std::ffi::c_void,
    acc_slot: usize,
    ml_slot: usize,
) -> Result<bool, String> {
    // Only the whole-group warp entries have a `_b` form (dk.kernel 4/5).
    if !matches!(dk.kernel, 4 | 5) {
        return Ok(false);
    }
    let qw = cfg.n_head * cfg.head_dim;
    let scale = 1.0f32 / (cfg.head_dim as f32).sqrt();
    let at = |p: *mut std::ffi::c_void, words: usize| unsafe {
        (p as *mut u8).add(words * 4) as *mut std::ffi::c_void
    };
    let mut c0 = 0usize;
    while c0 < cols {
        // A plain step at this depth sizes its split count from the position;
        // group the columns that share one so each column's arithmetic is the
        // one-column launch's.
        let splits = dk.splits(pos0 + c0 + 1);
        let mut c1 = c0 + 1;
        while c1 < cols && dk.splits(pos0 + c1 + 1) == splits {
            c1 += 1;
        }
        launch_kv::launch_cols_ptrs(
            l.module(),
            l.stream(),
            dk.kernel,
            at(pos_buf.as_ptr(), c0 * 2),
            at(q_ptr, c0 * qw),
            &dk.k,
            &dk.v,
            at(part_acc, c0 * acc_slot),
            at(part_ml, c0 * ml_slot),
            at(out_ptr, c0 * qw),
            cfg.n_head,
            cfg.n_head_kv,
            cfg.head_dim,
            dk.seq,
            scale,
            splits,
            c1 - c0,
            acc_slot,
            ml_slot,
            qw,
        )?;
        c0 = c1;
    }
    Ok(true)
}
