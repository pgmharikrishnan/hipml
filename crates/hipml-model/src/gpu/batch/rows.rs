//! Multi-row prefill over independent sequences (WP47).
//!
//! One weight pass serves the prompt tails of several slots at once: the rows
//! of every tail are concatenated into one `[M][n]` activation tile, the
//! projections and the FFN run as ordinary prefill GEMMs over `M` rows, and the
//! work that is per sequence stays per sequence:
//!
//! * attention reads and appends **each row's own slot cache**, with the
//!   slot's consecutive positions (one matrix-unit launch per slot),
//! * the GDN conv and recurrence advance **each slot's own window and matrix**
//!   in that slot's own token order (one launch per slot).
//!
//! The per-token arithmetic is the chunk path's, unchanged: a tail produces the
//! same KV, the same state and the same sampled id whether it runs alone
//! through `prefill_tile` or batched with other tails here. Only the projections
//! see a wider tile, so their last bits can move with `M` (as any prefill tile
//! length does).
//!
//! Layout: rows are concatenated tail by tail, so a slot's rows are contiguous.
//! `row_slot[r]` is row `r`'s slot and `row_local[r]` its token index inside
//! that slot.

use super::*;
use hipml_kernels::ssm::{SsmChunk, SsmDims};
use std::time::Instant;

/// A reusable `[cap][..]` activation block plus the per-row maps.
pub(crate) struct RowScratch {
    pub cap: usize,
    pub x: Buffer,
    pub x1: Buffer,
    pub t: Buffer,
    pub gate: Buffer,
    pub up: Buffer,
    pub act: Buffer,
    pub down: Buffer,
    pub qg: Buffer,
    pub q: Buffer,
    pub agate: Buffer,
    pub k: Buffer,
    pub v: Buffer,
    pub aout: Buffer,
    pub y: Buffer,
    /// fp16 copy of the widest activation tile, `(cap*max(n,ff,inner)+1)/2`.
    pub xh: Buffer,
    /// Split-K scratch for the ffn GEMMs; only touched for short blocks.
    pub sk: Buffer,
    pub ssm: SsmChunk,
    /// Per-row metadata, `[cap]` i32. The conv/gdn stages walk each tail from
    /// its base, so these are only needed by diagnostics today.
    pub row_slot: Buffer,
    pub row_local: Buffer,
}

impl RowScratch {
    fn new(cfg: &Config, cap: usize, _slots: usize) -> Result<RowScratch, String> {
        let n = cfg.n_embd;
        let ff = cfg.n_ff;
        let qd = cfg.n_head * cfg.head_dim;
        let w = cfg.n_head_kv * cfg.head_dim;
        let sd = SsmDims::from_config(cfg);
        let wide = n.max(ff).max(sd.inner());
        let zeros = |len: usize| Buffer::alloc(len.max(1));
        Ok(RowScratch {
            cap,
            x: zeros(cap * n)?,
            x1: zeros(cap * n)?,
            t: zeros(cap * n)?,
            gate: zeros(cap * ff)?,
            up: zeros(cap * ff)?,
            act: zeros(cap * ff)?,
            down: zeros(cap * n)?,
            qg: zeros(cap * 2 * qd)?,
            q: zeros(cap * qd)?,
            agate: zeros(cap * qd)?,
            k: zeros(cap * w)?,
            v: zeros(cap * w)?,
            aout: zeros(cap * qd)?,
            y: zeros(cap * n)?,
            xh: zeros((cap * wide + 1) / 2)?,
            sk: zeros(if cap <= 1024 { cap * wide } else { 1 })?,
            ssm: SsmChunk::new(&sd, cap)?,
            row_slot: zeros(cap)?,
            row_local: zeros(cap)?,
        })
    }

    pub(crate) fn bytes(&self) -> usize {
        let f32s = self.x.len()
            + self.x1.len()
            + self.t.len()
            + self.gate.len()
            + self.up.len()
            + self.act.len()
            + self.down.len()
            + self.qg.len()
            + self.q.len()
            + self.agate.len()
            + self.k.len()
            + self.v.len()
            + self.aout.len()
            + self.y.len()
            + self.xh.len()
            + self.sk.len()
            + self.row_slot.len()
            + self.row_local.len();
        f32s * 4 + self.ssm.bytes()
    }
}

/// Device pointer `words` f32 into `buf`.
fn at(buf: &Buffer, words: usize) -> *mut std::ffi::c_void {
    unsafe { (buf.as_ptr() as *mut u8).add(words * 4) as *mut std::ffi::c_void }
}

fn bytes_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Longest tail the row block accepts. A longer suffix is a real prefill and
/// stays on the chunked tile path, where one tile beats one step per token.
pub const ROW_TAIL_MAX: usize = 64;

/// Row cap of one block. Scratch is ~0.7 MB per row, so 512 bounds it near
/// 350 MB; a gather larger than this is split into several blocks.
pub const ROW_BLOCK_MAX: usize = 512;

impl<'a> BatchEngine<'a> {
    /// Shortest tail the per-tail conv can carry (its window is `d_conv-1`).
    pub fn row_min_tail(&self) -> usize {
        SsmDims::from_config(&self.cfg).d_conv.saturating_sub(1)
    }

    /// Longest tail a row block accepts.
    pub fn row_tail_max(&self) -> usize {
        ROW_TAIL_MAX
    }

    /// Rows one block may hold (`HIPML_ROWS_MAX`, default 512).
    pub fn row_block_max(&self) -> usize {
        std::env::var("HIPML_ROWS_MAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n: &usize| n > 0)
            .unwrap_or(ROW_BLOCK_MAX)
    }

impl<'a> BatchEngine<'a> {
    /// Rows the current scratch can hold.
    pub fn row_capacity(&self) -> usize {
        self.rows.as_ref().map(|r| r.cap).unwrap_or(0)
    }

    /// Rows the last `prefill_rows` call ran, and its wall time.
    pub fn last_row_block(&self) -> (usize, f64) {
        (self.row_login, self.row_ms)
    }

    /// Prefill one tail per slot in a single weight pass, returning each tail's
    /// first sampled id. `slot` must not be active and `positions[slot]` must
    /// be the tail's first position (the quiet adopt sets it).
    ///
    /// Advances each slot's KV, recurrent state and position exactly as
    /// `prefill_tile` would have for that tail alone.
    pub fn prefill_rows(&mut self, tails: &[(usize, &[u32])]) -> Result<Vec<u32>, String> {
        if tails.is_empty() {
            return Ok(Vec::new());
        }
        let cfg = self.cfg.clone();
        let dims = SsmDims::from_config(&cfg);
        if tails.len() > self.slots {
            return Err(format!("{} tails for {} slots", tails.len(), self.slots));
        }
        let mut total = 0usize;
        for (slot, tok) in tails {
            if *slot >= self.slots {
                return Err(format!("tail slot {slot} outside 0..{}", self.slots));
            }
            if self.active.contains(slot) {
                return Err(format!("tail slot {slot} is already active"));
            }
            if tok.is_empty() {
                return Err("empty tail".to_string());
            }
            if tok.len() > ROW_TAIL_MAX {
                return Err(format!(
                    "tail of {} tokens exceeds the row block's {ROW_TAIL_MAX}",
                    tok.len()
                ));
            }
            if tok.len() < dims.d_conv - 1 {
                return Err(format!(
                    "tail of {} tokens is shorter than the {} row conv window",
                    tok.len(),
                    dims.d_conv - 1
                ));
            }
            total += tok.len();
        }
        // Per-slot SSM/attention kernels need a shape the row kernels cover.
        for b in &self.layers {
            if let BatchKind::Full(f) = &b.kind {
                if !f.attn.q8_chunk_supported() {
                    return Err("row block: attention shape not covered".to_string());
                }
            }
            if let BatchKind::Ssm(s) = &b.kind {
                if !s.inst.rows_supported() {
                    return Err("row block: GDN head dims not covered".to_string());
                }
            }
        }
        self.ensure_rows(total)?;
        // SAFETY of the borrow split: the scratch, the layer weights and the
        // slot state are disjoint fields; `rows` is only read below.
        let rows = self.rows.take().expect("allocated above");

        let started = Instant::now();
        let result = self.forward_rows(&cfg, &dims, &rows, tails);
        self.rows = Some(rows);
        let out = result?;
        self.row_login = total;
        self.row_ms = started.elapsed().as_secs_f64() * 1e3;
        Ok(out)
    }

    /// Grow the row scratch to `rows` (rounded up to 64).
    fn ensure_rows(&mut self, rows: usize) -> Result<(), String> {
        let cap = (rows.max(1) + 63) / 64 * 64;
        if self.rows.as_ref().map(|r| r.cap).unwrap_or(0) >= rows {
            return Ok(());
        }
        self.rows = Some(RowScratch::new(&self.cfg, cap, self.slots)?);
        Ok(())
    }

    /// Bytes the row scratch holds (0 before the first row block).
    pub fn row_bytes(&self) -> usize {
        self.rows.as_ref().map(RowScratch::bytes).unwrap_or(0)
    }

    /// One slot's carried SSM state, layer by layer (conv window then matrix),
    /// for a bit-exact comparison against `GpuEngine::ssm_state_host`.
    pub fn slot_ssm_state_host(&self, slot: usize) -> Result<Vec<f32>, String> {
        let mut out = Vec::new();
        for b in &self.layers {
            if let BatchKind::Ssm(s) = &b.kind {
                let (conv, state) = s.inst.state_lens();
                let cw = s.conv_state.len() / self.slots;
                let sw = s.s_state.len() / self.slots;
                out.extend(bytes_to_f32(&s.conv_state.download_bytes(
                    slot * cw * 4,
                    conv * 4,
                )?));
                out.extend(bytes_to_f32(&s.s_state.download_bytes(
                    slot * sw * 4,
                    state * 4,
                )?));
            }
        }
        Ok(out)
    }

    fn forward_rows(
        &mut self,
        cfg: &Config,
        dims: &SsmDims,
        r: &RowScratch,
        tails: &[(usize, &[u32])],
    ) -> Result<Vec<u32>, String> {
        let l = self.model.launcher();
        let n = cfg.n_embd;
        let stream = l.stream();
        let embed = self.embed;
        let m: usize = tails.iter().map(|(_, t)| t.len()).sum();

        // Row maps: slot id and local token index per row, plus each tail's
        // first row / slot / length for the per-tail launches.
        let mut row_slot = vec![0i32; r.cap];
        let mut row_local = vec![0i32; r.cap];
        let mut tail_slot = Vec::with_capacity(tails.len());
        let mut tail_base = Vec::with_capacity(tails.len());
        let mut tail_len = Vec::with_capacity(tails.len());
        let mut base = 0usize;
        for (slot, tok) in tails {
            for t in 0..tok.len() {
                row_slot[base + t] = *slot as i32;
                row_local[base + t] = t as i32;
            }
            tail_slot.push(*slot);
            tail_base.push(base);
            tail_len.push(tok.len());
            base += tok.len();
        }
        let bytes = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        r.row_slot.upload_bytes_async(0, &bytes(&row_slot), stream)?;
        r.row_local.upload_bytes_async(0, &bytes(&row_local), stream)?;

        // Embed every tail token at its row.
        let mut row = 0usize;
        for (_, tok) in tails {
            for &id in tok.iter() {
                l.embed_row_q4_k_at(embed.0, id as usize, embed.1, &r.x, row * n, n)?;
                row += 1;
            }
        }

        let conv_stride = dims.d_conv.saturating_sub(1) * dims.conv_ch;
        let s_stride = dims.n_v * dims.head_v * dims.head_v;
        let hist = dims.d_conv - 1;

        let trace = std::env::var_os("HIPML_ROWS_TRACE").is_some();
        let mut marks: Vec<(&'static str, f64)> = Vec::new();
        let mut last = Instant::now();
        macro_rules! mark {
            ($name:expr) => {
                if trace {
                    l.synchronize()?;
                    let now = Instant::now();
                    marks.push(($name, now.duration_since(last).as_secs_f64() * 1e3));
                    last = now;
                }
            };
        }

        l.rmsnorm_rows(&r.x, &self.layers[0].attn_norm, &r.t, n, cfg.eps, m)?;
        mark!("norm");
        for i in 0..self.layers.len() {
            let block = &self.layers[i];
            let block_out: &Buffer = match &block.kind {
                BatchKind::Full(f) => {
                    f.attn.project_chunk(l, &r.t, &r.xh, &r.qg, &r.k, &r.v, m)?;
                    mark!("attn proj");
                    let kv = &f.kv;
                    let dtype = kv.dtype.id();
                    let qw = cfg.n_head * cfg.head_dim;
                    let kw = kv.n_kv * kv.head_dim;
                    let slot_words = kv.slot_bytes() / 4;
                    for i in 0..tails.len() {
                        let start = self.positions[tail_slot[i]];
                        let first = tail_base[i];
                        let len = tail_len[i];
                        let cache = tail_slot[i] * slot_words;
                        f.attn.chunk_front_rows(
                            l,
                            at(&r.qg, first * 2 * qw),
                            at(&r.k, first * kw),
                            at(&r.q, first * qw),
                            at(&r.agate, first * qw),
                            start,
                            len,
                        )?;
                        f.attn.kv_encode_bulk_rows(
                            l,
                            at(&r.k, first * kw),
                            at(&r.v, first * kw),
                            at(&kv.k, cache),
                            at(&kv.v, cache),
                            dtype,
                            kv.n_kv,
                            kv.head_dim,
                            kv.seq,
                            start,
                            len,
                        )?;
                        f.attn.chunk_attention_rows(
                            l,
                            at(&r.q, first * qw),
                            at(&kv.k, cache),
                            at(&kv.v, cache),
                            at(&r.aout, first * qw),
                            start,
                            len,
                            dtype + 1,
                            kv.seq,
                        )?;
                    }
                    mark!("attn core");
                    f.attn.apply_output_chunk(l, &r.aout, &r.agate, &r.y, m)?;
                    mark!("attn out");
                    &r.y
                }
                BatchKind::Ssm(s) => {
                    s.inst.project_chunk(l, &r.t, &r.xh, &r.ssm, m)?;
                    mark!("ssm proj");
                    s.inst.gates_rows(l, &r.ssm, m)?;
                    let conv_ch = dims.conv_ch;
                    for i in 0..tails.len() {
                        let slot = tail_slot[i];
                        let first = tail_base[i];
                        let len = tail_len[i];
                        s.inst.conv_chunk_at(
                            l,
                            at(&r.ssm.qkv, first * conv_ch),
                            at(&s.conv_state, slot * conv_stride),
                            at(&r.ssm.conv_next, 0),
                            at(&r.ssm.conv, first * conv_ch),
                            len,
                        )?;
                        r.ssm.conv_next.copy_range_on(
                            &s.conv_state,
                            0,
                            slot * conv_stride,
                            hist * conv_ch,
                            stream,
                        )?;
                    }
                    mark!("ssm conv");
                    s.inst.l2norm_rows(l, &r.ssm, m)?;
                    let inner = dims.inner();
                    for i in 0..tails.len() {
                        let slot = tail_slot[i];
                        let first = tail_base[i];
                        let len = tail_len[i];
                        s.inst.gdn_rows_at(
                            l,
                            at(&s.s_state, slot * s_stride),
                            at(&r.ssm.conv, first * conv_ch),
                            at(&r.ssm.g, first * dims.n_v),
                            at(&r.ssm.beta_sig, first * dims.n_v),
                            at(&r.ssm.o, first * inner),
                            len,
                        )?;
                    }
                    mark!("ssm rec");
                    s.inst.gate_norm_rows(l, &r.ssm, m)?;
                    s.inst.out_chunk(l, &r.xh, &r.ssm, m)?;
                    mark!("ssm out");
                    &r.ssm.out
                }
            };
            l.residual_buf(&r.x, block_out, &r.x1, m * n)?;
            l.rmsnorm_rows(&r.x1, &block.post_norm, &r.t, n, cfg.eps, m)?;
            if ::hipml_kernels::prefill_v2() {
                ::hipml_kernels::prefill_xhalf(l.module(), l.stream(), &r.t, &r.xh, m * n)?;
                block.gate.gemm_half(l, &r.xh, &r.gate, &r.sk, m)?;
                block.up.gemm_half(l, &r.xh, &r.up, &r.sk, m)?;
                l.silu_mul_buf(&r.gate, &r.up, &r.act, m * cfg.n_ff)?;
                ::hipml_kernels::prefill_xhalf(
                    l.module(),
                    l.stream(),
                    &r.act,
                    &r.xh,
                    m * cfg.n_ff,
                )?;
                block.down.gemm_half(l, &r.xh, &r.down, &r.sk, m)?;
            } else {
                block.gate.gemm(l, &r.t, &r.gate, m)?;
                block.up.gemm(l, &r.t, &r.up, m)?;
                l.silu_mul_buf(&r.gate, &r.up, &r.act, m * cfg.n_ff)?;
                block.down.gemm(l, &r.act, &r.down, m)?;
            }
            l.residual_buf(&r.down, &r.x1, &r.x, m * n)?;
            mark!("ffn");
        }

        // Only each tail's last row needs logits: gather those rows into the
        // first `tails.len()` rows, then one output projection and one argmax.
        for i in 0..tails.len() {
            let src = tail_base[i] + tail_len[i] - 1;
            r.x.copy_range_on(&self.final_h, src * n, i * n, n, stream)?;
        }
        l.rmsnorm_rows(&self.final_h, &self.output_norm, &r.t, n, cfg.eps, tails.len())?;
        let out = self
            .model
            .tensor(crate::OUTPUT)
            .ok_or_else(|| "model has no output.weight".to_string())?;
        let kind = crate::gpu_weights::batch_kind(gemv_entry(out.ty())?)
            .ok_or_else(|| "lm_head has no batched GEMV".to_string())?;
        l.gemv_batched(
            kind,
            out.buffer(),
            &r.t,
            &self.logits,
            cfg.n_vocab,
            n,
            tails.len(),
        )?;
        l.argmax_batched(
            &self.logits,
            &self.arg_v,
            &self.arg_i,
            &self.arg_out,
            cfg.n_vocab,
            ARGMAX_PARTS,
            tails.len(),
        )?;
        mark!("final");
        l.synchronize()?;
        if trace {
            let total: f64 = marks.iter().map(|(_, ms)| ms).sum();
            eprintln!(
                "rows: m={m} tails={} total {total:.2} ms | {}",
                tails.len(),
                marks
                    .iter()
                    .map(|(n, ms)| format!("{n} {ms:.2}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }

        for i in 0..tails.len() {
            self.positions[tail_slot[i]] += tail_len[i];
        }
        let bytes = self.arg_out.download_bytes(0, 4 * tails.len())?;
        Ok((0..tails.len())
            .map(|i| {
                let o = i * 4;
                u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]])
            })
            .collect())
    }
}
