//! Multi-sequence decode: B token rows through one weight pass.
//!
//! The single-stream engine assumes one sequence everywhere: one KV cache per
//! layer, one SSM conv window and delta-net matrix per layer, one position.
//! This module makes the state an array of slots and converts the decode step
//! to a batched GEMV over the B x-vectors, so the 17 GB weight stream serves B
//! tokens instead of one.
//!
//! Per-slot state:
//!   * full attention: one `DeepKv` quantized cache per (layer, slot),
//!   * SSM: one `SsmGpuLayer` per (layer, slot), which owns that slot's conv
//!     window and delta-net matrix (its weights are borrowed, not copied),
//!   * position: `positions[slot]`.
//!
//! The projections run as one batched launch per weight; attention, the SSM
//! recurrence, KV appends and sampling stay per slot. Every batched kernel
//! preserves its row's accumulation order, so slot b is numerically identical
//! to the single-sequence path on the same input.

use super::layers::{build_attn, build_ssm};
use super::*;

#[path = "batch/forward.rs"]
mod forward;

/// Slot bookkeeping. Capacity is fixed when the engine is built; allocation
/// just hands out the indices, so nothing in the forward path assumes slot 0
/// is the only sequence.
pub struct SlotPool {
    capacity: usize,
    used: Vec<bool>,
}

impl SlotPool {
    pub fn new(capacity: usize) -> SlotPool {
        SlotPool {
            capacity,
            used: vec![false; capacity],
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn in_use(&self) -> usize {
        self.used.iter().filter(|u| **u).count()
    }

    /// Lowest free slot, or `None` when the pool is full.
    pub fn alloc(&mut self) -> Option<usize> {
        let s = self.used.iter().position(|u| !u)?;
        self.used[s] = true;
        Some(s)
    }

    pub fn free(&mut self, slot: usize) {
        if slot < self.capacity {
            self.used[slot] = false;
        }
    }
}

struct BatchBlock<'a> {
    attn_norm: Buffer,
    post_norm: Buffer,
    gate: Proj<'a>,
    up: Proj<'a>,
    down: Proj<'a>,
    kind: BatchKind<'a>,
}

enum BatchKind<'a> {
    Full(FullBatch<'a>),
    Ssm(SsmBatch<'a>),
}

/// One full-attention layer: one `AttnLayer` (shared projections + row scratch)
/// plus one quantized KV cache per slot.
struct FullBatch<'a> {
    attn: AttnLayer<'a>,
    deep: Vec<DeepKv>,
    qg: Buffer,
    k: Buffer,
    v: Buffer,
    q: Buffer,
    gate: Buffer,
    out: Buffer,
    y: Buffer,
}

/// One SSM layer: a `SsmGpuLayer` per slot (state), and the batched projection
/// scratch shared by all slots.
struct SsmBatch<'a> {
    per_slot: Vec<SsmGpuLayer<'a>>,
    qkv: Buffer,
    z: Buffer,
    alpha: Buffer,
    beta: Buffer,
    g: Buffer,
    beta_sig: Buffer,
    conv: Buffer,
    o: Buffer,
    gated: Buffer,
    out: Buffer,
}

/// A whole-model batched decoder. `BatchEngine::new(model, n_ctx, kv, slots)`
/// allocates the per-slot state; `forward` decodes one token for each active
/// slot and returns the B greedy ids.
pub struct BatchEngine<'a> {
    model: &'a Model,
    pub cfg: Config,
    pub kv_mode: KvMode,
    pub slots: usize,
    layers: Vec<BatchBlock<'a>>,
    output_norm: Buffer,
    x: Buffer,
    x1: Buffer,
    t: Buffer,
    gate: Buffer,
    up: Buffer,
    act: Buffer,
    down: Buffer,
    final_h: Buffer,
    logits: Buffer,
    embed: (&'a ByteBuffer, usize),
    arg_v: Buffer,
    arg_i: Buffer,
    arg_out: Buffer,
    pool: SlotPool,
    positions: Vec<usize>,
    active: Vec<usize>,
    /// Stage timing (sync at group boundaries); costs a sync per group.
    pub trace: bool,
    /// Seconds per stage accumulated by `forward` when `trace` is on.
    pub stages: [f64; 9],
    /// Split alternating slots across the two streams. `HIPML_BATCH_NOSPLIT`
    /// disables it for an A/B.
    pub overlap: bool,
}

impl<'a> BatchEngine<'a> {
    pub fn new(
        model: &'a Model,
        n_ctx: usize,
        kv_mode: KvMode,
        slots: usize,
    ) -> Result<BatchEngine<'a>, String> {
        if slots == 0 || slots > 8 {
            return Err(format!("slots={slots}: batched kernels cover 1..=8"));
        }
        if kv_mode == KvMode::F32 {
            return Err("batch decode needs --kv q8 or q4 (the f32 cache is the single-stream oracle)".to_string());
        }
        let cfg = {
            let mut g = Gguf::open(model.path()).map_err(|e| format!("{e}"))?;
            Config::from_gguf(&mut g).map_err(|e| format!("config: {e}"))?
        };
        let l = model.launcher();
        let n = cfg.n_embd;
        let dims = SsmDims::from_config(&cfg);
        let embed = model
            .tensor(crate::TOKEN_EMBD)
            .ok_or_else(|| "model has no token_embd.weight".to_string())?;
        if embed.ty() != hipml_core::quant::TensorType::Q4K {
            return Err(format!("{}: expected Q4_K", crate::TOKEN_EMBD));
        }
        let embed = (embed.buffer(), (embed.nbytes() / embed.rows()) as usize);
        let zeros = |count: usize| Buffer::alloc(count);
        let overlap = std::env::var_os("HIPML_BATCH_NOSPLIT").is_none();
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let attn_norm = Buffer::from_host(&model.f32_vec(&format!("blk.{i}.attn_norm.weight"))?)?;
            let post_norm =
                Buffer::from_host(&model.f32_vec(&format!("blk.{i}.post_attention_norm.weight"))?)?;
            let gate = Proj::of(model, &format!("blk.{i}.ffn_gate.weight"))?;
            let up = Proj::of(model, &format!("blk.{i}.ffn_up.weight"))?;
            let down = Proj::of(model, &format!("blk.{i}.ffn_down.weight"))?;
            let kind = if cfg.is_recr[i] {
                let mut per_slot = Vec::with_capacity(slots);
                for b in 0..slots {
                    // Odd slots launch their recurrence on the side stream, so
                    // the two halves overlap instead of serialising 8 small
                    // kernels at a time.
                    per_slot.push(build_ssm(model, l, &cfg, i, overlap && b % 2 == 1)?);
                }
                BatchKind::Ssm(SsmBatch {
                    per_slot,
                    qkv: zeros(slots * dims.conv_ch)?,
                    z: zeros(slots * dims.inner())?,
                    alpha: zeros(slots * dims.n_v)?,
                    beta: zeros(slots * dims.n_v)?,
                    g: zeros(slots * dims.n_v)?,
                    beta_sig: zeros(slots * dims.n_v)?,
                    conv: zeros(slots * dims.conv_ch)?,
                    o: zeros(slots * dims.inner())?,
                    gated: zeros(slots * dims.inner())?,
                    out: zeros(slots * n)?,
                })
            } else {
                let mut deep = Vec::with_capacity(slots);
                for _ in 0..slots {
                    deep.push(DeepKv::new(&cfg, n_ctx, kv_mode.dtype().unwrap())?);
                }
                let qw = cfg.n_head * cfg.head_dim;
                let width = cfg.n_head_kv * cfg.head_dim;
                BatchKind::Full(FullBatch {
                    // 1 mirror row: the quantized path never reads the f32 cache.
                    attn: build_attn(model, l, &cfg, i, 1)?,
                    deep,
                    qg: zeros(slots * 2 * qw)?,
                    k: zeros(slots * width)?,
                    v: zeros(slots * width)?,
                    q: zeros(slots * qw)?,
                    gate: zeros(slots * qw)?,
                    out: zeros(slots * qw)?,
                    y: zeros(slots * n)?,
                })
            };
            layers.push(BatchBlock {
                attn_norm,
                post_norm,
                gate,
                up,
                down,
                kind,
            });
        }
        let output_norm = Buffer::from_host(&model.f32_vec("output_norm.weight")?)?;
        Ok(BatchEngine {
            model,
            cfg: cfg.clone(),
            kv_mode,
            slots,
            layers,
            output_norm,
            x: zeros(slots * n)?,
            x1: zeros(slots * n)?,
            t: zeros(slots * n)?,
            gate: zeros(slots * cfg.n_ff)?,
            up: zeros(slots * cfg.n_ff)?,
            act: zeros(slots * cfg.n_ff)?,
            down: zeros(slots * n)?,
            final_h: zeros(slots * n)?,
            logits: zeros(slots * cfg.n_vocab)?,
            embed,
            arg_v: zeros(slots * ARGMAX_PARTS)?,
            arg_i: zeros(slots * ARGMAX_PARTS)?,
            arg_out: zeros(slots)?,
            pool: SlotPool::new(slots),
            positions: vec![0; slots],
            active: Vec::new(),
            trace: false,
            stages: [0.0; 9],
            overlap,
        })
    }

    /// Allocate `n` slots, lowest indices first; the forward path then uses
    /// exactly those. Also used by the benchmark to reserve the whole pool.
    pub fn activate(&mut self, n: usize) -> Result<(), String> {
        if n > self.slots {
            return Err(format!("activate {n}: only {} slots", self.slots));
        }
        self.active.clear();
        for _ in 0..n {
            self.active.push(self.pool.alloc().ok_or("slot pool exhausted")?);
        }
        Ok(())
    }

    pub fn active(&self) -> &[usize] {
        &self.active
    }

    pub fn position(&self, slot: usize) -> usize {
        self.positions[slot]
    }

    /// Zero every slot's SSM state (a cold start on a reused engine).
    pub fn reset_state(&self) -> Result<(), String> {
        for b in &self.layers {
            if let BatchKind::Ssm(s) = &b.kind {
                for layer in &s.per_slot {
                    layer.conv_state_buf().zero()?;
                    layer.s_state_buf().zero()?;
                }
            }
        }
        self.model.launcher().synchronize()
    }

    /// Fill each active slot's KV cache to `depth` positions with valid blocks.
    /// `depth[b]` is the depth for active slot `b`.
    pub fn fill_synthetic(&self, depth: &[usize]) -> Result<(), String> {
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        for b in &self.layers {
            if let BatchKind::Full(f) = &b.kind {
                for (bi, &slot) in self.active.iter().enumerate() {
                    f.deep[slot].fill_synthetic(depth[bi], seed)?;
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                }
            }
        }
        Ok(())
    }

    pub fn set_positions(&mut self, pos: &[usize]) -> Result<(), String> {
        if pos.len() != self.active.len() {
            return Err(format!("{} positions for {} slots", pos.len(), self.active.len()));
        }
        for (bi, &p) in pos.iter().enumerate() {
            self.positions[self.active[bi]] = p;
        }
        Ok(())
    }

    /// Bytes streamed from the resident weights per decode step (one pass,
    /// shared by every slot).
    pub fn weight_bytes(&self) -> usize {
        let mut b = 0usize;
        for l in &self.layers {
            b += l.gate.bytes() + l.up.bytes() + l.down.bytes();
            b += match &l.kind {
                BatchKind::Full(f) => f.attn.weight_bytes(),
                BatchKind::Ssm(s) => s.per_slot[0].weight_bytes(),
            };
        }
        b + self
            .model
            .tensor(crate::OUTPUT)
            .map(|t| t.nbytes() as usize)
            .unwrap_or(0)
    }

    /// Quantized KV bytes for all slots at the engine's depth.
    pub fn kv_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|b| match &b.kind {
                BatchKind::Full(f) => f.deep.iter().map(|d| d.bytes()).sum::<usize>(),
                BatchKind::Ssm(_) => 0,
            })
            .sum()
    }

    /// Recurrent-state bytes (conv window + delta-net matrix) for all slots.
    pub fn ssm_state_bytes(&self) -> usize {
        let dims = SsmDims::from_config(&self.cfg);
        let n = self.layers.len();
        let _ = n;
        self.layers
            .iter()
            .filter(|b| matches!(b.kind, BatchKind::Ssm(_)))
            .count()
            * self.slots
            * dims.state_floats()
            * 4
    }

    pub fn model(&self) -> &Model {
        self.model
    }
}

/// Device pointer of row `row` of a `[rows][width]` f32 buffer.
pub(super) fn row_ptr(buf: &Buffer, row: usize, width: usize) -> *mut std::ffi::c_void {
    unsafe { (buf.as_ptr() as *mut u8).add(row * width * 4) as *mut std::ffi::c_void }
}
