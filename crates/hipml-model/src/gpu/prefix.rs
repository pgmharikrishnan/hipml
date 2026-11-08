//! Prefix snapshots for the hybrid prefill graph.
//!
//! A qwen35 prefix is summarised by two things: the 16 full-attention layers'
//! KV rows `[0, pos)` and the 48 SSM layers' recurrent state (conv window plus
//! delta-net matrix). The attention half is prefix-sliceable; the recurrent
//! half is the reason KV-only reuse is capped at ~1.38x. This module copies
//! both halves into device buffers keyed by the exact token prefix, so a hit
//! restores them and skips the corresponding prefill tiles entirely.
//!
//! Everything is a device-to-device copy: no state round-trips through the
//! host except the (optional) next-token logits, which are already on the host
//! side of a prefill and make an exact repeat a restore plus an argmax.
//!
//! Only `--kv f32` engines are snapshot targets: the quantized q8/q4 caches
//! have a head-major layout this cache does not copy yet, so a capture against
//! one returns an error rather than copying the wrong bytes.

use super::*;
use hipml_core::prefix_cache::{CacheStats, InsertOutcome, PrefixCache};

/// The two carried buffers of one SSM layer at the snapshot boundary.
pub struct SsmBufs {
    conv: Buffer,
    s: Buffer,
    conv_len: usize,
    s_len: usize,
}

/// The KV rows `[0, pos)` of one full-attention layer.
pub struct KvBufs {
    k: Buffer,
    v: Buffer,
    rows: usize,
    width: usize,
}

/// A complete prefix state, owned by the cache.
pub struct Snapshot {
    pub pos: usize,
    /// Logits after the last cached token, if the capture was at a token
    /// boundary the caller already computed. Reusing them is exact: the prefix
    /// key guarantees the same state.
    pub logits: Option<Vec<f32>>,
    /// Indexed by layer, `Some` for the 48 SSM layers.
    pub ssm: Vec<Option<SsmBufs>>,
    /// Indexed by layer, `Some` for the 16 f32 attention layers.
    pub kv: Vec<Option<KvBufs>>,
}

/// A cache hit as the engine sees it: where to restore from and whether the
/// entry can also answer the next token without a forward pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefixHit {
    pub index: usize,
    pub pos: usize,
    pub has_logits: bool,
}

impl<'a> GpuEngine<'a> {
    /// Turn the cache on. Both bounds are hard; LRU evicts to stay inside them.
    pub fn enable_prefix_cache(&mut self, max_entries: usize, max_bytes: usize) {
        self.prefix = Some(PrefixCache::new(max_entries, max_bytes));
    }

    pub fn prefix_enabled(&self) -> bool {
        self.prefix.is_some()
    }

    pub fn prefix_stats(&self) -> Option<CacheStats> {
        self.prefix.as_ref().map(|c| c.stats())
    }

    pub fn prefix_limits(&self) -> Option<(usize, usize)> {
        self.prefix.as_ref().map(|c| (c.max_entries(), c.max_bytes()))
    }

    /// `(prefix length, bytes, hash)` for every entry, in insertion slot order.
    pub fn prefix_entries(&self) -> Option<Vec<(usize, usize, u64)>> {
        let c = self.prefix.as_ref()?;
        Some(
            (0..c.len())
                .map(|i| {
                    let e = c.entry(i);
                    (e.tokens.len(), e.bytes, e.hash)
                })
                .collect(),
        )
    }

    /// Snapshot the live state as the entry for `prompt[..pos]`.
    ///
    /// The caller must have just prefilled exactly `prompt[..pos]`; `pos` is
    /// not validated against the graph because the graph does not retain the
    /// token ids. `logits` is the logits after token `pos-1`, if known.
    pub fn prefix_capture(
        &mut self,
        prompt: &[u32],
        pos: usize,
        logits: Option<&[f32]>,
    ) -> Result<InsertOutcome, String> {
        if pos == 0 || pos > prompt.len() {
            return Err(format!(
                "prefix capture at {pos} with a {}-token prompt",
                prompt.len()
            ));
        }
        if self.prefix.is_none() {
            return Err("prefix cache is not enabled".to_string());
        }
        // All state writes are kernels on the engine stream; finish them before
        // the copies so a snapshot never catches a half-written tile.
        self.model.launcher().synchronize()?;

        let mut ssm = Vec::with_capacity(self.layers.len());
        let mut kv = Vec::with_capacity(self.layers.len());
        let mut bytes = logits.map(|l| l.len() * 4).unwrap_or(0);

        for b in &self.layers {
            match &b.kind {
                BlockKind::Ssm(s) => {
                    let (cl, sl) = s.state_lens();
                    let conv = Buffer::alloc(cl.max(1))?;
                    let s_buf = Buffer::alloc(sl.max(1))?;
                    conv.copy_from_device(s.conv_state_buf(), cl)?;
                    s_buf.copy_from_device(s.s_state_buf(), sl)?;
                    bytes += (cl + sl) * 4;
                    ssm.push(Some(SsmBufs {
                        conv,
                        s: s_buf,
                        conv_len: cl,
                        s_len: sl,
                    }));
                    kv.push(None);
                }
                BlockKind::Full { deep: None, attn } => {
                    let width = attn.dims.width;
                    let n = pos * width;
                    let k = Buffer::alloc(n.max(1))?;
                    let v = Buffer::alloc(n.max(1))?;
                    k.copy_from_device(attn.kv_k_buf(), n)?;
                    v.copy_from_device(attn.kv_v_buf(), n)?;
                    bytes += 2 * n * 4;
                    ssm.push(None);
                    kv.push(Some(KvBufs {
                        k,
                        v,
                        rows: pos,
                        width,
                    }));
                }
                BlockKind::Full { deep: Some(_), .. } => {
                    return Err("prefix cache covers the f32 KV path, not q8/q4".to_string())
                }
            }
        }

        let snap = Snapshot {
            pos,
            logits: logits.map(|l| l.to_vec()),
            ssm,
            kv,
        };
        Ok(self
            .prefix
            .as_mut()
            .expect("checked above")
            .insert(prompt[..pos].to_vec(), bytes, snap))
    }

    /// Longest cached exact prefix of `prompt`, touching it for LRU.
    pub fn prefix_lookup(&mut self, prompt: &[u32]) -> Option<PrefixHit> {
        let hit = self.prefix.as_mut()?.find_longest(prompt)?;
        let has_logits = self
            .prefix
            .as_ref()
            .and_then(|c| c.entry(hit.index).payload.logits.as_ref())
            .is_some();
        Some(PrefixHit {
            index: hit.index,
            pos: hit.pos,
            has_logits,
        })
    }

    /// Restore the entry into the live layer buffers and set `st.pos`.
    /// Returns the cached logits, if this entry has them.
    pub fn prefix_restore(
        &self,
        hit: PrefixHit,
        st: &mut GpuState,
    ) -> Result<Option<Vec<f32>>, String> {
        let cache = self
            .prefix
            .as_ref()
            .ok_or_else(|| "prefix cache is not enabled".to_string())?;
        let e = cache
            .entry(hit.index)
            .tokens
            .len();
        if e != hit.pos {
            return Err(format!(
                "prefix restore: entry covers {e} tokens, hit says {}",
                hit.pos
            ));
        }
        let snap = &cache.entry(hit.index).payload;
        if snap.pos != hit.pos {
            return Err("prefix restore: snapshot position disagrees with the hit".to_string());
        }

        self.model.launcher().synchronize()?;
        for ((b, ss), kk) in self
            .layers
            .iter()
            .zip(snap.ssm.iter())
            .zip(snap.kv.iter())
        {
            match &b.kind {
                BlockKind::Ssm(s) => {
                    let bufs = ss
                        .as_ref()
                        .ok_or_else(|| "prefix restore: missing SSM snapshot".to_string())?;
                    s.conv_state_buf()
                        .copy_from_device(&bufs.conv, bufs.conv_len)?;
                    s.s_state_buf()
                        .copy_from_device(&bufs.s, bufs.s_len)?;
                }
                BlockKind::Full { deep: None, attn } => {
                    let bufs = kk
                        .as_ref()
                        .ok_or_else(|| "prefix restore: missing KV snapshot".to_string())?;
                    if bufs.width != attn.dims.width {
                        return Err("prefix restore: KV width changed".to_string());
                    }
                    let n = bufs.rows * bufs.width;
                    attn.kv_k_buf().copy_from_device(&bufs.k, n)?;
                    attn.kv_v_buf().copy_from_device(&bufs.v, n)?;
                }
                BlockKind::Full { deep: Some(_), .. } => {
                    return Err("prefix restore covers the f32 KV path".to_string())
                }
            }
        }
        st.pos = snap.pos;
        Ok(snap.logits.clone())
    }

    /// Host copy of the graph's current logits (the last token's).
    pub fn logits_host(&self) -> Result<Vec<f32>, String> {
        self.logits.to_host()
    }
}
