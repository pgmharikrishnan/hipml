// SPDX-License-Identifier: Apache-2.0
//! A bounded prefix cache for prefill state.
//!
//! A hybrid model's prefill state is not just the attention KV rows: the SSM
//! blocks carry a fixed-size recurrent state (conv window + delta-net matrix)
//! that also has to be restored at a prefix boundary. This module is the
//! host-side bookkeeping for that: it maps an exact token prefix to an opaque
//! payload (the device snapshot, in `hipml-model`), finds the longest cached
//! prefix of a prompt, and evicts least-recently-used entries under an entry
//! and byte bound.
//!
//! The key is the token prefix itself, not only its hash: a 64-bit hash is a
//! useful dedup/print field, but correctness cannot depend on no collision.
//! Lookups compare lengths and then the id slices, so a hash collision can
//! never serve the wrong state.

/// FNV-1a over each token's little-endian bytes, with the length folded in.
///
/// This is an identity tag for reporting and duplicate detection, not a
/// security boundary. Exact prefix comparison decides a hit.
pub fn prefix_hash(tokens: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    h ^= tokens.len() as u64;
    h = h.wrapping_mul(0x0000_0100_0000_01b3);
    for &t in tokens {
        for b in t.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// One cached prefix: the exact token ids and the payload they produced.
pub struct Entry<T> {
    pub tokens: Vec<u32>,
    pub hash: u64,
    pub bytes: usize,
    pub payload: T,
    last_used: u64,
}

/// One lookup result, before the payload is borrowed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hit {
    /// Index of the matching entry.
    pub index: usize,
    /// Prefix length the entry covers.
    pub pos: usize,
}

/// What an insert did.
///
/// Admission is all-or-nothing: a payload larger than `max_bytes` is refused
/// before it is published. It is never truncated, and the refusal is counted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InsertOutcome {
    /// An entry with the identical token prefix was replaced.
    pub replaced: bool,
    /// Number of entries evicted to stay inside the bound.
    pub evicted: usize,
    /// The payload alone exceeded `max_bytes`; nothing was stored.
    pub refused: bool,
    /// Bytes the entry would have occupied (also set on refusal).
    pub bytes: usize,
}

/// Cumulative counters, for the hit/miss report.
#[derive(Clone, Copy, Debug, Default)]
pub struct CacheStats {
    pub entries: usize,
    pub bytes: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    /// Captures refused because a single entry exceeded the byte bound.
    pub refused: u64,
}

/// LRU prefix cache with an entry bound and a byte bound.
pub struct PrefixCache<T> {
    entries: Vec<Entry<T>>,
    max_entries: usize,
    max_bytes: usize,
    bytes: usize,
    clock: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
    refused: u64,
}

impl<T> PrefixCache<T> {
    /// `max_entries` and `max_bytes` are both hard bounds; an entry larger than
    /// the whole cache is refused rather than silently violating the bound.
    pub fn new(max_entries: usize, max_bytes: usize) -> PrefixCache<T> {
        PrefixCache {
            entries: Vec::new(),
            max_entries: max_entries.max(1),
            max_bytes: max_bytes.max(1),
            bytes: 0,
            clock: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
            refused: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn max_entries(&self) -> usize {
        self.max_entries
    }

    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    pub fn entry(&self, index: usize) -> &Entry<T> {
        &self.entries[index]
    }

    pub fn stats(&self) -> CacheStats {
        CacheStats {
            entries: self.entries.len(),
            bytes: self.bytes,
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            refused: self.refused,
        }
    }

    /// Longest cached prefix of `prompt`, if any. Touches the entry (LRU).
    ///
    /// An entry qualifies only when its token ids are an exact prefix:
    /// `entry.tokens.len() <= prompt.len()` and the slices are equal. Ties
    /// cannot happen because identical prefixes are replaced on insert.
    pub fn find_longest(&mut self, prompt: &[u32]) -> Option<Hit> {
        self.find_longest_where(prompt, |_| true)
    }

    /// Like [`find_longest`](Self::find_longest), but an entry only qualifies
    /// when `valid` accepts it. The engine uses this to reject a snapshot whose
    /// recorded position does not match the entry's token length: a corrupt or
    /// half-published entry is a miss, never a hit.
    pub fn find_longest_where(
        &mut self,
        prompt: &[u32],
        valid: impl Fn(&Entry<T>) -> bool,
    ) -> Option<Hit> {
        let mut best: Option<(usize, usize)> = None; // (len, index)
        for (i, e) in self.entries.iter().enumerate() {
            let n = e.tokens.len();
            if n > prompt.len() || !valid(e) {
                continue;
            }
            if let Some((best_len, _)) = best {
                if n <= best_len {
                    continue;
                }
            }
            if prompt[..n] == e.tokens[..] {
                best = Some((n, i));
            }
        }
        match best {
            Some((pos, index)) => {
                self.clock += 1;
                self.entries[index].last_used = self.clock;
                self.hits += 1;
                Some(Hit { index, pos })
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    /// Does an entry with exactly these token ids already exist?
    pub fn contains(&self, tokens: &[u32]) -> bool {
        self.entries.iter().any(|e| e.tokens == tokens)
    }

    /// Make room for a `bytes`-sized entry, evicting LRU entries when that is
    /// enough. Returns the number evicted, or `None` when a single entry cannot
    /// fit in the whole cache (counted as a refusal, nothing evicted).
    ///
    /// Callers use this *before* allocating the payload, so an oversized
    /// capture costs no device memory and leaves no half-built entry behind.
    pub fn reserve(&mut self, bytes: usize) -> Option<usize> {
        if bytes > self.max_bytes {
            self.refused += 1;
            return None;
        }
        let mut evicted = 0;
        while (self.bytes + bytes > self.max_bytes || self.entries.len() + 1 > self.max_entries)
            && !self.entries.is_empty()
        {
            self.evict_lru();
            evicted += 1;
        }
        if self.bytes + bytes > self.max_bytes || self.entries.len() + 1 > self.max_entries {
            self.refused += 1;
            return None;
        }
        Some(evicted)
    }

    /// Insert or replace `tokens -> payload`, evicting LRU entries until the
    /// entry and byte bounds hold. An oversized payload is refused and counted.
    pub fn insert(&mut self, tokens: Vec<u32>, bytes: usize, payload: T) -> InsertOutcome {
        let mut out = InsertOutcome {
            bytes,
            ..InsertOutcome::default()
        };
        if bytes > self.max_bytes {
            self.refused += 1;
            out.refused = true;
            return out;
        }

        self.clock += 1;
        let hash = prefix_hash(&tokens);

        if let Some(i) = self.entries.iter().position(|e| e.tokens == tokens) {
            self.bytes -= self.entries[i].bytes;
            self.bytes += bytes;
            let e = &mut self.entries[i];
            e.hash = hash;
            e.bytes = bytes;
            e.payload = payload;
            e.last_used = self.clock;
            out.replaced = true;
            out.evicted = self.evict_if_needed();
            return out;
        }

        self.entries.push(Entry {
            tokens,
            hash,
            bytes,
            payload,
            last_used: self.clock,
        });
        self.bytes += bytes;
        out.evicted = self.evict_if_needed();
        out
    }

    fn evict_if_needed(&mut self) -> usize {
        let mut evicted = 0;
        while self.entries.len() > self.max_entries || self.bytes > self.max_bytes {
            if !self.evict_lru() {
                break;
            }
            evicted += 1;
        }
        evicted
    }

    /// Drop the least-recently-used entry. Returns false when the cache is
    /// empty. An oversized single entry is refused before it is pushed, so a
    /// victim always exists while the bounds are exceeded.
    fn evict_lru(&mut self) -> bool {
        let victim = self
            .entries
            .iter()
            .enumerate()
            .min_by_key(|(_, e)| e.last_used)
            .map(|(i, _)| i);
        let Some(victim) = victim else { return false };
        self.bytes -= self.entries[victim].bytes;
        self.entries.remove(victim);
        self.evictions += 1;
        true
    }

    /// Remove every entry (and its payload). Counters are kept.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.bytes = 0;
    }

    /// Remove every entry whose token ids are a prefix of `prompt`, including
    /// an exact match, and return how many went.
    ///
    /// Used when a request dies part-way through its prefill: the snapshots it
    /// captured are the ones a later request with the same prefix would
    /// restore, and a failed job's device state cannot be vouched for. Entries
    /// for unrelated prefixes are left alone, so one bad request does not cost
    /// every other conversation its cache.
    pub fn invalidate_prefixes(&mut self, prompt: &[u32]) -> usize {
        let mut removed = 0;
        let mut bytes = 0;
        self.entries.retain(|e| {
            let hit = e.tokens.len() <= prompt.len() && prompt[..e.tokens.len()] == e.tokens[..];
            if hit {
                removed += 1;
                bytes += e.bytes;
            }
            !hit
        });
        self.bytes -= bytes;
        removed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push(c: &mut PrefixCache<u32>, tokens: &[u32], payload: u32, bytes: usize) -> InsertOutcome {
        c.insert(tokens.to_vec(), bytes, payload)
    }

    #[test]
    fn hash_is_stable_and_length_sensitive() {
        assert_eq!(prefix_hash(&[1, 2, 3]), prefix_hash(&[1, 2, 3]));
        assert_ne!(prefix_hash(&[1, 2]), prefix_hash(&[1, 2, 0]));
        assert_ne!(prefix_hash(&[1, 2, 3]), prefix_hash(&[1, 2, 4]));
        assert_eq!(prefix_hash(&[]), prefix_hash(&[]));
    }

    #[test]
    fn lookup_returns_the_longest_exact_prefix() {
        let mut c = PrefixCache::new(4, 1 << 20);
        push(&mut c, &[1, 2], 20, 10);
        push(&mut c, &[1, 2, 3, 4], 40, 10);
        push(&mut c, &[9], 90, 10);

        let h = c.find_longest(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(h.pos, 4);
        assert_eq!(c.entry(h.index).payload, 40);

        // Exact repeat prefers the full-length entry.
        let h = c.find_longest(&[1, 2, 3, 4]).unwrap();
        assert_eq!(h.pos, 4);

        // A prompt shorter than the long entry falls back to the short one.
        let h = c.find_longest(&[1, 2, 9]).unwrap();
        assert_eq!(h.pos, 2);

        // No shared first token is a miss.
        assert!(c.find_longest(&[7, 7]).is_none());
        let s = c.stats();
        assert_eq!(s.hits, 3);
        assert_eq!(s.misses, 1);
    }

    #[test]
    fn prefix_must_match_the_whole_slice_not_a_hash() {
        let mut c = PrefixCache::new(4, 1 << 20);
        push(&mut c, &[1, 2, 3], 30, 10);
        // Same length, different ids: not a hit.
        assert!(c.find_longest(&[1, 2, 4]).is_none());
    }

    #[test]
    fn invalidate_drops_only_the_failed_prompts_prefixes() {
        let mut c = PrefixCache::new(8, 1 << 20);
        push(&mut c, &[1, 2], 20, 10);
        push(&mut c, &[1, 2, 3], 30, 10);
        push(&mut c, &[1, 9], 19, 10);
        push(&mut c, &[7], 70, 10);
        // A failed job for [1,2,3,4] leaves the two entries that are prefixes
        // of it, and nothing else.
        assert_eq!(c.invalidate_prefixes(&[1, 2, 3, 4]), 2);
        assert_eq!(c.len(), 2);
        assert_eq!(c.bytes(), 20);
        assert!(c.find_longest(&[1, 2, 3, 4]).is_none());
        assert!(c.find_longest(&[1, 9]).is_some());
        assert!(c.find_longest(&[7]).is_some());
        // A prompt that is a prefix of nothing removes nothing.
        assert_eq!(c.invalidate_prefixes(&[5]), 0);
    }

    #[test]
    fn lru_evicts_by_entry_bound() {
        let mut c = PrefixCache::new(2, 1 << 20);
        push(&mut c, &[1], 1, 10);
        push(&mut c, &[2], 2, 10);
        // Touch 1 so 2 is the LRU victim.
        assert_eq!(c.find_longest(&[1]).unwrap().pos, 1);
        let out = push(&mut c, &[3], 3, 10);
        assert_eq!(out.evicted, 1);
        assert_eq!(c.len(), 2);
        assert!(c.find_longest(&[2]).is_none());
        assert!(c.find_longest(&[1]).is_some());
        assert!(c.find_longest(&[3]).is_some());
        assert_eq!(c.stats().evictions, 1);
    }

    #[test]
    fn byte_bound_evicts_and_large_payload_is_refused() {
        let mut c = PrefixCache::new(10, 100);
        push(&mut c, &[1], 1, 60);
        push(&mut c, &[2], 2, 60);
        // 60 + 60 > 100, so the older entry leaves.
        assert_eq!(c.len(), 1);
        assert_eq!(c.bytes(), 60);

        let out = push(&mut c, &[3], 3, 101);
        assert!(out.refused);
        assert_eq!(out.bytes, 101);
        assert_eq!(c.len(), 1);
        assert_eq!(c.stats().refused, 1);
    }

    #[test]
    fn reserve_evicts_before_it_refuses() {
        let mut c = PrefixCache::new(10, 100);
        push(&mut c, &[1], 1, 40);
        push(&mut c, &[2], 2, 40);
        // Room is made by evicting LRU; the capture is then admitted.
        assert_eq!(c.reserve(80), Some(2));
        assert_eq!(c.stats().evictions, 2);
        assert_eq!(c.stats().refused, 0);
        // A single entry that cannot fit is refused without evicting.
        let before = c.stats().evictions;
        assert_eq!(c.reserve(101), None);
        assert_eq!(c.stats().evictions, before);
        assert_eq!(c.stats().refused, 1);
    }

    #[test]
    fn lookup_skips_entries_the_validator_rejects() {
        let mut c = PrefixCache::new(4, 1 << 20);
        push(&mut c, &[1, 2], 20, 10);
        push(&mut c, &[1, 2, 3], 30, 10);
        // Treat the longer entry as half-published: lookups fall back to the
        // shorter prefix instead of serving the invalid one.
        let h = c
            .find_longest_where(&[1, 2, 3, 4], |e| e.payload != 30)
            .unwrap();
        assert_eq!(h.pos, 2);
        assert_eq!(c.entry(h.index).payload, 20);
    }

    #[test]
    fn identical_prefix_replaces_instead_of_duplicating() {
        let mut c = PrefixCache::new(2, 1 << 20);
        push(&mut c, &[1, 2], 1, 10);
        let out = push(&mut c, &[1, 2], 2, 30);
        assert!(out.replaced);
        assert_eq!(out.evicted, 0);
        assert_eq!(c.len(), 1);
        assert_eq!(c.bytes(), 30);
        assert_eq!(c.entry(0).payload, 2);
    }
}
