//! Prefix reuse: a KV cache for the full-attention layers and the matcher that
//! decides how much of a prompt is already computed.
//!
//! Qwen3.8-27B is a hybrid: 16 of its 64 blocks are full attention (every fourth block
//! from `blk.3`), the other 48 are SSM blocks. Only the attention blocks have a
//! KV cache that can be addressed by token position, so this cache covers those
//! 16 and nothing else. The SSM blocks carry a fixed-size recurrent state that
//! cannot be sliced by prefix; the model bench re-runs them for the whole
//! prompt, which is the honest cost of caching the attention half.

/// One full-attention layer: `k` and `v` are `[n_kv, capacity, head_dim]`.
pub struct KvLayer {
    pub n_kv: usize,
    pub head_dim: usize,
    pub capacity: usize,
    pub len: usize,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

impl KvLayer {
    fn new(n_kv: usize, head_dim: usize, capacity: usize) -> Self {
        Self {
            n_kv,
            head_dim,
            capacity,
            len: 0,
            k: vec![0.0; n_kv * capacity * head_dim],
            v: vec![0.0; n_kv * capacity * head_dim],
        }
    }

    fn append(&mut self, tokens: usize, k: &[f32], v: &[f32]) -> Result<(), String> {
        let expect = self.n_kv * tokens * self.head_dim;
        if k.len() != expect || v.len() != expect {
            return Err(format!(
                "append {tokens} tokens: k/v must be {expect} values, got {}/{}",
                k.len(),
                v.len()
            ));
        }
        if self.len + tokens > self.capacity {
            return Err(format!(
                "KV cache overflow: {} cached + {tokens} > {}",
                self.len, self.capacity
            ));
        }
        let hd = self.head_dim;
        for h in 0..self.n_kv {
            let src = h * tokens * hd;
            let dst = (h * self.capacity + self.len) * hd;
            self.k[dst..dst + tokens * hd].copy_from_slice(&k[src..src + tokens * hd]);
            self.v[dst..dst + tokens * hd].copy_from_slice(&v[src..src + tokens * hd]);
        }
        self.len += tokens;
        Ok(())
    }
}

/// The cache for every full-attention layer of one model.
pub struct KvCache {
    pub layers: Vec<KvLayer>,
}

impl KvCache {
    /// `n_layers` full-attention blocks, each `n_kv` heads of `head_dim`
    /// values, holding up to `capacity` tokens.
    pub fn new(n_layers: usize, n_kv: usize, head_dim: usize, capacity: usize) -> Self {
        Self {
            layers: (0..n_layers)
                .map(|_| KvLayer::new(n_kv, head_dim, capacity))
                .collect(),
        }
    }

    /// Append `tokens` new positions to one layer. `k`/`v` are
    /// `[n_kv, tokens, head_dim]`, the layout the attention kernel reads.
    pub fn append(
        &mut self,
        layer: usize,
        tokens: usize,
        k: &[f32],
        v: &[f32],
    ) -> Result<(), String> {
        let n = self.layers.len();
        let l = self
            .layers
            .get_mut(layer)
            .ok_or_else(|| format!("layer {layer} is not an attention layer (0..{n})"))?;
        l.append(tokens, k, v)
    }

    /// Positions cached so far (all layers advance together).
    pub fn len(&self) -> usize {
        self.layers.first().map(|l| l.len).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Device-visible bytes the cache occupies: 2 (k and v) * layers * n_kv *
    /// capacity * head_dim * 4, at the configured capacity.
    pub fn bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| 2 * l.k.len() * std::mem::size_of::<f32>())
            .sum()
    }

    pub fn reset(&mut self) {
        for l in &mut self.layers {
            l.len = 0;
        }
    }
}

/// Length of the common prefix of `cached` and `prompt`.
///
/// This is the whole prefix-matching rule: an exact token match from position 0.
/// A mismatch anywhere ends the reusable prefix, because position `i` of the
/// cache was produced with a different token there and every later position
/// depends on it.
pub fn prefix_match(cached: &[u32], prompt: &[u32]) -> usize {
    cached
        .iter()
        .zip(prompt)
        .take_while(|(a, b)| a == b)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_match_stops_at_first_difference() {
        assert_eq!(prefix_match(&[1, 2, 3], &[1, 2, 3]), 3);
        assert_eq!(prefix_match(&[1, 2, 3], &[1, 2, 9]), 2);
        assert_eq!(prefix_match(&[1, 2, 3], &[9, 2, 3]), 0);
        assert_eq!(prefix_match(&[], &[1, 2]), 0);
        assert_eq!(prefix_match(&[1, 2], &[1, 2, 3]), 2);
    }

    #[test]
    fn append_advances_every_layer_and_bounds_the_capacity() {
        let mut c = KvCache::new(2, 2, 3, 4);
        let k = vec![1.0f32; 2 * 2 * 3];
        let v = vec![2.0f32; 2 * 2 * 3];
        c.append(0, 2, &k, &v).unwrap();
        c.append(1, 2, &k, &v).unwrap();
        assert_eq!(c.len(), 2);
        // Second head, first cached position, component 0.
        assert_eq!(c.layers[0].k[(1 * 4) * 3], 1.0);
        assert!(c.append(0, 3, &k, &v).is_err(), "2 + 3 > 4 must fail");
        c.append(0, 2, &k, &v).unwrap();
        assert_eq!(c.layers[0].len, 4);
        c.reset();
        assert_eq!(c.len(), 0);
    }
}
