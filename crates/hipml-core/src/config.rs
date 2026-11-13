// SPDX-License-Identifier: Apache-2.0
//! Architecture hyperparameters, read from the GGUF metadata table.
//!
//! Everything the forward pass needs is in `qwen35.*`; nothing here is
//! guessed from tensor shapes except the vocabulary size, which comes from the
//! embedding tensor because that is the one number the metadata does not carry.

use std::io::{self, Read, Seek};

use crate::gguf::Gguf;

/// Qwen3.5-style hybrid decoder configuration.
#[derive(Clone, Debug)]
pub struct Config {
    pub arch: String,
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_vocab: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub rope_base: f32,
    /// Rotated dimensions per head (`rope.dimension_count`, 64 here).
    pub n_rot: usize,
    /// M-RoPE section sizes; all-zero means plain rope.
    pub rope_sections: [usize; 4],
    /// True for gated-delta-net layers, false for full attention.
    pub is_recr: Vec<bool>,
    pub ssm_d_conv: usize,
    pub ssm_d_state: usize,
    pub ssm_n_group: usize,
    pub ssm_dt_rank: usize,
    pub ssm_d_inner: usize,
    pub context_length: usize,
    /// Trailing `nextn` (MTP) blocks in `block_count` that are not decoder
    /// layers. The stock Qwen3.8-27B files carry `blk.64` inline, so
    /// `block_count` is 65 while the decoder is 64 layers deep.
    pub nextn_layers: usize,
}

impl Config {
    pub fn from_gguf<R: Read + Seek>(g: &Gguf<R>) -> io::Result<Config> {
        let arch = g
            .meta_req("general.architecture")?
            .as_str()
            .unwrap_or("qwen35")
            .to_string();
        let key = |s: &str| format!("{arch}.{s}");
        let u = |s: &str| -> io::Result<usize> {
            g.meta_req(&key(s))?
                .as_usize()
                .ok_or_else(|| bad(&key(s), "integer"))
        };
        let f = |s: &str| -> io::Result<f32> {
            g.meta_req(&key(s))?
                .as_f32()
                .ok_or_else(|| bad(&key(s), "float"))
        };

        let n_layer_total = u("block_count")?;
        // `nextn_predict_layers` counts trailing MTP blocks inside
        // `block_count`. They are not decoder layers: the forward pass must
        // stop before them, and the MTP drafter reads the last one.
        let nextn_layers = g
            .meta(&key("nextn_predict_layers"))
            .and_then(|v| v.as_usize())
            .unwrap_or(0);
        if nextn_layers > n_layer_total {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("nextn_predict_layers {nextn_layers} > block_count {n_layer_total}"),
            ));
        }
        let n_layer = n_layer_total - nextn_layers;
        let n_embd = u("embedding_length")?;
        let n_ff = u("feed_forward_length")?;
        let n_head = u("attention.head_count")?;
        let n_head_kv = u("attention.head_count_kv")?;
        let head_dim = u("attention.key_length")?;
        let eps = f("attention.layer_norm_rms_epsilon")?;
        let rope_base = f("rope.freq_base")?;
        let n_rot = u("rope.dimension_count")?;

        let mut rope_sections = [0usize; 4];
        if let Some(v) = g.meta(&key("rope.dimension_sections")) {
            let s = v
                .as_usizes()
                .ok_or_else(|| bad("rope.dimension_sections", "array"))?;
            for (i, val) in s.iter().take(4).enumerate() {
                rope_sections[i] = *val;
            }
        }

        // The explicit recurrent-layer mask is authoritative; the interval is a
        // fallback for files that only carry that.
        let is_recr = if let Some(v) = g.meta(&key("attention.recurrent_layers")) {
            v.as_bools()
                .ok_or_else(|| bad("attention.recurrent_layers", "bool array"))?
        } else {
            let interval = u("full_attention_interval")?;
            (0..n_layer).map(|i| (i + 1) % interval != 0).collect()
        };
        // Some files declare the mask over every block, inline MTP included:
        // 65 entries for a 64-layer decoder with `nextn_predict_layers = 1`.
        // The MTP block is not a decoder layer, so its entry is dropped. Only
        // that exact length is accepted; any other length is still an error, so
        // a genuinely wrong mask cannot pass as a truncated one.
        let is_recr = trim_mtp_entry(is_recr, n_layer, nextn_layers).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "recurrent layer mask length != block_count",
            )
        })?;

        let ssm_d_conv = u("ssm.conv_kernel")?;
        let ssm_d_state = u("ssm.state_size")?;
        let ssm_n_group = u("ssm.group_count")?;
        let ssm_dt_rank = u("ssm.time_step_rank")?;
        let ssm_d_inner = u("ssm.inner_size")?;
        let context_length = u("context_length").unwrap_or(4096);

        let n_vocab = {
            let t = g
                .tensor("token_embd.weight")
                .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no token_embd.weight"))?;
            t.rows() as usize
        };

        if ssm_d_inner % ssm_dt_rank != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("ssm.inner_size {ssm_d_inner} is not a multiple of time_step_rank {ssm_dt_rank}"),
            ));
        }

        Ok(Config {
            arch,
            n_layer,
            n_embd,
            n_ff,
            n_vocab,
            n_head,
            n_head_kv,
            head_dim,
            eps,
            rope_base,
            n_rot,
            rope_sections,
            is_recr,
            ssm_d_conv,
            ssm_d_state,
            ssm_n_group,
            ssm_dt_rank,
            ssm_d_inner,
            context_length,
            nextn_layers,
        })
    }

    /// Value-head dimension of the gated delta net (`d_inner / n_v_heads`).
    pub fn ssm_head_v(&self) -> usize {
        self.ssm_d_inner / self.ssm_dt_rank
    }

    /// Key-head dimension of the gated delta net (`state_size`).
    pub fn ssm_head_k(&self) -> usize {
        self.ssm_d_state
    }

    /// Conv1d channel count: q | k | v.
    pub fn ssm_conv_channels(&self) -> usize {
        self.ssm_head_k() * self.ssm_n_group * 2 + self.ssm_head_v() * self.ssm_dt_rank
    }

    pub fn n_full_attn(&self) -> usize {
        self.is_recr.iter().filter(|r| !**r).count()
    }
}

fn bad(key: &str, want: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("metadata {key} is not a {want}"),
    )
}

/// Accept a declared mask that covers the inline MTP block as well, by dropping
/// that block's entry. `None` means the length is not one the MTP rule can
/// explain, so the caller reports it as the error it is.
fn trim_mtp_entry(mut mask: Vec<bool>, n_layer: usize, nextn: usize) -> Option<Vec<bool>> {
    if nextn > 0 && mask.len() == n_layer + nextn {
        mask.truncate(n_layer);
    }
    (mask.len() == n_layer).then_some(mask)
}

#[cfg(test)]
mod tests {
    use super::trim_mtp_entry;

    /// The declared shape of `Swift-1.5-Qwen3.8-27B` and `OrcaCyber-Nano-1.0`:
    /// 65 entries, full attention on 3, 7, ... 63 and on the MTP block 64.
    fn declared_65() -> Vec<bool> {
        let mut mask = vec![true; 65];
        for i in [
            3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 64,
        ] {
            mask[i] = false;
        }
        mask
    }

    #[test]
    fn a_mask_that_covers_the_mtp_block_is_truncated_to_the_decoder() {
        let got = trim_mtp_entry(declared_65(), 64, 1).expect("accepted");
        assert_eq!(got.len(), 64);
        let full: Vec<usize> = got
            .iter()
            .enumerate()
            .filter(|(_, r)| !**r)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(
            full,
            [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]
        );
    }

    #[test]
    fn a_mask_of_the_decoder_length_is_kept_as_it_is() {
        let got = trim_mtp_entry(declared_65()[..64].to_vec(), 64, 1).expect("accepted");
        assert_eq!(got.len(), 64);
        assert!(!got[63]);
        // A file with no MTP block keeps its own 64-entry mask.
        let got = trim_mtp_entry(declared_65()[..64].to_vec(), 64, 0).expect("accepted");
        assert_eq!(got.len(), 64);
    }

    #[test]
    fn any_other_length_is_still_rejected() {
        // One entry short, one entry long, and the same lengths with no MTP
        // block to explain them.
        assert!(trim_mtp_entry(vec![true; 63], 64, 1).is_none());
        assert!(trim_mtp_entry(vec![true; 66], 64, 1).is_none());
        assert!(trim_mtp_entry(vec![true; 63], 64, 0).is_none());
        assert!(trim_mtp_entry(vec![true; 65], 64, 0).is_none());
    }
}
