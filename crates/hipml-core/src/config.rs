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
                format!(
                    "nextn_predict_layers {nextn_layers} > block_count {n_layer_total}"
                ),
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
            let s = v.as_usizes().ok_or_else(|| bad("rope.dimension_sections", "array"))?;
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
        if is_recr.len() != n_layer {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "recurrent layer mask length != block_count",
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
