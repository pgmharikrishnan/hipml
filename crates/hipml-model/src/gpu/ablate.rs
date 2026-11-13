// SPDX-License-Identifier: Apache-2.0
//! Decode stage ablation: drop one stage's launches and time the rest.
//!
//! A stage span is not a kernel: [`decode-residual-gemv.md`] §4 deleted two
//! stage rows and moved the traced token 0.94 ms while the untraced step in the
//! same process moved 0.05 ms. This module is the other method — the stage is
//! not *timed*, it is *removed*, and the difference is one untraced end-to-end
//! wall time. A dropped stage issues no launch, records no event and costs
//! nothing on the host, so no instrument can contribute to the number it is used
//! to price.
//!
//! Default is [`Ablate::none`]: the shipped decode is untouched. `hipml
//! decode-ablate` sweeps the set; `HIPML_ABLATE=a,b` seeds it so `gpu-bench` can
//! reproduce one cell in a fresh process.
//!
//! Each stage is a *set of launches*, and each is one launch site in
//! `run_token`, so a stage name is exactly the calls it skips:
//!
//! | name | launches dropped |
//! | --- | --- |
//! | `embed` | the embedding row dequant |
//! | `attn_front` | q\|gate split, q/k head norm, M-RoPE |
//! | `kv_append` | the quantized KV row append |
//! | `attn_reduce` | the decode attention core (reads the KV cache) |
//! | `q_gate` | `attn_out *= silu(gate)` |
//! | `attn_gemv` | the four attention projections |
//! | `ssm_gates` | `softplus(alpha + dt) * a`, `sigmoid(beta)` |
//! | `ssm_conv` | the causal conv and its SiLU |
//! | `ssm_l2` | the two L2 norms |
//! | `ssm_rec` | the delta-net recurrence |
//! | `ssm_gn` | the gated RMS norm |
//! | `ssm_gemv` | the five GDN projections |
//! | `ffn_gemv` | `ffn_gate`, `ffn_up`, `ffn_down` |
//! | `norm` | every RMS norm on the residual stream |
//! | `ew` | the residual adds and `silu(gate)*up` |
//! | `lmhead` | the `output.weight` GEMV |
//! | `argmax` | the device argmax (the 4-byte read-back and its sync stay) |
//!
//! `attn_core` is `attn_front,kv_append,attn_reduce,q_gate`; `ssm_scan` is
//! `ssm_gates,ssm_conv,ssm_l2,ssm_rec,ssm_gn`; `attn` adds `attn_gemv` and
//! `ssm` adds `ssm_gemv`. So a whole block drops in one word.
//!
//! [`decode-residual-gemv.md`]: ../../../../docs/notes/decode-residual-gemv.md

/// One droppable set of decode launches.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Stage {
    Embed,
    AttnFront,
    KvAppend,
    AttnReduce,
    QGate,
    AttnGemv,
    SsmGates,
    SsmConv,
    SsmL2,
    SsmRec,
    SsmGateNorm,
    SsmGemv,
    FfnGemv,
    Norm,
    Ew,
    LmHead,
    Argmax,
}

/// The four launches between the Q/K/V projections and the output projection.
const ATTN_CORE: [Stage; 4] = [
    Stage::AttnFront,
    Stage::KvAppend,
    Stage::AttnReduce,
    Stage::QGate,
];

/// The five launches of the GDN recurrent block.
const SSM_SCAN: [Stage; 5] = [
    Stage::SsmGates,
    Stage::SsmConv,
    Stage::SsmL2,
    Stage::SsmRec,
    Stage::SsmGateNorm,
];

impl Stage {
    pub const ALL: [Stage; 17] = [
        Stage::Embed,
        Stage::AttnFront,
        Stage::KvAppend,
        Stage::AttnReduce,
        Stage::QGate,
        Stage::AttnGemv,
        Stage::SsmGates,
        Stage::SsmConv,
        Stage::SsmL2,
        Stage::SsmRec,
        Stage::SsmGateNorm,
        Stage::SsmGemv,
        Stage::FfnGemv,
        Stage::Norm,
        Stage::Ew,
        Stage::LmHead,
        Stage::Argmax,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Stage::Embed => "embed",
            Stage::AttnFront => "attn_front",
            Stage::KvAppend => "kv_append",
            Stage::AttnReduce => "attn_reduce",
            Stage::QGate => "q_gate",
            Stage::AttnGemv => "attn_gemv",
            Stage::SsmGates => "ssm_gates",
            Stage::SsmConv => "ssm_conv",
            Stage::SsmL2 => "ssm_l2",
            Stage::SsmRec => "ssm_rec",
            Stage::SsmGateNorm => "ssm_gn",
            Stage::SsmGemv => "ssm_gemv",
            Stage::FfnGemv => "ffn_gemv",
            Stage::Norm => "norm",
            Stage::Ew => "ew",
            Stage::LmHead => "lmhead",
            Stage::Argmax => "argmax",
        }
    }

    fn bit(self) -> u32 {
        1 << (self as u32)
    }
}

/// A set of stages to drop. Cheap to copy; one bit per [`Stage`].
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct Ablate(u32);

impl Ablate {
    pub const fn none() -> Ablate {
        Ablate(0)
    }

    /// Everything except the projection GEMVs: the isolated weight stream.
    pub fn gemv_only() -> Ablate {
        let mut a = Ablate::none();
        for s in [Stage::Embed, Stage::Norm, Stage::Ew, Stage::Argmax] {
            a.drop(s);
        }
        for s in ATTN_CORE {
            a.drop(s);
        }
        for s in SSM_SCAN {
            a.drop(s);
        }
        a
    }

    /// Everything: a token that issues no launch at all.
    pub fn all() -> Ablate {
        let mut a = Ablate::none();
        for s in Stage::ALL {
            a.drop(s);
        }
        a
    }

    pub fn drop(&mut self, s: Stage) {
        self.0 |= s.bit();
    }

    /// Add a stage back to a set that already drops it. `drop` only sets bits,
    /// so the "one stage back" cells need this to undo `gemv_only`.
    pub fn keep(&mut self, s: Stage) {
        self.0 &= !s.bit();
    }

    pub fn dropped(self, s: Stage) -> bool {
        self.0 & s.bit() != 0
    }

    /// True when any launch of the attention block's non-GEMV part is dropped,
    /// so `run_token` has to compose it launch by launch instead of calling the
    /// block's own entry point.
    pub fn splits_attn_core(self) -> bool {
        ATTN_CORE.iter().any(|s| self.dropped(*s))
    }

    /// Same for the GDN recurrent block.
    pub fn splits_ssm_scan(self) -> bool {
        SSM_SCAN.iter().any(|s| self.dropped(*s))
    }

    pub fn is_none(self) -> bool {
        self.0 == 0
    }

    /// The dropped stages, in table order, for a log line.
    pub fn names(self) -> Vec<&'static str> {
        Stage::ALL
            .iter()
            .filter(|s| self.dropped(**s))
            .map(|s| s.name())
            .collect()
    }

    /// Parse a comma list. Accepts the stage names above plus the block aliases
    /// `attn_core`, `ssm_scan`, `attn`, `ssm`, and the whole-token `all`,
    /// `none` and `gemv_only`.
    pub fn parse(spec: &str) -> Result<Ablate, String> {
        let mut a = Ablate::none();
        for word in spec.split(',').map(str::trim).filter(|w| !w.is_empty()) {
            match word {
                "none" => {}
                "all" => a = Ablate::all(),
                "gemv_only" => a = Ablate::gemv_only(),
                "attn_core" => {
                    for s in ATTN_CORE {
                        a.drop(s);
                    }
                }
                "ssm_scan" => {
                    for s in SSM_SCAN {
                        a.drop(s);
                    }
                }
                "attn" => {
                    for s in ATTN_CORE {
                        a.drop(s);
                    }
                    a.drop(Stage::AttnGemv);
                }
                "ssm" => {
                    for s in SSM_SCAN {
                        a.drop(s);
                    }
                    a.drop(Stage::SsmGemv);
                }
                other => {
                    let s = Stage::ALL
                        .iter()
                        .find(|s| s.name() == other)
                        .ok_or_else(|| format!("unknown ablation stage {other}"))?;
                    a.drop(*s);
                }
            }
        }
        Ok(a)
    }
}
