// SPDX-License-Identifier: Apache-2.0
//! Per-stage timing and per-stage byte accounting for one token.
//!
//! `gpu-profile --trace` uses this: [`mark`] opens a stage on the launcher's
//! stream, and the timeline behind it (see `hipml_hip::probe`) also brackets
//! every launch, so a stage reports both what it cost and how many launches it
//! issued. Folding runs after one synchronize.
//!
//! The name on a stage is the *start* boundary, so a stage's weight bytes are
//! attributed to the stage that begins just before it runs (see `stage_bytes`).

use super::*;

/// Per-stage GPU timing for one token.
///
/// A stage mark is a recorded HIP event; `fold` runs after one synchronize and
/// folds the spans by stage name.
#[derive(Default)]
pub struct Trace {
    /// Stages folded by the last [`Trace::fold`].
    rows: Vec<hipml_hip::probe::StageRow>,
}

impl Trace {
    pub(crate) fn tick(&mut self, l: &Launcher, name: &'static str) -> Result<(), String> {
        hipml_hip::probe::mark(name, l.stream());
        Ok(())
    }

    /// Folded stage rows, largest span first. The caller must have
    /// synchronised the stream so every event has completed.
    pub fn fold(&mut self) -> &[hipml_hip::probe::StageRow] {
        self.rows = hipml_hip::probe::report();
        &self.rows
    }

    pub fn clear(&mut self) {
        self.rows.clear();
    }
}

impl<'a> GpuEngine<'a> {
    /// Streamed weight bytes per traced stage, so a trace can report GB/s.
    pub fn stage_bytes(&self) -> Vec<(&'static str, usize)> {
        let (mut gate, mut up, mut down) = (0usize, 0usize, 0usize);
        let mut moe = 0usize;
        let (mut attn_bytes, mut ssm_proj, mut ssm_step) = (0usize, 0usize, 0usize);
        for l in &self.layers {
            match &l.ffn {
                Ffn::Dense {
                    gate: g,
                    up: u,
                    down: d,
                    ..
                } => {
                    gate += g.bytes();
                    up += u.bytes();
                    down += d.bytes();
                }
                Ffn::Moe(m) => {
                    // One row for the whole FFN: the router, the gather, the
                    // combine and the shared expert are one stage, so its bytes
                    // are the layer's entire MoE stream.
                    moe += m.bytes();
                }
            }
            match &l.kind {
                BlockKind::Full { attn, .. } => attn_bytes += attn.weight_bytes(),
                BlockKind::Ssm(s) => {
                    ssm_proj += s.proj_bytes();
                    ssm_step += s.step_bytes();
                }
            }
        }
        let lm = self
            .model
            .tensor(crate::OUTPUT)
            .map(|t| t.nbytes() as usize)
            .unwrap_or(0);
        vec![
            ("gate", gate),
            ("up", up),
            ("down", down),
            ("moe", moe),
            ("attn", attn_bytes),
            ("ssm_proj", ssm_proj),
            ("ssm_step", ssm_step),
            ("lm_head", lm),
        ]
    }
}

/// Close the previous stage and open `name` (no-op unless tracing).
pub(super) fn mark(tr: &mut Option<Trace>, l: &Launcher, name: &'static str) -> Result<(), String> {
    match tr {
        Some(t) => t.tick(l, name),
        None => Ok(()),
    }
}
