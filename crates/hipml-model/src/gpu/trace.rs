//! Per-stage timing and per-stage byte accounting for one token.
//!
//! `gpu-profile --trace` uses this: every stage boundary records a HIP event,
//! and after one synchronize the spans are folded by stage name. Recording is a
//! queue operation, so the graph's execution order is unchanged.
//!
//! The name on a span is the *start* boundary, so a stage's weight bytes are
//! attributed to the span that begins just before it runs (see `stage_bytes`).

use super::*;

/// Per-stage GPU timing for one token.
///
/// Every stage mark is a recorded HIP event; `fold` runs after one synchronize
/// and folds the spans by stage name.
#[derive(Default)]
pub struct Trace {
    /// Boundary events in issue order; `names[i]` is the stage that starts at
    /// `events[i]`. A span is the gap between consecutive boundaries.
    events: Vec<Event>,
    names: Vec<&'static str>,
}

impl Trace {
    pub(crate) fn tick(&mut self, l: &Launcher, name: &'static str) -> Result<(), String> {
        let e = Event::new()?;
        e.record(l.stream())?;
        self.events.push(e);
        self.names.push(name);
        Ok(())
    }

    /// Folded `(stage, microseconds)` totals, largest first. The caller must
    /// have synchronised the stream so every event has completed.
    pub fn fold(&mut self) -> Vec<(&'static str, f64)> {
        let mut out: Vec<(&'static str, f64)> = Vec::new();
        let mut bad = 0usize;
        for i in 0..self.events.len().saturating_sub(1) {
            let us = match self.events[i].since(&self.events[i + 1]) {
                Ok(ms) => ms as f64 * 1e3,
                Err(_) => {
                    bad += 1;
                    0.0
                }
            };
            match out.iter().position(|(n, _)| *n == self.names[i]) {
                Some(j) => out[j].1 += us,
                None => out.push((self.names[i], us)),
            }
        }
        if bad > 0 {
            println!("trace: {bad} event pairs did not complete");
        }
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    pub fn clear(&mut self) {
        self.events.clear();
        self.names.clear();
    }
}

impl<'a> GpuEngine<'a> {
    /// Streamed weight bytes per traced stage, so a trace can report GB/s.
    pub fn stage_bytes(&self) -> Vec<(&'static str, usize)> {
        let (mut gate, mut up, mut down) = (0usize, 0usize, 0usize);
        let (mut attn_bytes, mut ssm_proj, mut ssm_step) = (0usize, 0usize, 0usize);
        for l in &self.layers {
            gate += l.gate.bytes();
            up += l.up.bytes();
            down += l.down.bytes();
            match &l.kind {
                BlockKind::Full { attn, .. } => attn_bytes += attn.weight_bytes(),
                BlockKind::Ssm(s) => {
                    ssm_proj += s.proj_bytes();
                    ssm_step += s.step_bytes();
                }
            }
        }
        let lm = self.model.tensor(crate::OUTPUT).map(|t| t.nbytes() as usize).unwrap_or(0);
        vec![
            ("gate", gate),
            ("up", up),
            ("down", down),
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

