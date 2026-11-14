// SPDX-License-Identifier: Apache-2.0
//! The captured token graph: one `hipGraphLaunch` per decode step.
//!
//! Child module of `gpu` so it can reach the engine's scratch buffers and the
//! private token body. Everything that changes between steps (position, sequence
//! length, token id, the four M-RoPE positions) is already in the launcher's
//! device step buffer, so the graph is captured once and replayed.

use super::*;

/// One measured graph-vs-eager comparison for a token.
#[derive(Clone, Copy, Debug)]
pub struct GraphTiming {
    pub eager_ms: f64,
    pub eager_submit_ms: f64,
    pub graph_ms: f64,
    pub graph_submit_ms: f64,
    pub launches: usize,
}

/// Launches one token issues, from the graph's structure (kept here so the
/// probe and `gpu-profile` cannot disagree).
pub fn launches_per_token(n_layer: usize, is_recr: &[bool], moe: bool) -> usize {
    let mut n = 1; // embedding dequant
    for &recr in is_recr.iter().take(n_layer) {
        // attn_norm, the block, then the ffn. The GDN block's four input
        // projections are one job-list launch and the attention block's q/k/v
        // are one (12 -> 9 and 13 -> 11); a dense ffn gate+up is one as well
        // (7 -> 6), while a MoE ffn is 17 launches because the router, top-k,
        // gather, combine and shared expert are each their own.
        n += 1;
        n += if recr { 9 } else { 11 };
        // Dense: residual, rmsnorm, gate+up, silu, down, residual.
        // MoE: residual, rmsnorm, router, top-k, gate, up, silu, down,
        // combine, shared gate, shared up, shared silu, shared down, shared
        // gate dot, sigmoid, shared add, residual.
        n += if moe { 17 } else { 6 };
    }
    n + 2
}

impl GpuEngine<'_> {
    /// Capture one token, argmax included, into a reusable graph.
    ///
    /// Every argument that changes between steps lives in the launcher's device
    /// step buffer, so the graph is launched once per token and the host only
    /// writes 32 bytes and reads 4 back. The position and token id are therefore
    /// *not* baked: `graph_token` supplies them.
    ///
    /// `pos` is the position of the first token the graph will replay. It has
    /// to be passed because a launch *shape* can depend on the position: the
    /// quantized decode attention divides the key range into one split per 64
    /// cached positions, and a graph bakes its grid. Capturing at 0 and then
    /// replaying at 8K would run the mono kernel over the whole cache.
    pub fn build_token_graph(&mut self, pos: usize) -> Result<(), String> {
        let l = self.model.launcher();
        let mut work = GpuState { pos };
        // No warm-up token here: running one would append to every layer's KV
        // cache at the current position and change the tokens. Capture records
        // launches without executing them, so the first real launch both warms
        // the code objects and decodes the token.
        l.stream().begin_capture()?;
        let captured = self.run_token_and_argmax(1, &mut work, false);
        let graph = match captured {
            Ok(()) => l.stream().end_capture(),
            Err(e) => {
                // The stream is left in capture mode; end it before reporting.
                let _ = l.stream().end_capture();
                return Err(e);
            }
        }?;
        graph.upload(l.stream())?;
        l.synchronize()?;
        self.graph = Some(graph);
        Ok(())
    }

    /// True once `build_token_graph` has run.
    pub fn has_graph(&self) -> bool {
        self.graph.is_some()
    }

    /// One token through the captured graph: write the step scalars, launch,
    /// sync, read the id. Returns the same id the eager path produces.
    pub fn graph_token(&mut self, token: u32, st: &mut GpuState) -> Result<u32, String> {
        let l = self.model.launcher();
        let graph = self
            .graph
            .as_ref()
            .ok_or_else(|| "no token graph: call build_token_graph".to_string())?;
        l.set_step(st.pos, st.pos + 1, token as usize)?;
        graph.launch(l.stream())?;
        l.synchronize()?;
        st.pos += 1;
        let bytes = self.arg_out.download_bytes(0, 4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    /// `run_token` followed by the device argmax, with no synchronize.
    fn run_token_and_argmax(
        &mut self,
        token: u32,
        st: &mut GpuState,
        set_step: bool,
    ) -> Result<(), String> {
        self.run_token(token, st, set_step, true)?;
        let l = self.model.launcher();
        l.argmax_buf(
            &self.logits,
            &self.arg_v,
            &self.arg_i,
            &self.arg_out,
            self.cfg.n_vocab,
            ARGMAX_PARTS,
        )
    }

    /// Eager versus captured step, with the host submit measured on its own.
    ///
    /// `submit` is the time the host spends issuing one token's work with no
    /// synchronize, so it is the cost a graph replaces. `ms` is wall time over
    /// `reps` tokens, one synchronize at the end, so it is what the token costs.
    pub fn graph_probe(
        &mut self,
        token: u32,
        st: &mut GpuState,
        reps: usize,
    ) -> Result<GraphTiming, String> {
        let l = self.model.launcher();
        let mut work = GpuState { pos: st.pos };

        let mut submit = 0.0f64;
        let t = std::time::Instant::now();
        for _ in 0..reps {
            let h = std::time::Instant::now();
            self.run_token(token, &mut work, true, true)?;
            submit += h.elapsed().as_secs_f64();
        }
        l.synchronize()?;
        let eager_s = t.elapsed().as_secs_f64();

        self.build_token_graph(work.pos)?;
        let mut submit_g = 0.0f64;
        let graph = self.graph.as_ref().expect("just captured");
        let t = std::time::Instant::now();
        for _ in 0..reps {
            let h = std::time::Instant::now();
            l.set_step(work.pos, work.pos + 1, token as usize)?;
            graph.launch(l.stream())?;
            work.pos += 1;
            submit_g += h.elapsed().as_secs_f64();
        }
        l.synchronize()?;
        let graph_s = t.elapsed().as_secs_f64();

        Ok(GraphTiming {
            eager_ms: eager_s * 1e3 / reps as f64,
            eager_submit_ms: submit * 1e3 / reps as f64,
            graph_ms: graph_s * 1e3 / reps as f64,
            graph_submit_ms: submit_g * 1e3 / reps as f64,
            launches: launches_per_token(
                self.cfg.n_layer,
                &self.cfg.is_recr,
                self.cfg.n_expert > 0,
            ),
        })
    }
}
