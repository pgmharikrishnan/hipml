// SPDX-License-Identifier: Apache-2.0
//! Greedy speculative decode: the MTP drafter plus the R-column verify.
//!
//! One step produces `1 + a` tokens where `a` is the length of the accepted
//! draft prefix:
//!
//! ```text
//!   draft:  d1, d2, .. dD   from (h_q, t_{q+1}) recursively
//!   verify: [t_{q+1}, d1, .. dD] through all 64 layers, one weight pass
//!   accept: ids[c] == d_{c+1} for c = 0..a-1
//!   emit:   d1..da, then the correction ids[a]
//! ```
//!
//! With temperature 0 the emitted ids are exactly plain greedy decode's: every
//! verify column is bit-identical to a one-column launch, so the accepted
//! prefix and the correction are the same tokens the non-speculative path
//! would have produced.
//!
//! Rejected drafts roll back by restoring the recurrent-state snapshot taken
//! after column `a` and moving `pos` back; the KV rows past `pos` are simply
//! overwritten by the next append.

use super::*;
use std::time::Instant;

/// Per-draft-position counters, for the acceptance table.
#[derive(Default, Clone, Debug)]
pub struct SpecStats {
    pub steps: usize,
    /// Tokens emitted (accepted drafts + corrections, including the seed).
    pub tokens: usize,
    /// Per draft position: drafted, accepted.
    pub drafted: [usize; 3],
    pub accepted: [usize; 3],
}

impl SpecStats {
    /// Conditional acceptance `P(draft k correct | drafts 1..k correct)`.
    pub fn accept_rate(&self, k: usize) -> f64 {
        let denom = if k == 0 {
            self.drafted[0]
        } else {
            self.accepted[k - 1]
        };
        if denom == 0 {
            0.0
        } else {
            self.accepted[k] as f64 / denom as f64
        }
    }

    /// Mean tokens produced per verify step.
    pub fn tokens_per_step(&self) -> f64 {
        if self.steps == 0 {
            0.0
        } else {
            self.tokens as f64 / self.steps as f64
        }
    }
}

/// One speculative run: the ids, the counters, and where the time went.
pub struct SpecRun {
    pub ids: Vec<u32>,
    pub stats: SpecStats,
    /// True when a stop id (EOS) was emitted.
    pub stopped: bool,
    /// Prompt prefill (chunked), including the hidden-state capture.
    pub prefill_s: f64,
    /// Building the drafter's prompt KV.
    pub draft_prefill_s: f64,
    /// The speculative decode loop only (draft + verify + rollback).
    pub decode_s: f64,
    /// Inside `decode_s`: the drafter, the verify pass, the rollback/KV rewrite.
    pub draft_s: f64,
    pub verify_s: f64,
    pub rollback_s: f64,
}

/// One speculative step's output.
pub struct SpecStep {
    /// Tokens to append to the reply: the accepted drafts followed by the
    /// correction (unless a draft was itself a stop token).
    pub ids: Vec<u32>,
    /// True when a stop id was emitted.
    pub stopped: bool,
    /// Draft tokens accepted in this step.
    pub accepted: usize,
    /// Where the step's wall time went, for the bench breakdown.
    pub draft_s: f64,
    pub verify_s: f64,
    pub rollback_s: f64,
}

impl<'a> GpuEngine<'a> {
    /// Build the drafter's prompt KV from the target's captured hidden rows and
    /// seed the first recursive draft.
    ///
    /// The caller must have just prefilled `prompt` with
    /// [`GpuEngine::prefill_tile_capture`] into `capture`, one row per token.
    /// Splitting this from the prefill lets a server check for new requests
    /// between prefill tiles and abandon the MTP path without throwing away
    /// work it has already done.
    pub fn spec_seed(
        &mut self,
        mtp: &mut MtpGpu<'_>,
        prompt: &[u32],
        capture: &Buffer,
    ) -> Result<(), String> {
        for q in 0..prompt.len().saturating_sub(1) {
            mtp.set_hidden_row(capture, q)?;
            mtp.prefill_kv(prompt[q + 1], q)?;
        }
        if prompt.is_empty() {
            return Err("empty prompt".to_string());
        }
        mtp.set_hidden_row(capture, prompt.len() - 1)?;
        Ok(())
    }

    /// One draft + verify + rollback step.
    ///
    /// `next` is the token that will be appended at `st.pos` (the last emitted
    /// token); `snaps` must hold at least `drafts` columns and is reused across
    /// steps. `stats` accumulates the acceptance counters. The target state
    /// advances to `st.pos + a + 1` and the drafter KV is rewritten for the
    /// accepted positions, ready for the next call.
    pub fn spec_step(
        &mut self,
        mtp: &mut MtpGpu<'_>,
        st: &mut GpuState,
        snaps: &mut Snap,
        next: u32,
        drafts: usize,
        stop_ids: &[u32],
        stats: &mut SpecStats,
    ) -> Result<SpecStep, String> {
        if drafts == 0 || drafts > 3 {
            return Err(format!("drafts {drafts} outside 1..=3"));
        }
        if next as usize >= self.cfg.n_vocab {
            return Err(format!("seed token {next} outside the vocabulary"));
        }
        // The verify appends 1 + drafts KV rows at `st.pos..`. Near the end of
        // the cache, draft fewer: nothing checks the row index on the device,
        // and a row at `n_ctx` lands in the next head (or past the buffer).
        // Fewer drafts cannot change the emitted ids.
        if st.pos >= self.n_ctx {
            return Err(format!("position {} is past ctx {}", st.pos, self.n_ctx));
        }
        let drafts = drafts.min(self.n_ctx - st.pos - 1);

        // 1. Draft recursively from the current position's hidden state.
        let q = st.pos.saturating_sub(1);
        let mut draft_ids = Vec::with_capacity(drafts);
        let mut tok = next;
        let t = Instant::now();
        for k in 0..drafts {
            tok = mtp.step(tok, q + k)?;
            draft_ids.push(tok);
        }
        let draft_s = t.elapsed().as_secs_f64();

        // 2. Verify all R = 1 + drafts columns in one weight pass.
        let mut cols = Vec::with_capacity(1 + drafts);
        cols.push(next);
        cols.extend_from_slice(&draft_ids);
        let pos0 = st.pos;
        let t = Instant::now();
        let ids = self.forward_columns(&cols, st, snaps)?;
        let verify_s = t.elapsed().as_secs_f64();
        let t = Instant::now();

        // 3. Longest accepted prefix.
        let mut a = 0usize;
        let mut stopped = false;
        let mut stopped_on_draft = false;
        for k in 0..drafts {
            stats.drafted[k] += 1;
            if ids[k] == draft_ids[k] {
                a += 1;
                stats.accepted[k] += 1;
                if stop_ids.contains(&ids[k]) {
                    stopped = true;
                    stopped_on_draft = true;
                    break;
                }
            } else {
                break;
            }
        }
        stats.steps += 1;

        // 4. Roll back to the state after column `a` and move `pos` to the
        //    first position the next step will append.
        if a < drafts {
            self.restore_ssm(snaps, a)?;
        }
        st.pos = pos0 + a + 1;

        // 5. Emit the accepted drafts, then the correction.
        let mut emitted = Vec::with_capacity(a + 1);
        emitted.extend_from_slice(&draft_ids[..a]);
        let correction = ids[a];
        if !stopped_on_draft {
            if stop_ids.contains(&correction) {
                stopped = true;
            }
            emitted.push(correction);
        }

        // 6. Rewrite the drafter's KV for the accepted positions from the
        //    target's hidden state and seed the next draft.
        if !stopped {
            for k in 0..a {
                mtp.set_hidden_row(self.column_hidden()?, k)?;
                mtp.prefill_kv(emitted[k], pos0 + k)?;
            }
            mtp.set_hidden_row(self.column_hidden()?, a)?;
        }

        Ok(SpecStep {
            ids: emitted,
            stopped,
            accepted: a,
            draft_s,
            verify_s,
            rollback_s: t.elapsed().as_secs_f64(),
        })
    }

    /// Greedy speculative decode of `tokens` ids after `prompt`.
    ///
    /// `drafts` is 1..=3. The engine's caches must be sized for the prompt plus
    /// `tokens`. Returns the generated ids (excluding the prompt) and the
    /// acceptance counters.
    pub fn spec_generate(
        &mut self,
        mtp: &mut MtpGpu<'_>,
        prompt: &[u32],
        tokens: usize,
        drafts: usize,
        chunk: usize,
        stop_ids: &[u32],
    ) -> Result<SpecRun, String> {
        if drafts == 0 || drafts > 3 {
            return Err(format!("drafts {drafts} outside 1..=3"));
        }
        if prompt.is_empty() {
            return Err("empty prompt".to_string());
        }
        let n = self.cfg.n_embd;
        let n_ctx = self.n_ctx;
        if prompt.len() + tokens > n_ctx {
            return Err(format!(
                "prompt {} + {tokens} tokens exceeds ctx {n_ctx}",
                prompt.len()
            ));
        }
        let mut stats = SpecStats::default();

        // 1. Prompt, keeping every row's post-norm hidden for the drafter.
        let capture = Buffer::alloc(prompt.len() * n)?;
        let mut st = self.new_state();
        let t_prefill = Instant::now();
        let mut next = self.prefill_chunked_capture(prompt, &mut st, chunk, &capture)?;
        let prefill_s = t_prefill.elapsed().as_secs_f64();

        // 2. The drafter's own KV over the prompt: position q pairs hidden h_q
        //    with the token at q+1.
        let t_draft = Instant::now();
        self.spec_seed(mtp, prompt, &capture)?;
        let draft_prefill_s = t_draft.elapsed().as_secs_f64();

        let mut gen = Vec::with_capacity(tokens + 1);
        gen.push(next);
        let mut stopped = stop_ids.contains(&next);
        let mut snaps = self.alloc_snapshots(drafts)?;
        let (mut draft_s, mut verify_s, mut rollback_s) = (0.0f64, 0.0f64, 0.0f64);
        let t_decode = Instant::now();

        while gen.len() < tokens && !stopped {
            let step =
                self.spec_step(mtp, &mut st, &mut snaps, next, drafts, stop_ids, &mut stats)?;
            draft_s += step.draft_s;
            verify_s += step.verify_s;
            rollback_s += step.rollback_s;
            for id in step.ids {
                if gen.len() < tokens {
                    gen.push(id);
                    stats.tokens += 1;
                }
            }
            if let Some(&last) = gen.last() {
                next = last;
            }
            stopped = step.stopped;
        }

        let decode_s = t_decode.elapsed().as_secs_f64();
        gen.truncate(tokens);
        Ok(SpecRun {
            ids: gen,
            stats,
            stopped,
            prefill_s,
            draft_prefill_s,
            decode_s,
            draft_s,
            verify_s,
            rollback_s,
        })
    }
}
