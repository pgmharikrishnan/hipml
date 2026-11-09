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

impl<'a> GpuEngine<'a> {
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
        let p0 = prompt.len();
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
        for q in 0..prompt.len().saturating_sub(1) {
            mtp.set_hidden_row(&capture, q)?;
            mtp.prefill_kv(prompt[q + 1], q)?;
        }
        let draft_prefill_s = t_draft.elapsed().as_secs_f64();
        // The first draft is seeded from the last prompt token's hidden state.
        mtp.set_hidden_row(&capture, prompt.len() - 1)?;

        let mut gen = Vec::with_capacity(tokens + 1);
        gen.push(next);
        let mut stopped = stop_ids.contains(&next);
        let mut snaps = self.alloc_snapshots(drafts)?;
        let mut draft_ids = Vec::with_capacity(drafts);
        let (mut draft_s, mut verify_s, mut rollback_s) = (0.0f64, 0.0f64, 0.0f64);
        let t_decode = Instant::now();

        while gen.len() < tokens && !stopped {
            // 3. Draft `drafts` tokens from the current position's hidden
            //    state (`hnext` was seeded above).
            let q = st.pos - 1;
            draft_ids.clear();
            let t = Instant::now();
            let mut tok = next;
            for k in 0..drafts {
                tok = mtp.step(tok, q + k)?;
                draft_ids.push(tok);
            }
            draft_s += t.elapsed().as_secs_f64();

            // 4. Verify all R = 1 + drafts columns in one weight pass.
            let mut cols = Vec::with_capacity(1 + drafts);
            cols.push(next);
            cols.extend_from_slice(&draft_ids);
            let pos0 = st.pos;
            let t = Instant::now();
            let ids = self.forward_columns(&cols, &mut st, &mut snaps)?;
            verify_s += t.elapsed().as_secs_f64();
            let t = Instant::now();

            // 5. Longest matching prefix. A stop token ends the completion even
            //    if the draft would have continued past it: with EOS ignored the
            //    model closes the turn, invents a user turn and repeats, and
            //    every acceptance figure is measured on degenerate text.
            let mut a = 0usize;
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

            // 6. Roll back to the state after column `a`, then move `pos` to
            //    the first position the next step will append.
            if a < drafts {
                self.restore_ssm(&snaps, a)?;
            }
            st.pos = pos0 + a + 1;

            // 7. Emit the accepted drafts, then the correction unless the stop
            //    token was itself the last accepted draft.
            for &d in &draft_ids[..a] {
                if gen.len() < tokens {
                    gen.push(d);
                    stats.tokens += 1;
                }
            }
            next = ids[a];
            if !stopped_on_draft {
                if stop_ids.contains(&next) {
                    stopped = true;
                }
                if gen.len() < tokens || stopped {
                    gen.push(next);
                    stats.tokens += 1;
                }
            }

            rollback_s += t.elapsed().as_secs_f64();
            if stopped {
                break;
            }
            // 8. Rewrite the drafter's KV for the accepted positions from the
            //    target's hidden state (the recursive draft wrote its own), as
            //    the reference does. `gen[p + 1 - p0]` is the token after `p`.
            //    Skipped once the requested token count is reached.
            let t = Instant::now();
            if gen.len() < tokens {
                for p in pos0..pos0 + a {
                    mtp.set_hidden_row(self.column_hidden()?, p - pos0)?;
                    mtp.prefill_kv(gen[p + 1 - p0], p)?;
                }
                // Seed the next draft from the last accepted column's hidden.
                mtp.set_hidden_row(self.column_hidden()?, a)?;
            }
            rollback_s += t.elapsed().as_secs_f64();
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
