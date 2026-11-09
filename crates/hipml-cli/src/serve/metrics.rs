//! Server counters, served as Prometheus text on `GET /metrics`.

use std::sync::Mutex;

#[derive(Clone, Default, Debug)]
pub struct Stats {
    pub requests_total: u64,
    pub requests_done: u64,
    /// Requests ended early: client gone, or a stop string matched.
    pub requests_cancelled: u64,
    /// Requests that ended with an error event (admission or prefill).
    pub requests_failed: u64,
    /// Live occupancy, refreshed every scheduler iteration, so a leaked slot
    /// or a stuck queue is visible from outside.
    pub active_slots: u64,
    pub queue_depth: u64,
    pub prompt_tokens: u64,
    pub generated_tokens: u64,
    pub decode_steps: u64,
    pub batch_steps: u64,
    pub batch_requests: u64,
    pub mtp_requests: u64,
    pub spec_steps: u64,
    pub spec_drafted: u64,
    pub spec_accepted: u64,
    pub prefix_lookups: u64,
    pub prefix_hits: u64,
    pub prefix_hit_tokens: u64,
    pub prefix_misses: u64,
    pub prefix_restore_ms: f64,
    pub prefix_captures: u64,
    pub prefix_capture_ms: f64,
    pub prefill_chunks: u64,
    pub prefill_ms: f64,
    /// Prefill chunks that ran while at least one request was decoding.
    pub prefill_interleaved: u64,
    pub interleave_stall_ms: f64,
    pub max_interleave_stall_ms: f64,
    /// Row-block tail prefill (WP47): blocks run, tails served and wall time.
    pub tail_row_blocks: u64,
    pub tail_rows: u64,
    pub tail_rows_ms: f64,
}

#[derive(Default)]
pub struct Metrics {
    inner: Mutex<Stats>,
}

impl Metrics {
    pub fn update(&self, f: impl FnOnce(&mut Stats)) {
        if let Ok(mut s) = self.inner.lock() {
            f(&mut s);
        }
    }

    pub fn snapshot(&self) -> Stats {
        self.inner.lock().map(|s| s.clone()).unwrap_or_default()
    }

    /// Prometheus text. `vram` is `(free, total)` bytes when available.
    pub fn render(&self, vram: Option<(usize, usize)>) -> String {
        let s = self.snapshot();
        let hit_rate = if s.prefix_lookups == 0 {
            0.0
        } else {
            s.prefix_hits as f64 / s.prefix_lookups as f64
        };
        let mut out = String::new();
        let mut c = |name: &str, help: &str, kind: &str, value: String| {
            out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} {kind}\n{name} {value}\n"));
        };
        c("hipml_requests_total", "Requests admitted", "counter", s.requests_total.to_string());
        c("hipml_requests_done_total", "Requests completed", "counter", s.requests_done.to_string());
        c("hipml_requests_cancelled_total", "Requests ended early (client gone or stop string)", "counter", s.requests_cancelled.to_string());
        c("hipml_requests_failed_total", "Requests ended with an error", "counter", s.requests_failed.to_string());
        c("hipml_active_slots", "Requests prefilling or decoding (batch slots, a lone MTP request, the prefill)", "gauge", s.active_slots.to_string());
        c("hipml_queue_depth", "Requests waiting for admission", "gauge", s.queue_depth.to_string());
        c("hipml_prompt_tokens_total", "Prompt tokens processed", "counter", s.prompt_tokens.to_string());
        c("hipml_generated_tokens_total", "Tokens generated", "counter", s.generated_tokens.to_string());
        c("hipml_decode_steps_total", "Scheduler decode steps", "counter", s.decode_steps.to_string());
        c("hipml_batch_steps_total", "Batched forward passes", "counter", s.batch_steps.to_string());
        c("hipml_batch_requests_total", "Requests decoded through the batch engine", "counter", s.batch_requests.to_string());
        c("hipml_mtp_requests_total", "Requests decoded with MTP speculation", "counter", s.mtp_requests.to_string());
        c("hipml_spec_steps_total", "MTP draft+verify steps", "counter", s.spec_steps.to_string());
        c("hipml_spec_drafted_total", "Draft tokens proposed", "counter", s.spec_drafted.to_string());
        c("hipml_spec_accepted_total", "Draft tokens accepted", "counter", s.spec_accepted.to_string());
        c("hipml_prefix_lookups_total", "Prefix-cache lookups", "counter", s.prefix_lookups.to_string());
        c("hipml_prefix_hits_total", "Prefix-cache hits (any length)", "counter", s.prefix_hits.to_string());
        c("hipml_prefix_misses_total", "Prefix-cache misses", "counter", s.prefix_misses.to_string());
        c("hipml_prefix_hit_tokens_total", "Prompt tokens restored from cache", "counter", s.prefix_hit_tokens.to_string());
        c("hipml_prefix_hit_rate", "Fixture hit rate over lookups", "gauge", format!("{hit_rate:.6}"));
        c("hipml_prefix_restore_seconds_total", "Device time restoring a prefix", "counter", format!("{:.6}", s.prefix_restore_ms / 1e3));
        c("hipml_prefix_captures_total", "Prefix snapshots taken", "counter", s.prefix_captures.to_string());
        c("hipml_prefix_capture_seconds_total", "Device time capturing a prefix", "counter", format!("{:.6}", s.prefix_capture_ms / 1e3));
        c("hipml_prefill_chunks_total", "Prefill tiles run by the scheduler", "counter", s.prefill_chunks.to_string());
        c("hipml_prefill_seconds_total", "Wall time in prefill tiles", "counter", format!("{:.6}", s.prefill_ms / 1e3));
        c("hipml_prefill_interleaved_total", "Prefill tiles run while a batch was decoding", "counter", s.prefill_interleaved.to_string());
        c("hipml_interleave_stall_seconds_total", "Decode stall charged to interleaved prefill", "counter", format!("{:.6}", s.interleave_stall_ms / 1e3));
        c("hipml_interleave_stall_seconds_max", "Longest single interleaved prefill tile", "gauge", format!("{:.6}", s.max_interleave_stall_ms / 1e3));
        c("hipml_tail_row_blocks_total", "Row-block tail prefills run", "counter", s.tail_row_blocks.to_string());
        c("hipml_tail_rows_total", "Prompt tail rows served by row blocks", "counter", s.tail_rows.to_string());
        c("hipml_tail_rows_seconds_total", "Wall time in row-block tail prefill", "counter", format!("{:.6}", s.tail_rows_ms / 1e3));
        if let Some((free, total)) = vram {
            c("hipml_vram_free_bytes", "Device free memory", "gauge", free.to_string());
            c("hipml_vram_total_bytes", "Device total memory", "gauge", total.to_string());
        }
        out
    }
}
