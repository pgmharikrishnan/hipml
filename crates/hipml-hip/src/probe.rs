// SPDX-License-Identifier: Apache-2.0
//! An opt-in per-stage GPU timeline, so a stage table can carry launch counts.
//!
//! `HIPML_KERNEL_TRACE=1` turns it on for the process, or a tool can call
//! [`set_enabled`]. A stage is opened with [`mark`] — a recorded HIP event — and
//! every `Kernel::launch` is counted, not timed. After the stream has been
//! synchronised, [`report`] folds the timeline into one row per stage: the span
//! between its marks and how many launches it issued.
//!
//! Only one event per stage is recorded, because one token issues ~1,300
//! launches and an event pair per launch would cost more than the token. The
//! span is wall time on the stream, so it includes any gap between the stage's
//! kernels; a stage whose bytes and span agree with the stream rate is
//! kernel-bound and needs no further look. Recording is a queue operation, so
//! the order of work is unchanged, and nothing is recorded unless the flag is
//! set.
//!
//! `HIPML_KERNEL_SPAN=1` adds the other axis: an event pair around *every*
//! launch, folded into one row per kernel name by [`kernel_report`]. That is
//! two events per launch, so it costs more than the stage table and is only
//! worth it when the question is which kernel, not which stage. Both can be on
//! at once.

use std::cell::RefCell;
use std::sync::atomic::{AtomicU8, Ordering};

use crate::{Event, Stream};

struct Timeline {
    events: Vec<Event>,
    names: Vec<&'static str>,
    /// Launch count of the stage opened by `names[i]`.
    launches: Vec<usize>,
    open: usize,
    failed: usize,
}

thread_local! {
    static TIMELINE: RefCell<Option<Timeline>> = const { RefCell::new(None) };
    static SPANS: RefCell<Spans> = const { RefCell::new(Spans { ev: Vec::new(), names: Vec::new(), next: 0, open: false, failed: 0 }) };
}

struct Spans {
    /// Two events per recorded launch: `ev[2*i]` opens, `ev[2*i+1]` closes.
    /// Kept and reused across tokens so a launch costs no allocation.
    ev: Vec<Event>,
    names: Vec<String>,
    next: usize,
    open: bool,
    failed: usize,
}

/// 0 = unset, 1 = on, 2 = off. Read once and kept, because this is on the path
/// of every launch.
static STATE: AtomicU8 = AtomicU8::new(0);

/// True when the timeline is collecting.
pub fn enabled() -> bool {
    match STATE.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            let on = std::env::var("HIPML_KERNEL_TRACE")
                .map(|v| v != "0")
                .unwrap_or(false);
            STATE.store(if on { 1 } else { 2 }, Ordering::Relaxed);
            on
        }
    }
}

/// Turn the timeline on or off, overriding the env var.
pub fn set_enabled(on: bool) {
    STATE.store(if on { 1 } else { 2 }, Ordering::Relaxed);
}

/// Count one launch into the stage that is open.
pub fn count_launch() {
    TIMELINE.with(|t| {
        if let Some(line) = t.borrow_mut().as_mut() {
            if let Some(n) = line.launches.get_mut(line.open) {
                *n += 1;
            }
        }
    });
}

/// 0 = unset, 1 = on, 2 = off, for [`span_enabled`].
static SPAN_STATE: AtomicU8 = AtomicU8::new(0);

/// True when every launch is being bracketed by events.
pub fn span_enabled() -> bool {
    match SPAN_STATE.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            let on = std::env::var("HIPML_KERNEL_SPAN")
                .map(|v| v != "0")
                .unwrap_or(false);
            SPAN_STATE.store(if on { 1 } else { 2 }, Ordering::Relaxed);
            on
        }
    }
}

/// Open a launch span. Paired with [`span_end`] on the same thread.
pub fn span_begin(name: &str, stream: &Stream) {
    SPANS.with(|s| {
        let s = &mut *s.borrow_mut();
        let i = s.next;
        while s.ev.len() < 2 * i + 2 {
            match Event::new() {
                Ok(e) => s.ev.push(e),
                Err(_) => {
                    s.failed += 1;
                    return;
                }
            }
        }
        if s.ev[2 * i].record(stream).is_err() {
            s.failed += 1;
            return;
        }
        if i == s.names.len() {
            s.names.push(name.to_string());
        } else {
            s.names[i].clear();
            s.names[i].push_str(name);
        }
        s.open = true;
    });
}

/// Close the span opened by the last [`span_begin`].
pub fn span_end(stream: &Stream) {
    SPANS.with(|s| {
        let s = &mut *s.borrow_mut();
        if !s.open {
            return;
        }
        s.open = false;
        let i = s.next;
        if s.ev[2 * i + 1].record(stream).is_err() {
            s.failed += 1;
        }
        s.next += 1;
    });
}

/// Open a stage: the row runs from here to the next mark.
pub fn mark(name: &'static str, stream: &Stream) {
    TIMELINE.with(|t| {
        let mut t = t.borrow_mut();
        let line = t.get_or_insert_with(|| Timeline {
            events: Vec::new(),
            names: Vec::new(),
            launches: Vec::new(),
            open: 0,
            failed: 0,
        });
        let i = line.names.len();
        if i == line.events.len() {
            match Event::new() {
                Ok(e) => line.events.push(e),
                Err(_) => {
                    line.failed += 1;
                    return;
                }
            }
        }
        if let Err(_) = line.events[i].record(stream) {
            line.failed += 1;
            return;
        }
        line.names.push(name);
        line.launches.push(0);
        line.open = i;
    });
}

/// One row of the folded timeline: the span a stage marker opens and the
/// launches it issued.
#[derive(Clone, Copy, Debug, Default)]
pub struct StageRow {
    pub name: &'static str,
    pub span_us: f64,
    pub launches: usize,
}

/// One row of the per-launch fold: everything one kernel name cost in a token.
#[derive(Clone, Debug, Default)]
pub struct KernelRow {
    pub name: String,
    pub span_us: f64,
    pub launches: usize,
}

/// Fold the per-launch spans by kernel name, largest first. The caller must
/// have synchronised the stream first.
pub fn kernel_report() -> Vec<KernelRow> {
    SPANS.with(|s| {
        let s = &mut *s.borrow_mut();
        let mut rows: Vec<KernelRow> = Vec::new();
        for i in 0..s.next {
            let span = match s.ev[2 * i].since(&s.ev[2 * i + 1]) {
                Ok(ms) => ms as f64 * 1e3,
                Err(_) => {
                    s.failed += 1;
                    0.0
                }
            };
            let name = &s.names[i];
            match rows.iter_mut().find(|r| &r.name == name) {
                Some(r) => {
                    r.span_us += span;
                    r.launches += 1;
                }
                None => rows.push(KernelRow {
                    name: name.clone(),
                    span_us: span,
                    launches: 1,
                }),
            }
        }
        if s.failed > 0 {
            eprintln!("kernel span: {} event pairs failed", s.failed);
        }
        s.next = 0;
        s.open = false;
        rows.sort_by(|a, b| {
            b.span_us
                .partial_cmp(&a.span_us)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        rows
    })
}

/// Fold the timeline. The caller must have synchronised the stream first.
///
/// A stage with no closing mark (the last one) runs to the final recorded mark,
/// which is where the token's tail ends.
pub fn report() -> Vec<StageRow> {
    TIMELINE.with(|t| {
        let mut t = t.borrow_mut();
        let Some(line) = t.as_mut() else {
            return Vec::new();
        };
        let mut rows: Vec<StageRow> = Vec::new();
        for i in 0..line.names.len().saturating_sub(1) {
            let span = match line.events[i].since(&line.events[i + 1]) {
                Ok(ms) => ms as f64 * 1e3,
                Err(_) => {
                    line.failed += 1;
                    0.0
                }
            };
            let name = line.names[i];
            match rows.iter_mut().find(|r| r.name == name) {
                Some(r) => {
                    r.span_us += span;
                    r.launches += line.launches[i];
                }
                None => rows.push(StageRow {
                    name,
                    span_us: span,
                    launches: line.launches[i],
                }),
            }
        }
        if line.failed > 0 {
            eprintln!("kernel trace: {} event pairs failed", line.failed);
        }
        line.events.clear();
        line.names.clear();
        line.launches.clear();
        line.open = 0;
        rows.sort_by(|a, b| {
            b.span_us
                .partial_cmp(&a.span_us)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        rows
    })
}
