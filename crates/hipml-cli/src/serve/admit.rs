// SPDX-License-Identifier: Apache-2.0
//! Admission control: a hard bound on concurrent generation requests and on
//! connection threads.
//!
//! The real limit is VRAM slots, not threads. A slot costs its KV cache plus
//! its SSM recurrent state (hundreds of MiB at 8K ctx); a queued request costs
//! only its token ids on the host (4 bytes a token). So the queue may be a
//! small multiple of the slot count, but it must be bounded: without a bound a
//! burst grows the queue and the connection count without limit, and every
//! request waits for a slot instead of being told to come back.
//!
//! `inflight = slots + queue_max` is the whole admission bound. A request that
//! cannot get in is answered `503` with `Retry-After` *before* it is tokenized
//! or queued, so the scheduler never sees it.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use super::json;
use super::metrics::{Metrics, Stats};

/// One counted resource: a semaphore with a Prometheus gauge attached.
pub struct Bound {
    used: AtomicUsize,
    max: usize,
    metrics: Arc<Metrics>,
    field: fn(&mut Stats) -> &mut u64,
}

impl Bound {
    pub fn new(max: usize, metrics: Arc<Metrics>, field: fn(&mut Stats) -> &mut u64) -> Arc<Bound> {
        Arc::new(Bound {
            used: AtomicUsize::new(0),
            max: max.max(1),
            metrics,
            field,
        })
    }

    pub fn max(&self) -> usize {
        self.max
    }

    pub fn used(&self) -> usize {
        self.used.load(Ordering::Relaxed)
    }

    /// Take one unit, or `None` when the bound is full. The unit is returned
    /// when the ticket drops.
    pub fn try_enter(self: &Arc<Self>) -> Option<Ticket> {
        let mut cur = self.used.load(Ordering::Relaxed);
        loop {
            if cur >= self.max {
                return None;
            }
            match self.used.compare_exchange_weak(
                cur,
                cur + 1,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    let n = cur as u64 + 1;
                    let f = self.field;
                    self.metrics.update(|s| *f(s) = n);
                    return Some(Ticket { bound: self.clone() });
                }
                Err(v) => cur = v,
            }
        }
    }
}

pub struct Ticket {
    bound: Arc<Bound>,
}

impl Drop for Ticket {
    fn drop(&mut self) {
        let n = self.bound.used.fetch_sub(1, Ordering::AcqRel) - 1;
        let f = self.bound.field;
        self.bound.metrics.update(|s| *f(s) = n as u64);
    }
}

/// Wait until every admitted request has closed, or `max` elapses.
///
/// Used after a device-level failure: the scheduler has already sent every
/// stream an error, and the process may exit only once the last frame is
/// written. The connection thread holds its ticket until its reply is complete,
/// so a zero count means every socket has been closed.
pub fn await_close(bound: &Bound, max: std::time::Duration) {
    let t = std::time::Instant::now();
    while bound.used() > 0 && t.elapsed() < max {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

/// Answer `503 Service Unavailable` with `Retry-After`, so a client backs off
/// instead of retrying in a tight loop.
pub fn busy(
    stream: &mut std::net::TcpStream,
    msg: &str,
    retry_after: u32,
    metrics: &Metrics,
    connection: bool,
) -> Result<(), String> {
    metrics.update(|s| {
        if connection {
            s.connections_rejected += 1;
        } else {
            s.requests_rejected_busy += 1;
        }
    });
    let body = format!(
        "{{\"error\":{{\"message\":\"{}\",\"type\":\"overloaded\",\"code\":503}}}}",
        json::escape(msg)
    );
    let head = format!(
        "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\n\
         Retry-After: {retry_after}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    use std::io::Write;
    stream
        .write_all(head.as_bytes())
        .and_then(|_| stream.write_all(body.as_bytes()))
        .and_then(|_| stream.flush())
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bound(max: usize) -> Arc<Bound> {
        Bound::new(max, Arc::new(Metrics::default()), |s| &mut s.inflight)
    }

    #[test]
    fn refuses_past_the_bound_and_returns_the_unit_on_drop() {
        let b = bound(2);
        let a = b.try_enter().expect("first");
        let c = b.try_enter().expect("second");
        assert!(b.try_enter().is_none());
        assert_eq!(b.used(), 2);
        drop(a);
        assert_eq!(b.used(), 1);
        let d = b.try_enter().expect("room again");
        drop((c, d));
        assert_eq!(b.used(), 0);
    }

    #[test]
    fn the_gauge_follows_the_tickets() {
        let m = Arc::new(Metrics::default());
        let b = Bound::new(3, Arc::clone(&m), |s| &mut s.inflight);
        let t = b.try_enter().unwrap();
        assert_eq!(m.snapshot().inflight, 1);
        drop(t);
        assert_eq!(m.snapshot().inflight, 0);
    }
}
