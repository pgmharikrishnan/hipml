//! Deterministic RNG so parity failures reproduce exactly.
//!
//! xorshift64* plus Box–Muller. No dependencies, and the same seed gives the
//! same inputs on every machine, which is the whole point: a failing parity case
//! has to be re-runnable on the GPU host.

#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // Any non-zero state works; keep the seed visible for reproduction.
        Self { state: seed | 1 }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f32 {
        // 24 bits of mantissa is all f32 can hold.
        ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32)
    }

    /// Uniform in [lo, hi).
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.uniform()
    }

    /// Standard normal via Box–Muller.
    pub fn normal(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-7);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }

    /// Normal scaled to `std`.
    pub fn normal_scaled(&mut self, std: f32) -> f32 {
        self.normal() * std
    }
}
