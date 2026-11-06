//! Host-side foundation for hipml: tensors, deterministic RNG, reference kernels
//! and the parity harness.
//!
//! No external crates. Every kernel in `hipml-kernels` has a reference
//! implementation here with the same signature and the same memory layout, so a
//! device result can be compared against it directly.

pub mod ops;
pub mod parity;
pub mod rng;
pub mod tensor;

pub use parity::{compare, OpCase, Stats, Tol};
pub use rng::Rng;
pub use tensor::Tensor;
