//! Host-side foundation for hipml: tensors, deterministic RNG, reference kernels
//! and the parity harness.
//!
//! No external crates. Every kernel in `hipml-kernels` has a reference
//! implementation here with the same signature and the same memory layout, so a
//! device result can be compared against it directly.

pub mod attention;
pub mod attn_parity;
pub mod cache;
pub mod config;
pub mod delta;
pub mod gen;
pub mod gguf;
pub mod gguf_value;
pub mod kv;
pub mod mat;
pub mod mmap;
pub mod model;
pub mod ops;
pub mod parity;
pub mod parity_kv;
pub mod prefill_parity;
pub mod quant;
pub mod quant_parity;
pub mod rng;
pub mod tensor;
pub mod tokenizer;

pub use kv::KvDtype;
pub use parity::{compare, OpCase, Stats, Tol};
pub use rng::Rng;
pub use tensor::Tensor;
