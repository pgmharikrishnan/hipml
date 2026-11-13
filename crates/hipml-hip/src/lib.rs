// SPDX-License-Identifier: Apache-2.0
//! Minimal HIP runtime bindings for hipml.
//!
//! One vendor API, no wrappers on top of it. Build with `--features hip` on a
//! machine with the ROCm/HIP runtime; without the feature everything here
//! returns a readable error so the rest of the workspace still builds and tests.

pub mod device;
pub mod memory;
pub mod probe;

#[cfg(feature = "hip")]
pub mod ffi;

pub use device::{Device, Event, Graph, Kernel, LaunchConfig, Module, Stream};
pub use memory::{Buffer, ByteBuffer};

/// True when this build can talk to a GPU.
pub const HAS_HIP: bool = cfg!(feature = "hip");
