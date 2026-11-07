//! The kernel manifest: the one table that maps an op to its HIP entry point and
//! launch geometry.
//!
//! Adding a kernel means adding a `.hip` source, a reference in `hipml-core`, a
//! parity case, and a row here. The CLI walks this table, so nothing can be
//! launched that is not also testable.

use std::path::PathBuf;

pub mod launch;

pub use launch::Launcher;

/// How a kernel is gridded.
#[derive(Clone, Copy, Debug)]
pub enum KernelKind {
    /// One block per row of the first dimension.
    PerRow { block: u32 },
    /// One block per `rows` output rows, with the x k-tile staged in shared
    /// memory. The kernel's `ROW_TILE`/`BLOCK` defines must match.
    RowTile { rows: u32, block: u32 },
    /// One warp (or one block) per head.
    PerHead { block: u32 },
    /// One block per head, sized to `head_dim` at launch.
    PerHeadOnDim,
    /// 1-D coverage of `n` elements.
    Linear { block: u32 },
    /// No device kernel: this op stays on the host by design.
    HostOnly(&'static str),
}

/// One entry in the manifest.
#[derive(Clone, Copy, Debug)]
pub struct KernelSpec {
    /// Op name, matching the parity case and the reference function.
    pub name: &'static str,
    /// Entry point inside the code object.
    pub entry: &'static str,
    pub kind: KernelKind,
}

pub const KERNELS: &[KernelSpec] = &[
    KernelSpec {
        name: "gemv_f32",
        entry: "gemv_f32",
        kind: KernelKind::PerRow { block: 256 },
    },
    KernelSpec {
        name: "gemv_q4_k",
        entry: "gemv_q4_k",
        // Two rows per warp, 8 warps: 16 rows per block, K_TILE 4096 in shared.
        kind: KernelKind::RowTile {
            rows: 16,
            block: 256,
        },
    },
    KernelSpec {
        name: "gemv_q6_k",
        entry: "gemv_q6_k",
        kind: KernelKind::RowTile {
            rows: 16,
            block: 256,
        },
    },
    KernelSpec {
        name: "gemv_q8_0",
        entry: "gemv_q8_0",
        kind: KernelKind::PerRow { block: 256 },
    },
    KernelSpec {
        name: "gemm_f32",
        entry: "gemm_f32",
        kind: KernelKind::PerRow { block: 256 },
    },
    KernelSpec {
        name: "rmsnorm_f32",
        entry: "rmsnorm_f32",
        kind: KernelKind::PerRow { block: 256 },
    },
    KernelSpec {
        name: "rope_neox_f32",
        entry: "rope_neox_f32",
        kind: KernelKind::PerHead { block: 32 },
    },
    KernelSpec {
        name: "silu_mul_f32",
        entry: "silu_mul_f32",
        kind: KernelKind::Linear { block: 256 },
    },
    KernelSpec {
        name: "softmax_f32",
        entry: "softmax_f32",
        kind: KernelKind::PerRow { block: 256 },
    },
    KernelSpec {
        name: "residual_add_f32",
        entry: "residual_add_f32",
        kind: KernelKind::Linear { block: 256 },
    },
    KernelSpec {
        name: "attn_decode_f32",
        entry: "attn_decode_f32",
        kind: KernelKind::PerHeadOnDim,
    },
    KernelSpec {
        name: "sample_top_p_f32",
        entry: "top_p_cdf_f32",
        kind: KernelKind::HostOnly(
            "nucleus walk is one token per step; the sort lands on device at M2",
        ),
    },
];

pub fn spec(name: &str) -> Option<&'static KernelSpec> {
    KERNELS.iter().find(|k| k.name == name)
}

/// Where `build.rs` put the code object, if it compiled one.
pub fn kernel_dir() -> Option<PathBuf> {
    option_env!("HIPML_KERNEL_DIR").map(PathBuf::from)
}

/// The `--offload-arch` the kernels were built for.
pub fn kernel_arch() -> &'static str {
    option_env!("HIPML_KERNEL_ARCH").unwrap_or("not-built")
}

/// Number of ops that have a device kernel.
pub fn device_kernel_count() -> usize {
    KERNELS
        .iter()
        .filter(|k| !matches!(k.kind, KernelKind::HostOnly(_)))
        .count()
}
