//! The kernel manifest: the one table that maps an op to its HIP entry point and
//! launch geometry.
//!
//! Adding a kernel means adding a `.hip` source, a reference in `hipml-core`, a
//! parity case, and a row here. The CLI walks this table, so nothing can be
//! launched that is not also testable.

use std::path::PathBuf;

pub mod attn_gpu;
pub mod attn_ops;
pub mod launch;
#[cfg(feature = "hip")]
pub mod kvbench;
#[cfg(feature = "hip")]
pub mod ssm;
#[cfg(feature = "hip")]
pub mod ssm_launch;
#[cfg(feature = "hip")]
mod ssm_ops;
#[cfg(feature = "hip")]
mod ssm_chunk;

pub use attn_gpu::AttnLayer;
#[cfg(feature = "hip")]
pub mod launch_kv;
pub use launch::{
    prefill_entry_for, prefill_half, prefill_path, set_prefill_half, set_prefill_path, PrefillPath,
};
#[cfg(feature = "hip")]
pub use launch::{gemm_rows_f32, prefill_resident};
#[cfg(feature = "hip")]
pub use launch::{gemv_job_tiles, GemvJob, ReadJob};
#[cfg(feature = "hip")]
pub use launch::{GEMV_B_Q4K, GEMV_B_Q5K, GEMV_B_Q6K, GEMV_B_Q80};
#[cfg(feature = "hip")]
pub use launch::{GEMV_JOB_Q4K, GEMV_JOB_Q5K, GEMV_JOB_Q6K, GEMV_JOB_Q80};
pub use launch::Launcher;

/// The batched-GEMV kind for a resident matrix kind, when one exists.
#[cfg(feature = "hip")]
pub fn batch_kind_for(kind: hipml_core::mat::Kind) -> Option<i32> {
    use hipml_core::mat::Kind;
    Some(match kind {
        Kind::Q4K => launch::GEMV_B_Q4K,
        Kind::Q6K => launch::GEMV_B_Q6K,
        Kind::Q5K => launch::GEMV_B_Q5K,
        Kind::Q8_0 => launch::GEMV_B_Q80,
        Kind::F32 => return None,
    })
}

/// Without the `hip` feature no batched kernel exists.
#[cfg(not(feature = "hip"))]
pub fn batch_kind_for(_kind: hipml_core::mat::Kind) -> Option<i32> {
    None
}

/// How a kernel is gridded.
#[derive(Clone, Copy, Debug)]
pub enum KernelKind {
    /// One block per row of the first dimension.
    PerRow { block: u32 },
    /// One block per `rows` output rows, with the x k-tile staged in shared
    /// memory. The kernel's `ROW_TILE`/`BLOCK` defines must match.
    RowTile { rows: u32, block: u32 },
    /// One block per `rows` output rows x `tokens` prompt tokens, dequantising
    /// the weight tile once into shared. The kernel's geometry defines must
    /// match.
    Prefill {
        rows: u32,
        tokens: u32,
        block: u32,
    },
    /// One warp (or one block) per head.
    PerHead { block: u32 },
    /// One block per head, sized to `head_dim` at launch.
    PerHeadOnDim,
    /// One block per (KV head, contiguous split of positions), 32 * GQA-group
    /// threads. Paired with a reduce kernel, launched from the same arm.
    KvSplit,
    /// 1-D coverage of `n` elements.
    Linear { block: u32 },
    /// Several launches end to end (one whole SSM layer).
    Composite,
    /// Split-K grid sized by the launcher (sequence slices x KV heads).
    SplitK,
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
        // Two rows per warp, 8 warps: 16 rows per block, 16-value lanes,
        // K_TILE 2048 in shared.
        kind: KernelKind::RowTile {
            rows: 16,
            block: 256,
        },
    },
    KernelSpec {
        name: "gemv_q6_k",
        entry: "gemv_q6_k",
        // Two rows per warp, 8 warps: 16 rows per block, 8-value lanes (Q6_K
        // has more quant bytes per value than Q4_K), K_TILE 2048 in shared.
        kind: KernelKind::RowTile {
            rows: 16,
            block: 256,
        },
    },
    KernelSpec {
        name: "gemv_q5_k",
        entry: "gemv_q5_k",
        // One row per warp, like gemv_q4_k_r1: Q5_K only appears as a
        // 5120-row ffn_down, and 320 blocks leaves the machine under-occupied.
        kind: KernelKind::RowTile {
            rows: 8,
            block: 256,
        },
    },
    KernelSpec {
        name: "gemv_q4_k_u4",
        entry: "gemv_q4_k_u4",
        // Same geometry as _r1, one 128-bit load per 16 values instead of four
        // dwords. A/B entry for the load-issue hypothesis.
        kind: KernelKind::RowTile {
            rows: 8,
            block: 256,
        },
    },
    KernelSpec {
        name: "gemv_q4_k_nt",
        entry: "gemv_q4_k_nt",
        // `_r1` geometry with a streaming (sc1) 128-bit payload load.
        kind: KernelKind::RowTile {
            rows: 8,
            block: 256,
        },
    },
    KernelSpec {
        name: "gemv_q4_k_split",
        entry: "gemv_q4_k_split",
        // grid.y = splits; the launcher sizes it.
        kind: KernelKind::RowTile {
            rows: 8,
            block: 256,
        },
    },
    KernelSpec {
        name: "gemv_split_reduce",
        entry: "gemv_split_reduce",
        kind: KernelKind::Linear { block: 256 },
    },
    KernelSpec {
        name: "gemv_q4_k_r1",
        entry: "gemv_q4_k_r1",
        // One row per warp: 8 rows per block, twice the grid of the two-row
        // entry and the same arithmetic.
        kind: KernelKind::RowTile {
            rows: 8,
            block: 256,
        },
    },
    KernelSpec {
        name: "gemv_q6_k_r1",
        entry: "gemv_q6_k_r1",
        kind: KernelKind::RowTile {
            rows: 8,
            block: 256,
        },
    },
    KernelSpec {
        name: "gemv_q8_0",
        entry: "gemv_q8_0",
        kind: KernelKind::PerRow { block: 256 },
    },
    KernelSpec {
        name: "dequant_q4_k_f32",
        entry: "dequant_q4_k_f32",
        // One 256-thread block per Q4_K super-block, one output value per thread.
        kind: KernelKind::Linear { block: 256 },
    },
    KernelSpec {
        name: "prefill_q4_k",
        entry: "prefill_q4_k",
        // 256 tokens x 64 rows per block, 256 threads, 8x8 microtile; the
        // weight tile is dequantised once into shared for all 256 tokens.
        kind: KernelKind::Prefill {
            rows: 64,
            tokens: 256,
            block: 256,
        },
    },
    KernelSpec {
        name: "prefill_q6_k",
        entry: "prefill_q6_k",
        // Same 256-token tile as Q4_K: 8x8 microtile, 32-value k-tiles.
        kind: KernelKind::Prefill {
            rows: 64,
            tokens: 256,
            block: 256,
        },
    },
    KernelSpec {
        name: "prefill_q8_0",
        entry: "prefill_q8_0",
        kind: KernelKind::Prefill {
            rows: 64,
            tokens: 256,
            block: 256,
        },
    },
    // Packed-fp16 compute path: half weights and activations in shared, f32
    // accumulators, `v_dot2_f32_f16` in the inner loop. 128 output rows per
    // block, 512 threads.
    KernelSpec {
        name: "prefill_q4_k_h",
        entry: "prefill_q4_k_h",
        kind: KernelKind::Prefill {
            rows: 128,
            tokens: 256,
            block: 512,
        },
    },
    KernelSpec {
        name: "prefill_q6_k_h",
        entry: "prefill_q6_k_h",
        kind: KernelKind::Prefill {
            rows: 128,
            tokens: 256,
            block: 512,
        },
    },
    KernelSpec {
        name: "prefill_q8_0_h",
        entry: "prefill_q8_0_h",
        kind: KernelKind::Prefill {
            rows: 128,
            tokens: 256,
            block: 512,
        },
    },
    KernelSpec {
        name: "prefill_halfw_h",
        entry: "prefill_halfw_h",
        kind: KernelKind::Prefill {
            rows: 128,
            tokens: 256,
            block: 512,
        },
    },
    // Matrix-unit path (gfx12 `wmma_f32_16x16x16_f16`): 256 tokens x 128 output
    // rows per block, 512 threads, f32 accumulators, f32 epilogue.
    KernelSpec {
        name: "prefill_q4_k_w",
        entry: "prefill_q4_k_w",
        kind: KernelKind::Prefill {
            rows: 128,
            tokens: 256,
            block: 512,
        },
    },
    KernelSpec {
        name: "prefill_q6_k_w",
        entry: "prefill_q6_k_w",
        kind: KernelKind::Prefill {
            rows: 128,
            tokens: 256,
            block: 512,
        },
    },
    KernelSpec {
        name: "prefill_q8_0_w",
        entry: "prefill_q8_0_w",
        kind: KernelKind::Prefill {
            rows: 128,
            tokens: 256,
            block: 512,
        },
    },
    KernelSpec {
        name: "prefill_halfw_w",
        entry: "prefill_halfw_w",
        kind: KernelKind::Prefill {
            rows: 128,
            tokens: 256,
            block: 512,
        },
    },
    KernelSpec {
        name: "gemm_f32",
        entry: "gemm_f32",
        kind: KernelKind::PerRow { block: 256 },
    },
    KernelSpec {
        name: "gemm_rows_f32",
        entry: "gemm_rows_f32",
        kind: KernelKind::Linear { block: 256 },
    },
    // Skinny quantized GEMM: R <= 16 activation rows, one 16-wide WMMA tile in
    // the token dimension, `PWS_RT` output rows x `PWS_KW` k values per block.
    // The bench and the MTP verify share these; every kind runs the same
    // parity case.
    KernelSpec {
        name: "gemm_skinny_q4_k",
        entry: "gemm_skinny_q4_k",
        kind: KernelKind::RowTile {
            rows: 64,
            block: 128,
        },
    },
    KernelSpec {
        name: "gemm_skinny_q6_k",
        entry: "gemm_skinny_q6_k",
        kind: KernelKind::RowTile {
            rows: 64,
            block: 128,
        },
    },
    KernelSpec {
        name: "gemm_skinny_q8_0",
        entry: "gemm_skinny_q8_0",
        kind: KernelKind::RowTile {
            rows: 64,
            block: 128,
        },
    },
    KernelSpec {
        name: "stream_read",
        entry: "stream_read",
        kind: KernelKind::Linear { block: 256 },
    },
    KernelSpec {
        name: "stream_read_jobs",
        entry: "stream_read_jobs",
        kind: KernelKind::Linear { block: 256 },
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
        name: "ssm_conv1d_f32",
        entry: "ssm_conv1d_f32",
        kind: KernelKind::Linear { block: 256 },
    },
    KernelSpec {
        name: "ssm_silu_inplace_f32",
        entry: "ssm_silu_inplace_f32",
        kind: KernelKind::Linear { block: 256 },
    },
    KernelSpec {
        name: "ssm_l2norm_f32",
        entry: "l2norm_f32",
        kind: KernelKind::PerHead { block: 256 },
    },
    KernelSpec {
        name: "gdn_ar_f32",
        entry: "gdn_ar_f32",
        kind: KernelKind::PerHead { block: 128 },
    },
    KernelSpec {
        name: "ssm_gate_norm_f32",
        entry: "ssm_gate_norm_f32",
        kind: KernelKind::PerHead { block: 128 },
    },
    KernelSpec {
        name: "ssm_gates_f32",
        entry: "ssm_gates_f32",
        kind: KernelKind::Linear { block: 64 },
    },
    KernelSpec {
        name: "ssm_layer_f32",
        entry: "ssm_layer_f32",
        kind: KernelKind::Composite,
    },
    // Chunked GDN chain: one launch per stage per prompt tile instead of one
    // per token. Grids are set by `SsmGpuLayer::steps_chunk`.
    // Chunked attention chain: four launches per layer per prompt tile.
    KernelSpec {
        name: "qg_split_chunk_f32",
        entry: "qg_split_chunk_f32",
        kind: KernelKind::PerHead { block: 256 },
    },
    KernelSpec {
        name: "norm_rope_chunk_f32",
        entry: "norm_rope_chunk_f32",
        kind: KernelKind::PerHead { block: 256 },
    },
    KernelSpec {
        name: "kv_append_chunk_f32",
        entry: "kv_append_chunk_f32",
        kind: KernelKind::Linear { block: 256 },
    },
    KernelSpec {
        name: "attn_chunk_f32",
        entry: "attn_chunk_f32",
        kind: KernelKind::PerHeadOnDim,
    },
    KernelSpec {
        name: "attn_chunk_tile_f32",
        entry: "attn_chunk_tile_f32",
        kind: KernelKind::PerHead { block: 256 },
    },
    KernelSpec {
        name: "ssm_gates_chunk_f32",
        entry: "ssm_gates_chunk_f32",
        kind: KernelKind::Linear { block: 256 },
    },
    KernelSpec {
        name: "ssm_conv1d_chunk_f32",
        entry: "ssm_conv1d_chunk_f32",
        kind: KernelKind::Linear { block: 256 },
    },
    KernelSpec {
        name: "l2norm_chunk_f32",
        entry: "l2norm_chunk_f32",
        kind: KernelKind::PerHead { block: 128 },
    },
    KernelSpec {
        name: "gdn_chunk_f32",
        entry: "gdn_chunk_f32",
        kind: KernelKind::PerHead { block: 1024 },
    },
    KernelSpec {
        name: "gdn_rows_f32",
        entry: "gdn_rows_f32",
        kind: KernelKind::PerHead { block: 256 },
    },
    KernelSpec {
        name: "gate_norm_chunk_f32",
        entry: "gate_norm_chunk_f32",
        kind: KernelKind::PerHead { block: 128 },
    },
    KernelSpec {
        name: "copy_f32",
        entry: "copy_f32",
        kind: KernelKind::Linear { block: 256 },
    },
    KernelSpec {
        name: "attn_decode_pm_f32",
        entry: "attn_decode_f32",
        kind: KernelKind::PerHeadOnDim,
    },
    KernelSpec {
        name: "attn_decode_split_f32",
        entry: "attn_decode_split_f32",
        kind: KernelKind::SplitK,
    },
    KernelSpec {
        name: "attn_reduce_f32",
        entry: "attn_reduce_f32",
        kind: KernelKind::SplitK,
    },
    KernelSpec {
        name: "qg_split_f32",
        entry: "qg_split_f32",
        kind: KernelKind::PerHead { block: 256 },
    },
    KernelSpec {
        name: "head_rmsnorm_f32",
        entry: "head_rmsnorm_f32",
        kind: KernelKind::PerHead { block: 256 },
    },
    KernelSpec {
        name: "mrope_f32",
        entry: "mrope_f32",
        kind: KernelKind::PerHead { block: 32 },
    },
    KernelSpec {
        name: "q_gate_f32",
        entry: "q_gate_f32",
        kind: KernelKind::Linear { block: 256 },
    },
    KernelSpec {
        name: "kv_append_f32",
        entry: "kv_append_f32",
        kind: KernelKind::Linear { block: 256 },
    },
    KernelSpec {
        name: "kv_encode_q8",
        entry: "kv_encode_q8",
        kind: KernelKind::PerRow { block: 256 },
    },
    KernelSpec {
        name: "kv_encode_q4",
        entry: "kv_encode_q4",
        kind: KernelKind::PerRow { block: 256 },
    },
    KernelSpec {
        name: "kv_encode_bulk_q8",
        entry: "kv_encode_bulk_q8",
        kind: KernelKind::Linear { block: 256 },
    },
    KernelSpec {
        name: "kv_encode_bulk_q4",
        entry: "kv_encode_bulk_q4",
        kind: KernelKind::Linear { block: 256 },
    },
    // One row per case: the op is the pair (partial pass, final fold), and the
    // parity case compares the single index it produces.
    KernelSpec {
        name: "argmax_f32",
        entry: "argmax_part_f32",
        kind: KernelKind::PerRow { block: 256 },
    },
    KernelSpec {
        name: "argmax_final_f32",
        entry: "argmax_final_f32",
        kind: KernelKind::PerRow { block: 32 },
    },
    KernelSpec {
        name: "sample_top_p_f32",
        entry: "top_p_cdf_f32",
        kind: KernelKind::HostOnly(
            "nucleus walk is one token per step; the sort lands on device at M2",
        ),
    },
    // Quantized-KV decode attention. One row per parity case so the CLI table
    // names the shape that was actually compared; every row launches the same
    // two-kernel pair (decode over splits, then reduce).
    kv_case("attn_decode_kv_q8_hd256_s1", "attn_decode_kv_q8"),
    kv_case("attn_decode_kv_q8_hd256_s37", "attn_decode_kv_q8"),
    kv_case("attn_decode_kv_q8_hd256_s1000", "attn_decode_kv_q8"),
    kv_case("attn_decode_kv_q8_hd256_s4096", "attn_decode_kv_q8"),
    kv_case("attn_decode_kv_q4_hd256_s1", "attn_decode_kv_q4"),
    kv_case("attn_decode_kv_q4_hd256_s37", "attn_decode_kv_q4"),
    kv_case("attn_decode_kv_q4_hd256_s1000", "attn_decode_kv_q4"),
    kv_case("attn_decode_kv_q4_hd256_s4096", "attn_decode_kv_q4"),
    kv_case("attn_decode_kv_q8_hd64_s37", "attn_decode_kv_q8"),
    kv_case("attn_decode_kv_q8_hd64_s1000", "attn_decode_kv_q8"),
    kv_case("attn_decode_kv_q4_hd64_s37", "attn_decode_kv_q4"),
    kv_case("attn_decode_kv_q4_hd64_s1000", "attn_decode_kv_q4"),
    // 100K depth, one head: the code path that the bench uses, compared
    // against the reference. Kept out of `cases()` because the host reference is
    // a second of work; `hipml parity --deep` runs it.
    kv_case("attn_decode_kv_q8_hd256_s100000", "attn_decode_kv_q8"),
    kv_case("attn_decode_kv_q4_hd256_s100000", "attn_decode_kv_q4"),
];

const fn kv_case(name: &'static str, entry: &'static str) -> KernelSpec {
    KernelSpec {
        name,
        entry,
        kind: KernelKind::KvSplit,
    }
}

/// The decode-path variant of a row-tile GEMV.
///
/// One row per warp instead of two: the lane owns the same columns and the
/// reduction tree is unchanged, so the result is bit-identical, but a block
/// covers 8 rows instead of 16 and the grid doubles. Shapes whose row count
/// leaves the machine under-occupied (5120 rows = 320 blocks) stream 14%
/// faster; 17408-row shapes gain ~3%.
pub fn stream_entry(name: &'static str) -> &'static str {
    match name {
        "gemv_q4_k" => "gemv_q4_k_r1",
        "gemv_q6_k" => "gemv_q6_k_r1",
        other => other,
    }
}

/// The skinny-GEMM entry for a resident block format (`None` for formats with
/// no such kernel, e.g. Q5_K). `R <= 16` columns, `y[r, :] = W @ x[r, :]`.
pub fn skinny_entry(name: &str) -> Option<&'static str> {
    Some(match name {
        "gemv_q4_k" | "gemm_skinny_q4_k" => "gemm_skinny_q4_k",
        "gemv_q6_k" | "gemm_skinny_q6_k" => "gemm_skinny_q6_k",
        "gemv_q8_0" | "gemm_skinny_q8_0" => "gemm_skinny_q8_0",
        _ => return None,
    })
}

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
