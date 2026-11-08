//! Quantized-KV decode attention: the case driver and the two-kernel ABI.
//!
//! Split out of `launch.rs` by size. The split pass and the reduce are one
//! operation, so they share one launcher and one argument-array definition;
//! `launch` is also the entry the KV bench calls with persistent buffers.

use hipml_core::parity::OpCase;
use hipml_core::tensor::Tensor;
use hipml_hip::{Buffer, LaunchConfig, Module, Stream};


pub fn run(
    module: &Module,
    stream: &Stream,
    step: &Buffer,
    case: &OpCase,
    inputs: &[Tensor],
    reps: usize,
) -> Result<Vec<f32>, String> {
            let n_heads = case.scalar("n_heads") as usize;
            let n_kv = case.scalar("n_kv") as usize;
            let head_dim = case.scalar("head_dim") as usize;
            let seq = case.scalar("seq") as usize;
            let scale = case.scalar("scale");
            let splits = (case.scalar("splits") as usize).max(1);
            let dtype = case.scalar("dtype") as usize;

            let q = Buffer::from_host(inputs[0].data())?;
            let k = Buffer::from_host(inputs[1].data())?;
            let v = Buffer::from_host(inputs[2].data())?;
            let out = Buffer::alloc(n_heads * head_dim)?;
            let part_acc = Buffer::alloc(n_heads * splits * head_dim)?;
            let part_ml = Buffer::alloc(n_heads * splits * 2)?;
            // The kernel reads the position count from `step`; this arm decodes
            // `seq` positions against a cache of `seq` rows.
            let mut step_bytes = [0u8; 32];
            for (i, v) in [0i32, seq as i32, 0, 0, 0, 0, 0, 0].iter().enumerate() {
                step_bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
            step.upload_bytes(0, &step_bytes)?;

            for _ in 0..reps {
                launch(
                    module,
                    stream,
                    step,
                    dtype,
                    &q,
                    &k,
                    &v,
                    &part_acc,
                    &part_ml,
                    &out,
                    n_heads,
                    n_kv,
                    head_dim,
                    seq,
                    scale,
                    splits,
                    0,
                    0,
                )?;
            }
            stream.synchronize()?;
            out.to_host()
        }

/// Launch the quantized decode pair once: `attn_decode_kv_{q8,q4}` over
/// `n_kv * splits` blocks, then `attn_decode_kv_reduce` over `n_heads`.
///
/// Shared by the parity arm and the KV bench so the argument array — the kernel
/// ABI — is written out in exactly one place. `part_acc` and `part_ml` are the
/// split scratch: `[n_heads][splits][head_dim]` and `[n_heads][splits][2]` f32.
#[cfg(feature = "hip")]
#[allow(clippy::too_many_arguments)]
pub fn launch(
    module: &Module,
    stream: &Stream,
    step: &Buffer,
    dtype: usize,
    q: &Buffer,
    k: &Buffer,
    v: &Buffer,
    part_acc: &Buffer,
    part_ml: &Buffer,
    out: &Buffer,
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
    seq: usize,
    scale: f32,
    splits: usize,
    k_off: usize,
    v_off: usize,
) -> Result<(), String> {
    // `seq` is both the number of valid positions and the cache's per-head row
    // stride; the count comes from the step buffer at run time and the stride is
    // baked here, so one captured graph serves every step.
    launch_ptrs(
        module,
        stream,
        step,
        dtype,
        q.as_ptr(),
        k,
        v,
        part_acc,
        part_ml,
        out.as_ptr(),
        n_heads,
        n_kv,
        head_dim,
        seq,
        scale,
        splits,
        k_off,
        v_off,
    )
}

/// Same pair, with the query row and output row given as raw device pointers.
/// The chunked q8 prefill walks one token at a time over the chunk's projection
/// buffers, so it needs row pointers instead of a whole-buffer `Buffer`.
#[cfg(feature = "hip")]
#[allow(clippy::too_many_arguments)]
pub fn launch_ptrs(
    module: &Module,
    stream: &Stream,
    step: &Buffer,
    dtype: usize,
    q_ptr: *mut std::ffi::c_void,
    k: &Buffer,
    v: &Buffer,
    part_acc: &Buffer,
    part_ml: &Buffer,
    out_ptr: *mut std::ffi::c_void,
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
    seq: usize,
    scale: f32,
    splits: usize,
    k_off: usize,
    v_off: usize,
) -> Result<(), String> {
    if n_kv == 0 || n_heads % n_kv != 0 {
        return Err(format!("n_heads {n_heads} is not a multiple of n_kv {n_kv}"));
    }
    let group = n_heads / n_kv;
    if group > 8 {
        return Err(format!("GQA group {group} > 8: one block holds at most 8 warps"));
    }
    if head_dim % 32 != 0 || head_dim > 256 {
        return Err(format!("head_dim {head_dim}: must be 32..=256, multiple of 32"));
    }
    // The shipped q8 path at head_dim 256 is the warp kernel: the parity cases
    // and the engine both take it, so `parity --deep` exercises what runs.
    // Id 6 keeps the staged kernel reachable for A/B.
    let dtype = if dtype == 0 && head_dim == 256 && group <= 8 {
        5
    } else {
        dtype
    };
    let entry = match dtype {
        0 | 6 => "attn_decode_kv_q8",
        1 => "attn_decode_kv_q4",
        2 => "attn_decode_kv_q8s",
        3 => "attn_decode_kv_q8g",
        4 => "attn_decode_kv_w2",
        5 => "attn_decode_kv_w4",
        8 => "attn_decode_kv_lab_full",
        9 => "attn_decode_kv_lab_nov",
        10 => "attn_decode_kv_lab_nok",
        11 => "attn_decode_kv_lab_stage",
        12 => "attn_decode_kv_lab_nostage",
        other => return Err(format!("unknown KV dtype {other}")),
    };
    if dtype >= 2 && dtype <= 3 && head_dim % 128 != 0 {
        return Err(format!("head_dim {head_dim}: this variant needs a multiple of 128"));
    }
    // The warp-per-KV-head kernel owns eight components per lane, so it is
    // head_dim 256 only; other widths keep the staged kernel.
    if dtype >= 4 && head_dim != 256 {
        return Err(format!("head_dim {head_dim}: the warp kernel is head_dim 256 only"));
    }
    // Same constants as KV_TILE_Q8 / KV_TILE_Q4 in attn_decode_kv.hip and
    // Q8S_TILE in attn_decode_kv_q8s.hip.
    let tile = match dtype {
        1 => 8,
        2 => 16,
        0 | 6 | 8..=12 => 4,
        _ => 0,
    };
    let row_bytes = (head_dim >> 5) * 34;
    let shared = match dtype {
        2 => 2 * tile * row_bytes,
        4 | 5 => 0,
        _ => 2 * tile * head_dim * 4,
    };

    let kernel = module.kernel(entry)?;
    let mut p_q = q_ptr;
    let mut p_k = (k.as_ptr() as *mut u8).wrapping_add(k_off) as *mut std::ffi::c_void;
    let mut p_v = (v.as_ptr() as *mut u8).wrapping_add(v_off) as *mut std::ffi::c_void;
    let mut p_acc = part_acc.as_ptr();
    let mut p_ml = part_ml.as_ptr();
    let mut v_heads = n_heads as i32;
    let mut v_kv = n_kv as i32;
    let mut v_dim = head_dim as i32;
    let mut p_step = step.as_ptr();
    let mut v_stride = seq as i32;
    let mut v_scale = scale;
    let mut v_splits = splits as i32;
    let mut params: [*mut std::ffi::c_void; 12] = [
        &mut p_q as *mut _ as *mut std::ffi::c_void,
        &mut p_k as *mut _ as *mut std::ffi::c_void,
        &mut p_v as *mut _ as *mut std::ffi::c_void,
        &mut p_acc as *mut _ as *mut std::ffi::c_void,
        &mut p_ml as *mut _ as *mut std::ffi::c_void,
        &mut v_heads as *mut _ as *mut std::ffi::c_void,
        &mut v_kv as *mut _ as *mut std::ffi::c_void,
        &mut v_dim as *mut _ as *mut std::ffi::c_void,
        &mut p_step as *mut _ as *mut std::ffi::c_void,
        &mut v_stride as *mut _ as *mut std::ffi::c_void,
        &mut v_scale as *mut _ as *mut std::ffi::c_void,
        &mut v_splits as *mut _ as *mut std::ffi::c_void,
    ];
    // The warp kernel launches one warp per (KV head, split) task, four to a
    // block; the staged kernels launch one block per task.
    let warps = if dtype == 4 || dtype == 5 { 4 } else { 0 };
    let tasks = n_kv * splits;
    let grid = if warps > 0 {
        tasks.div_ceil(warps) as u32
    } else {
        tasks as u32
    };
    let block = if warps > 0 { 32 * warps } else { 32 * group };
    let cfg = LaunchConfig {
        grid: (grid, 1, 1),
        block: (block as u32, 1, 1),
        // KV_TILE rows of K and of V. The f32-staging kernels stage
        // dequantized rows (q8: 4 rows = 8 KB; q4: 8 rows = 16 KB); q8s stages
        // the 272 B quantized rows (16 rows = 8.7 KB); q8g uses none. Must
        // match the constants in the kernel.
        shared_mem_bytes: shared as u32,
    };
    kernel.launch(cfg, stream, &mut params)?;

    let reduce = module.kernel("attn_decode_kv_reduce")?;
    let mut p_acc = part_acc.as_ptr();
    let mut p_ml = part_ml.as_ptr();
    let mut p_out = out_ptr;
    let mut v_dim = head_dim as i32;
    let mut v_splits = splits as i32;
    let mut params: [*mut std::ffi::c_void; 5] = [
        &mut p_acc as *mut _ as *mut std::ffi::c_void,
        &mut p_ml as *mut _ as *mut std::ffi::c_void,
        &mut p_out as *mut _ as *mut std::ffi::c_void,
        &mut v_dim as *mut _ as *mut std::ffi::c_void,
        &mut v_splits as *mut _ as *mut std::ffi::c_void,
    ];
    let cfg = LaunchConfig {
        grid: (n_heads as u32, 1, 1),
        block: (head_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    reduce.launch(cfg, stream, &mut params)
}
