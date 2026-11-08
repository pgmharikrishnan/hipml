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
    let entry = match dtype {
        0 => "attn_decode_kv_q8",
        1 => "attn_decode_kv_q4",
        other => return Err(format!("unknown KV dtype {other}")),
    };
    // Same constants as KV_TILE_Q8 / KV_TILE_Q4 in attn_decode_kv.hip.
    let tile = if dtype == 0 { 4 } else { 8 };

    let kernel = module.kernel(entry)?;
    let mut p_q = q.as_ptr();
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
    let cfg = LaunchConfig {
        grid: ((n_kv * splits) as u32, 1, 1),
        block: ((32 * group) as u32, 1, 1),
        // KV_TILE rows of K and of V, dequantized to f32. q8 stages twice the
        // bytes per row, so it runs a 4-row tile (8 KB per block, 8 blocks in
        // the 64 KB of LDS per CU); q4 runs an 8-row tile (16 KB, 4 blocks) for
        // half the barriers. Must match KV_TILE_Q8 / KV_TILE_Q4 in the kernel.
        shared_mem_bytes: (2 * tile * head_dim * 4) as u32,
    };
    kernel.launch(cfg, stream, &mut params)?;

    let reduce = module.kernel("attn_decode_kv_reduce")?;
    let mut p_acc = part_acc.as_ptr();
    let mut p_ml = part_ml.as_ptr();
    let mut p_out = out.as_ptr();
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
