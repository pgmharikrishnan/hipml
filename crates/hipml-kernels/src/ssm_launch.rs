//! Parity / bench dispatch for the gated-delta-net cases.
//!
//! Each function mirrors one row of `parity::cases()`: it uploads the named
//! inputs, launches the same kernels the layer uses, and returns the flat
//! output the reference builder produced (results followed by carried state).

use hipml_core::parity::OpCase;
use hipml_core::tensor::Tensor;
use hipml_hip::{Buffer, LaunchConfig, Module, Stream};
use std::ffi::c_void;

use crate::ssm::{SsmDims, SsmGpuLayer, SsmLayerRefs};

fn buf(t: &Tensor) -> Result<Buffer, String> {
    Buffer::from_host(t.data())
}

/// Copy a buffer back and append it to `out`.
fn append(out: &mut Vec<f32>, b: &Buffer) -> Result<(), String> {
    out.extend_from_slice(&b.to_host()?);
    Ok(())
}

/// `ssm_conv1d_f32`: inputs [x, w, state] -> y | state'.
pub(crate) fn conv(
    module: &Module,
    stream: &Stream,
    case: &OpCase,
    inputs: &[Tensor],
    reps: usize,
) -> Result<Vec<f32>, String> {
    let channels = case.scalar("channels") as usize;
    let d_conv = case.scalar("d_conv") as usize;
    let x = buf(&inputs[0])?;
    let w = buf(&inputs[1])?;
    let state = buf(&inputs[2])?;
    let y = Buffer::alloc(channels)?;
    let kernel = module.kernel("ssm_conv1d_f32")?;

    let mut p_x = x.as_ptr();
    let mut p_w = w.as_ptr();
    let mut p_state = state.as_ptr();
    let mut p_y = y.as_ptr();
    let mut v_c = channels as i32;
    let mut v_k = d_conv as i32;
    let mut params: [*mut c_void; 6] = [
        &mut p_x as *mut _ as *mut c_void,
        &mut p_w as *mut _ as *mut c_void,
        &mut p_state as *mut _ as *mut c_void,
        &mut p_y as *mut _ as *mut c_void,
        &mut v_c as *mut _ as *mut c_void,
        &mut v_k as *mut _ as *mut c_void,
    ];
    let cfg = LaunchConfig::linear(channels, 256);
    for _ in 0..reps {
        kernel.launch(cfg, stream, &mut params)?;
    }
    stream.synchronize()?;

    let mut out = y.to_host()?;
    append(&mut out, &state)?;
    Ok(out)
}

/// `gdn_ar_f32`: inputs [q, k, v, g, beta, state] -> o | state'.
pub(crate) fn gdn(
    module: &Module,
    stream: &Stream,
    case: &OpCase,
    inputs: &[Tensor],
    reps: usize,
) -> Result<Vec<f32>, String> {
    let n_v = case.scalar("n_v") as usize;
    let n_k = case.scalar("n_k") as usize;
    let sd = case.scalar("sd") as usize;
    if sd > 128 {
        return Err(format!("gdn_ar_f32: sd {sd} > 128"));
    }
    let q = buf(&inputs[0])?;
    let k = buf(&inputs[1])?;
    let v = buf(&inputs[2])?;
    let g = buf(&inputs[3])?;
    let beta = buf(&inputs[4])?;
    let s = buf(&inputs[5])?;
    let out = Buffer::alloc(n_v * sd)?;
    let kernel = module.kernel("gdn_ar_f32")?;

    let mut p_s = s.as_ptr();
    let mut p_q = q.as_ptr();
    let mut p_k = k.as_ptr();
    let mut p_v = v.as_ptr();
    let mut p_g = g.as_ptr();
    let mut p_beta = beta.as_ptr();
    let mut p_out = out.as_ptr();
    let mut v_nv = n_v as i32;
    let mut v_nk = n_k as i32;
    let mut v_sd = sd as i32;
    let mut params: [*mut c_void; 10] = [
        &mut p_s as *mut _ as *mut c_void,
        &mut p_q as *mut _ as *mut c_void,
        &mut p_k as *mut _ as *mut c_void,
        &mut p_v as *mut _ as *mut c_void,
        &mut p_g as *mut _ as *mut c_void,
        &mut p_beta as *mut _ as *mut c_void,
        &mut p_out as *mut _ as *mut c_void,
        &mut v_nv as *mut _ as *mut c_void,
        &mut v_nk as *mut _ as *mut c_void,
        &mut v_sd as *mut _ as *mut c_void,
    ];
    let cfg = LaunchConfig {
        grid: (n_v as u32, 1, 1),
        block: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    for _ in 0..reps {
        kernel.launch(cfg, stream, &mut params)?;
    }
    stream.synchronize()?;

    let mut result = out.to_host()?;
    append(&mut result, &s)?;
    Ok(result)
}

/// `l2norm_f32`: inputs [x] -> normalised.
pub(crate) fn l2norm(
    module: &Module,
    stream: &Stream,
    case: &OpCase,
    inputs: &[Tensor],
    reps: usize,
) -> Result<Vec<f32>, String> {
    let heads = case.scalar("heads") as usize;
    let dim = case.scalar("dim") as usize;
    let eps = case.scalar("eps");
    let x = buf(&inputs[0])?;
    let y = Buffer::alloc(heads * dim)?;
    let kernel = module.kernel("l2norm_f32")?;

    let mut p_x = x.as_ptr();
    let mut p_y = y.as_ptr();
    let mut v_h = heads as i32;
    let mut v_d = dim as i32;
    let mut v_eps = eps;
    let mut params: [*mut c_void; 5] = [
        &mut p_x as *mut _ as *mut c_void,
        &mut p_y as *mut _ as *mut c_void,
        &mut v_h as *mut _ as *mut c_void,
        &mut v_d as *mut _ as *mut c_void,
        &mut v_eps as *mut _ as *mut c_void,
    ];
    let cfg = LaunchConfig {
        grid: (heads as u32, 1, 1),
        block: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    for _ in 0..reps {
        kernel.launch(cfg, stream, &mut params)?;
    }
    stream.synchronize()?;
    y.to_host()
}

/// `ssm_gate_norm_f32`: inputs [o, norm, z] -> gated.
pub(crate) fn gate_norm(
    module: &Module,
    stream: &Stream,
    case: &OpCase,
    inputs: &[Tensor],
    reps: usize,
) -> Result<Vec<f32>, String> {
    let n_v = case.scalar("n_v") as usize;
    let head_v = case.scalar("head_v") as usize;
    let eps = case.scalar("eps");
    let o = buf(&inputs[0])?;
    let norm = buf(&inputs[1])?;
    let z = buf(&inputs[2])?;
    let y = Buffer::alloc(n_v * head_v)?;
    let kernel = module.kernel("ssm_gate_norm_f32")?;

    let mut p_o = o.as_ptr();
    let mut p_norm = norm.as_ptr();
    let mut p_z = z.as_ptr();
    let mut p_y = y.as_ptr();
    let mut v_h = n_v as i32;
    let mut v_d = head_v as i32;
    let mut v_eps = eps;
    let mut params: [*mut c_void; 7] = [
        &mut p_o as *mut _ as *mut c_void,
        &mut p_norm as *mut _ as *mut c_void,
        &mut p_z as *mut _ as *mut c_void,
        &mut p_y as *mut _ as *mut c_void,
        &mut v_h as *mut _ as *mut c_void,
        &mut v_d as *mut _ as *mut c_void,
        &mut v_eps as *mut _ as *mut c_void,
    ];
    let cfg = LaunchConfig {
        grid: (n_v as u32, 1, 1),
        block: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    for _ in 0..reps {
        kernel.launch(cfg, stream, &mut params)?;
    }
    stream.synchronize()?;
    y.to_host()
}

/// The input order `parity::build_ssm_layer` writes.
fn layer_refs<'a>(inputs: &'a [Tensor]) -> SsmLayerRefs<'a> {
    SsmLayerRefs {
        wqkv: inputs[1].data(),
        wgate: inputs[2].data(),
        conv_w: inputs[3].data(),
        alpha: inputs[4].data(),
        beta: inputs[5].data(),
        dt_bias: inputs[6].data(),
        a: inputs[7].data(),
        norm: inputs[8].data(),
        wout: inputs[9].data(),
    }
}

/// `ssm_layer_f32`: inputs are x, weights, conv state, delta-net state.
pub(crate) fn layer(
    module: &Module,
    stream: &Stream,
    case: &OpCase,
    inputs: &[Tensor],
    reps: usize,
) -> Result<Vec<f32>, String> {
    let dims = SsmDims::from_case(case);
    let w = layer_refs(inputs);
    let layer = SsmGpuLayer::new(module, stream, dims, &w)?;
    layer.write_x(inputs[0].data())?;
    layer.write_state(inputs[10].data(), inputs[11].data())?;
    for _ in 0..reps {
        layer.step()?;
    }
    layer.synchronize()?;

    let mut out = layer.read_out()?;
    let (conv, s) = layer.read_state()?;
    out.extend_from_slice(&conv);
    out.extend_from_slice(&s);
    Ok(out)
}
