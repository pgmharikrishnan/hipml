//! Layer construction: the resident weights one block borrows.
//!
//! Child module of `gpu` so the engine can build a block without exposing the
//! weight views.

use super::*;

pub(super) fn build_attn<'a>(
    model: &'a Model,
    l: &Launcher,
    cfg: &Config,
    i: usize,
    n_ctx: usize,
) -> Result<AttnLayer<'a>, String> {
    let q = attn_mat(model, &format!("blk.{i}.attn_q.weight"))?;
    let k = attn_mat(model, &format!("blk.{i}.attn_k.weight"))?;
    let v = attn_mat(model, &format!("blk.{i}.attn_v.weight"))?;
    let o = attn_mat(model, &format!("blk.{i}.attn_output.weight"))?;
    let qn = model.f32_vec(&format!("blk.{i}.attn_q_norm.weight"))?;
    let kn = model.f32_vec(&format!("blk.{i}.attn_k_norm.weight"))?;
    AttnLayer::new_resident(l, q, k, v, o, &qn, &kn, cfg, n_ctx)
}

/// `(resident payload, rows, cols, kind)` for one full-attention projection.
fn attn_mat<'a>(
    model: &'a Model,
    name: &str,
) -> Result<(&'a ByteBuffer, usize, usize, Kind), String> {
    let t = model.tensor(name).ok_or_else(|| format!("model has no tensor {name}"))?;
    let shape = t.shape();
    if shape.len() != 2 {
        return Err(format!("{name}: expected a matrix, got {shape:?}"));
    }
    Ok((
        t.buffer(),
        shape[0] as usize,
        shape[1] as usize,
        mat_kind(t.ty())?,
    ))
}

pub(super) fn build_ssm<'a>(
    model: &'a Model,
    l: &'a Launcher,
    cfg: &Config,
    i: usize,
) -> Result<SsmGpuLayer<'a>, String> {
    let dims = SsmDims::from_config(cfg);
    let proj = |name: &str| -> Result<(&'a ByteBuffer, &'static str), String> {
        let t = model.tensor(name).ok_or_else(|| format!("model has no tensor {name}"))?;
        Ok((t.buffer(), gemv_entry(t.ty())?))
    };
    let conv = model.f32_vec(&format!("blk.{i}.ssm_conv1d.weight"))?;
    let dt = model.f32_vec(&format!("blk.{i}.ssm_dt.bias"))?;
    let a = model.f32_vec(&format!("blk.{i}.ssm_a"))?;
    let norm = model.f32_vec(&format!("blk.{i}.ssm_norm.weight"))?;
    let refs = SsmResidentRefs {
        wqkv: proj(&format!("blk.{i}.attn_qkv.weight"))?,
        wgate: proj(&format!("blk.{i}.attn_gate.weight"))?,
        alpha: proj(&format!("blk.{i}.ssm_alpha.weight"))?,
        beta: proj(&format!("blk.{i}.ssm_beta.weight"))?,
        wout: proj(&format!("blk.{i}.ssm_out.weight"))?,
        conv_w: &conv,
        dt_bias: &dt,
        a: &a,
        norm: &norm,
    };
    // The small vectors live as long as the layer; `new_resident` uploads them.
    let layer = SsmGpuLayer::new_resident(l.module(), l.stream(), dims, &refs)?;
    let _ = l;
    Ok(layer)
}
