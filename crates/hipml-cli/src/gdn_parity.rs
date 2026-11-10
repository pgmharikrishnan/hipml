// SPDX-License-Identifier: Apache-2.0
//! `hipml gdn-parity`: the row-split chunk recurrence against the per-head one.
//!
//! `gdn_rows_f32` claims to do the arithmetic of `gdn_chunk_f32` element for
//! element, so the check is bit equality of both the per-token output and the
//! carried state, on Qwen3.8-27B's GDN shape (48 value heads, 16 key heads, 128 wide)
//! at a token count that is not a multiple of the 16-token tile. It then times
//! both kernels on a 512-token tile.

use std::ffi::c_void;
use std::time::Instant;

use hipml_core::rng::Rng;
use hipml_hip::{Buffer, LaunchConfig};
use hipml_kernels::Launcher;

const N_V: usize = 48;
const N_K: usize = 16;
const SD: usize = 128;
const KEY_DIM: usize = N_K * SD;
const CONV_CH: usize = 2 * KEY_DIM + N_V * SD;

struct Case {
    s0: Vec<f32>,
    qkv: Buffer,
    g: Buffer,
    beta: Buffer,
}

fn case(tokens: usize, seed: u64) -> Result<Case, String> {
    let mut rng = Rng::new(seed);
    let s0: Vec<f32> = (0..N_V * SD * SD).map(|_| rng.normal_scaled(0.1)).collect();
    let mut qkv: Vec<f32> = (0..tokens * CONV_CH).map(|_| rng.normal()).collect();
    // q and k are L2-normalised per head upstream: scale to that magnitude.
    for t in 0..tokens {
        for x in &mut qkv[t * CONV_CH..t * CONV_CH + 2 * KEY_DIM] {
            *x *= 0.088;
        }
    }
    let g: Vec<f32> = (0..tokens * N_V).map(|_| -2.0 * rng.uniform()).collect();
    let beta: Vec<f32> = (0..tokens * N_V).map(|_| rng.uniform()).collect();
    Ok(Case {
        s0,
        qkv: Buffer::from_host(&qkv)?,
        g: Buffer::from_host(&g)?,
        beta: Buffer::from_host(&beta)?,
    })
}

/// Launch one recurrence kernel over `tokens` rows, state in `s`.
fn launch(
    l: &Launcher,
    rows: bool,
    c: &Case,
    s: &Buffer,
    o: &Buffer,
    tokens: usize,
) -> Result<(), String> {
    let mut p_s = s.as_ptr();
    let mut p_qkv = c.qkv.as_ptr();
    let mut p_g = c.g.as_ptr();
    let mut p_b = c.beta.as_ptr();
    let mut p_o = o.as_ptr();
    let mut v_nv = N_V as i32;
    let mut v_nk = N_K as i32;
    let mut v_sd = SD as i32;
    let mut v_cc = CONV_CH as i32;
    let mut v_kd = KEY_DIM as i32;
    let mut v_t = tokens as i32;
    // gdn_rows_f32 has the same arguments minus `sd` (fixed at 128).
    let mut params: Vec<*mut c_void> = vec![
        &mut p_s as *mut _ as *mut c_void,
        &mut p_qkv as *mut _ as *mut c_void,
        &mut p_g as *mut _ as *mut c_void,
        &mut p_b as *mut _ as *mut c_void,
        &mut p_o as *mut _ as *mut c_void,
    ];
    params.push(&mut v_nv as *mut _ as *mut c_void);
    params.push(&mut v_nk as *mut _ as *mut c_void);
    if !rows {
        params.push(&mut v_sd as *mut _ as *mut c_void);
    }
    params.push(&mut v_cc as *mut _ as *mut c_void);
    params.push(&mut v_kd as *mut _ as *mut c_void);
    params.push(&mut v_t as *mut _ as *mut c_void);
    let (name, cfg) = if rows {
        let grid = (N_V as u32, (SD / 32) as u32, 1);
        (
            "gdn_rows_f32",
            LaunchConfig {
                grid,
                block: (256, 1, 1),
                shared_mem_bytes: 0,
            },
        )
    } else {
        let grid = (N_V as u32, 1, 1);
        (
            "gdn_chunk_f32",
            LaunchConfig {
                grid,
                block: (1024, 1, 1),
                shared_mem_bytes: 0,
            },
        )
    };
    l.module()
        .kernel(name)?
        .launch(cfg, l.stream(), &mut params)
}

/// `gdn_cols_f32` (the verify kernel) against `gdn_ar_f32` (the plain decode
/// kernel) on one token: output and carried state must be bit-identical.
fn run_ar_cols(l: &Launcher) -> Result<(usize, usize), String> {
    let c = case(1, 0xC015)?;
    let mut out = Vec::new();
    for cols in [false, true] {
        let s = Buffer::from_host(&c.s0)?;
        let o = Buffer::alloc(N_V * SD)?;
        if cols {
            let mut p_s = s.as_ptr();
            let mut p_qkv = c.qkv.as_ptr();
            let mut p_g = c.g.as_ptr();
            let mut p_b = c.beta.as_ptr();
            let mut p_o = o.as_ptr();
            let mut p_snap: *mut c_void = std::ptr::null_mut();
            let mut v_stride = 0i32;
            let mut v_nv = N_V as i32;
            let mut v_nk = N_K as i32;
            let mut v_cc = CONV_CH as i32;
            let mut v_kd = KEY_DIM as i32;
            let mut v_sd = SD as i32;
            let mut v_rows = 1i32;
            let mut params: [*mut c_void; 13] = [
                &mut p_s as *mut _ as *mut c_void,
                &mut p_qkv as *mut _ as *mut c_void,
                &mut p_g as *mut _ as *mut c_void,
                &mut p_b as *mut _ as *mut c_void,
                &mut p_o as *mut _ as *mut c_void,
                &mut p_snap as *mut _ as *mut c_void,
                &mut v_stride as *mut _ as *mut c_void,
                &mut v_nv as *mut _ as *mut c_void,
                &mut v_nk as *mut _ as *mut c_void,
                &mut v_cc as *mut _ as *mut c_void,
                &mut v_kd as *mut _ as *mut c_void,
                &mut v_sd as *mut _ as *mut c_void,
                &mut v_rows as *mut _ as *mut c_void,
            ];
            let cfg = LaunchConfig {
                grid: (N_V as u32, 1, 1),
                block: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            l.module()
                .kernel("gdn_cols_f32")?
                .launch(cfg, l.stream(), &mut params)?;
        } else {
            let mut p_s = s.as_ptr();
            let mut p_q = c.qkv.as_ptr();
            let mut p_k = unsafe { (c.qkv.as_ptr() as *mut u8).add(KEY_DIM * 4) } as *mut c_void;
            let mut p_v =
                unsafe { (c.qkv.as_ptr() as *mut u8).add(2 * KEY_DIM * 4) } as *mut c_void;
            let mut p_g = c.g.as_ptr();
            let mut p_b = c.beta.as_ptr();
            let mut p_o = o.as_ptr();
            let mut v_nv = N_V as i32;
            let mut v_nk = N_K as i32;
            let mut v_sd = SD as i32;
            let mut params: [*mut c_void; 10] = [
                &mut p_s as *mut _ as *mut c_void,
                &mut p_q as *mut _ as *mut c_void,
                &mut p_k as *mut _ as *mut c_void,
                &mut p_v as *mut _ as *mut c_void,
                &mut p_g as *mut _ as *mut c_void,
                &mut p_b as *mut _ as *mut c_void,
                &mut p_o as *mut _ as *mut c_void,
                &mut v_nv as *mut _ as *mut c_void,
                &mut v_nk as *mut _ as *mut c_void,
                &mut v_sd as *mut _ as *mut c_void,
            ];
            let cfg = LaunchConfig {
                grid: (N_V as u32, 1, 1),
                block: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            l.module()
                .kernel("gdn_ar_f32")?
                .launch(cfg, l.stream(), &mut params)?;
        }
        l.synchronize()?;
        out.push((o.to_host()?, s.to_host()?));
    }
    let diff = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count()
    };
    Ok((diff(&out[0].0, &out[1].0), diff(&out[0].1, &out[1].1)))
}

fn run_both(l: &Launcher, tokens: usize) -> Result<(usize, usize), String> {    let c = case(tokens, 0x6D17_0000 + tokens as u64)?;
    let mut out = Vec::new();
    for rows in [false, true] {
        let s = Buffer::from_host(&c.s0)?;
        let o = Buffer::alloc(tokens * N_V * SD)?;
        launch(l, rows, &c, &s, &o, tokens)?;
        l.synchronize()?;
        out.push((o.to_host()?, s.to_host()?));
    }
    let diff = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count()
    };
    Ok((diff(&out[0].0, &out[1].0), diff(&out[0].1, &out[1].1)))
}

fn time(l: &Launcher, rows: bool, tokens: usize, reps: usize) -> Result<f64, String> {
    let c = case(tokens, 7)?;
    let s = Buffer::from_host(&c.s0)?;
    let o = Buffer::alloc(tokens * N_V * SD)?;
    launch(l, rows, &c, &s, &o, tokens)?;
    l.synchronize()?;
    let t0 = Instant::now();
    for _ in 0..reps {
        launch(l, rows, &c, &s, &o, tokens)?;
    }
    l.synchronize()?;
    Ok(t0.elapsed().as_secs_f64() * 1e6 / reps as f64)
}

pub fn run(_args: &[String]) -> i32 {
    let l = match Launcher::new() {
        Ok(l) => l,
        Err(e) => {
            eprintln!("gdn-parity: {e}");
            return 1;
        }
    };
    let mut ok = true;
    match run_ar_cols(&l) {
        Ok((dout, dstate)) => {
            let pass = dout == 0 && dstate == 0;
            ok &= pass;
            println!(
                "gdn_ar vs gdn_cols (1 tok): output bits differ {dout}, state bits differ {dstate}  {}",
                if pass { "PASS" } else { "FAIL" }
            );
        }
        Err(e) => {
            eprintln!("gdn-parity: {e}");
            return 1;
        }
    }
    for tokens in [1usize, 77, 512] {
        match run_both(&l, tokens) {
            Ok((dout, dstate)) => {
                let pass = dout == 0 && dstate == 0;
                ok &= pass;
                println!(
                    "tokens {tokens:>4}: output bits differ {dout}, state bits differ {dstate}  {}",
                    if pass { "PASS" } else { "FAIL" }
                );
            }
            Err(e) => {
                eprintln!("gdn-parity: {e}");
                return 1;
            }
        }
    }
    for tokens in [512usize] {
        let (Ok(head), Ok(rows)) = (time(&l, false, tokens, 20), time(&l, true, tokens, 20)) else {
            eprintln!("gdn-parity: timing launch failed");
            return 1;
        };
        println!(
            "tokens {tokens}: gdn_chunk_f32 {head:.1} us, gdn_rows_f32 {rows:.1} us, {:.2}x \
             ({:.2} vs {:.2} us/token)",
            head / rows,
            head / tokens as f64,
            rows / tokens as f64
        );
    }
    if ok {
        0
    } else {
        1
    }
}
