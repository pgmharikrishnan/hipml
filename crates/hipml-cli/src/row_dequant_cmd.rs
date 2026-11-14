// SPDX-License-Identifier: Apache-2.0
//! `hipml row-dequant`: the embedding row-gather kernel against the host
//! reference, for every format the kernel decodes.
//!
//! `token_embd.weight` is a row gather, so the only thing a format has to
//! provide is a decoder — there is no shape argument for a per-type allowlist.
//! This command proves that claim rather than asserting it: for each format it
//! builds deterministic block bytes with the reference generators in
//! `hipml_core::quant`, dequantises one row on the device with
//! `dequant_rows.hip`, and compares against `dequant_*` on the host, value for
//! value.
//!
//! `--row` sets the row width in values (default 2048, this file's `n_embd`);
//! it must be a whole number of 256-value blocks.

use hipml_core::quant::{
    bf16_to_f32, dequant_iq3_s, dequant_iq4_nl, dequant_iq4_xs, dequant_q3_k, dequant_q4_k,
    dequant_q5_k, dequant_q6_k, dequant_q8_0, random_iq3_s_bytes, random_iq4_nl_bytes,
    random_iq4_xs_bytes, random_q3_k_bytes, random_q4_k_bytes, random_q5_k_bytes,
    random_q6_k_bytes, random_q8_0_bytes, TensorType, QK_K,
};
use hipml_core::Rng;
use hipml_hip::ByteBuffer;
use hipml_kernels::{Launcher, ROW_BLOCK_VALUES};

/// One format's test: the type, its row bytes and the host reference.
struct Case {
    ty: TensorType,
    bytes: Vec<u8>,
    /// Values per block, for the host loop.
    block_values: usize,
    block_bytes: usize,
    dequant: Option<fn(&[u8], &mut [f32])>,
}

fn build_cases(row: usize, seed: u64) -> Vec<Case> {
    let units = row / ROW_BLOCK_VALUES;
    let mut rng = Rng::new(seed);
    let mut out = Vec::new();
    // The eight block formats have deterministic generators in hipml-core; F32
    // and BF16 have no block structure and are built here.
    let kquant = |ty: TensorType, bytes: Vec<u8>, f: fn(&[u8], &mut [f32])| Case {
        ty,
        bytes,
        block_values: QK_K,
        block_bytes: ty.block_bytes(),
        dequant: Some(f),
    };
    out.push(kquant(
        TensorType::Q4K,
        random_q4_k_bytes(&mut rng, units),
        dequant_q4_k,
    ));
    out.push(kquant(
        TensorType::Q5K,
        random_q5_k_bytes(&mut rng, units),
        dequant_q5_k,
    ));
    out.push(kquant(
        TensorType::Q6K,
        random_q6_k_bytes(&mut rng, units),
        dequant_q6_k,
    ));
    out.push(kquant(
        TensorType::Q3K,
        random_q3_k_bytes(&mut rng, units),
        dequant_q3_k,
    ));
    out.push(kquant(
        TensorType::IQ3S,
        random_iq3_s_bytes(&mut rng, units),
        dequant_iq3_s,
    ));
    out.push(kquant(
        TensorType::IQ4XS,
        random_iq4_xs_bytes(&mut rng, units),
        dequant_iq4_xs,
    ));
    // 32-value blocks: a 256-value unit is eight of them.
    out.push(Case {
        ty: TensorType::Q8_0,
        bytes: random_q8_0_bytes(&mut rng, row / 32),
        block_values: 32,
        block_bytes: TensorType::Q8_0.block_bytes(),
        dequant: Some(dequant_q8_0),
    });
    out.push(Case {
        ty: TensorType::IQ4NL,
        bytes: random_iq4_nl_bytes(&mut rng, row / 32),
        block_values: 32,
        block_bytes: TensorType::IQ4NL.block_bytes(),
        dequant: Some(dequant_iq4_nl),
    });
    // Unblocked: the host reference is the widening itself.
    let f32s: Vec<f32> = (0..row).map(|i| (i as f32) * 1e-4 - 0.1).collect();
    out.push(Case {
        ty: TensorType::F32,
        bytes: f32s.iter().flat_map(|v| v.to_le_bytes()).collect(),
        block_values: 1,
        block_bytes: 4,
        dequant: None,
    });
    let bf: Vec<u16> = (0..row).map(|i| ((i * 37) & 0xFFFF) as u16).collect();
    out.push(Case {
        ty: TensorType::BF16,
        bytes: bf.iter().flat_map(|v| v.to_le_bytes()).collect(),
        block_values: 1,
        block_bytes: 2,
        dequant: None,
    });
    out
}

/// The host reference for one row: block by block for a block format, the
/// widening itself for F32 / BF16.
fn host_row(c: &Case, row: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; row];
    match (c.ty, c.dequant) {
        (TensorType::F32, _) => {
            for (i, chunk) in c.bytes.chunks_exact(4).enumerate() {
                out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }
        (TensorType::BF16, _) => {
            for (i, chunk) in c.bytes.chunks_exact(2).enumerate() {
                out[i] = bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
        }
        (_, Some(f)) => {
            for (b, block) in c.bytes.chunks_exact(c.block_bytes).enumerate() {
                let base = b * c.block_values;
                f(block, &mut out[base..base + c.block_values]);
            }
        }
        (_, None) => unreachable!("every block format has a host reference"),
    }
    out
}

pub fn run(args: &[String]) -> i32 {
    let mut row = 2048usize;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--row" => {
                let Some(v) = args.get(i + 1) else {
                    eprintln!("row-dequant: --row needs a value");
                    return 2;
                };
                match v.parse() {
                    Ok(n) => row = n,
                    Err(e) => {
                        eprintln!("row-dequant: bad --row: {e}");
                        return 2;
                    }
                }
                i += 2;
            }
            other => {
                eprintln!("row-dequant: unknown argument {other}");
                return 2;
            }
        }
    }
    if row == 0 || row % ROW_BLOCK_VALUES != 0 {
        eprintln!("row-dequant: --row must be a multiple of {ROW_BLOCK_VALUES}");
        return 2;
    }

    let l = match Launcher::new() {
        Ok(l) => l,
        Err(e) => {
            eprintln!("row-dequant: device: {e}");
            return 1;
        }
    };
    println!(
        "row-dequant: {} values per row, {} blocks, device {}",
        row,
        row / ROW_BLOCK_VALUES,
        l.device_name
    );
    println!(
        "  {:<8} {:>14} {:>12} {:>12}",
        "type", "exact/values", "max_abs", "status"
    );

    let mut failed = 0;
    for c in build_cases(row, 0x524F_575F_4445_5100) {
        let buf = match ByteBuffer::from_host(&c.bytes) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("row-dequant: {}: upload: {e}", c.ty.name());
                return 1;
            }
        };
        let nblocks = row / ROW_BLOCK_VALUES;
        let device = match l.dequant_row_typed(c.ty, &buf, 0, nblocks) {
            Ok(v) => v,
            Err(e) => {
                println!("  {:<8} {:>14} {:>12} {}", c.ty.name(), "-", "-", e);
                failed += 1;
                continue;
            }
        };
        let host = host_row(&c, row);
        let mut exact = 0usize;
        let mut worst = 0.0f32;
        let mut shown = 0usize;
        for (i, (d, h)) in device.iter().zip(&host).enumerate() {
            if d.to_bits() == h.to_bits() {
                exact += 1;
            } else {
                worst = worst.max((d - h).abs());
                if std::env::var_os("HIPML_ROW_DEBUG").is_some() && shown < 8 {
                    println!("    [{i}] device {d:.6e} host {h:.6e}");
                    shown += 1;
                }
            }
        }
        // f32 arithmetic on both sides is the same order here (the kernel walks
        // the block the way the host does), so exactness is the expected result
        // and a tolerance would only hide a decode error.
        let ok = exact == row;
        if !ok {
            failed += 1;
        }
        println!(
            "  {:<8} {:>14} {:>12.3e} {}",
            c.ty.name(),
            format!("{exact}/{row}"),
            worst,
            if ok { "exact" } else { "DIFFERS" }
        );
    }
    if failed > 0 {
        println!("FAIL: {failed} format(s) differ from the host reference");
        return 1;
    }
    println!("PASS: every format decodes the same bits on device and host");
    0
}
