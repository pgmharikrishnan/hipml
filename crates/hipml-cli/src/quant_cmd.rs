//! `hipml gguf` and `hipml decode-bench`: inspect the real model file and
//! stream every quantized weight tensor through the matching GEMV once.
//!
//! Kept out of `main.rs` so both stay readable. Nothing here allocates the whole
//! model: tensor metadata is cloned out of the reader, and payloads are pulled
//! in 64 MiB row-aligned chunks.

use std::fs::File;

use hipml_core::gguf::Gguf;
use hipml_core::parity::quant_bench_cases;
use hipml_core::quant::{dequant_q4_k, dequant_q6_k, dequant_q8_0, TensorType};
use hipml_kernels::Launcher;

/// Bytes streamed per chunk in `decode-bench`.
const CHUNK_BYTES: u64 = 64 * 1024 * 1024;
/// Default kernel repetitions per chunk: one, so the figure is a true single
/// pass over the model (one launch and one stream sync per chunk). `--reps N`
/// amortises launch/sync latency, but small tensors then hit cache and flatter
/// the number.
const DECODE_REPS: usize = 1;

fn shape_string(dims: &[u64]) -> String {
    dims.iter()
        .rev()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join("x")
}

/// `hipml gguf <model.gguf> [read <tensor>]`
pub fn gguf(args: &[String]) -> i32 {
    let Some(path) = args.first() else {
        eprintln!("usage: hipml gguf <model.gguf> [read <tensor>]");
        return 2;
    };
    let mut g = match Gguf::open(path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("{path}: {e}");
            return 1;
        }
    };

    if args.get(1).map(String::as_str) == Some("read") {
        let Some(name) = args.get(2) else {
            eprintln!("usage: hipml gguf <model.gguf> read <tensor>");
            return 2;
        };
        return read_one(&mut g, name);
    }

    println!(
        "version {}  tensors {}  kv {}  alignment {} bytes  data_offset {}",
        g.version, g.tensor_count, g.kv_count, g.alignment, g.data_offset
    );
    println!(
        "{:<52} {:<22} {:<8} {:>14}",
        "name", "shape", "type", "bytes"
    );
    for t in &g.tensors {
        println!(
            "{:<52} {:<22} {:<8} {:>14}",
            t.name,
            shape_string(&t.dims),
            t.ty.name(),
            t.nbytes
        );
    }

    // The three shapes the quantized GEMV work cares about, at blk.0.
    // blk.0 is an SSM block in this hybrid model, so it has attn_qkv rather than
    // attn_q; the first full-attention block (blk.3) carries attn_q.
    println!();
    for suffix in ["ffn_up.weight", "ffn_down.weight", "attn_qkv.weight"] {
        if let Some(t) = g
            .tensors
            .iter()
            .find(|t| t.name.starts_with("blk.0.") && t.name.ends_with(suffix))
        {
            println!(
                "blk.0 {}: shape {} type {} bytes {}",
                suffix,
                shape_string(&t.dims),
                t.ty.name(),
                t.nbytes
            );
        } else {
            println!("blk.0 {}: not found", suffix);
        }
    }
    if let Some(t) = g.tensors.iter().find(|t| t.name.ends_with("attn_q.weight")) {
        println!(
            "first attn_q {}: shape {} type {} bytes {}",
            t.name,
            shape_string(&t.dims),
            t.ty.name(),
            t.nbytes
        );
    } else {
        println!("attn_q.weight: not found");
    }
    0
}

fn read_one(g: &mut Gguf<File>, name: &str) -> i32 {
    let Some(info) = g.tensor(name).cloned() else {
        eprintln!("no tensor {name}");
        return 1;
    };
    let raw = match g.read_tensor(name) {
        Ok(raw) => raw,
        Err(e) => {
            eprintln!("{name}: {e}");
            return 1;
        }
    };
    println!(
        "{name}: shape {} type {} values {} bytes {} read {}",
        shape_string(&info.dims),
        info.ty.name(),
        info.n_elements(),
        info.nbytes,
        raw.len()
    );
    match info.ty {
        TensorType::Q4K => {
            let mut out = [0.0f32; 256];
            dequant_q4_k(&raw[..144], &mut out);
            println!("first 8 dequantised: {:?}", &out[..8]);
        }
        TensorType::Q6K => {
            let mut out = [0.0f32; 256];
            dequant_q6_k(&raw[..210], &mut out);
            println!("first 8 dequantised: {:?}", &out[..8]);
        }
        TensorType::Q8_0 => {
            let mut out = [0.0f32; 32];
            dequant_q8_0(&raw[..34], &mut out);
            println!("first 8 dequantised: {:?}", &out[..8]);
        }
        TensorType::F32 => {
            let vals: Vec<f32> = raw
                .chunks_exact(4)
                .take(8)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            println!("first 8 values: {vals:?}");
        }
        TensorType::Q5K => println!("Q5_K read OK (no dequant preview)"),
        TensorType::Unsupported(id) => println!("type {id}: raw read only"),
    }
    0
}

#[derive(Default)]
struct Acc {
    tensors: usize,
    bytes: u64,
    secs: f64,
}

impl Acc {
    fn add(&mut self, bytes: u64, secs: f64) {
        self.tensors += 1;
        self.bytes += bytes;
        self.secs += secs;
    }

    fn gbs(&self) -> f64 {
        if self.secs > 0.0 {
            self.bytes as f64 / self.secs / 1e9
        } else {
            0.0
        }
    }
}

/// `hipml decode-bench <model.gguf>`
///
/// One pass over every Q4_K/Q6_K/Q8_0 tensor, 64 MiB at a time, launching the
/// matching GEMV once per chunk. Reports the bytes actually streamed, kernel
/// GB/s, and the tok/s that would follow if a decode step streamed exactly one
/// model's worth of quantized weights.
pub fn decode_bench(args: &[String]) -> i32 {
    let Some(path) = args.first().filter(|a| !a.starts_with('-')) else {
        eprintln!("usage: hipml decode-bench <model.gguf> [-v] [--reps N]");
        return 2;
    };
    let verbose = args.iter().any(|a| a == "-v");
    let reps = args
        .iter()
        .position(|a| a == "--reps")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DECODE_REPS)
        .max(1);
    let mut g = match Gguf::open(path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("{path}: {e}");
            return 1;
        }
    };
    let launcher = match Launcher::new() {
        Ok(l) => l,
        Err(e) => {
            println!("device: none ({e})");
            return 1;
        }
    };
    println!("device: {}", launcher.device_name);
    println!(
        "chunk {} MiB, {} kernel reps per chunk (file read once; copies outside the timer)",
        CHUNK_BYTES / (1024 * 1024),
        reps
    );

    let tensors: Vec<_> = g
        .tensors
        .iter()
        .filter(|t| matches!(t.ty, TensorType::Q4K | TensorType::Q6K | TensorType::Q8_0))
        .cloned()
        .collect();
    let all_weight_bytes: u64 = g.tensors.iter().map(|t| t.nbytes).sum();
    let quant_weight_bytes: u64 = tensors.iter().map(|t| t.nbytes).sum();

    let mut q4 = Acc::default();
    let mut q6 = Acc::default();
    let mut q8 = Acc::default();
    let mut total = Acc::default();

    for t in &tensors {
        let (op, acc) = match t.ty {
            TensorType::Q4K => ("gemv_q4_k", &mut q4),
            TensorType::Q6K => ("gemv_q6_k", &mut q6),
            TensorType::Q8_0 => ("gemv_q8_0", &mut q8),
            _ => unreachable!(),
        };
        let k = t.ne0() as usize;
        let m = t.rows() as usize;
        let bv = t.ty.block_values() as u64;
        if k == 0 || m == 0 || (k as u64) % bv != 0 {
            eprintln!(
                "skip {}: ne0 {} is not a whole {} block",
                t.name,
                k,
                t.ty.name()
            );
            continue;
        }
        let row_bytes = t.nbytes / m as u64;
        let rows_per_chunk = ((CHUNK_BYTES / row_bytes).max(1)) as usize;
        let x = vec![0.01f32; k];
        let mut buf: Vec<u8> = Vec::new();
        let mut offset = 0u64;
        let mut t_bytes = 0u64;
        let mut t_secs = 0f64;
        let mut row = 0usize;
        while row < m {
            let rows = (m - row).min(rows_per_chunk);
            let nbytes = rows as u64 * row_bytes;
            buf.resize(nbytes as usize, 0);
            if let Err(e) = g.read_tensor_range(&t.name, offset, nbytes as usize, &mut buf) {
                eprintln!("{}: {e}", t.name);
                break;
            }
            match launcher.run_quant_reps(op, &buf, &x, rows, k, reps) {
                Ok((_, secs)) => t_secs += secs / reps as f64,
                Err(e) => {
                    eprintln!("{}: launch failed: {e}", t.name);
                    break;
                }
            }
            offset += nbytes;
            t_bytes += nbytes;
            row += rows;
        }
        acc.add(t_bytes, t_secs);
        total.add(t_bytes, t_secs);
        if verbose {
            let us = t_secs * 1e6;
            let gbs = if t_secs > 0.0 {
                t_bytes as f64 / t_secs / 1e9
            } else {
                0.0
            };
            println!(
                "{:<44} {:>6}x{:<6} {:>9.2} MB {:>9.1} us {:>7.1} GB/s",
                t.name,
                m,
                k,
                t_bytes as f64 / 1e6,
                us,
                gbs
            );
        }
    }

    println!(
        "\n{:<8} {:>8} {:>16} {:>10} {:>10}",
        "type", "tensors", "bytes", "GB", "GB/s"
    );
    for (name, a) in [("Q4_K", &q4), ("Q6_K", &q6), ("Q8_0", &q8)] {
        println!(
            "{:<8} {:>8} {:>16} {:>10.1} {:>10.1}",
            name,
            a.tensors,
            a.bytes,
            a.bytes as f64 / 1e9,
            a.gbs()
        );
    }
    println!(
        "{:<8} {:>8} {:>16} {:>10.1} {:>10.1}",
        "total",
        total.tensors,
        total.bytes,
        total.bytes as f64 / 1e9,
        total.gbs()
    );

    let bw = total.gbs();
    println!(
        "\nquantized weight bytes: {:.3} GB  (all tensor bytes: {:.3} GB)",
        quant_weight_bytes as f64 / 1e9,
        all_weight_bytes as f64 / 1e9
    );
    if total.secs > 0.0 {
        let implied = 1.0 / total.secs;
        let implied_all = bw * 1e9 / all_weight_bytes as f64;
        println!(
            "implied decode: {:.2} tok/s (quantized bytes only), {:.2} tok/s (all tensor bytes at the same GB/s)",
            implied, implied_all
        );
        println!("llama.cpp HIP reference: 31.4 tok/s");
    }
    0
}

/// Print the quantized GEMV bench rows. Called from `hipml bench`.
pub fn bench_quant(launcher: &Launcher, reps: usize) -> i32 {
    for case in quant_bench_cases() {
        let (w, x, _) = case.reference();
        if let Err(e) = launcher.run_quant_reps(case.name, &w, &x, case.m, case.k, 1) {
            println!("{:<28} failed: {e}", case.label);
            continue;
        }
        let (_, secs) = match launcher.run_quant_reps(case.name, &w, &x, case.m, case.k, reps) {
            Ok(v) => v,
            Err(e) => {
                println!("{:<28} failed: {e}", case.label);
                continue;
            }
        };
        let per = secs / reps as f64;
        let bytes = case.weight_bytes();
        println!(
            "{:<28} {:>10.2} {:>10.1} {:>12}",
            case.label,
            per * 1e6,
            bytes as f64 / per / 1e9,
            bytes
        );
    }
    0
}
