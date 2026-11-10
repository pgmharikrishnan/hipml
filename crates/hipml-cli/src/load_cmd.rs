// SPDX-License-Identifier: Apache-2.0
//! `hipml load` and `hipml logits`: put the whole model in VRAM and prove it.
//!
//! `load` is the acceptance gate: residency table, per-dtype totals,
//! `hipMemGetInfo` before/after, a 20-tensor byte-identical readback, five
//! embedding rows checked against the CPU dequant, and one lm_head pass.
//! `logits` skips the checks and runs lm_head on a vector supplied by the caller.

use std::path::Path;

use hipml_model::verify::{
    compare_rows, cpu_embedding_row, fixed_vector, read_f32_file, round_trip, top_k, RoundTrip,
};
use hipml_model::{LoadReport, Model};

/// Fixed token ids for the embedding check, plus the vocabulary's last row.
/// The last row used to be the literal 248319 ("the 248320-row
/// vocabulary"); it now comes from the loaded embedding shape, so a model with
/// a different vocabulary probes a valid id.
fn probe_tokens(rows: u64) -> Vec<u32> {
    let mut out = Vec::new();
    for id in [0u32, 1, 1234, 100_000] {
        if (id as u64) < rows && !out.contains(&id) {
            out.push(id);
        }
    }
    if rows > 0 {
        let last = (rows - 1) as u32;
        if !out.contains(&last) {
            out.push(last);
        }
    }
    out
}
const ROUNDTRIP_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

fn arg<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

/// Group digits, because 17,192,421,376 is easier to check than 17192421376.
fn commas(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// `hipml load --gguf PATH [--roundtrip N]`
pub fn load(args: &[String]) -> i32 {
    let Some(path) = arg(args, "--gguf") else {
        eprintln!("usage: hipml load --gguf <model.gguf> [--roundtrip N]");
        return 2;
    };
    let count = arg(args, "--roundtrip")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(20);
    match run_load(Path::new(path), count) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("load failed: {e}");
            1
        }
    }
}

fn run_load(path: &Path, count: usize) -> Result<(), String> {
    let (model, report) = Model::load(path)?;
    print_report(&model, &report, path);
    print_config(path)?;

    let trips = round_trip(&model, path, count, ROUNDTRIP_SEED)?;
    print_round_trip(&trips);

    let rows = model
        .tensor(hipml_model::TOKEN_EMBD)
        .map(|t| t.rows())
        .unwrap_or(0);
    print_embeddings(&model, path, &probe_tokens(rows))?;

    let k = model
        .tensor(hipml_model::OUTPUT)
        .map(|t| t.ne0() as usize)
        .unwrap_or(5120);
    let x = fixed_vector(k);
    print_lm_head(&model, &x, 1)?;
    Ok(())
}

fn print_config(path: &Path) -> Result<(), String> {
    use hipml_core::config::Config;
    use hipml_core::gguf::Gguf;
    let mut g = Gguf::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let c = Config::from_gguf(&mut g).map_err(|e| format!("config: {e}"))?;
    let full: Vec<usize> = c
        .is_recr
        .iter()
        .enumerate()
        .filter(|(_, r)| !**r)
        .map(|(i, _)| i)
        .collect();
    println!(
        "\nconfig ({}): {} blocks ({} decoder + {} nextn), hidden {}, ffn {}, \
         heads {} / kv {}, head_dim {}, vocab {}, eps {:e}, rope base {:e} rot {} sections {:?}, \
         ctx {}",
        c.arch,
        c.n_layer + c.nextn_layers,
        c.n_layer,
        c.nextn_layers,
        c.n_embd,
        c.n_ff,
        c.n_head,
        c.n_head_kv,
        c.head_dim,
        c.n_vocab,
        c.eps,
        c.rope_base,
        c.n_rot,
        c.rope_sections,
        c.context_length
    );
    println!(
        "  ssm: conv {}, state {}, groups {}, dt_rank {}, inner {} (head_v {})",
        c.ssm_d_conv,
        c.ssm_d_state,
        c.ssm_n_group,
        c.ssm_dt_rank,
        c.ssm_d_inner,
        c.ssm_head_v()
    );
    println!(
        "  full-attention layers ({}): {:?}",
        full.len(),
        full
    );
    Ok(())
}

fn print_report(model: &Model, r: &LoadReport, path: &Path) {
    println!("device:  {}", model.device_name());
    println!(
        "gguf:    {}  {} B ({:.2} GiB)",
        path.display(),
        commas(r.file_bytes),
        gib(r.file_bytes)
    );
    println!(
        "tensors: {}  data {} B ({:.2} GiB)  read+upload {:.2} s ({:.2} GB/s)",
        r.tensor_count,
        commas(r.tensor_bytes),
        gib(r.tensor_bytes),
        r.upload_secs,
        r.tensor_bytes as f64 / r.upload_secs.max(1e-9) / 1e9
    );
    println!(
        "hipMemGetInfo free: {} B before, {} B after; resident delta {} B ({:.2} GiB)",
        commas(r.free_before as u64),
        commas(r.free_after as u64),
        commas(r.resident_delta().max(0) as u64),
        gib(r.resident_delta().max(0) as u64)
    );

    println!(
        "\n{:<8} {:>8} {:>16} {:>16} {:>9}  {}",
        "dtype", "tensors", "bytes", "resident", "GiB", "status"
    );
    for t in &r.per_type {
        let status = if r.unsupported.iter().any(|(_, ty, _)| ty == &t.ty) {
            "sized, no dequant"
        } else {
            "supported"
        };
        println!(
            "{:<8} {:>8} {:>16} {:>16} {:>9.2}  {}",
            t.ty,
            t.tensors,
            commas(t.bytes),
            commas(t.bytes),
            gib(t.bytes),
            status
        );
    }
    println!(
        "{:<8} {:>8} {:>16} {:>16} {:>9.2}",
        "total",
        r.tensor_count,
        commas(r.tensor_bytes),
        commas(r.tensor_bytes),
        gib(r.tensor_bytes)
    );

    if r.unsupported.is_empty() {
        println!("\nunsupported: none");
    } else {
        println!("\nloaded but not numerically supported:");
        for (name, ty, bytes) in &r.unsupported {
            println!("  {:<52} {:<6} {} B", name, ty, commas(*bytes));
        }
    }
}

fn print_round_trip(trips: &[RoundTrip]) {
    println!(
        "\nround-trip ({} tensors, seed 0x{ROUNDTRIP_SEED:016X}):",
        trips.len()
    );
    println!(
        "  {:<48} {:>14} {:>18} {:>18}  {}",
        "tensor", "bytes", "file fnv1a", "device fnv1a", "result"
    );
    let mut matches = 0usize;
    for t in trips {
        let verdict = if t.matched { "MATCH" } else { "MISMATCH" };
        if t.matched {
            matches += 1;
        }
        println!(
            "  {:<48} {:>14} {:>18} {:>18}  {}",
            t.name,
            commas(t.bytes),
            format!("0x{:016X}", t.file_hash),
            format!("0x{:016X}", t.device_hash),
            verdict
        );
    }
    println!(
        "  checksum matches: {matches}/{}  byte-identical: {matches}/{}",
        trips.len(),
        trips.len()
    );
}

fn print_embeddings(model: &Model, path: &Path, probes: &[u32]) -> Result<(), String> {
    let t = model
        .tensor(hipml_model::TOKEN_EMBD)
        .ok_or_else(|| format!("no tensor {}", hipml_model::TOKEN_EMBD))?;
    println!(
        "\nembedding rows ({} {:?} {}; device dequant vs CPU dequant):",
        t.name(),
        t.shape(),
        t.ty().name()
    );
    println!(
        "  {:<10} {:>14} {:>12} {:>12}",
        "token", "exact/values", "max_abs", "max_rel"
    );
    for &token in probes {
        let device = model.embedding_row(token)?;
        let cpu = cpu_embedding_row(path, token)?;
        let c = compare_rows(&device, &cpu)
            .ok_or_else(|| format!("token {token}: length mismatch"))?;
        println!(
            "  {:<10} {:>14} {:>12.3e} {:>12.3e}",
            token,
            format!("{}/{}", c.exact, c.values),
            c.max_abs,
            c.max_rel()
        );
    }
    println!("  tolerance: max_abs <= 1e-6 (f32 arithmetic order differs only by fma)");
    Ok(())
}

fn print_lm_head(model: &Model, x: &[f32], reps: usize) -> Result<(), String> {
    let t = model
        .tensor(hipml_model::OUTPUT)
        .ok_or_else(|| format!("no tensor {}", hipml_model::OUTPUT))?;
    let (logits, secs) = model.lm_head_reps(x, reps)?;
    let per = secs / reps as f64;
    println!(
        "\nlm_head {} {:?} {} against {} f32:",
        t.name(),
        t.shape(),
        t.ty().name(),
        x.len()
    );
    println!(
        "  {} logits in {:.3} ms ({:.1} GB/s over {} B)",
        commas(logits.len() as u64),
        per * 1e3,
        t.nbytes() as f64 / per.max(1e-12) / 1e9,
        commas(t.nbytes())
    );
    println!("  top-5:");
    for (rank, (id, logit)) in top_k(&logits, 5).iter().enumerate() {
        println!("    {}. token {:>6}  logit {:>12.6}", rank + 1, id, logit);
    }
    Ok(())
}

/// `hipml logits --gguf PATH --vector f32.bin [--reps N]`
pub fn logits(args: &[String]) -> i32 {
    let Some(path) = arg(args, "--gguf") else {
        eprintln!("usage: hipml logits --gguf <model.gguf> --vector <f32.bin> [--reps N]");
        return 2;
    };
    let Some(vec_path) = arg(args, "--vector") else {
        eprintln!("usage: hipml logits --gguf <model.gguf> --vector <f32.bin> [--reps N]");
        return 2;
    };
    let reps = arg(args, "--reps")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1)
        .max(1);
    match run_logits(Path::new(path), Path::new(vec_path), reps) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("logits failed: {e}");
            1
        }
    }
}

fn run_logits(path: &Path, vec_path: &Path, reps: usize) -> Result<(), String> {
    let (model, report) = Model::load(path)?;
    let delta = report.resident_delta().max(0) as u64;
    println!(
        "device: {}  tensors {}  resident {} B ({:.2} GiB)",
        model.device_name(),
        report.tensor_count,
        commas(delta),
        gib(delta)
    );
    let x = read_f32_file(vec_path)?;
    print_lm_head(&model, &x, reps)?;
    Ok(())
}
