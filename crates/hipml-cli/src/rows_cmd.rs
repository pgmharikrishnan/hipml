//! `hipml rows-check`: prove the multi-row tail prefill (WP47) matches the
//! single-sequence chunk path token for token and state for state.
//!
//! For each slot it builds a distinct prefix, prefills it on the single-stream
//! engine, and copies that state into a quiet batch slot. Then all tails
//! (variable length) run through one `prefill_rows`, and the check compares:
//!
//! * every tail's first sampled id against its own `prefill_tile` run,
//! * every slot's carried SSM state (conv window + delta-net matrix) against
//!   the single-sequence engine's, bit for bit,
//! * `--decode` further greedy ids through the batched decode.
//!
//! Exits non-zero on the first mismatch.

use hipml_model::gpu::{BatchEngine, GpuEngine, KvMode};
use hipml_model::Model;

struct Args {
    gguf: String,
    slots: usize,
    prefix: usize,
    tail: usize,
    decode: usize,
}

fn parse(argv: &[String]) -> Result<Args, String> {
    let mut a = Args {
        gguf: String::new(),
        slots: 8,
        prefix: 96,
        tail: 17,
        decode: 8,
    };
    let mut i = 0;
    while i < argv.len() {
        let need = |i: usize| -> Result<&str, String> {
            argv.get(i + 1)
                .map(String::as_str)
                .ok_or_else(|| format!("{} needs a value", argv[i]))
        };
        match argv[i].as_str() {
            "--gguf" => {
                a.gguf = need(i)?.to_string();
                i += 2;
            }
            "--slots" => {
                a.slots = need(i)?.parse().map_err(|e| format!("bad --slots: {e}"))?;
                i += 2;
            }
            "--prefix" => {
                a.prefix = need(i)?.parse().map_err(|e| format!("bad --prefix: {e}"))?;
                i += 2;
            }
            "--tail" => {
                a.tail = need(i)?.parse().map_err(|e| format!("bad --tail: {e}"))?;
                i += 2;
            }
            "--decode" => {
                a.decode = need(i)?.parse().map_err(|e| format!("bad --decode: {e}"))?;
                i += 2;
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if a.gguf.is_empty() {
        return Err("--gguf PATH is required".to_string());
    }
    if a.slots == 0 || a.slots > 32 {
        return Err("--slots must be 1..=32".to_string());
    }
    if a.prefix < 2 {
        return Err("--prefix must be at least 2".to_string());
    }
    if a.decode == 0 {
        return Err("--decode must be at least 1".to_string());
    }
    if !(3..=60).contains(&a.tail) {
        return Err("--tail must be 3..=60".to_string());
    }
    Ok(a)
}

fn main_run(a: &Args) -> Result<(), String> {
    let (model, report) = Model::load(&a.gguf).map_err(|e| format!("load: {e}"))?;
    println!(
        "resident: {:.2} GiB in {} tensors on {} (upload {:.1}s)",
        report.tensor_bytes as f64 / (1u64 << 30) as f64,
        report.tensor_count,
        model.device_name(),
        report.upload_secs
    );
    let ctx = a.prefix + a.tail + a.decode + 16;
    let mut single = GpuEngine::new(&model, ctx, KvMode::Q8)?;
    let mut batch = BatchEngine::new(&model, ctx, KvMode::Q8, a.slots)?;
    let vocab = single.n_vocab() as u32;
    let cap = (a.prefix + a.tail + 16).next_power_of_two().max(64);

    // Distinct token streams per slot, so a state bleed between slots would
    // change an id instead of hiding behind identical inputs.
    let tok = |b: usize, i: usize| -> u32 { (97 + b as u32 * 1013 + i as u32 * 37) % vocab };
    let tail_len = |b: usize| -> usize { a.tail + b % 3 };

    struct Expect {
        tail: Vec<u32>,
        ids: Vec<u32>,
        ssm: Vec<f32>,
    }
    let mut expect: Vec<Expect> = Vec::with_capacity(a.slots);
    for b in 0..a.slots {
        let prefix: Vec<u32> = (0..a.prefix).map(|i| tok(b, i)).collect();
        let tail: Vec<u32> = (0..tail_len(b))
            .map(|i| tok(b, a.prefix + i))
            .collect();
        single.reset_state()?;
        let mut st = single.new_state();
        single.prefill_tile(&prefix, &mut st, cap, false)?;
        let first = single.prefill_tile(&tail, &mut st, cap, false)?;
        let ssm = single.ssm_state_host()?;
        let mut ids = vec![first];
        for _ in 1..a.decode {
            let last = *ids.last().expect("decode >= 1");
            let id = single.forward_token_id(last, &mut st)?;
            ids.push(id);
        }
        expect.push(Expect { tail, ids, ssm });
    }

    // Second pass: restore each prefix into its quiet slot.
    for b in 0..a.slots {
        let prefix: Vec<u32> = (0..a.prefix).map(|i| tok(b, i)).collect();
        single.reset_state()?;
        let mut st = single.new_state();
        single.prefill_tile(&prefix, &mut st, cap, false)?;
        batch.adopt_quiet(&single, batch.next_slot(), st.pos)?;
        if std::env::var_os("HIPML_ROWS_DEBUG").is_some() {
            let got = batch.slot_ssm_state_host(b)?;
            let want = single.ssm_state_host()?;
            let bad = got
                .iter()
                .zip(want.iter())
                .position(|(x, y)| x.to_bits() != y.to_bits());
            println!("adopt slot {b}: first differing word {bad:?}");
        }
    }

    // One weight pass over every tail.
    let tails: Vec<(usize, &[u32])> = expect
        .iter()
        .enumerate()
        .map(|(b, e)| (b, e.tail.as_slice()))
        .collect();
    let firsts = batch.prefill_rows(&tails)?;
    let (rows, ms) = batch.last_row_block();
    println!(
        "row block: {} tails, {} rows, {:.2} ms ({:.3} ms/tail, {:.2} row/ms)",
        firsts.len(),
        rows,
        ms,
        ms / firsts.len() as f64,
        rows as f64 / ms
    );

    let mut ok = true;
    for (b, e) in expect.iter().enumerate() {
        if firsts[b] != e.ids[0] {
            println!(
                "slot {b} (tail {}): first id row {} vs tile {} -> MISMATCH",
                e.tail.len(),
                firsts[b],
                e.ids[0]
            );
            ok = false;
            continue;
        }
        let state = batch.slot_ssm_state_host(b)?;
        let same = state.len() == e.ssm.len()
            && state
                .iter()
                .zip(e.ssm.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits());
        if !same {
            let first_bad = state
                .iter()
                .zip(e.ssm.iter())
                .position(|(a, b)| a.to_bits() != b.to_bits());
            let diffs = state
                .iter()
                .zip(e.ssm.iter())
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let worst = state
                .iter()
                .zip(e.ssm.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!(
                "slot {b} (tail {}): SSM state differs at word {:?}, {} words, max |d| {worst:e} (len {}/{}) -> MISMATCH",
                e.tail.len(),
                first_bad,
                diffs,
                state.len(),
                e.ssm.len()
            );
            ok = false;
            continue;
        }
        println!(
            "slot {b} (tail {}): first id {}, SSM state {} words identical",
            e.tail.len(),
            firsts[b],
            state.len()
        );
    }
    if !ok {
        return Err("row block differs from the single-sequence path".to_string());
    }

    // Decode the same stream in the batch and compare every id.
    let mut feeds = firsts.clone();
    let mut matched = 0usize;
    for step in 1..a.decode {
        let ids = batch.forward(&feeds)?;
        for (b, e) in expect.iter().enumerate() {
            if ids[b] != e.ids[step] {
                println!(
                    "slot {b} decode step {step}: batch {} vs single {} -> MISMATCH",
                    ids[b], e.ids[step]
                );
                ok = false;
            } else {
                matched += 1;
            }
        }
        feeds = ids;
    }
    if !ok {
        return Err("batched decode differs from the single-sequence path".to_string());
    }
    println!(
        "rows-check: {} tails, {} decode ids, all match the single-sequence path",
        a.slots, matched
    );
    Ok(())
}

pub fn run(argv: &[String]) -> i32 {
    let a = match parse(argv) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("rows-check: {e}");
            return 2;
        }
    };
    match main_run(&a) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("rows-check: {e}");
            1
        }
    }
}
