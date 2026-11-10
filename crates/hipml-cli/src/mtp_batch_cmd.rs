//! `hipml mtp-batch-bench`: speculative decode over a batch of slots.
//!
//! Runs the same prompts through the batch engine twice — plain batched decode
//! and the draft/verify schedule — and reports, per cell: acceptance per draft
//! position per slot, steps, ms/step, aggregate and per-stream tok/s, and a
//! token-for-token comparison of every slot's 256-token stream against the plain
//! batched run.
//!
//! ```text
//! hipml mtp-batch-bench --gguf Qwen3.8-27B.gguf --mtp mtp.gguf \
//!     --ids-file copy_200_words.ids --slots 8 --drafts 3 --tokens 256
//! ```

use std::time::Instant;

use hipml_hip::Buffer;
use hipml_model::gpu::{BatchEngine, GpuEngine, KvMode, MtpBatch, VerifySnap};
use hipml_model::Model;

struct Args {
    gguf: String,
    mtp: String,
    ids_file: String,
    slots: usize,
    active: usize,
    drafts: usize,
    tokens: usize,
    kv: KvMode,
    adapt: bool,
    skinny: bool,
    admit_at: Option<usize>,
    seed: u64,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut a = Args {
        gguf: String::new(),
        mtp: String::new(),
        ids_file: String::new(),
        slots: 8,
        active: 0,
        drafts: 3,
        tokens: 256,
        kv: KvMode::Q8,
        adapt: false,
        skinny: false,
        admit_at: None,
        seed: 7,
    };
    let mut i = 0;
    while i < args.len() {
        let need = |i: usize| -> Result<&str, String> {
            args.get(i + 1)
                .map(String::as_str)
                .ok_or_else(|| format!("{} needs a value", args[i]))
        };
        let num = |i: usize| -> Result<usize, String> {
            need(i)?.parse().map_err(|e| format!("{}: {e}", args[i]))
        };
        match args[i].as_str() {
            "--gguf" => {
                a.gguf = need(i)?.to_string();
                i += 2;
            }
            "--mtp" => {
                a.mtp = need(i)?.to_string();
                i += 2;
            }
            "--ids-file" => {
                a.ids_file = need(i)?.to_string();
                i += 2;
            }
            "--slots" => {
                a.slots = num(i)?;
                i += 2;
            }
            "--active" => {
                a.active = num(i)?;
                i += 2;
            }
            "--drafts" => {
                a.drafts = num(i)?;
                i += 2;
            }
            "--tokens" => {
                a.tokens = num(i)?;
                i += 2;
            }
            "--kv" => {
                a.kv = KvMode::parse(need(i)?)?;
                i += 2;
            }
            "--seed" => {
                a.seed = need(i)?.parse().map_err(|e| format!("bad --seed: {e}"))?;
                i += 2;
            }
            "--admit-at" => {
                a.admit_at = Some(num(i)?);
                i += 2;
            }
            "--adapt" => {
                a.adapt = true;
                i += 1;
            }
            "--skinny" => {
                a.skinny = true;
                i += 1;
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if a.gguf.is_empty() || a.mtp.is_empty() || a.ids_file.is_empty() {
        return Err("--gguf, --mtp and --ids-file are required".to_string());
    }
    if a.active == 0 {
        a.active = a.slots;
    }
    if a.active > a.slots {
        return Err(format!("--active {} exceeds --slots {}", a.active, a.slots));
    }
    if a.drafts == 0 || a.drafts > 3 {
        return Err("--drafts must be 1..=3".to_string());
    }
    Ok(a)
}

/// One configuration's measured numbers.
struct Run {
    steps: usize,
    ms_per_step: f64,
    agg: f64,
    per_stream: f64,
    streams: Vec<Vec<u32>>,
    drafted: Vec<Vec<usize>>,
    accepted: Vec<Vec<usize>>,
    draft_ms: f64,
    verify_ms: f64,
    rewrite_ms: f64,
    zero_steps: usize,
    zero_step_ms: f64,
    accept_step_ms: f64,
    admitted_ms: Option<f64>,
    admit_step_ms: Option<f64>,
}

/// Rotate the fixture per slot so the slots do not share a token stream; the
/// decoder still sees valid ids, and slot interference shows as a divergence
/// from that slot's own plain run.
fn prompt_for(ids: &[u32], slot: usize, shift: u64) -> Vec<u32> {
    let n = ids.len().min(320);
    let off = ((slot as u64 * 37 + shift) as usize) % n;
    let mut p = Vec::with_capacity(n);
    p.extend_from_slice(&ids[off..]);
    p.extend_from_slice(&ids[..off]);
    p
}

pub fn run(args: &[String]) -> i32 {
    let a = match parse(args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("mtp-batch-bench: {e}");
            return 2;
        }
    };
    if a.skinny {
        std::env::set_var("HIPML_BATCH_SKINNY", "1");
    }
    let ids = match crate::device_cmd::read_ids_file(&a.ids_file) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("mtp-batch-bench: {e}");
            return 1;
        }
    };
    let (mut model, report) = match Model::load(&a.gguf) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("mtp-batch-bench: load: {e}");
            return 1;
        }
    };
    if let Err(e) = model.load_sidecar(&a.mtp) {
        eprintln!("mtp-batch-bench: sidecar: {e}");
        return 1;
    }
    let cfg = match hipml_core::gguf::Gguf::open(model.path())
        .map_err(|e| format!("{e}"))
        .and_then(|mut g| hipml_core::config::Config::from_gguf(&mut g).map_err(|e| format!("{e}")))
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("mtp-batch-bench: config: {e}");
            return 1;
        }
    };
    let prompts: Vec<Vec<u32>> = (0..a.slots).map(|b| prompt_for(&ids, b, a.seed)).collect();
    let n_ctx = prompts.iter().map(|p| p.len()).max().unwrap_or(0) + a.tokens + a.drafts + 8;
    let n = cfg.n_embd;
    println!(
        "resident: {:.2} GiB in {} tensors on {} | slots {} active {} | {} tokens | drafts {} | kv {}",
        report.tensor_bytes as f64 / (1u64 << 30) as f64,
        report.tensor_count,
        model.device_name(),
        a.slots,
        a.active,
        a.tokens,
        a.drafts,
        a.kv.name()
    );
    let mut single = match GpuEngine::new(&model, n_ctx, a.kv) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("mtp-batch-bench: single engine: {e}");
            return 1;
        }
    };

    // Plain batched decode: the exactness reference and the throughput floor.
    let (mut batch, _mtp, seeds, _capture) = match setup(
        &model,
        &cfg,
        &prompts[..a.active],
        a.active,
        n_ctx,
        a.kv,
        &mut single,
        n,
    ) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("mtp-batch-bench: setup: {e}");
            return 1;
        }
    };
    let base = match plain_run(&mut batch, &seeds, a.tokens) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("mtp-batch-bench: plain: {e}");
            return 1;
        }
    };
    println!(
        "plain B={}: {:.2} tok/s aggregate | {:.2}/stream | {:.2} ms/step",
        a.active, base.agg, base.per_stream, base.ms_per_step
    );
    drop(batch);
    drop(_mtp);

    // Speculative run, fresh engines.
    let (mut batch, mut mtp, seeds, cap) = match setup(
        &model,
        &cfg,
        &prompts[..a.active],
        a.slots,
        n_ctx,
        a.kv,
        &mut single,
        n,
    ) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("mtp-batch-bench: setup: {e}");
            return 1;
        }
    };
    drop(cap);
    let snap = match batch.alloc_verify_snapshots(a.drafts + 1) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("mtp-batch-bench: snapshots: {e}");
            return 1;
        }
    };
    let spec = match spec_run(
        &mut batch,
        &mut mtp,
        &mut single,
        &prompts,
        &seeds,
        &snap,
        a.drafts,
        a.tokens,
        a.adapt,
        a.admit_at,
        n,
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("mtp-batch-bench: spec: {e}");
            return 1;
        }
    };
    report_run("spec", &a, &base, &spec, &seeds);
    // Exactness over the full 256-token horizon, slot by slot. An admission run
    // stops a short window after the join, so it compares the shorter horizon.
    let mut exact_ok = true;
    let mut horizon = 0usize;
    for b in 0..a.active {
        let n = spec.streams[b]
            .len()
            .min(base.streams[b].len())
            .min(a.tokens);
        horizon = n;
        let same = spec.streams[b]
            .iter()
            .zip(&base.streams[b])
            .take(n)
            .filter(|(x, y)| x == y)
            .count();
        let first = spec.streams[b]
            .iter()
            .zip(&base.streams[b])
            .position(|(x, y)| x != y);
        if same != n || first.is_some() || (a.admit_at.is_none() && n < a.tokens) {
            exact_ok = false;
            println!(
                "exact slot {b}: {same}/{n}{}",
                first.map(|i| format!(" @{i}")).unwrap_or_default()
            );
        }
    }
    println!(
        "exact: {}",
        if exact_ok {
            format!(
                "all {} slots reproduce the plain batched stream over {} tokens{}",
                a.active,
                horizon,
                if horizon < a.tokens { " (admission window)" } else { "" }
            )
        } else {
            "TOKENS DIFFER".to_string()
        }
    );
    if !exact_ok {
        return 1;
    }
    0
}

#[allow(clippy::too_many_arguments)]
fn setup<'a>(
    model: &'a Model,
    cfg: &hipml_core::config::Config,
    prompts: &[Vec<u32>],
    capacity: usize,
    n_ctx: usize,
    kv: KvMode,
    single: &mut GpuEngine<'a>,
    n: usize,
) -> Result<(BatchEngine<'a>, MtpBatch<'a>, Vec<u32>, Buffer), String> {
    let mut batch = BatchEngine::new(model, n_ctx, kv, capacity)?;
    let mut mtp = MtpBatch::new(model, n_ctx, capacity, cfg.clone())?;
    let cap = Buffer::alloc(prompts.iter().map(|p| p.len()).max().unwrap_or(1) * n)?;
    let mut seeds = Vec::with_capacity(prompts.len());
    for (b, p) in prompts.iter().enumerate() {
        single.reset_state()?;
        let mut st = single.new_state();
        let first = single.prefill_chunked_capture(p, &mut st, 256, &cap)?;
        batch.adopt(single, p.len())?;
        mtp.prefill_prompt(b, p, &cap, n)?;
        mtp.set_seed_hidden(&cap, &[(b, p.len() - 1)], n)?;
        seeds.push(first);
    }
    Ok((batch, mtp, seeds, cap))
}

fn plain_run(batch: &mut BatchEngine<'_>, seeds: &[u32], tokens: usize) -> Result<Run, String> {
    let mut streams: Vec<Vec<u32>> = seeds.iter().map(|&s| vec![s]).collect();
    let mut feeds = seeds.to_vec();
    let t = Instant::now();
    for _ in 1..tokens {
        let ids = batch.forward(&feeds)?;
        for (b, &id) in ids.iter().enumerate() {
            streams[b].push(id);
            feeds[b] = id;
        }
    }
    let s = t.elapsed().as_secs_f64();
    let steps = tokens - 1;
    let b = seeds.len();
    Ok(Run {
        steps,
        ms_per_step: s * 1e3 / steps as f64,
        agg: (b * steps) as f64 / s,
        per_stream: steps as f64 / s,
        streams,
        drafted: Vec::new(),
        accepted: Vec::new(),
        draft_ms: 0.0,
        verify_ms: 0.0,
        rewrite_ms: 0.0,
        zero_steps: 0,
        zero_step_ms: 0.0,
        accept_step_ms: 0.0,
        admitted_ms: None,
        admit_step_ms: None,
    })
}

#[allow(clippy::too_many_arguments)]
fn spec_run<'a>(
    batch: &mut BatchEngine<'a>,
    mtp: &mut MtpBatch<'a>,
    single: &mut GpuEngine<'a>,
    prompts: &[Vec<u32>],
    seeds: &[u32],
    snap: &VerifySnap,
    drafts_max: usize,
    tokens: usize,
    adapt: bool,
    admit_at: Option<usize>,
    n: usize,
) -> Result<Run, String> {
    let active0 = seeds.len();
    let mut streams: Vec<Vec<u32>> = seeds.iter().map(|&s| vec![s]).collect();
    let mut seed = seeds.to_vec();
    let mut drafted = vec![vec![0usize; drafts_max]; active0];
    let mut accepted = vec![vec![0usize; drafts_max]; active0];
    let mut plan = vec![drafts_max; active0];
    let mut zero_run = vec![0usize; active0];
    let (mut draft_s, mut verify_s, mut rewrite_s) = (0.0f64, 0.0f64, 0.0f64);
    let mut zero_steps = 0usize;
    let mut zero_step_ms = 0.0f64;
    let mut accept_step_ms = 0.0f64;
    let mut steps = 0usize;
    let mut admitted_ms = None;
    let mut admit_step_ms = None;
    let started = Instant::now();
    let mut admit_cap: Option<Buffer> = None;

    loop {
        let active = batch.active().len();
        plan.resize(active, drafts_max);
        drafted.resize(active, vec![0usize; drafts_max]);
        accepted.resize(active, vec![0usize; drafts_max]);
        zero_run.resize(active, 0);
        streams.resize(active, Vec::new());
        seed.resize(active, 0);
        let done = (0..active).all(|b| streams[b].len() >= tokens);
        // With admission the new slot starts 256 tokens behind, so the run is
        // bounded by the original slots plus a short window after the join.
        let done = done
            || admit_at.map(|at| steps > at + 24).unwrap_or(false);
        if done {
            break;
        }

        if let Some(at) = admit_at {
            if steps == at && active < prompts.len() {
                let t = Instant::now();
                let b = active;
                let p = &prompts[b];
                single.reset_state()?;
                let mut st = single.new_state();
                let buf = admit_cap.get_or_insert(Buffer::alloc(p.len() * n)?);
                let first = single.prefill_chunked_capture(p, &mut st, 256, buf)?;
                batch.adopt(single, p.len())?;
                mtp.prefill_prompt(b, p, buf, n)?;
                mtp.set_seed_hidden(buf, &[(b, p.len() - 1)], n)?;
                let elapsed = t.elapsed().as_secs_f64() * 1e3;
                admitted_ms = Some(elapsed);
                plan.push(drafts_max);
                drafted.push(vec![0usize; drafts_max]);
                accepted.push(vec![0usize; drafts_max]);
                zero_run.push(0);
                streams.push(vec![first]);
                seed.push(first);
                let step_start = Instant::now();
                let (d, v, r, z, a) = spec_step(
                    batch, mtp, snap, &mut seed, &mut plan, &mut drafted, &mut accepted,
                    drafts_max, &mut zero_run, adapt, tokens, &mut streams, n,
                )?;
                let step_ms = step_start.elapsed().as_secs_f64() * 1e3;
                draft_s += d;
                verify_s += v;
                rewrite_s += r;
                zero_steps += z;
                if z == 1 { zero_step_ms += step_ms } else { accept_step_ms += step_ms }
                admit_step_ms = Some(step_start.elapsed().as_secs_f64() * 1e3);
                let _ = a;
                steps += 1;
                continue;
            }
        }
        let step_start = Instant::now();
        let (d, v, r, z, _) = spec_step(
            batch, mtp, snap, &mut seed, &mut plan, &mut drafted, &mut accepted, drafts_max,
            &mut zero_run, adapt, tokens, &mut streams, n,
        )?;
        let step_ms = step_start.elapsed().as_secs_f64() * 1e3;
        draft_s += d;
        verify_s += v;
        rewrite_s += r;
        zero_steps += z;
        if z == 1 { zero_step_ms += step_ms } else { accept_step_ms += step_ms }
        steps += 1;
    }
    let secs = started.elapsed().as_secs_f64();
    let emitted: usize = (0..active0)
        .map(|b| streams[b].len().min(tokens) - 1)
        .sum();
    Ok(Run {
        steps,
        ms_per_step: secs * 1e3 / steps.max(1) as f64,
        agg: emitted as f64 / secs,
        per_stream: emitted as f64 / (active0 as f64 * secs),
        streams,
        drafted,
        accepted,
        draft_ms: draft_s * 1e3 / steps.max(1) as f64,
        verify_ms: verify_s * 1e3 / steps.max(1) as f64,
        rewrite_ms: rewrite_s * 1e3 / steps.max(1) as f64,
        zero_steps,
        zero_step_ms,
        accept_step_ms,
        admitted_ms,
        admit_step_ms,
    })
}

/// One draft/verify/accept step. Returns `(draft, verify, rewrite, zero, accepted)` ms.
#[allow(clippy::too_many_arguments)]
fn spec_step(
    batch: &mut BatchEngine<'_>,
    mtp: &mut MtpBatch<'_>,
    snap: &VerifySnap,
    seed: &mut [u32],
    plan: &mut [usize],
    drafted: &mut [Vec<usize>],
    accepted: &mut [Vec<usize>],
    drafts_max: usize,
    zero_run: &mut [usize],
    adapt: bool,
    tokens: usize,
    streams: &mut [Vec<u32>],
    n: usize,
) -> Result<(f64, f64, f64, usize, usize), String> {
    let active = batch.active().len();
    let pos: Vec<usize> = (0..active).map(|b| batch.position(b)).collect();
    let t_draft = Instant::now();
    let rmax = plan[..active].iter().copied().max().unwrap_or(0);
    let mut drafts: Vec<Vec<u32>> = vec![Vec::new(); active];
    for k in 0..rmax {
        let toks: Vec<u32> = (0..active)
            .map(|b| {
                if k == 0 {
                    seed[b]
                } else {
                    drafts[b].get(k - 1).copied().unwrap_or(seed[b])
                }
            })
            .collect();
        let posk: Vec<usize> = pos.iter().map(|&p| p - 1 + k).collect();
        let ids = mtp.draft(&toks, &posk)?;
        for b in 0..active {
            if plan[b] > k {
                drafts[b].push(ids[b]);
            }
        }
    }
    let draft_s = t_draft.elapsed().as_secs_f64();

    let cols: Vec<Vec<u32>> = (0..active)
        .map(|b| {
            let mut v = Vec::with_capacity(1 + drafts[b].len());
            v.push(seed[b]);
            v.extend_from_slice(&drafts[b]);
            v
        })
        .collect();
    let t_verify = Instant::now();
    let ids = batch.forward_verify(&cols, snap)?;
    let verify_s = t_verify.elapsed().as_secs_f64();

    let t_rw = Instant::now();
    let mut nacc = vec![0usize; active];
    for b in 0..active {
        for k in 0..drafts[b].len() {
            drafted[b][k] += 1;
            if ids[b][k] == drafts[b][k] {
                nacc[b] += 1;
                accepted[b][k] += 1;
            } else {
                break;
            }
        }
    }
    batch.accept_verify(&cols, &nacc, snap)?;

    // Emit the accepted drafts then the correction.
    let mut base_rows = Vec::with_capacity(active);
    let mut row = 0usize;
    for b in 0..active {
        base_rows.push(row);
        row += cols[b].len();
    }
    for b in 0..active {
        let room = tokens.saturating_sub(streams[b].len());
        let emitted: Vec<u32> = drafts[b][..nacc[b]]
            .iter()
            .copied()
            .chain(std::iter::once(ids[b][nacc[b]]))
            .take(room)
            .collect();
        if let Some(&last) = emitted.last() {
            seed[b] = last;
        }
        streams[b].extend(emitted);
    }
    let rows: Vec<(usize, usize)> = (0..active).map(|b| (b, base_rows[b] + nacc[b])).collect();
    mtp.set_seed_hidden(batch.verify_hidden()?, &rows, n)?;
    let mut hr = Vec::new();
    let mut tk = Vec::new();
    let mut ps = Vec::new();
    let mut counts = Vec::with_capacity(active);
    for b in 0..active {
        counts.push(nacc[b]);
        for k in 0..nacc[b] {
            hr.push(base_rows[b] + k);
            tk.push(drafts[b][k]);
            ps.push(pos[b] + k);
        }
    }
    mtp.write_kv(batch.verify_hidden()?, &hr, &tk, &ps, &counts, n)?;
    let rw_s = t_rw.elapsed().as_secs_f64();

    // Adapt: a slot that has accepted nothing for eight steps stops drafting.
    let zero = nacc.iter().all(|&a| a == 0) as usize;
    for b in 0..active {
        if nacc[b] == 0 {
            zero_run[b] += 1;
        } else {
            zero_run[b] = 0;
        }
        if adapt && zero_run[b] >= 8 {
            plan[b] = 0;
        }
    }
    let _ = drafts_max;
    Ok((draft_s, verify_s, rw_s, zero, nacc.iter().sum()))
}

fn report_run(label: &str, a: &Args, base: &Run, r: &Run, seeds: &[u32]) {
    println!();
    println!(
        "{label}: {} slots, drafts {}, {} tokens | {:.2} ms/step | {:.1} tok/s aggregate | {:.2}/stream",
        a.active, a.drafts, a.tokens, r.ms_per_step, r.agg, r.per_stream
    );
    println!(
        "  breakdown/step: draft {:.2} ms + verify {:.2} ms + rewrite {:.2} ms",
        r.draft_ms, r.verify_ms, r.rewrite_ms
    );
    println!(
        "  no-accept: {} of {} steps had no slot accept ({:.1}%); a no-accept step is {:.2} ms ({:.2}x a plain step), an accepting step {:.2} ms; spec/plain {:.2}x",
        r.zero_steps,
        r.steps,
        100.0 * r.zero_steps as f64 / r.steps.max(1) as f64,
        if r.zero_steps > 0 { r.zero_step_ms / r.zero_steps as f64 } else { 0.0 },
        if r.zero_steps > 0 { r.zero_step_ms / r.zero_steps as f64 / base.ms_per_step.max(1e-9) } else { 0.0 },
        if r.steps > r.zero_steps { r.accept_step_ms / (r.steps - r.zero_steps) as f64 } else { 0.0 },
        r.ms_per_step / base.ms_per_step.max(1e-9)
    );
    // Acceptance per draft position, per slot, then the distribution.
    let slots = r.drafted.len();
    let cols = r.drafted.first().map(|v| v.len()).unwrap_or(0);
    for k in 0..cols {
        let rates: Vec<f64> = (0..slots)
            .map(|b| {
                let d = r.drafted[b][k];
                if d == 0 {
                    0.0
                } else {
                    r.accepted[b][k] as f64 / d as f64
                }
            })
            .collect();
        if slots <= 8 {
            let per: Vec<String> = rates.iter().map(|x| format!("{x:.2}")).collect();
            println!("  accept d{}: [{}]", k + 1, per.join(" "));
        } else {
            let mut s = rates.clone();
            s.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let mean = rates.iter().sum::<f64>() / slots as f64;
            println!(
                "  accept d{}: mean {:.3} | min {:.2} p25 {:.2} median {:.2} p75 {:.2} max {:.2}",
                k + 1,
                mean,
                s[0],
                s[slots / 4],
                s[slots / 2],
                s[3 * slots / 4],
                s[slots - 1]
            );
        }
    }
    let _ = seeds;
    if let Some(ms) = r.admitted_ms {
        println!(
            "  admission: prefill+adopt+seed {:.1} ms, the joining step {:.1} ms (a normal step {:.1} ms)",
            ms,
            r.admit_step_ms.unwrap_or(0.0),
            r.ms_per_step
        );
    }
    println!(
        "  vs plain: aggregate {:.2}x, per-stream {:.2}x, step cost {:.2}x",
        r.agg / base.agg.max(1e-9),
        r.per_stream / base.per_stream.max(1e-9),
        r.ms_per_step / base.ms_per_step.max(1e-9)
    );
}
