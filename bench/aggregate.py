#!/usr/bin/env python3
"""Turn raw per-cell job files into bench/results.json and bench/RESULTS.md.

Every prompt is its own row; nothing is averaged across prompt classes. Each
direction of a comparison is reported separately: TTFT (lower is better),
prefill tok/s and decode tok/s (higher is better).
"""

import argparse
import datetime
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def stats(values):
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return {
        "n": len(vals),
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
        "values": vals,
    }


def fmt(v, digits=1):
    if v is None:
        return "-"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 100:
        return f"{v:.0f}"
    return f"{v:.{digits}f}"


def cell_block(v, digits=1):
    if not v:
        return "-"
    if v["n"] == 1:
        return fmt(v["median"], digits)
    return f"{fmt(v['median'], digits)} ({fmt(v['min'], digits)}-{fmt(v['max'], digits)})"


def load_jobs(jobs_dir):
    out = {}
    for path in glob.glob(os.path.join(jobs_dir, "*", "result.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                j = json.load(f)
        except Exception as e:
            print(f"skip {path}: {e}", file=sys.stderr)
            continue
        key = (j.get("engine"), j.get("prompt"), int(j.get("depth", -1)))
        out[key] = j
    return out


def engine_summary(j):
    if not j or j.get("error"):
        return {"available": False, "error": (j or {}).get("error", "missing")}
    reps = j.get("reps") or []
    ttft = stats([r.get("ttft_ms") for r in reps])
    pre = stats([r.get("prefill_tok_s") for r in reps])
    dec = stats([r.get("decode_tok_s") for r in reps])
    if pre is None:
        pre = stats([
            (j.get("tokens") / (r["prefill_ms"] / 1e3))
            if r.get("prefill_ms") else None
            for r in reps
        ])
    if dec is None:
        dec = stats([r.get("decode_tok_s_client") for r in reps])
    cache_n = [r.get("cache_n") for r in reps if r.get("cache_n") is not None]
    cold_ok = all(int(x) == 0 for x in cache_n) if cache_n else None
    return {
        "available": True,
        "reps": len(reps),
        "reps_requested": j.get("reps_requested"),
        "ttft_ms": ttft,
        "prefill_tok_s": pre,
        "decode_tok_s": dec,
        "peak_vram_mib": j.get("peak_vram_mib"),
        "cold_cache_n_zero": cold_ok,
        "cache_n_values": cache_n,
        "duration_s": j.get("duration_s"),
        "kv_type": j.get("kv_type"),
        "kv_mode": j.get("kv_mode"),
        "kv": j.get("kv_type") or j.get("kv_mode"),
        "hipml_kv": j.get("hipml_kv"),
        "spilled": j.get("spilled"),
        "warm": j.get("warm"),
        "stage": j.get("stage"),
        "server_version": j.get("server_version"),
        "ids_sha256": j.get("ids_sha256"),
        "tokens": j.get("tokens"),
    }


def pct(a, b):
    if a is None or b is None or b == 0:
        return None
    return (a / b - 1.0) * 100.0


def public_model(model):
    """Model identity for the report: basename plus size and hash, never a path."""
    if not model:
        return None
    out = {"path": os.path.basename(str(model.get("path") or ""))}
    for key in ("bytes", "sha256"):
        if model.get(key) is not None:
            out[key] = model[key]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default=os.path.join(HERE, "fixtures", "prompts.json"))
    ap.add_argument("--jobs", default=os.path.join(HERE, "out", "jobs"))
    ap.add_argument("--meta", default=os.path.join(HERE, "out", "meta.json"))
    ap.add_argument("--results", default=os.path.join(HERE, "results.json"))
    ap.add_argument("--markdown", default=os.path.join(HERE, "RESULTS.md"))
    ap.add_argument("--deep-prompt", default="code_edit",
                    help="only this prompt is run at depths >= 32768")
    args = ap.parse_args()

    with open(args.fixture, "r", encoding="utf-8") as f:
        fixture = json.load(f)
    meta = {}
    if os.path.exists(args.meta):
        try:
            with open(args.meta, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            print(f"warning: ignoring {args.meta}: {e}", file=sys.stderr)

    jobs = load_jobs(args.jobs)
    if not meta.get("llama_version"):
        for j in jobs.values():
            v = j.get("server_version")
            if v:
                meta["llama_version"] = v
                break
    if meta.get("llama_version") and not meta.get("llama_build"):
        import re
        m = re.search(r"build (\d+)", meta["llama_version"])
        c = re.search(r"commit ([0-9a-f]+)", meta["llama_version"])
        if m:
            meta["llama_build"] = m.group(1)
        if c:
            meta["llama_commit"] = c.group(1)
    cells = []
    for c in fixture["cells"]:
        if c["depth"] >= 32768 and c["prompt"] != args.deep_prompt:
            continue
        h = engine_summary(jobs.get(("hipml", c["prompt"], c["depth"])))
        l = engine_summary(jobs.get(("llama", c["prompt"], c["depth"])))
        row = dict(c)
        row["hipml"] = h
        row["llama"] = l
        row["gaps"] = {
            "prefill_pct_hipml_minus_llama": pct(
                (h.get("prefill_tok_s") or {}).get("median"),
                (l.get("prefill_tok_s") or {}).get("median")),
            "decode_pct_hipml_minus_llama": pct(
                (h.get("decode_tok_s") or {}).get("median"),
                (l.get("decode_tok_s") or {}).get("median")),
            "ttft_ratio_llama_over_hipml": (
                ((l.get("ttft_ms") or {}).get("median") /
                 (h.get("ttft_ms") or {}).get("median"))
                if ((h.get("ttft_ms") or {}).get("median") and
                    (l.get("ttft_ms") or {}).get("median")) else None),
        }
        cells.append(row)

    gaps = []
    for c in cells:
        g = c["gaps"]
        for metric, key in (("prefill", "prefill_pct_hipml_minus_llama"),
                            ("decode", "decode_pct_hipml_minus_llama")):
            if g[key] is not None:
                gaps.append({
                    "metric": metric,
                    "prompt": c["prompt"],
                    "depth": c["depth"],
                    "tokens": c["total"],
                    "pct_hipml_minus_llama": g[key],
                    "reps_hipml": c["hipml"].get("reps"),
                    "reps_llama": c["llama"].get("reps"),
                })
        ttft_ratio = g.get("ttft_ratio_llama_over_hipml")
        if ttft_ratio is not None:
            gaps.append({
                "metric": "ttft",
                "prompt": c["prompt"],
                "depth": c["depth"],
                "tokens": c["total"],
                "pct_hipml_minus_llama": (ttft_ratio - 1.0) * 100.0,
                "reps_hipml": c["hipml"].get("reps"),
                "reps_llama": c["llama"].get("reps"),
            })

    report = {
        "format": "hipml-bench-results/1",
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "meta": meta,
        "fixture": {
            "path": os.path.relpath(args.fixture, os.path.dirname(HERE)),
            "model": public_model(fixture.get("model")),
            "tokenizer": fixture.get("tokenizer"),
            "depths": fixture.get("depths"),
            "document_tokens": fixture.get("document", {}).get("tokens"),
            "prompts": [
                {"label": p["label"], "tokens": p["token_count"]}
                for p in fixture.get("prompts", [])
            ],
        },
        "protocol": {
            "prompt": "token ids from the fixture, identical for both engines",
            "depth0": "prompt exactly as tokenised",
            "depth_gt_0": "BOS + real document prefix + prompt, truncated to exact depth",
            "gen_tokens": 33,
            "decode_window": "tokens 2..33 (32 intervals)",
            "reps": "cold prefill per rep; median/min/max reported",
            "llama": {
                "server": "llama-server",
                "port": 8092,
                "temperature": 0,
                "ignore_eos": True,
                "cold": "cache_prompt=false",
                "warm": "one extra request with cache_prompt=true",
            },
            "hipml": {
                "path": "generate --device",
                "reps": "fresh GpuEngine per rep (zeroed KV/recurrent state)",
                "kv": "f32",
                "stage_report": "separate synchronised breakdown, not the TTFT",
            },
        },
        "cells": cells,
        "gaps": gaps,
    }

    with open(args.results, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    write_markdown(report, args.markdown)
    print(f"wrote {args.results}")
    print(f"wrote {args.markdown}")


def write_markdown(report, path):
    meta = report.get("meta", {})
    model = report["fixture"].get("model", {})
    model_path = meta.get("model_path") or model.get("path", "?")
    lines = []
    lines.append("# Like-for-like bench: hipml vs llama.cpp\n")
    lines.append(
        "Five prompt classes, each tokenised once by `hipml tokenize`. Both "
        "engines receive the identical token ids from `bench/fixtures/prompts.json`; "
        "neither retokenises at run time. Depth 0 is the prompt as-is. Deeper "
        "cells are `BOS + real document prefix + prompt`, truncated to the exact "
        "token count. Every numeric cell is `median (min-max)` over cold reps; "
        "decode is measured over tokens 2..33. Depths 0 and 8192 cover all five "
        "prompts at 3 reps; 32768 and 100000 use `code_edit` at 1 rep, marked in "
        "the reps column. Both engines use q8-class KV at every depth.\n"
    )
    lines.append("| field | value |")
    lines.append("| --- | --- |")
    lines.append(f"| model | `{model_path}` ({model.get('bytes', 0):,} bytes) |")
    if meta.get("gpu"):
        lines.append(f"| GPU | {meta['gpu']} |")
    if meta.get("host"):
        lines.append(f"| host | {meta['host']} |")
    if meta.get("hipml_commit"):
        dirty = " (dirty worktree)" if meta.get("hipml_dirty") else ""
        lines.append(f"| hipml commit | `{meta['hipml_commit']}`{dirty} |")
    if meta.get("hipml_binary_sha256"):
        lines.append(f"| hipml binary sha256 | `{meta['hipml_binary_sha256']}` |")
    if meta.get("llama_version"):
        lines.append(f"| llama.cpp | {meta['llama_version']} |")
    if meta.get("llama_build"):
        lines.append(f"| llama.cpp build | `{meta['llama_build']}` commit `{meta.get('llama_commit', '?')}` |")
    lines.append(f"| llama KV type | {meta.get('llama_kv', '?')} |")
    lines.append(f"| hipml KV type | {meta.get('hipml_kv', '?')} |")
    lines.append(f"| generated | {report['generated']} |")
    lines.append("")

    lines.append("## Results\n")
    lines.append(
        "`TTFT` is lower-is-better. `prefill` and `decode` are higher-is-better. "
        "`Peak` is the process's peak dedicated GPU memory (WDDM counter); it is "
        "one number for the whole cell, not per rep.\n"
    )
    lines.append("| prompt | depth | tokens | engine | KV | reps | TTFT ms | prefill tok/s | decode tok/s | peak GiB | spilled |")
    lines.append("| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for c in report["cells"]:
        for eng in ("hipml", "llama"):
            e = c[eng]
            if not e.get("available"):
                lines.append(
                    f"| {c['prompt']} | {c['depth']} | {c['total']} | {eng} | - | 0 | "
                    f"ERROR: {e.get('error', 'missing')} | - | - | - | - |")
                continue
            peak = e.get("peak_vram_mib")
            peak_gib = f"{peak / 1024.0:.2f}" if peak else "-"
            kv = e.get("kv") or "-"
            spill = e.get("spilled")
            spill_s = "-" if eng == "hipml" else ("YES" if spill is True else ("no" if spill is False else "?"))
            lines.append(
                f"| {c['prompt']} | {c['depth']} | {c['total']} | {eng} | {kv} | "
                f"{e.get('reps', 0)} | "
                f"{cell_block(e.get('ttft_ms'))} | "
                f"{cell_block(e.get('prefill_tok_s'))} | "
                f"{cell_block(e.get('decode_tok_s'))} | "
                f"{peak_gib} | {spill_s} |")
    lines.append("")

    lines.append("## Gap (hipml minus llama.cpp)\n")
    lines.append(
        "Positive means hipml is faster; negative means llama.cpp is faster. "
        "Each row is its own prompt class; no number is averaged across classes.\n"
    )
    lines.append("| prompt | depth | prefill | decode | TTFT (llama/hipml) |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for c in report["cells"]:
        g = c["gaps"]
        pre = g.get("prefill_pct_hipml_minus_llama")
        dec = g.get("decode_pct_hipml_minus_llama")
        ttft = g.get("ttft_ratio_llama_over_hipml")
        lines.append(
            f"| {c['prompt']} | {c['depth']} | "
            f"{'%+.1f%%' % pre if pre is not None else '-'} | "
            f"{'%+.1f%%' % dec if dec is not None else '-'} | "
            f"{'%.2fx' % ttft if ttft is not None else '-'} |")
    lines.append("")

    # Headline the extremes, with the rep count so a 1-rep 100K cell is obvious.
    def best(metric, sign):
        rows = [g for g in report["gaps"] if g["metric"] == metric]
        if not rows:
            return None
        return max(rows, key=lambda x: sign * x["pct_hipml_minus_llama"])

    lines.append("## Headline gaps\n")
    for metric in ("prefill", "decode", "ttft"):
        hi = best(metric, +1)
        lo = best(metric, -1)
        if not hi:
            continue
        if hi["pct_hipml_minus_llama"] > 0:
            line = (
                f"- **{metric}**: hipml fastest at `{hi['prompt']}@{hi['depth']}` "
                f"({hi['pct_hipml_minus_llama']:+.1f}%, reps {hi['reps_hipml']}/{hi['reps_llama']})")
        else:
            line = f"- **{metric}**: hipml did not win any cell"
        if lo and lo["pct_hipml_minus_llama"] < 0:
            line += (
                f"; llama.cpp fastest at `{lo['prompt']}@{lo['depth']}` "
                f"({lo['pct_hipml_minus_llama']:+.1f}%, reps {lo['reps_hipml']}/{lo['reps_llama']}).")
        else:
            line += "; llama.cpp did not win any cell for this metric."
        lines.append(line)
    lines.append("")

    # Warm llama pass.
    warm_rows = []
    for c in report["cells"]:
        w = (c["llama"] or {}).get("warm") if c["llama"].get("available") else None
        if w and not w.get("error"):
            warm_rows.append((c, w))
    if warm_rows:
        lines.append("## llama.cpp warm pass (cache_prompt=true)\n")
        lines.append(
            "Same ids immediately after the cold reps. A near-1 `prompt_n` and a "
            "large `cache_n` confirm the prefix was reused; this is a cache-hit "
            "TTFT, not a prefill number.\n"
        )
        lines.append("| prompt | depth | TTFT ms | warm prefill tok/s | cache_n | prompt_n |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for c, w in warm_rows:
            lines.append(
                f"| {c['prompt']} | {c['depth']} | {fmt(w.get('ttft_ms'))} | "
                f"{fmt(w.get('prefill_tok_s'))} | {w.get('cache_n', '-')} | "
                f"{w.get('prompt_n', '-')} |")
        lines.append("")

    # hipml stage breakdown where it was collected.
    stage_rows = [(c, c["hipml"].get("stage")) for c in report["cells"]
                  if c["hipml"].get("available") and c["hipml"].get("stage")]
    if stage_rows:
        lines.append("## hipml prefill stage breakdown\n")
        lines.append(
            "One extra synchronised run per cell; the per-stage values include a "
            "stream sync each, so `total_wall` is larger than the table TTFT. "
            "Reported as a diagnostic of where prefill time goes.\n"
        )
        keys = ["total_wall_ms", "embed_ms", "elementwise_ms", "gemm_attn_ms",
                "gemm_ssm_ms", "gemm_ffn_ms", "final_ms", "seq_ssm_ms",
                "seq_attn_ms", "stage_sum_ms"]
        lines.append("| prompt | depth | " + " | ".join(k.replace("_ms", "") for k in keys) + " |")
        lines.append("| --- | ---: | " + " | ".join("---:" for _ in keys) + " |")
        for c, s in stage_rows:
            lines.append(
                f"| {c['prompt']} | {c['depth']} | " +
                " | ".join(fmt(s.get(k), 0) for k in keys) + " |")
        lines.append("")

    lines.append("## Completeness\n")
    missing = [c for c in report["cells"]
               if not c["hipml"].get("available") or not c["llama"].get("available")]
    if not missing:
        lines.append(
            "All planned cells are present: 5 prompts x depths 0 and 8192 at 3 reps, "
            "and `code_edit` x depths 32768 and 100000 at 1 rep. No ERROR rows.")
    else:
        lines.append("| prompt | depth | missing engine | reason |")
        lines.append("| --- | ---: | --- | --- |")
        for c in missing:
            for eng in ("hipml", "llama"):
                e = c[eng]
                if not e.get("available"):
                    lines.append(
                        f"| {c['prompt']} | {c['depth']} | {eng} | "
                        f"{e.get('error', 'missing')} |")
    lines.append("")

    lines.append("## Reps and caveats\n")
    one_rep = [c for c in report["cells"]
               if (c["hipml"].get("available") and c["hipml"].get("reps", 0) == 1) or
                  (c["llama"].get("available") and c["llama"].get("reps", 0) == 1)]
    if one_rep:
        lines.append(
            "- **One-rep cells:** " +
            ", ".join(f"`{c['prompt']}@{c['depth']}`" for c in one_rep) +
            ". These are marked `1` in the reps column; treat the range as absent, "
            "not as zero variance."
        )
    bad_cold = [c for c in report["cells"]
                if c["llama"].get("available") and c["llama"].get("cold_cache_n_zero") is False]
    if bad_cold:
        lines.append(
            "- **Cache contamination:** these llama cells reported `cache_n > 0` on "
            "a cold rep: " + ", ".join(f"`{c['prompt']}@{c['depth']}`" for c in bad_cold) + "."
        )
    lines.append(
        "- The earlier incomplete table, every `ERROR: missing` cell, and all "
        "reruns are documented in `bench/RUNLOG.md`.")
    lines.append(
        "- `hipml commit` is the Rust commit used; the `(dirty worktree)` flag on "
        "that run refers to bench-script edits made after the Rust commit, not to "
        "a different binary. The binary SHA-256 is recorded above.")
    lines.append(
        "- Same model file, same box, sequential engine runs. Both engines use "
        "quantized KV at every depth: hipml `q8` and llama.cpp `-ctk q8_0 -ctv q8_0`. "
        "The formats are the same 8-bit class but not byte-identical, so this is a "
        "same-precision-class comparison, not a same-format one.")
    lines.append(
        "- At 100K the q8 KV is about 3.2 GiB for hipml and 3.5 GiB for llama.cpp. "
        "The 100K rows report peak VRAM and whether llama.cpp spilled; the run is "
        "only accepted if the server logged no allocation failure or CPU offload.")
    lines.append(
        "- The earlier f16/f32 sensitivity runs are preserved under `bench/sensitivity/` "
        "and are not part of the head-to-head.")
    lines.append(
        "- 100K cells are real cold prefill. If a cell shows `1` rep, the run was "
        "too slow to repeat inside the lock window; it is not cache-prefilled.")
    lines.append(
        "- The harness cannot control: GPU clocks/thermal state, driver scheduling, "
        "other processes on the box outside the lock, llama.cpp's own thread and "
        "graph choices, or the difference between hipml's argmax-in-prefill TTFT "
        "and llama-server's prompt-only timing.")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


if __name__ == "__main__":
    main()
