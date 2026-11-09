#!/usr/bin/env python3
"""Benchmark the hipml serving loop against llama-server on one machine.

One invocation runs one (engine, cell) so the caller can keep each GPU-lock
hold short. Cells:

  a  one request at a time, 5 conversations, 32K context per request
  b  8 concurrent requests sharing a ~2K-token system prompt, 32K per slot
  c  16 concurrent requests sharing the same system prompt, 8K per slot

Context: llama-server's `-c` is the total KV budget and `--parallel B` splits
it, so the script passes `-c (ctx * slots)` to give llama the same per-slot
context hipml gets from `--ctx`. 16 x 32K per slot does not fit on the 32 GiB
card (about 19.4 GiB of q8 KV + SSM on top of 17.4 GiB of weights), and 8 x 32K
per slot already pages (32.2 GiB measured), so b uses 16K per slot and c 8K,
on both engines. Cell a keeps 32K.

Every cell sends the same messages to both engines and uses greedy decoding.
The measured prompt always contains the shared system prompt, so the prefix
cache is exercised; a warm wave with a throwaway user turn fills it on all
slots first.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PORT_HIPML = 8093
PORT_LLAMA = 8092
MODEL = r"C:\models\Qwen3.8-27B-Q4_K_M.gguf"
MTP = r"C:\models\mtp-Qwen3.8-27B-Q4_0.gguf"
HIPML = os.path.join(ROOT, "target", "release", "hipml.exe")
LLAMA = r"C:\llama.cpp\rocm-gfx120x\llama-server.exe"

CELLS = {
    "a": {"slots": 1, "ctx": 32768, "requests": 5, "reps": 3},
    # 8 x 32K per slot does not fit beside the prefill engine and weights
    # (measured 32.2 GiB of 31.9 GiB and paging), so 16K per slot.
    "b": {"slots": 8, "ctx": 16384, "requests": 8, "reps": 3},
    "c": {"slots": 16, "ctx": 8192, "requests": 16, "reps": 3},
}

# ~2K tokens of shared system prompt. Repetition is deliberate: it is the
# workload a chat product actually sends, and it keeps the text deterministic.
_CLAUSE = (
    "Clause {i}. State the assumption first, name the file or interface under "
    "discussion, and keep the answer under two hundred words. Prefer a "
    "concrete example over a general claim. If the request is ambiguous, ask "
    "one clarifying question and proceed with the most conservative reading. "
    "Do not invent APIs, configuration keys, or measurements; mark anything "
    "unverified as unverified. When two systems are compared, give both the "
    "same workload, report the metric that was asked for, and state the "
    "context settings that produced it. A performance number includes the "
    "method, the observed variance, and the part of the system that was not "
    "measured. Write short paragraphs, avoid marketing language, and never "
    "pad an answer with a restatement of the question."
)
SYSTEM = "You are a senior engineer reviewing work in progress.\n\n" + "\n\n".join(
    _CLAUSE.format(i=i) for i in range(1, 17)
)

USERS = [
    "Explain why a continuous-batching server should retire a sequence as soon as it emits EOS.",
    "Describe how a prefix cache can reduce time to first token for a shared system prompt.",
    "Give one reason a batched decode step is bandwidth bound rather than compute bound.",
    "Show how to measure p95 time to first token for eight concurrent streaming requests.",
    "Outline the difference between chunked prefill and speculative decoding.",
    "Explain what happens to a batch slot when one request finishes before the others.",
    "Name two ways a quantized KV cache changes the memory budget of a server.",
    "Describe a safe way to reuse an SSM state across requests with the same system prompt.",
    "Explain why a request arriving mid-generation should join the batched step.",
    "Give the arithmetic for how many 32K q8 slots fit beside 16 GiB of weights.",
    "Describe how to compare aggregate token throughput across two servers fairly.",
    "Explain the tradeoff between speculate-when-alone and always-batch policies.",
    "Give one way to detect that a benchmark was silently paging VRAM to system memory.",
    "Describe how to verify that speculative decode emits the same ids as plain decode.",
    "Outline how to report the decode latency cost of an interleaved prefill tile.",
    "Explain why a per-request HTTP thread must not touch the GPU directly.",
]


def pct(values, p):
    if not values:
        return None
    xs = sorted(values)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


def median(values):
    if not values:
        return None
    xs = sorted(values)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def port_open(port):
    s = socket.socket()
    s.settimeout(0.5)
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


def wait_health(port, proc, timeout_s):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(1.0)
    raise RuntimeError(f"server not healthy after {timeout_s}s")


def start_hipml(cell, out_dir):
    if port_open(PORT_HIPML):
        raise RuntimeError(f"port {PORT_HIPML} is already in use")
    env = dict(os.environ)
    env["HIP_PATH"] = r"C:\rocm-sdk\_rocm_sdk_core"
    env["HIPML_ARCH"] = "gfx1201"
    env["PATH"] = r"C:\rocm-sdk\_rocm_sdk_core\bin;" + env.get("PATH", "")
    cmd = [HIPML, "serve", "--gguf", MODEL, "--mtp", MTP,
           "--port", str(PORT_HIPML),
           "--ctx", str(cell["ctx"]), "--slots", str(cell["slots"]),
           "--chunk", "256", "--drafts", "1",
           "--prefix-entries", "16", "--prefix-mb", "2048"]
    log = open(os.path.join(out_dir, "server.log"), "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    return proc, log, cmd, PORT_HIPML


def start_llama(cell, out_dir):
    if port_open(PORT_LLAMA):
        raise RuntimeError(f"port {PORT_LLAMA} is already in use")
    total_ctx = cell["ctx"] * cell["slots"]
    cmd = [LLAMA, "-m", MODEL, "--host", "127.0.0.1", "--port", str(PORT_LLAMA),
           "-ngl", "99", "-c", str(total_ctx), "--parallel", str(cell["slots"]),
           "-ctk", "q8_0", "-ctv", "q8_0", "--no-webui", "--no-cache-idle-slots"]
    log = open(os.path.join(out_dir, "server.log"), "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    return proc, log, cmd, PORT_LLAMA


def start_sampler(pid, out_path):
    ps = os.path.join(ROOT, "bench", "ps", "peak-vram.ps1")
    cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ps,
           "-TargetPid", str(pid), "-Out", out_path]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def read_peak(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            v = int(f.read().strip() or "0")
        return v / (1024.0 * 1024.0) if v > 0 else None
    except Exception:
        return None


def chat_once(port, user, max_tokens, timeout_s, cache=True):
    body = {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "ignore_eos": True,
        "stream": True,
        "cache_prompt": cache,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    first = None
    last = None
    chunks = 0
    final = None
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if "timings" in obj or "usage" in obj:
                final = obj
            try:
                delta = obj["choices"][0]["delta"]
            except (KeyError, IndexError):
                delta = {}
            # llama-server may stream a reasoning phase before content when the
            # chat template enables it; either is a generated token.
            content = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
            if content:
                chunks += 1
                t = time.perf_counter()
                if first is None:
                    first = t
                last = t
    if first is None:
        raise RuntimeError("streamed no content")
    if last is None:
        last = first
    final = final or {}
    usage = final.get("usage") or {}
    timings = final.get("timings") or {}
    n = usage.get("completion_tokens") or timings.get("predicted_n") or chunks
    n = int(n)
    return {
        "ttft_ms": (first - t0) * 1e3,
        "decode_ms": (last - first) * 1e3,
        "decode_tokens": max(n - 1, 0),
        "tokens": n,
        "chunks": chunks,
        "prompt_tokens": usage.get("prompt_tokens") or timings.get("prompt_n"),
        "server_tps": timings.get("predicted_per_second"),
        "prefix_hit_tokens": timings.get("prefix_hit_tokens"),
        "first": first,
        "last": last,
    }


def wave(port, users, max_tokens, timeout_s, barrier, cache=True):
    out = [None] * len(users)

    def one(i, user):
        barrier.wait()
        try:
            out[i] = chat_once(port, user, max_tokens, timeout_s, cache)
        except Exception as e:
            out[i] = {"error": str(e)}

    ts = [threading.Thread(target=one, args=(i, u)) for i, u in enumerate(users)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return out


def run_waves(port, users, cell, max_tokens, timeout_s, reps):
    results = []
    warm_user = "Reply with the single word ready."
    # One warm wave occupies every slot, so both engines cache the shared
    # system prefix before the measured waves.
    try:
        wave(port, [warm_user] * cell["slots"], 2, timeout_s,
             threading.Barrier(cell["slots"]))
    except Exception:
        pass
    # Cell a isolates the single-stream decoder, so it does not consult the
    # prefix cache; b and c are the shared-prefix workload.
    cache = cell["slots"] > 1
    for r in range(reps):
        if cell["slots"] == 1:
            rows = [chat_once(port, users[i % len(users)], max_tokens, timeout_s, cache)
                    for i in range(len(users))]
        else:
            rows = wave(port, users, max_tokens, timeout_s,
                        threading.Barrier(len(users)), cache)
        ok = [x for x in rows if x and "error" not in x]
        for x in ok:
            x["rep"] = r
        results.append({"rep": r, "rows": rows, "ok": ok})
    return results


def summarize(results, cell):
    rows = [x for w in results for x in w["ok"]]
    ttfts = [x["ttft_ms"] for x in rows]
    streams = [x["decode_tokens"] / (x["decode_ms"] / 1e3) if x["decode_ms"] > 0 else None
               for x in rows]
    streams = [s for s in streams if s is not None]
    aggs = []
    for w in results:
        ok = w["ok"]
        if not ok:
            continue
        if cell["slots"] == 1:
            # Sequential requests: aggregate == per-stream decode. Including
            # the cold prefills would report wall-clock goodput instead.
            aggs.append(median(streams))
            continue
        t_first = min(x["first"] for x in ok)
        t_last = max(x["last"] for x in ok)
        toks = sum(x["decode_tokens"] for x in ok)
        if t_last > t_first:
            aggs.append(toks / (t_last - t_first))
    return {
        "requests": len(rows),
        "errors": sum(1 for w in results for x in w["rows"] if x and "error" in x),
        "ttft_p50_ms": pct(ttfts, 50),
        "ttft_p95_ms": pct(ttfts, 95),
        "ttft_min_ms": min(ttfts) if ttfts else None,
        "ttft_max_ms": max(ttfts) if ttfts else None,
        "stream_decode_p50_tok_s": median(streams),
        "stream_decode_min_tok_s": min(streams) if streams else None,
        "aggregate_tok_s_p50": median(aggs),
        "aggregate_tok_s_best": max(aggs) if aggs else None,
        "prompt_tokens": median([x["prompt_tokens"] for x in rows if x.get("prompt_tokens")]),
        "prefix_hit_tokens_p50": median([x["prefix_hit_tokens"] or 0 for x in rows]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["hipml", "llama"])
    ap.add_argument("--cell", required=True, choices=sorted(CELLS))
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--reps", type=int, default=0, help="override the cell's wave count")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "bench", "serve", "out"))
    ap.add_argument("--load-timeout", type=int, default=420)
    ap.add_argument("--request-timeout", type=int, default=900)
    args = ap.parse_args()

    cell = dict(CELLS[args.cell])
    if args.reps:
        cell["reps"] = args.reps
    out_dir = os.path.join(args.out_dir, f"{args.engine}_{args.cell}")
    os.makedirs(out_dir, exist_ok=True)
    peak_path = os.path.join(out_dir, "peak.txt")
    if os.path.exists(peak_path):
        os.remove(peak_path)

    if args.engine == "hipml":
        proc, log, cmd, port = start_hipml(cell, out_dir)
    else:
        proc, log, cmd, port = start_llama(cell, out_dir)
    sampler = start_sampler(proc.pid, peak_path)
    started = time.time()
    try:
        wait_health(port, proc, args.load_timeout)
        results = run_waves(port, USERS[: cell["requests"]], cell, args.tokens,
                            args.request_timeout, cell["reps"])
        metrics = None
        if args.engine == "hipml":
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/metrics", timeout=15) as r:
                    text = r.read().decode("utf-8", errors="replace")
                metrics = {}
                for line in text.splitlines():
                    if line.startswith("#") or not line.strip():
                        continue
                    name, _, value = line.partition(" ")
                    if any(k in name for k in ("prefix", "interleave", "spec_", "mtp_", "prefill", "tail_")):
                        metrics[name] = float(value)
            except Exception:
                metrics = None
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=45)
        except Exception:
            proc.kill()
        log.close()
        try:
            sampler.wait(timeout=15)
        except subprocess.TimeoutExpired:
            sampler.kill()

    summary = summarize(results, cell)
    payload = {
        "engine": args.engine,
        "cell": args.cell,
        "config": cell,
        "tokens_per_request": args.tokens,
        "system_prompt_chars": len(SYSTEM),
        "summary": summary,
        "metrics": metrics,
        "peak_vram_mib": read_peak(peak_path),
        "command": cmd,
        "duration_s": time.time() - started,
        "waves": results,
    }
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    print(json.dumps(summary, indent=1))
    print(f"peak VRAM {payload['peak_vram_mib']} MiB; wrote {out_dir}/result.json")
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
