#!/usr/bin/env python3
"""Run one (engine, prompt, depth) cell and write its raw measurements.

This runs on the GPU host, inside the GPU-lock window. It does exactly one
cell so `bench/run.sh` can chunk the slow 100K work and keep every lock hold
short. Raw output goes to bench/out/jobs/<engine>_<prompt>_<depth>.json; the
aggregator turns those files into results.json and RESULTS.md.

hipml: one `generate --device --reps N` process. The engine builds a fresh
GpuEngine per rep, so each rep has zeroed KV and zeroed recurrent state: a
cold prefill without a 17 GB reload.

llama.cpp: llama-server on 8092. N cold requests with cache_prompt=false,
then one warm request with cache_prompt=true. Only 8092 is ever touched.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import _paths

HERE = os.path.dirname(os.path.abspath(__file__))
GEN_TOKENS = 33          # tokens 1..33; decode rate is over 2..33
DEFAULT_PORT = 8092


def now():
    return time.time()


def assemble(fixture, cell):
    p = next(x for x in fixture["prompts"] if x["label"] == cell["prompt"])
    if cell["depth"] == 0:
        return list(p["raw_ids"])
    tok = fixture["tokenizer"]
    bos = [tok["bos_id"]] if tok["add_bos"] else []
    doc = fixture["document"]["ids"]
    return bos + doc[: cell["prefix_len"]] + p["ids"]


def wait_health(port, proc, timeout_s):
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return
        except Exception as e:  # connection refused while loading
            last = e
        time.sleep(1.0)
    raise RuntimeError(f"llama-server not healthy after {timeout_s}s: {last}")


def port_open(port):
    s = socket.socket()
    s.settimeout(0.5)
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


def start_peak_sampler(pid, out_path):
    ps = os.path.join(HERE, "ps", "peak-vram.ps1")
    cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ps,
           "-TargetPid", str(pid), "-Out", out_path]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def read_peak(out_path):
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            v = int(f.read().strip() or "0")
        return v / (1024.0 * 1024.0) if v > 0 else None
    except Exception:
        return None


def parse_hipml_stdout(text):
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and '"engine":"hipml"' in line:
            return json.loads(line)
    raise RuntimeError("no hipml JSON line in stdout:\n" + text[-2000:])


def run_hipml(args, fixture, cell, ids, out):
    ids_path = os.path.join(args.tmp, f"ids_{cell['prompt']}_{cell['depth']}.json")
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump({"ids": ids}, f, separators=(",", ":"))

    cmd = [args.hipml, "generate", "--device", "--gguf", args.model,
           "--ids-file", ids_path, "--tokens", str(GEN_TOKENS),
           "--ctx", str(len(ids) + 64), "--reps", str(args.reps), "--json",
           "--kv", args.hipml_kv]
    if args.stage_report:
        cmd.append("--stage-report")

    started = now()
    peak_path = os.path.join(out, "peak.txt")
    if os.path.exists(peak_path):
        os.remove(peak_path)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)
    sampler = start_peak_sampler(proc.pid, peak_path)
    try:
        stdout, stderr = proc.communicate(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        raise RuntimeError(f"hipml timed out after {args.timeout}s")
    duration = now() - started
    try:
        sampler.wait(timeout=15)
    except subprocess.TimeoutExpired:
        sampler.kill()

    if proc.returncode != 0:
        raise RuntimeError(f"hipml exit {proc.returncode}\nstderr:\n{stderr[-4000:]}")

    payload = parse_hipml_stdout(stdout)
    result = {
        "engine": "hipml",
        "prompt": cell["prompt"],
        "depth": cell["depth"],
        "tokens": len(ids),
        "reps_requested": args.reps,
        "reps": payload.get("reps", []),
        "stage": payload.get("stage"),
        "prompt_tokens_reported": payload.get("prompt_tokens"),
        "n_ctx": payload.get("n_ctx"),
        "kv_mode": payload.get("kv_mode"),
        "hipml_kv": args.hipml_kv,
        "weights_bytes": payload.get("weights_bytes"),
        "kv_bytes": payload.get("kv_bytes"),
        "model_load_s": payload.get("model_load_s"),
        "peak_vram_mib": read_peak(peak_path),
        "command": _paths.public_argv(cmd, [
            (args.hipml, "<HIPML_EXE>"),
            (args.model, "<HIPML_GGUF>"),
            (ids_path, "<IDS_FILE>"),
        ]),
        "duration_s": duration,
        "started": started,
        "stderr_tail": stderr[-2000:],
    }
    return result


def llama_request(port, ids, cache_prompt, timeout_s):
    payload = {
        "prompt": ids,
        "n_predict": GEN_TOKENS,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "min_p": 0.0,
        "repeat_penalty": 1.0,
        "ignore_eos": True,
        "stream": True,
        "cache_prompt": bool(cache_prompt),
        "return_tokens": True,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    first = None
    last = None
    n = 0
    final = None
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if "timings" in obj:
                final = obj
            toks = obj.get("tokens") or []
            if toks:
                n += len(toks)
                t = time.perf_counter()
                if first is None:
                    first = t
                if n >= GEN_TOKENS and last is None:
                    last = t
            if obj.get("stop"):
                break
    if first is None:
        raise RuntimeError("llama-server streamed no tokens")
    if last is None:
        last = first
    timings = (final or {}).get("timings", {})
    decode_tokens = max(n - 1, 0)
    client_decode_ms = (last - first) * 1e3
    client_tps = (decode_tokens / (client_decode_ms / 1e3)) if client_decode_ms > 0 else None
    # Prefer the server's own decode clock. It times the same generation loop
    # hipml times internally; the client arrival clock also pays HTTP/TCP
    # buffering, so keep it only as a cross-check.
    server_tps = timings.get("predicted_per_second")
    server_ms = timings.get("predicted_ms")
    return {
        "ttft_ms": (first - t0) * 1e3,
        "prefill_ms": timings.get("prompt_ms"),
        "prefill_tok_s": timings.get("prompt_per_second"),
        "decode_ms": server_ms if server_ms is not None else client_decode_ms,
        "decode_tok_s": server_tps if server_tps is not None else client_tps,
        "decode_tok_s_client": client_tps,
        "client_decode_ms": client_decode_ms,
        "decode_tokens": decode_tokens,
        "streamed_tokens": n,
        "prompt_n": timings.get("prompt_n"),
        "cache_n": timings.get("cache_n"),
        "predicted_n": timings.get("predicted_n"),
        "predicted_ms": timings.get("predicted_ms"),
        "server_predicted_per_second": timings.get("predicted_per_second"),
        "stop_type": (final or {}).get("stop_type"),
        "truncated": (final or {}).get("truncated"),
        "wall_ms": (time.perf_counter() - t0) * 1e3,
    }


def llama_version(server):
    try:
        out = subprocess.check_output([server, "--version"], text=True,
                                      stderr=subprocess.STDOUT, timeout=20)
        return out.strip().splitlines()[0] if out.strip() else ""
    except Exception:
        return ""


def start_server(args, ctx, log_path):
    cmd = [args.llama_server, "-m", args.model,
           "--host", "127.0.0.1", "--port", str(args.port),
           "-ngl", "99", "-c", str(ctx), "--parallel", "1",
           "-ctk", args.llama_kv, "-ctv", args.llama_kv,
           "--no-webui", "--no-cache-idle-slots"]
    log = open(log_path, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    return proc, log, cmd


def run_llama(args, fixture, cell, ids, out):
    if port_open(args.port):
        raise RuntimeError(
            f"port {args.port} is already in use; refusing to touch it"
        )
    ctx = max(len(ids) + 64, 512)
    log_path = os.path.join(out, "server.log")
    peak_path = os.path.join(out, "peak.txt")
    if os.path.exists(peak_path):
        os.remove(peak_path)
    started = now()
    proc, log, cmd = start_server(args, ctx, log_path)
    sampler = start_peak_sampler(proc.pid, peak_path)
    try:
        wait_health(args.port, proc, args.load_timeout)
        reps = []
        for r in range(args.reps):
            rep = llama_request(args.port, ids, False, args.request_timeout)
            rep["rep"] = r
            reps.append(rep)
        try:
            warm = llama_request(args.port, ids, True, args.request_timeout)
        except Exception as e:
            warm = {"error": str(e)}
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
        log.close()
    duration = now() - started
    try:
        sampler.wait(timeout=15)
    except subprocess.TimeoutExpired:
        sampler.kill()

    peak = read_peak(peak_path)
    spilled = None
    try:
        log_text = open(log_path, "r", encoding="utf-8", errors="replace").read().lower()
        if ("failed to allocate" in log_text or "out of memory" in log_text
                or "to cpu" in log_text or "cpu buffer" in log_text):
            spilled = True
        elif peak is not None and peak < 24 * 1024:
            # All weights and the q8 KV fit; llama-server would have logged an
            # allocation failure or a CPU offload before running this low.
            spilled = False
    except Exception:
        pass

    result = {
        "engine": "llama",
        "prompt": cell["prompt"],
        "depth": cell["depth"],
        "tokens": len(ids),
        "reps_requested": args.reps,
        "reps": reps,
        "warm": warm,
        "ctx": ctx,
        "kv_type": args.llama_kv,
        "spilled": spilled,
        "server_version": llama_version(args.llama_server),
        "peak_vram_mib": peak,
        "command": _paths.public_argv(cmd, [
            (args.llama_server, "<LLAMA_SERVER>"),
            (args.model, "<HIPML_GGUF>"),
            (log_path, "<SERVER_LOG>"),
        ]),
        "duration_s": duration,
        "started": started,
    }
    return result


def main():
    ap = argparse.ArgumentParser(
        description="Run one (engine, prompt, depth) bench cell. The paths come "
                    "from the environment so a run is reproducible anywhere.")
    ap.add_argument("--engine", required=True, choices=["hipml", "llama"])
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--depth", required=True, type=int)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--fixture", default=os.path.join(HERE, "fixtures", "prompts.json"))
    ap.add_argument("--model", default=os.environ.get("HIPML_GGUF"),
                    help="target GGUF (default: $HIPML_GGUF)")
    ap.add_argument("--hipml", default=None,
                    help="hipml binary (default: $HIPML_EXE or "
                         "target/release/hipml[.exe])")
    ap.add_argument("--llama-server", default=os.environ.get("LLAMA_SERVER"),
                    help="llama.cpp server binary (default: $LLAMA_SERVER; "
                         "required only for --engine llama)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--llama-kv", default="q8_0", choices=["f32", "f16", "q8_0"])
    ap.add_argument("--hipml-kv", default="q8", choices=["f32", "q8", "q4"])
    ap.add_argument("--stage-report", action="store_true")
    ap.add_argument("--timeout", type=int, default=1080,
                    help="hipml per-cell wall timeout (keeps the lock under 20 min)")
    ap.add_argument("--load-timeout", type=int, default=300)
    ap.add_argument("--request-timeout", type=int, default=900)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "out", "jobs"))
    ap.add_argument("--tmp", default=os.path.join(HERE, "out", "tmp"))
    args = ap.parse_args()

    if not args.model:
        sys.exit("error: no target model; pass --model or set HIPML_GGUF")
    args.hipml = _paths.hipml_exe(args.hipml)
    if args.engine == "llama" and not args.llama_server:
        sys.exit("error: engine llama needs --llama-server or LLAMA_SERVER")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.tmp, exist_ok=True)
    out = os.path.join(args.out_dir, f"{args.engine}_{args.prompt}_{args.depth}")
    os.makedirs(out, exist_ok=True)

    with open(args.fixture, "r", encoding="utf-8") as f:
        fixture = json.load(f)
    cell = next((c for c in fixture["cells"]
                 if c["prompt"] == args.prompt and c["depth"] == args.depth), None)
    if cell is None:
        sys.exit(f"no cell {args.prompt}@{args.depth} in fixture")
    ids = assemble(fixture, cell)
    if len(ids) != cell["total"]:
        sys.exit(f"assembled {len(ids)} ids, fixture says {cell['total']}")

    started = now()
    try:
        if args.engine == "hipml":
            res = run_hipml(args, fixture, cell, ids, out)
        else:
            res = run_llama(args, fixture, cell, ids, out)
    except Exception as e:
        err = {
            "engine": args.engine,
            "prompt": args.prompt,
            "depth": args.depth,
            "tokens": len(ids),
            "error": str(e),
            "started": started,
            "duration_s": now() - started,
        }
        with open(os.path.join(out, "result.json"), "w", encoding="utf-8") as f:
            json.dump(err, f, indent=2)
        print(f"FAIL {args.engine} {args.prompt}@{args.depth}: {e}", file=sys.stderr)
        sys.exit(1)

    res["ids_sha256"] = cell["ids_sha256"]
    with open(os.path.join(out, "result.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"OK {args.engine} {args.prompt}@{args.depth} in {res['duration_s']:.1f}s")


if __name__ == "__main__":
    main()
