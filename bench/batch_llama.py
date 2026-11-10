#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Measure llama.cpp aggregate decode throughput at equal B.

Starts llama-server on 8092 with `-np B` and q8_0 K/V, then issues B identical
greedy requests concurrently and reports the aggregate token rate. Only 8092 is
ever touched, and the server is terminated before returning.

  python batch_llama.py --np 4 --predict 32
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

import _paths

PORT = 8092
PROMPT = [1] + list(range(100, 116))


def wait_health(proc, timeout_s):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(1.0)
    raise RuntimeError("server not healthy")


def one_request(predict, out, idx):
    payload = {
        "prompt": PROMPT,
        "n_predict": predict,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "min_p": 0.0,
        "repeat_penalty": 1.0,
        "ignore_eos": True,
        "stream": True,
        "cache_prompt": False,
        "return_tokens": True,
        "id_slot": idx,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/completion",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    final = None
    first = last = None
    n = 0
    with urllib.request.urlopen(req, timeout=600) as resp:
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
                last = t
            if obj.get("stop"):
                break
    out[idx] = {
        "n": n,
        "first": first,
        "last": last,
        "timings": (final or {}).get("timings", {}),
        "ids": (final or {}).get("tokens", []),
    }


def port_open():
    s = socket.socket()
    s.settimeout(0.5)
    try:
        return s.connect_ex(("127.0.0.1", PORT)) == 0
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser(
        description="Measure llama.cpp aggregate decode throughput at equal B. "
                    "The model and server paths come from the environment.")
    ap.add_argument("--np", type=int, default=4)
    ap.add_argument("--predict", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--model", default=os.environ.get("HIPML_GGUF"),
                    help="target GGUF (default: $HIPML_GGUF)")
    ap.add_argument("--llama-server", default=os.environ.get("LLAMA_SERVER"),
                    help="llama.cpp server binary (default: $LLAMA_SERVER)")
    args = ap.parse_args()
    model = _paths.model(args.model)
    server = _paths.llama_server(args.llama_server)
    if port_open():
        print(f"port {PORT} already in use; refusing to touch it", file=sys.stderr)
        return 2
    cmd = [server, "-m", model, "--host", "127.0.0.1", "--port", str(PORT),
           "-ngl", "99", "-c", str(args.ctx), "--parallel", str(args.np),
           "-ctk", "q8_0", "-ctv", "q8_0",
           "--no-webui", "--no-cache-idle-slots"]
    log = open("llama-batch.log", "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_health(proc, 300)
        # Warm: one token each, then drain.
        warm = {}
        ts = [threading.Thread(target=one_request, args=(2, warm, i))
              for i in range(args.np)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()

        best = None
        for _ in range(args.reps):
            out = {}
            t0 = time.perf_counter()
            ts = [threading.Thread(target=one_request, args=(args.predict, out, i))
                  for i in range(args.np)]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
            wall = time.perf_counter() - t0
            server = 0.0
            tokens = 0
            for r in out.values():
                tm = r["timings"]
                dec = max(tm.get("predicted_n", 1) - 1, 1)
                ms = tm.get("predicted_ms") or 1.0
                server += dec / (ms / 1e3)
                tokens += dec
            first = min(r["first"] for r in out.values() if r["first"])
            last = max(r["last"] for r in out.values() if r["last"])
            wall_decode = tokens / (last - first) if last > first else 0.0
            row = (server, wall_decode, tokens, wall)
            if best is None or row[0] > best[0]:
                best = row
        server, wall_decode, tokens, wall = best
        print(
            f"llama.cpp -np {args.np} q8 KV: aggregate {server:.2f} tok/s "
            f"(server clocks), {wall_decode:.2f} tok/s (client overlap), "
            f"{tokens} tokens, {server/args.np:.2f} tok/s per stream"
        )
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
        log.close()


if __name__ == "__main__":
    sys.exit(main())
