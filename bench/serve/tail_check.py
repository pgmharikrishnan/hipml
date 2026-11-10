#!/usr/bin/env python3
"""End-to-end server check for the WP47 row-block tail path.

Starts a hipml server (no MTP, so every request takes the batched engine),
then asks the same 169-token prompt three ways:

  A  full prompt, no prefix cache           -> ordinary chunked prefill + batch
  W  full prompt, cache on, cache_cut=K=len-26 -> captures the prefix at K
  C  identical prompt, cache on             -> restores K, runs the 26-token
                                               tail through the row block

A and C must emit the same 20 greedy ids. Exit non-zero otherwise, and print
the row-block counters from /metrics.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "bench"))
import _paths  # noqa: E402

PORT = 8093
IDS = os.path.join(ROOT, "bench", "fixtures", "p2_math_169.ids")


def port_open(port):
    s = socket.socket()
    s.settimeout(0.5)
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


def wait_health(proc, timeout_s):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early ({proc.returncode})")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(1.0)
    raise RuntimeError("server not healthy")


def post(body):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/completion",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read().decode())


def metrics():
    out = {}
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=15) as r:
        for line in r.read().decode().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            name, _, value = line.partition(" ")
            if "tail_" in name:
                out[name] = float(value)
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Check the row-block tail path end to end. Paths come from "
                    "HIPML_GGUF / HIPML_EXE or the flags below.")
    ap.add_argument("--model", default=None,
                    help="target GGUF (default: $HIPML_GGUF)")
    ap.add_argument("--hipml", default=None,
                    help="hipml binary (default: $HIPML_EXE or target/release/hipml)")
    args = ap.parse_args()
    model = _paths.model(args.model)
    hipml = _paths.hipml_exe(args.hipml)

    if port_open(PORT):
        raise SystemExit(f"port {PORT} is already in use")
    ids = json.load(open(IDS, encoding="utf-8"))["ids"]
    cut = len(ids) - 26
    env = _paths.rocm_env()
    env["HIPML_ARCH"] = env.get("HIPML_ARCH", "gfx1201")
    cmd = [hipml, "serve", "--gguf", model, "--port", str(PORT),
           "--ctx", "4096", "--slots", "4", "--chunk", "256",
           "--prefix-entries", "4", "--prefix-mb", "512"]
    out_dir = _paths.out_dir("serve")
    log = open(os.path.join(out_dir, "tail_check.log"), "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        wait_health(proc, 420)
        base = {"prompt": ids, "max_tokens": 20, "ignore_eos": True}
        a = post(dict(base, cache_prompt=False))["tokens"]
        post({"prompt": ids, "max_tokens": 1, "ignore_eos": True,
              "cache_prompt": True, "cache_cut": cut})
        c = post(dict(base, cache_prompt=True, cache_cut=cut))["tokens"]
        m = metrics()
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=45)
        except Exception:
            proc.kill()
        log.close()

    print("A full prefill :", a)
    print("C row tail     :", c)
    print("metrics        :", json.dumps(m, indent=1))
    if a != c:
        print("MISMATCH: the row-block tail path differs from the ordinary prefill")
        return 1
    if m.get("hipml_tail_rows_total", 0) < 26:
        print("the row block did not serve the tail (check cache_cut / gather)")
        return 1
    print(f"tail-check: {len(a)}/{len(a)} ids identical, row block served the 26-row tail")
    return 0


if __name__ == "__main__":
    sys.exit(main())
