#!/usr/bin/env python3
"""One llama-server request on the synthetic 100K prompt; print the first token.

The synthetic prompt matches `hipml gpu-bench --prefill N` exactly:
ids `100 + (i % 1000)` for i in 0..N. Uses 8092 only.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench"))
import _paths  # noqa: E402

PORT = 8092


def main():
    ap = argparse.ArgumentParser(
        description="One llama-server request on the synthetic 100K prompt. "
                    "Paths come from HIPML_GGUF / LLAMA_SERVER.")
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--ctx", type=int, default=0)
    ap.add_argument("--model", default=None,
                    help="target GGUF (default: $HIPML_GGUF)")
    ap.add_argument("--llama-server", default=None,
                    help="llama.cpp server binary (default: $LLAMA_SERVER)")
    args = ap.parse_args()
    model = _paths.model(args.model)
    server = _paths.llama_server(args.llama_server)
    ctx = args.ctx or (args.n + 64)
    ids = [100 + (i % 1000) for i in range(args.n)]
    proc = subprocess.Popen(
        [server, "-m", model, "--host", "127.0.0.1", "--port", str(PORT),
         "-ngl", "99", "-c", str(ctx), "-np", "1",
         "-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on",
         "--no-webui", "--no-cache-idle-slots"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 300
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{PORT}/health", timeout=3
                ) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            if proc.poll() is not None:
                raise RuntimeError("server exited")
            time.sleep(1.0)
        payload = {
            "prompt": ids,
            "n_predict": 1,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "min_p": 0.0,
            "repeat_penalty": 1.0,
            "ignore_eos": True,
            "cache_prompt": False,
            "return_tokens": True,
        }
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/completion",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        t = time.time()
        with urllib.request.urlopen(req, timeout=600) as r:
            body = json.loads(r.read())
        dt = time.time() - t
        print(json.dumps({
            "n": args.n,
            "seconds": round(dt, 3),
            "tokens": body.get("tokens"),
            "timings": {k: body.get("timings", {}).get(k) for k in
                        ("prompt_n", "prompt_per_second", "predicted_n")},
        }))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
