#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""One request at a time against `hipml serve`, with the server's own timings.

`serve_bench.py` cell a reports the aggregate over five conversations; this
prints the per-request split (`prompt_n`, `prompt_ms`, `prefix_hit_tokens`, and
both the server's and the client's tokens/s) so a regression can be attributed
to prefill, cache geometry or decode. `PROBE_CTX` sets the cache context,
`PROBE_REQS` the number of requests, and the first request is uncached.
"""

import json, os, subprocess, sys, time, urllib.request

sys.path.insert(0, os.path.join("bench", "serve"))
from serve_bench import SYSTEM, USERS  # noqa: E402

PORT = 8093


def wait_health(proc, timeout=420):
    end = time.time() + timeout
    while time.time() < end:
        if proc.poll() is not None:
            raise RuntimeError("server exited early")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(1.0)
    raise RuntimeError("no health")


def chat(user, max_tokens=64, cache=False):
    body = {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "cache_prompt": cache,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    first = last = None
    final = None
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
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
            d = obj.get("choices", [{}])[0].get("delta", {})
            if (d.get("content") or "") or (d.get("reasoning_content") or ""):
                t = time.perf_counter()
                if first is None:
                    first = t
                last = t
    tm = (final or {}).get("timings", {})
    n = (final or {}).get("usage", {}).get("completion_tokens")
    return {
        "client_tps": (n - 1) / (last - first) if n and last > first else None,
        "server_tps": tm.get("predicted_per_second"),
        "prompt_n": tm.get("prompt_n"),
        "prompt_ms": tm.get("prompt_ms"),
        "hit": tm.get("prefix_hit_tokens"),
    }


def main():
    gguf = os.environ["HIPML_GGUF"]
    exe = os.environ["HIPML_EXE"]
    log = open("probe_a.log", "w")
    proc = subprocess.Popen(
        [exe, "serve", "--gguf", gguf, "--port", str(PORT), "--ctx", os.environ.get("PROBE_CTX", "32768"),
         "--slots", "1", "--prefix-entries", "16", "--prefix-mb", os.environ.get("PROBE_PREFIX", "2048")],
        stdout=log, stderr=subprocess.STDOUT,
    )
    try:
        wait_health(proc)
        for i, cache in enumerate([False, True, True][: int(os.environ.get("PROBE_REQS", "3"))]):
            r = chat(USERS[i], cache=cache)
            print("req %d cache=%s %s" % (i, cache, json.dumps(r)), flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
        log.close()


main()
