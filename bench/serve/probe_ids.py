#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Five conversations through `hipml serve` at a fixed context, hashed.

The point is to compare two builds' answers without diffing text: each reply's
sha1 and length is printed, so a byte-level difference shows up as a different
hash. `PROBE_CTX` sets the cache context (the same prompt at 32K is the cell
that exposed the attention split-count defect).
"""

import hashlib, json, os, subprocess, sys, time, urllib.request

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


def chat(user, max_tokens=64):
    body = {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": False,
        "cache_prompt": False,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        obj = json.load(r)
    return obj["choices"][0]["message"]["content"]


def main():
    ctx = os.environ.get("PROBE_CTX", "32768")
    exe = os.environ["HIPML_EXE"]
    log = open("probe_ids.log", "w")
    proc = subprocess.Popen(
        [exe, "serve", "--gguf", os.environ["HIPML_GGUF"], "--port", str(PORT),
         "--ctx", ctx, "--slots", "1", "--prefix-entries", "16"],
        stdout=log, stderr=subprocess.STDOUT,
    )
    try:
        wait_health(proc)
        for i, u in enumerate(USERS):
            t = chat(u)
            print("chat %d sha1 %s len %d" % (i, hashlib.sha1(t.encode()).hexdigest()[:16], len(t)),
                  flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
        log.close()


main()
