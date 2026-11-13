#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Serve-path exactness: speculation on must reproduce speculation off, id for id.

`spec-bench` exercises the CLI engine, so it cannot see a server-only dispatch
change: `serve` sets `HIPML_BATCH_SKINNY=1` for the whole process, and that is
the flag the review-4 audit found routing the 4-column MTP verify onto the
non-bit-identical WMMA tile. This probe runs the real server both ways and
compares the ids `/completion` returns.

One server per draft setting (the flag is process-wide), the same 16
conversations through each, greedy, `ignore_eos` so every reply is the same
length. Prints each reply's sha1 and length, then the verdict. Exits 1 on any
difference.

    python bench/serve/probe_exact.py --exe target/release/hipml.exe \
        --gguf model.gguf --mtp mtp.gguf --ctx 32768 --tokens 128
"""

import argparse, hashlib, json, os, subprocess, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_bench import SYSTEM, USERS  # noqa: E402


def wait_health(proc, port, timeout=420):
    end = time.time() + timeout
    while time.time() < end:
        if proc.poll() is not None:
            raise RuntimeError("server exited early")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(1.0)
    raise RuntimeError("no health")


def complete(port, prompt, tokens):
    body = {
        "prompt": prompt,
        "n_predict": tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": False,
        "cache_prompt": False,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)["tokens"]


def run_serve(args, drafts, port):
    """One server, all 16 conversations, in order. Returns [[ids], ...]."""
    cmd = [args.exe, "serve", "--gguf", args.gguf, "--port", str(port),
           "--ctx", str(args.ctx), "--slots", "1", "--prefix-entries", "16"]
    if args.chunk:
        cmd += ["--chunk", str(args.chunk)]
    if drafts:
        cmd += ["--mtp", args.mtp, "--drafts", str(drafts)]
    log = open(f"probe_exact_d{drafts}.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_health(proc, port)
        out = []
        for u in USERS:
            ids = complete(port, f"{SYSTEM}\n\nUser: {u}\n\nAssistant:", args.tokens)
            out.append(ids)
        return out
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
        log.close()


def sha(ids):
    return hashlib.sha1(json.dumps(ids).encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--mtp", default="")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--drafts", default="1,2,3")
    ap.add_argument("--port", type=int, default=8093)
    ap.add_argument("--chunk", type=int, default=None,
                    help="serve --chunk (default: the server's own default). "
                         "Pass it to compare a fixed chunk across builds.")
    ap.add_argument("--dump", default="",
                    help="write the ids per draft as JSON, so two runs (two "
                         "chunks) can be diffed id for id")
    a = ap.parse_args()

    runs = {}
    for d in [0] + [int(x) for x in a.drafts.split(",") if x.strip()]:
        if d and not a.mtp:
            sys.exit("probe_exact: --mtp is required for drafts > 0")
        runs[d] = run_serve(a, d, a.port)
        for i, ids in enumerate(runs[d]):
            print("drafts %d chat %d sha1 %s len %d" % (d, i, sha(ids), len(ids)),
                  flush=True)
    if a.dump:
        with open(a.dump, "w", encoding="utf-8") as f:
            json.dump({str(d): ids for d, ids in runs.items()}, f)

    base = runs[0]
    bad = 0
    for d in sorted(runs):
        if d == 0:
            continue
        for i, (x, y) in enumerate(zip(base, runs[d])):
            if x != y:
                at = next((k for k, (p, q) in enumerate(zip(x, y)) if p != q), None)
                print("DIFF drafts %d chat %d first id %s: plain %s spec %s" % (
                    d, i, at,
                    x[at:at + 4] if at is not None else "len",
                    y[at:at + 4] if at is not None else "len"), flush=True)
                bad += 1
    if bad:
        print("serve exactness: FAIL, %d of %d replies differ" % (
            bad, (len(runs) - 1) * len(base)))
        return 1
    print("serve exactness: PASS, %d replies id-for-id across drafts %s" % (
        len(base) * (len(runs) - 1), ",".join(str(d) for d in sorted(runs) if d)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
