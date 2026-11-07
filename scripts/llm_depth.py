#!/usr/bin/env python3
"""llama.cpp reference target: build a prompt of a target depth, then measure
prefill and decode on it.

The server must already be listening on --port with the context we want (this
script only talks HTTP). One call per server instance, because the first prefill
of a fresh slot is the one whose `timings` we want.

  # build a ~100K-token prompt from the calibration corpus and measure
  python scripts/llm_depth.py --port 8092 --corpus C:\\models\\calib_longctx.txt \
      --target 100000 --prompt C:\\models\\prompt100k.txt --out f16.json
"""
import argparse
import json
import sys
import urllib.request

TARGET_TOKENS = 100_000


def post(port, path, payload, timeout=1800):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def tokenize(port, text):
    out = post(port, "/tokenize", {"content": text, "add_special": False})
    return out["tokens"]


def build_prompt(port, corpus, target, out_path):
    text = open(corpus, encoding="utf-8", errors="ignore").read()
    lo, hi = 1024, len(text)
    best = text
    for _ in range(6):
        n = (lo + hi) // 2
        cut = text.rfind("\n", 0, n)
        if cut < lo:
            cut = n
        chunk = text[:cut]
        ntok = len(tokenize(port, chunk))
        print(f"  bytes {cut} -> {ntok} tokens")
        if abs(ntok - target) < 500:
            best = chunk
            break
        if ntok < target:
            lo = cut
        else:
            hi = cut
        best = chunk
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(best)
    toks = tokenize(port, best)
    print(f"prompt written: {out_path}, {len(best)} bytes, {len(toks)} tokens")
    return best, toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--prompt", required=True, help="prompt file to read/write")
    ap.add_argument("--corpus", help="build the prompt from this text first")
    ap.add_argument("--target", type=int, default=TARGET_TOKENS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-predict", type=int, default=128)
    args = ap.parse_args()

    if args.corpus:
        text, toks = build_prompt(args.port, args.corpus, args.target, args.prompt)
    else:
        text = open(args.prompt, encoding="utf-8", errors="ignore").read()
        toks = tokenize(args.port, text)
        print(f"prompt {len(text)} bytes -> {len(toks)} tokens")

    body = {
        "prompt": toks,
        "n_predict": args.n_predict,
        "temperature": 0.0,
        "top_k": 1,
        "cache_prompt": False,
        "ignore_eos": True,
        "stream": False,
    }
    res = post(args.port, "/completion", body)
    t = res.get("timings", {})
    row = {
        "prompt_bytes": len(text),
        "prompt_tokens": t.get("prompt_n"),
        "prompt_ms": t.get("prompt_ms"),
        "prefill_tok_s": t.get("prompt_per_second"),
        "predicted_tokens": t.get("predicted_n"),
        "predicted_ms": t.get("predicted_ms"),
        "decode_tok_s": t.get("predicted_per_second"),
        "content_head": (res.get("content") or "")[:80],
    }
    print(json.dumps(row, indent=2))
    with open(args.out, "w") as f:
        json.dump(row, f, indent=2)


if __name__ == "__main__":
    main()
