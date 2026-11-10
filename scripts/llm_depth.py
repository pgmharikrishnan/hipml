#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""llama.cpp reference target: build a prompt of a target depth, then measure
prefill and decode on it.

The server must already be listening on --port with the context we want (this
script only talks HTTP). One call per server instance, because the first prefill
of a fresh slot is the one whose `timings` we want.

  # build a ~100K-token prompt from the calibration corpus and measure
  python scripts/llm_depth.py --port 8092 --corpus %HIPML_LONGCTX_CORPUS% \
      --target 100000 --prompt bench\\out\\prompt100k.txt --out f16.json

  --corpus defaults to $HIPML_LONGCTX_CORPUS; the committed corpus is
  bench/prompts/longdoc.txt. --prompt defaults to bench/out/prompt100k.txt.
"""
import argparse
import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PROMPT = os.path.join(ROOT, "bench", "out", "prompt100k.txt")

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
    ap.add_argument("--prompt", default=DEFAULT_PROMPT,
                    help="prompt file to read/write (default: "
                         "bench/out/prompt100k.txt)")
    ap.add_argument("--corpus", default=os.environ.get("HIPML_LONGCTX_CORPUS"),
                    help="build the prompt from this text first (default: "
                         "$HIPML_LONGCTX_CORPUS or bench/prompts/longdoc.txt)")
    ap.add_argument("--out", default=None,
                    help="result JSON (default: <prompt>.json)")
    ap.add_argument("--target", type=int, default=TARGET_TOKENS)
    ap.add_argument("--n-predict", type=int, default=128)
    args = ap.parse_args()

    if not args.out:
        args.out = os.path.splitext(args.prompt)[0] + ".json"
    os.makedirs(os.path.dirname(os.path.abspath(args.prompt)), exist_ok=True)

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
