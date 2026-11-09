#!/usr/bin/env python3
"""One realistic coding-agent request through both engines.

The conversation is multi-turn, carries an assistant tool call with JSON
arguments, and the tool response is a Rust file whose identifiers and strings
are non-ASCII. The prompt is rendered by llama's /apply-template, tokenized
once, and both engines generate greedily from the same ids. Reports the first
output divergence and the margin llama assigns to it.

Requires llama-server (GPU, -fa off, f32 KV) on 8092 and a hip build of hipml.

Usage:
    python3 bench/tokenizer/coding_task.py --gguf C:\\models\\Qwen3.8-27B-Q4_K_M.gguf \
        --hipml target/release/hipml.exe --tokens 96 --out coding_task.json
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8092"

CODE = """// src/caf\u00e9/\u4e16\u754c.rs
fn saluer(nom: &str) -> String {
    format!("Bonjour, {nom} \U0001F44B")
}

fn main() {
    let valeurs = vec![1, 2, 3];
    println!("{}", valeurs[3]); // \u26a0\ufe0f index hors limites
}
"""

REQUEST = {
    "messages": [
        {"role": "system",
         "content": "You are a coding agent in a Rust repository. Be terse."},
        {"role": "user",
         "content": "Open src/caf\u00e9/\u4e16\u754c.rs and fix the panic."},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_read",
            "type": "function",
            "function": {"name": "read_file",
                         "arguments": '{"path":"src/caf\\u00e9/\\u4e16\\u754c.rs"}'}}]},
        {"role": "tool", "content": CODE},
        {"role": "user",
         "content": "Explain the panic in one sentence and give the minimal diff."},
    ],
}


def post(path, obj, timeout=1800):
    data = json.dumps(obj).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as f:
            return True, json.load(f)
    except urllib.error.HTTPError as e:
        return False, e.read().decode("utf-8", "replace")


def first_diff(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--hipml", required=True)
    ap.add_argument("--tokens", type=int, default=96)
    ap.add_argument("--out", default="coding_task.json")
    args = ap.parse_args()

    ok, at = post("/apply-template", REQUEST)
    if not ok:
        raise SystemExit(f"/apply-template: {at}")
    prompt = at["prompt"]
    ids = post("/tokenize", {"content": prompt, "add_special": False})[1]["tokens"]

    req_path = args.out + ".request.json"
    with open(req_path, "w", encoding="utf-8") as f:
        json.dump(REQUEST, f, ensure_ascii=False)
    proc = subprocess.run(
        [args.hipml, "chat", "--gguf", args.gguf, "--json-file", req_path, "--json"],
        capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        raise SystemExit(f"hipml chat: {proc.stderr}")
    hip = json.loads(proc.stdout)
    print(f"prompt: {len(prompt.encode('utf-8'))} bytes, {len(ids)} ids; "
          f"text {'match' if hip['text'] == prompt else 'DIFF'}, "
          f"ids {'match' if hip['ids'] == ids else 'DIFF'}")

    token_ids = args.out + ".ids.json"
    with open(token_ids, "w", encoding="utf-8") as f:
        json.dump({"ids": ids}, f)

    ok, comp = post("/completion", {
        "prompt": ids, "n_predict": args.tokens, "temperature": 0.0,
        "top_k": 1, "top_p": 1.0, "min_p": 0.0, "repeat_penalty": 1.0,
        "ignore_eos": False, "cache_prompt": False, "n_probs": 8,
    })
    if not ok:
        raise SystemExit(f"/completion: {comp}")
    probs = comp.get("completion_probabilities") or []
    llama_out = [p["id"] for p in probs]
    llama_text = comp.get("content", "")

    gen = subprocess.run(
        [args.hipml, "generate", "--device", "--gguf", args.gguf,
         "--ids-file", token_ids, "--tokens", str(args.tokens),
         "--kv", "f32", "--chunk", "0", "--stop-eos", "--json"],
        capture_output=True, text=True, encoding="utf-8")
    if gen.returncode != 0:
        raise SystemExit(f"hipml generate: {gen.stderr}")
    hr = json.loads(gen.stdout)["reps"][0]
    hip_out = hr["ids"]

    d = first_diff(llama_out, hip_out)
    margin = None
    top = None
    if d < len(probs) and d < len(hip_out) and llama_out[d] != hip_out[d]:
        pl = probs[d]
        cand = [x for x in (pl.get("top_logprobs") or []) if x["id"] == hip_out[d]]
        if cand:
            margin = pl["logprob"] - cand[0]["logprob"]
        top = [(x["id"], x["logprob"]) for x in (pl.get("top_logprobs") or [])[:3]]

    result = {
        "prompt_bytes": len(prompt.encode("utf-8")),
        "prompt_tokens": len(ids),
        "prompt_text_match": hip["text"] == prompt,
        "prompt_ids_match": hip["ids"] == ids,
        "llama_ids": llama_out,
        "hipml_ids": hip_out,
        "first_diff": d,
        "llama_token": llama_out[d] if d < len(llama_out) else None,
        "hipml_token": hip_out[d] if d < len(hip_out) else None,
        "llama_top3_at_diff": top,
        "margin_nats": margin,
        "llama_text": llama_text,
        "hipml_text": hr.get("text", ""),
        "stopped": hr.get("stopped"),
        "llama_stop_type": comp.get("stop_type"),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)

    identical = d >= min(len(llama_out), len(hip_out)) and llama_out == hip_out
    print(f"generated: llama {len(llama_out)} ids, hipml {len(hip_out)} ids")
    if identical:
        print("output ids identical")
    else:
        print(f"first divergence at id {d}: llama={result['llama_token']} "
              f"hipml={result['hipml_token']}")
        print(f"  llama top-3 at that step: {top}")
        if margin is not None:
            print(f"  margin (llama tok - hipml tok) = {margin:.4f} nats")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
