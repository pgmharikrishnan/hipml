#!/usr/bin/env python3
"""MTP acceptance on chat-formatted prompts, with EOS stop.

Runs `hipml spec-bench` (plain greedy + 1..3 drafts) on the prompts rendered by
`hipml chat`, so every acceptance figure is measured on text that starts at the
assistant turn and ends at the model's own EOS. Without the stop the model
closes the turn and invents a user turn; that degenerate repetition is trivially
draftable and inflates acceptance.

Usage:
    python C:\\hipml-chat\\bench\\chat\\mtp_chat.py --tokens 384 --drafts 3
"""

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHAT = os.path.join(ROOT, "bench", "chat")
OUT = r"C:\Users\Hari\wp43\chat"
MODEL = r"C:\models\Qwen3.8-27B-Q4_K_M.gguf"
MTP = r"C:\models\mtp-Qwen3.8-27B-Q4_0.gguf"
HIPML = os.path.join(ROOT, "target", "release", "hipml.exe")
CONVS = [
    ("01_single_turn_code", "01_single_turn_code.txt", None),
    ("02_multi_turn", "02_multi_turn.txt", None),
    ("03_system_tool", "03_system_tool.txt", "03_tools.jsonl"),
    ("04_longdoc_qa", "04_longdoc_qa.txt", None),
    ("05_copy", "05_copy.txt", None),
]


def ids_for(name, message_file, tools_file):
    path = os.path.join(OUT, name + ".ids.json")
    if os.path.exists(path):
        return path
    cmd = [HIPML, "chat", "--gguf", MODEL,
           "--messages", os.path.join(CHAT, message_file), "--json",
           "--ids-out", path]
    if tools_file:
        cmd += ["--tools", os.path.join(CHAT, tools_file)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(f"hipml chat {name}: {r.stderr}")
    return path


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--drafts", type=int, default=3)
    ap.add_argument("--kv", default="f32")
    ap.add_argument("--ctx", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    OUT = args.out
    os.makedirs(OUT, exist_ok=True)
    all_rows = {}
    for name, message_file, tools_file in CONVS:
        ids = ids_for(name, message_file, tools_file)
        n_ctx = args.ctx or (len(json.load(open(ids))["ids"]) + args.tokens + 64)
        cmd = [HIPML, "spec-bench", "--gguf", MODEL, "--mtp", MTP,
               "--ids-file", ids, "--tokens", str(args.tokens),
               "--drafts", str(args.drafts), "--kv", args.kv,
               "--ctx", str(n_ctx), "--json", "--stop-eos"]
        print(f"running {name} ...", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        if r.returncode != 0:
            raise RuntimeError(f"spec-bench {name}: {r.stderr[-3000:]}")
        row = json.loads(r.stdout)
        all_rows[name] = row
        json.dump(row, open(os.path.join(OUT, name + ".mtp.json"), "w"), indent=1)
        print(json.dumps({k: row[k] for k in
                          ("prompt_tokens", "kv")}, sort_keys=True))
    json.dump(all_rows, open(os.path.join(OUT, "mtp_chat.json"), "w"), indent=1)

    print("\n=== MTP acceptance on chat prompts, EOS stop ===")
    print(f"{'prompt':<22} {'toks':>5} {'plain':>7} {'drafts':>6} "
          f"{'tok/s':>7} {'x':>5} {'accept':<22} {'exact':>10} {'eos':>4}")
    for name, _m, _t in CONVS:
        row = all_rows[name]
        rows = row["rows"]
        plain = rows[0]
        best = max(rows[1:], key=lambda r: r["tok_s"])
        acc = ",".join(f"{a:.3f}" for a in best["accept"])
        base = rows[0]["ids"]
        n = min(len(best["ids"]), len(base))
        same = sum(1 for a, b in zip(best["ids"], base) if a == b)
        print(f"{name:<22} {row['prompt_tokens']:>5} {plain['tok_s']:>7.2f} "
              f"{best['drafts']:>6} {best['tok_s']:>7.2f} "
              f"{best['tok_s']/plain['tok_s']:>5.2f} {acc:<22} "
              f"{str(same)+'/'+str(n):>10} {str(best['stopped']):>4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
