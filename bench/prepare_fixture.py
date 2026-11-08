#!/usr/bin/env python3
"""Build the shared token fixture for the like-for-like bench.

Every prompt is tokenised once, by `hipml tokenize`, on the same GGUF the two
engines will run. The fixture stores the raw prompt ids, one long document
token stream, and, for every (prompt, depth) cell, the exact prefix length and
a SHA-256 of the assembled ids. Both engines read that fixture and send those
ids; neither retokenises at run time.

Depth 0 is the prompt exactly as tokenised. At depth D the sequence is

    [BOS] + document_tokens[:D - 1 - len(prompt_without_bos)] + prompt_without_bos

so the total is exactly D tokens and the prompt is unchanged at the tail.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.path.join(HERE, "prompts")
DEPTHS = [0, 8192, 32768, 100000]

# (label, file, how to build the text)
PROMPT_FILES = [
    ("code_edit", "01_code_edit.txt", None),
    ("json_tool_call", "02_json_tool_call.txt", None),
    ("prose_explanation", "03_prose_explanation.txt", None),
    ("longdoc_qa", None, "longdoc"),
    ("copy_200_words", "05_copy_200_words.txt", None),
]

LONGDOC = "longdoc.txt"
LONGDOC_HEADER = "04_longdoc_qa_header.txt"


def sha_ids(ids):
    h = hashlib.sha256()
    for i in ids:
        h.update(int(i).to_bytes(4, "little", signed=False))
    return h.hexdigest()[:16]


def tokenize(hipml, gguf, text_path, tmpdir, name):
    out = os.path.join(tmpdir, name + ".json")
    cmd = [hipml, "tokenize", "--gguf", gguf, "--text-file", text_path,
           "--json", "--out", out]
    subprocess.run(cmd, check=True)
    with open(out, "r", encoding="utf-8") as f:
        return json.load(f)


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hipml", required=True, help="hipml executable")
    ap.add_argument("--gguf", required=True, help="model both engines will use")
    ap.add_argument("--out", default=os.path.join(HERE, "fixtures", "prompts.json"))
    ap.add_argument("--depths", default=",".join(str(d) for d in DEPTHS))
    ap.add_argument("--max-depth", type=int, default=100000)
    args = ap.parse_args()

    depths = [int(x) for x in args.depths.split(",") if x.strip()]
    if args.max_depth < max(depths):
        args.max_depth = max(depths)

    tmpdir = tempfile.mkdtemp(prefix="hipml-bench-fixture-")
    doc_path = os.path.join(PROMPTS, LONGDOC)
    doc = read_text(doc_path)
    header = read_text(os.path.join(PROMPTS, LONGDOC_HEADER))
    longdoc_prompt_path = os.path.join(tmpdir, "04_longdoc_qa.txt")
    write_text(longdoc_prompt_path, header + doc + "\n")

    doc_tok = tokenize(args.hipml, args.gguf, doc_path, tmpdir, "document")
    add_bos = bool(doc_tok.get("add_bos", False))
    bos_id = int(doc_tok.get("bos_id", 0))
    bos = [bos_id] if add_bos else []

    doc_ids = list(doc_tok["ids"])
    if add_bos and doc_ids and doc_ids[0] == bos_id:
        doc_ids = doc_ids[1:]
    if not doc_ids:
        sys.exit("document tokenised to zero ids")

    # Repeat the real document token stream until it covers the deepest
    # prefix. Repetition is deterministic and both engines see the result.
    while len(doc_ids) < args.max_depth:
        doc_ids = doc_ids + doc_ids
    doc_ids = doc_ids[: args.max_depth]

    prompts = []
    for label, fname, kind in PROMPT_FILES:
        path = longdoc_prompt_path if kind == "longdoc" else os.path.join(PROMPTS, fname)
        tok = tokenize(args.hipml, args.gguf, path, tmpdir, label)
        raw = [int(x) for x in tok["ids"]]
        ids = list(raw)
        if add_bos and ids and ids[0] == bos_id:
            ids = ids[1:]
        if not ids:
            sys.exit(f"{label}: tokenised to zero ids")
        prompts.append({
            "label": label,
            "source": os.path.basename(path) if kind == "longdoc" else fname,
            "raw_ids": raw,
            "ids": ids,
            "token_count": len(raw),
        })

    cells = []
    for p in prompts:
        for d in depths:
            if d == 0:
                seq = list(p["raw_ids"])
            else:
                prefix_len = d - len(bos) - len(p["ids"])
                if prefix_len < 0:
                    sys.exit(
                        f"{p['label']}: prompt is {len(p['ids'])} tokens (+BOS); "
                        f"too long for depth {d}. Shorten the prompt."
                    )
                seq = bos + doc_ids[:prefix_len] + p["ids"]
            if d != 0 and len(seq) != d:
                sys.exit(f"{p['label']}@{d}: assembled {len(seq)} tokens, wanted {d}")
            cells.append({
                "id": f"{p['label']}@{d}",
                "prompt": p["label"],
                "depth": d,
                "total": len(seq),
                "prompt_len": len(p["ids"]),
                "prefix_len": 0 if d == 0 else d - len(bos) - len(p["ids"]),
                "ids_sha256": sha_ids(seq),
            })

    fixture = {
        "format": "hipml-bench-fixture/1",
        "generated_by": "hipml tokenize",
        "model": {
            "path": os.path.abspath(args.gguf),
            "bytes": os.path.getsize(args.gguf),
        },
        "tokenizer": {"add_bos": add_bos, "bos_id": bos_id},
        "depths": depths,
        "document": {
            "source": LONGDOC,
            "tokens": len(doc_ids),
            "ids": doc_ids,
        },
        "prompts": prompts,
        "cells": cells,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(fixture, f, separators=(",", ":"))
        f.write("\n")
    print(f"wrote {args.out}")
    for p in prompts:
        print(f"  {p['label']:20s} {p['token_count']:6d} tokens")
    print(f"  document            {len(doc_ids):6d} tokens")
    for c in cells:
        print(f"  {c['id']:26s} total {c['total']:6d}  sha {c['ids_sha256']}")
    print(f"  fixture bytes       {os.path.getsize(args.out):,}")


if __name__ == "__main__":
    main()
