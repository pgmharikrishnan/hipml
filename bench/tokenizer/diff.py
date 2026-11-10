#!/usr/bin/env python3
"""Tokenizer differential harness: hipml vs llama.cpp on a hex corpus.

Reads the corpus written by gen_corpus.py (``hex(text) TAB label``), gets
llama's ids from llama-server's /tokenize, gets hipml's ids from
``hipml tokenize-batch``, and reports every mismatch class with a minimal
reproducer. With --save it writes ``ids.txt`` (``label TAB hex TAB [ids]``)
for the permanent fixture.

Usage:
    python3 bench/tokenizer/diff.py --corpus corpus.txt --hipml-ids hipml.txt
    python3 bench/tokenizer/diff.py --corpus corpus.txt --save fixture.txt
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = "http://127.0.0.1:8092"


def post(path, obj, tries=3):
    data = json.dumps(obj).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    last = None
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=120) as f:
                return json.load(f)
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            time.sleep(1 + attempt)
    raise last


def read_corpus(path):
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            continue
        h, _, label = line.partition("\t")
        out.append((label, bytes.fromhex(h).decode("utf-8"), h))
    return out


def read_hipml(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        _, label, ids = line.rstrip("\n").split("\t", 2)
        out.setdefault(label, []).append(json.loads(ids))
    return out


def llama_ids(text):
    r = post("/tokenize", {"content": text, "add_special": False})
    return r["tokens"]


def cls(label):
    return label.split("-")[0] if label else "?"


def main():
    global BASE
    ap = argparse.ArgumentParser(
        description="Diff hipml tokenizer output against llama-server's.")
    ap.add_argument("--base", default=os.environ.get("HIPML_BASE_URL", BASE),
                    help="llama-server base URL (default: %(default)s)")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--hipml-ids", help="precomputed tokenize-batch output")
    ap.add_argument("--save", help="write the fixture file")
    ap.add_argument("--max-per-class", type=int, default=0,
                    help="when saving, keep at most N evenly spaced records per class")
    ap.add_argument("--max-text-chars", type=int, default=0,
                    help="when saving, skip records longer than N characters")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    BASE = args.base.rstrip("/")

    corpus = read_corpus(args.corpus)
    if args.limit:
        corpus = corpus[:args.limit]
    if args.hipml_ids:
        hip = read_hipml(args.hipml_ids)
    else:
        raise SystemExit("--hipml-ids is required (run hipml tokenize-batch)")

    # Which records go into the committed fixture: evenly spaced per class so
    # hard cases from every block survive without a 10 MB file.
    keep = set(range(len(corpus)))
    if args.save and args.max_text_chars:
        keep = {i for i in keep if len(corpus[i][1]) <= args.max_text_chars}
    if args.save and args.max_per_class:
        allowed = keep
        by_class = {}
        for i, (label, _, _) in enumerate(corpus):
            if i in allowed:
                by_class.setdefault(cls(label), []).append(i)
        keep = set()
        for idxs in by_class.values():
            n = len(idxs)
            if n <= args.max_per_class:
                keep.update(idxs)
            else:
                step = (n - 1) / (args.max_per_class - 1)
                keep.update(idxs[round(k * step)] for k in range(args.max_per_class))

    used = {}
    bad = []
    errors = []
    fixture = []
    for i, (label, text, h) in enumerate(corpus):
        k = used.get(label, 0)
        used[label] = k + 1
        mine = hip[label][k]
        if i % 500 == 0:
            print(f"  {i}/{len(corpus)}", file=sys.stderr)
        try:
            ref = llama_ids(text)
        except Exception as e:  # noqa: BLE001
            errors.append((label, text, repr(e)))
            continue
        if args.save and i in keep:
            fixture.append(f"{h}\t{label}\t{json.dumps(ref)}")
        if ref != mine:
            pos = next((j for j, (a, b) in enumerate(zip(ref, mine)) if a != b),
                       min(len(ref), len(mine)))
            bad.append({
                "label": label, "text": text, "hex": h,
                "llama": ref, "hipml": mine, "pos": pos,
            })

    print(f"\nstrings: {len(corpus) - len(errors)} compared, {len(errors)} llama errors")
    print(f"mismatches: {len(bad)}")
    by = {}
    for b in bad:
        by.setdefault(cls(b["label"]), []).append(b)
    for k in sorted(by, key=lambda x: -len(by[x])):
        b = by[k][0]
        print(f"  {k}: {len(by[k])}  e.g. {b['label']} at id {b['pos']}: "
              f"llama={b['llama'][max(0,b['pos']-2):b['pos']+3]} "
              f"hipml={b['hipml'][max(0,b['pos']-2):b['pos']+3]}")

    if bad:
        with open("mismatches.json", "w", encoding="utf-8") as f:
            json.dump(bad, f, ensure_ascii=False, indent=1)
        print("wrote mismatches.json")
        for b in bad[:10]:
            print(f"\n-- {b['label']} hex={b['hex'][:120]}")
            print(f"   text={b['text'][:120]!r}")
            print(f"   llama={b['llama'][:40]}")
            print(f"   hipml={b['hipml'][:40]}")
    if errors:
        for label, text, err in errors[:5]:
            print(f"llama error {label}: {text[:60]!r}: {err}")
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            f.write("\n".join(fixture) + "\n")
        print(f"wrote {args.save} ({len(fixture)} records)")
    return 1 if bad or errors else 0


if __name__ == "__main__":
    sys.exit(main())
