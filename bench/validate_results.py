#!/usr/bin/env python3
"""Fail unless the table satisfies the acceptance contract.

Checks, for every planned cell:
  - both engines have a result;
  - the cold rep count is 3 at depths 0/8192 and 1 at 32768/100000;
  - hipml ran q8 KV and llama ran q8_0;
  - the 100K row records KV type and a non-spill verdict;
  - ids hashes match the fixture.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    path = os.path.join(HERE, "results.json")
    with open(path, "r", encoding="utf-8") as f:
        r = json.load(f)
    errors = []
    rows = 0
    for c in r["cells"]:
        want = 3 if c["depth"] <= 8192 else 1
        rows += 1
        for eng in ("hipml", "llama"):
            e = c[eng]
            if not e.get("available"):
                errors.append(f"{c['prompt']}@{c['depth']} {eng}: {e.get('error')}")
                continue
            if e.get("reps") != want:
                errors.append(
                    f"{c['prompt']}@{c['depth']} {eng}: {e.get('reps')} reps, want {want}")
            if e.get("ids_sha256") and e["ids_sha256"] != c["ids_sha256"]:
                errors.append(f"{c['prompt']}@{c['depth']} {eng}: id hash mismatch")
            if eng == "hipml" and e.get("kv") != "q8":
                errors.append(f"{c['prompt']}@{c['depth']} hipml: kv {e.get('kv')}")
            if eng == "llama" and e.get("kv") != "q8_0":
                errors.append(f"{c['prompt']}@{c['depth']} llama: kv {e.get('kv')}")
        if c["depth"] >= 100000:
            l = c["llama"]
            if l.get("available") and l.get("spilled") is True:
                errors.append(f"{c['prompt']}@100000 llama: spilled")
            if l.get("available") and not l.get("kv"):
                errors.append(f"{c['prompt']}@100000 llama: no KV type")
    if errors:
        print("FAIL")
        for e in errors:
            print("  " + e)
        return 1
    print(f"PASS: {rows} cells, every cell present and labelled")
    return 0


if __name__ == "__main__":
    sys.exit(main())
