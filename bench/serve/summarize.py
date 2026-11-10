#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Print the serving table from bench/serve/results/*.json."""

import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    rows = {}
    for path in sorted(glob.glob(os.path.join(HERE, "results", "*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        rows[(d["cell"], d["engine"])] = d

    print("| cell | engine | TTFT p50 | TTFT p95 | stream tok/s | aggregate | VRAM |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for cell in "abc":
        for engine in ("hipml", "llama"):
            d = rows.get((cell, engine))
            if not d:
                continue
            s = d["summary"]
            print(f"| {cell} | {engine} | {s['ttft_p50_ms']:.0f} ms | "
                  f"{s['ttft_p95_ms']:.0f} ms | {s['stream_decode_p50_tok_s']:.1f} | "
                  f"{s['aggregate_tok_s_p50']:.1f} | "
                  f"{d['peak_vram_mib'] / 1024:.1f} GiB |")

    print()
    for cell in "abc":
        d = rows.get((cell, "hipml"))
        if not d or not d.get("metrics"):
            continue
        m = d["metrics"]
        print(f"hipml {cell}: prefix hits {m.get('hipml_prefix_hits_total', 0):.0f}/"
              f"{m.get('hipml_prefix_lookups_total', 0):.0f}, "
              f"restore {m.get('hipml_prefix_restore_seconds_total', 0):.3f}s, "
              f"interleaved tiles {m.get('hipml_prefill_interleaved_total', 0):.0f}, "
              f"stall {m.get('hipml_interleave_stall_seconds_total', 0):.3f}s "
              f"(max {m.get('hipml_interleave_stall_seconds_max', 0) * 1e3:.0f} ms)")


if __name__ == "__main__":
    main()
