#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Normalise machine paths in committed results and gate logs, without touching a number.

Results are measurement records: this script rewrites only path strings and the
recorded host name. It is idempotent, so it can run after every re-measure, and
`--check` exits 1 if anything would still change (the CI use).

    python bench/tools/scrub_paths.py            # rewrite the default set
    python bench/tools/scrub_paths.py --check    # exit 1 if a file needs it
    python bench/tools/scrub_paths.py FILE...    # explicit files

JSON is rewritten value by value; plain-text logs (`.out`, `.txt`, `.log`) are
rewritten in place, path string by path string, and are otherwise byte-for-byte.

What it does:
  * a key named `host` becomes `gpu-host`;
  * a Windows or macOS absolute path becomes a placeholder or a basename:
      *.gguf                 -> <HIPML_GGUF> / <HIPML_MTP_GGUF>
      llama-server[.exe]     -> <LLAMA_SERVER>
      hipml[.exe]            -> <HIPML_EXE>
      gpu-lock.ps1           -> <HIPML_GPU_LOCK>
      C:\\hipml-*\\...        -> <REPO>\\<relative>
      C:\\rocm-sdk\\...       -> <ROCM>\\<relative>
      C:\\models\\...         -> <MODELS>\\<relative>
      C:\\Users\\<name>\\...  -> <HOME>\\<relative>
      anything else          -> <REPO>\\<relative> or the basename
  * every other value, and every number, is left byte-identical.

Indentation, key order, non-ASCII escaping and the trailing newline of each
file are preserved. A path that is already backslash-escaped inside a quoted
JSON fragment in a log keeps its doubled separators.
"""

import argparse
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_FILES = [
    "bench/fixtures/prompts.json",
    "bench/sensitivity/*.json",
    "bench/serve/results/**/*.json",
    "bench/integrate/*.json",
    "bench/release/gates2/*.out",
    "bench/release/gates3/*.out",
]

JSON_SUFFIXES = (".json", ".jsonl")
TEXT_SUFFIXES = (".out", ".txt", ".log")

WIN_PATH = re.compile(r"[A-Za-z]:\\[^\s\"']*")
POSIX_PATH = re.compile(r"/Users/[^/\s\"']+/[^\s\"']*")

REPO_PREFIXES = (
    "hipml-serve", "hipml-release", "hipml-bench", "hipml-", "src",
)


def _gguf_placeholder(basename):
    return "<HIPML_MTP_GGUF>" if "mtp" in basename.lower() else "<HIPML_GGUF>"


def _join(prefix, parts, sep="\\"):
    return prefix + sep + sep.join(parts) if parts else prefix


def _scrub_windows(match):
    raw = match.group(0)
    # A path quoted inside a JSON fragment in a log has doubled separators;
    # keep that spelling so the fragment stays valid JSON.
    sep = "\\\\" if "\\\\" in raw else "\\"
    rest = raw[3:]  # drop "C:\"
    parts = [p for p in rest.replace("\\\\", "\\").split("\\") if p]
    if not parts:
        return "<PATH>"
    base = parts[-1]
    low = base.lower()
    if low.endswith(".gguf"):
        return _gguf_placeholder(base)
    if low in ("llama-server.exe", "llama-server"):
        return "<LLAMA_SERVER>"
    if low in ("hipml.exe", "hipml"):
        return "<HIPML_EXE>"
    if low == "gpu-lock.ps1":
        return "<HIPML_GPU_LOCK>"
    head = parts[0].lower()
    if head.startswith("hipml-") or head == "src":
        tail = parts[1:] if head.startswith("hipml-") else parts[2:]
        return _join("<REPO>", tail, sep)
    if head == "rocm-sdk":
        return _join("<ROCM>", parts[1:], sep)
    if head == "models":
        return _join("<MODELS>", parts[1:], sep)
    if head == "users":
        return _join("<HOME>", parts[2:], sep)
    return base


def _scrub_posix(match):
    raw = match.group(0)
    for marker in ("/bench/", "/docs/", "/crates/", "/scripts/"):
        if marker in raw:
            return marker.strip("/") + "/" + raw.split(marker, 1)[1]
    return raw.rsplit("/", 1)[-1]


def scrub_string(value):
    if not isinstance(value, str):
        return value
    out = WIN_PATH.sub(_scrub_windows, value)
    out = POSIX_PATH.sub(_scrub_posix, out)
    return out


def scrub(obj, key=None):
    if isinstance(obj, dict):
        return {k: scrub(v, k) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub(v, key) for v in obj]
    if isinstance(obj, str):
        if key == "host" and obj:
            return "gpu-host"
        return scrub_string(obj)
    return obj


def read_shape(text):
    """Indentation, ASCII escaping, line ending and trailing newline, from the file."""
    indent = None
    m = re.search(r"\n( +)\"", text)
    if m:
        indent = len(m.group(1))
    return {
        "indent": indent,
        "ensure_ascii": "\\u" in text,
        "crlf": "\r\n" in text,
        "newline": text.endswith("\n"),
    }


def rewrite(path, check=False):
    with open(path, "r", encoding="utf-8", newline="") as f:
        text = f.read()
    if path.lower().endswith(TEXT_SUFFIXES):
        body = POSIX_PATH.sub(_scrub_posix, WIN_PATH.sub(_scrub_windows, text))
    else:
        data = json.loads(text)
        shape = read_shape(text)
        clean = scrub(data)
        body = json.dumps(clean, indent=shape["indent"],
                          ensure_ascii=shape["ensure_ascii"])
        if shape["newline"]:
            body += "\n"
        if shape["crlf"]:
            body = body.replace("\n", "\r\n")
    if body == text:
        return False
    if not check:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(body)
    return True


def expand(patterns):
    files = []
    for pattern in patterns:
        full = os.path.join(ROOT, pattern)
        matches = sorted(glob.glob(full, recursive=True))
        files.extend(matches if matches else [full])
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if any file still needs scrubbing")
    ap.add_argument("files", nargs="*", help="files or globs (repo-relative)")
    args = ap.parse_args()

    files = args.files or expand(DEFAULT_FILES)
    changed = []
    for path in files:
        if not os.path.exists(path):
            print(f"skip (missing): {path}", file=sys.stderr)
            continue
        if rewrite(path, check=args.check):
            changed.append(os.path.relpath(path, ROOT))

    if args.check:
        for path in changed:
            print(f"needs scrub: {path}")
        return 1 if changed else 0

    for path in changed:
        print(f"scrubbed {path}")
    print(f"{len(changed)} file(s) changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
