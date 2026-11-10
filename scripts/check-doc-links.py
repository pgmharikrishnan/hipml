#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check that every relative Markdown link points at something in the tree.

Scope, by default:

    README.md
    docs/**/*.md

Two areas are excluded on purpose:

    docs/reference/     verbatim llama.cpp sources, whose README links upstream
    docs/internal-notes.md  the internal internal notes, never published

Every link target is resolved from the directory of the file that holds it.
Links to `http:`, `https:`, `mailto:` and absolute paths are skipped; the
`#fragment` and `?query` parts are dropped, so this checks files, not anchors.

    python3 scripts/check-doc-links.py          # CI: exit 1 on any broken link
    python3 scripts/check-doc-links.py --fix    # rewrite case-only mismatches

`--fix` only rewrites a link when the target resolves on a case-insensitive
filesystem but not on a case-sensitive one (a link written on Windows or macOS
for a file spelled differently). It never invents a target, so it cannot repair
a link to a file that does not exist. CI does not use it.

No third-party imports: this must run on a bare `ubuntu-latest` with python3.
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Directories and files the check does not walk. See the module docstring.
SKIP_PREFIXES = ("docs/reference/", "docs/internal-notes.md", "target/")

# An inline link or image: [text](target) / ![alt](target "title") / [a](<a b>).
INLINE = re.compile(r"(!?\[[^\]\n]*\]\()(<[^>\n]+>|[^)\s]+)((?:\s+\"[^\"]*\")?\))")
# A reference definition: [id]: target
REFDEF = re.compile(r"^(\s{0,3}\[[^\]\n]+\]:\s*)(\S+)")
FENCE = re.compile(r"^\s{0,3}(?:```|~~~)")


def is_external(target: str) -> bool:
    return bool(
        re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target) or target.startswith("#")
    )


def split_target(target: str) -> tuple[str, str]:
    """Return (path part, trailing suffix) where suffix is '#frag' or '?q'."""
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    cut = len(target)
    for ch in "#?":
        i = target.find(ch)
        if i != -1:
            cut = min(cut, i)
    return target[:cut], target[cut:]


def files_to_check(explicit: list[str]) -> list[Path]:
    if explicit:
        return [Path(p) for p in explicit]
    paths = []
    readme = ROOT / "README.md"
    if readme.is_file():
        paths.append(readme)
    paths.extend(sorted((ROOT / "docs").rglob("*.md")))
    return paths


def rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def scan(path: Path) -> list[dict]:
    """Return every link found in `path`, with its line number and validity."""
    text = path.read_text(encoding="utf-8")
    in_fence = False
    out = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for m in INLINE.finditer(line):
            out.append({"line": lineno, "target": m.group(2), "span": m.span(2)})
        m = REFDEF.match(line)
        if m:
            out.append({"line": lineno, "target": m.group(2), "span": m.span(2)})
    for link in out:
        link["resolved"] = resolve(path.parent.resolve(), link["target"])
    return out


def resolve(base: Path, target: str) -> dict:
    """Resolve a link target from `base`, comparing names byte for byte.

    The comparison is done against directory entries rather than `Path.exists`
    so the answer is the same on a case-insensitive filesystem (macOS, Windows)
    as it is on Ubuntu, where CI runs.
    """
    if is_external(target):
        return {"kind": "external"}
    path_part, suffix = split_target(target)
    path_part = urllib.parse.unquote(path_part)
    if path_part == "":
        return {"kind": "external"}
    if path_part.startswith("/"):
        return {"kind": "absolute"}

    cur = base
    mismatch: list[str] = []
    parts = [p for p in path_part.split("/") if p not in ("", ".")]
    for part in parts:
        if part == "..":
            cur = cur.parent
            continue
        if not cur.is_dir():
            return {"kind": "broken", "suffix": suffix, "fix": None}
        exact = None
        loose = None
        for entry in cur.iterdir():
            if entry.name == part:
                exact = entry
                break
            if loose is None and entry.name.lower() == part.lower():
                loose = entry
        if exact is not None:
            cur = exact
        elif loose is not None:
            mismatch.append(loose.name)
            cur = loose
        else:
            return {"kind": "broken", "suffix": suffix, "fix": None}

    if not mismatch:
        return {"kind": "ok", "path": cur}
    # The files only resolve because this filesystem folds case: CI would fail.
    return {"kind": "broken", "suffix": suffix, "fix": cur, "case_only": True}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", help="Markdown files (default: README.md docs/**/*.md)")
    ap.add_argument("--fix", action="store_true", help="rewrite case-only mismatches")
    args = ap.parse_args(argv)

    broken = 0
    fixed = 0
    checked_files = 0
    checked_links = 0

    for path in files_to_check(args.paths):
        name = rel(path)
        if any(name == p or name.startswith(p) for p in SKIP_PREFIXES):
            continue
        checked_files += 1
        links = scan(path)
        bad = [l for l in links if l["resolved"]["kind"] == "broken"]
        checked_links += len(links) - sum(
            1 for l in links if l["resolved"]["kind"] in ("external", "absolute")
        )
        if not bad:
            continue

        lines = path.read_text(encoding="utf-8").splitlines()
        changed = False
        for link in bad:
            where = f"{name}:{link['line']}"
            fix = link["resolved"]["fix"]
            if args.fix and fix is not None and fix.exists():
                new_target = urllib.parse.quote(fix.relative_to(path.parent.resolve()).as_posix())
                new_target += link["resolved"].get("suffix", "")
                start, end = link["span"]
                lines[link["line"] - 1] = (
                    lines[link["line"] - 1][:start] + new_target + lines[link["line"] - 1][end:]
                )
                print(f"FIXED {where}: {link['target']} -> {new_target}")
                fixed += 1
                changed = True
            else:
                print(f"BROKEN {where}: {link['target']}")
                if link["resolved"].get("case_only"):
                    print(f"       wrong case: the file in the tree is {rel(fix)}")
                else:
                    part = split_target(link["target"])[0]
                    missing = (path.parent / urllib.parse.unquote(part)).resolve()
                    print(f"       no such file: {rel(missing)}")
                broken += 1
        if changed:
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if broken:
        print(f"\nFAIL: {broken} broken link(s) in {checked_files} files.", file=sys.stderr)
        return 1
    if fixed:
        print(f"OK: rewrote {fixed} case-only link(s); everything else resolves.")
    else:
        print(f"OK: {checked_links} relative links in {checked_files} files resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
