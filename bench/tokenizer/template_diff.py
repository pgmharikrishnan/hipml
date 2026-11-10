#!/usr/bin/env python3
"""Chat-template branch verification: hipml vs llama-server /apply-template.

Each case is the exact request body llama accepts (messages, tools,
chat_template_kwargs, add_generation_prompt). The same bytes go to
/apply-template and to `hipml chat --json-file`, then the rendered text and
the token ids are compared. Cases that must be rejected (image/video parts)
assert a clear error on the hipml side.

Usage: python3 bench/tokenizer/template_diff.py --gguf MODEL [--save fixture]
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "bench"))
import _paths  # noqa: E402

BASE = "http://127.0.0.1:8092"
HIPML = None
WS = "Get the current weather for a city."

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}]


def tool_msg(content, arguments, name="get_weather", call_type="function"):
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [{
            "id": "call_1",
            "type": call_type,
            "function": {"name": name, "arguments": arguments},
        }],
    }


CASES = {
    "text-single": {"messages": [{"role": "user", "content": "hi"}]},
    "text-system": {"messages": [
        {"role": "system", "content": "  be terse  "},
        {"role": "user", "content": "hi"}]},
    "text-parts": {"messages": [
        {"role": "user", "content": [
            {"type": "text", "text": "hello "},
            {"type": "text", "text": "world"}]}]},
    "text-multiturn-reasoning": {"messages": [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "reasoning_content": "thinking1"},
        {"role": "user", "content": "q2"}]},
    "tools-schema": {"tools": TOOLS, "messages": [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "weather in Paris?"}]},
    "tools-unicode-schema": {"tools": [{
        "type": "function",
        "function": {"name": "caf\u00e9", "description": "\u5929\u6c14 \U0001F326\uFE0F",
                     "parameters": {"type": "object", "properties": {
                         "ville": {"type": "string", "description": "\u57ce\u5e02"}}}}}],
        "messages": [{"role": "user", "content": "q"}]},
    "toolcall-args-string": {"messages": [
        {"role": "user", "content": "hi"},
        tool_msg("", '{"city":"Paris","unit":"celsius"}'),
        {"role": "tool", "content": "18C"}]},
    "toolcall-args-object": {"messages": [
        {"role": "user", "content": "hi"},
        tool_msg("", {"city": "Paris", "unit": "celsius", "days": 3,
                      "ratio": 0.5, "opts": {"hourly": True}}),
        {"role": "tool", "content": "18C"}]},
    "toolcall-args-escapes": {"messages": [
        {"role": "user", "content": "hi"},
        tool_msg("checking", {
            "s": "line1\nline2 \"q\" \\ x\ttab",
            "n": 1,
            "neg": -2,
            "f": 1.0,
            "exp": 1e3,
            "tiny": 1e-7,
            "b": True,
            "nil": None,
            "arr": [1, "x", False, None],
            "obj": {"k": "v", "u": "h\u00e9llo \u4e16\u754c \U0001F600"},
            "ctrl": "\u0001\u001f",
            "sep": "a\u2028b\u2029c",
        }),
        {"role": "tool", "content": "ok"}]},
    "toolcall-two": {"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "two calls", "tool_calls": [
            {"id": "a", "type": "function", "function": {
                "name": "f", "arguments": '{"x":1}'}},
            {"id": "b", "type": "function", "function": {
                "name": "g", "arguments": "{}"}}]},
        {"role": "tool", "content": "1"},
        {"role": "tool", "content": "2"}]},
    "toolcall-empty-content-no-args": {"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "type": "function",
             "function": {"name": "ping", "arguments": "{}"}}]},
        {"role": "tool", "content": "pong"}]},
    "preserve-thinking-false": {
        "chat_template_kwargs": {"preserve_thinking": False},
        "messages": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1", "reasoning_content": "secret"},
            {"role": "user", "content": "q2"}]},
    "preserve-thinking-true": {
        "chat_template_kwargs": {"preserve_thinking": True},
        "messages": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1", "reasoning_content": "secret"},
            {"role": "user", "content": "q2"}]},
    "thinking-off": {"chat_template_kwargs": {"enable_thinking": False},
                     "messages": [{"role": "user", "content": "hi"}]},
    "effort-low": {"chat_template_kwargs": {"reasoning_effort": "low"},
                   "messages": [{"role": "user", "content": "hi"}]},
    "effort-medium": {"chat_template_kwargs": {"reasoning_effort": "medium"},
                      "messages": [{"role": "user", "content": "hi"}]},
    "image-rejected": {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "what is this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}]},
    "video-rejected": {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "what is this"},
        {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AA"}}]}]},
}


def post(path, obj, timeout=120):
    data = json.dumps(obj).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as f:
            return True, json.load(f)
    except urllib.error.HTTPError as e:
        return False, e.read().decode("utf-8", "replace")


def hipml(gguf, body, out_prefix):
    path = out_prefix + ".request.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(body, f, ensure_ascii=False)
    cmd = [HIPML, "chat", "--gguf", gguf, "--json-file", path, "--json",
           "--out", out_prefix + ".prompt.txt", "--ids-out", out_prefix + ".ids.json"]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        return False, r.stderr.strip()
    return True, json.loads(r.stdout)


def main():
    global BASE, HIPML
    ap = argparse.ArgumentParser(
        description="Diff hipml's chat template against llama-server's "
                    "/apply-template. Needs HIPML_GGUF or --gguf.")
    ap.add_argument("--base", default=os.environ.get("HIPML_BASE_URL", BASE),
                    help="llama-server base URL (default: %(default)s)")
    ap.add_argument("--gguf", default=os.environ.get("HIPML_GGUF"),
                    help="target GGUF (default: $HIPML_GGUF)")
    ap.add_argument("--hipml", default=None,
                    help="hipml binary (default: $HIPML_EXE or target/release/hipml)")
    ap.add_argument("--out", default=os.path.join(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "out", "template"),
                    help="output directory (default: bench/out/template)")
    ap.add_argument("--save", help="write the permanent fixture here")
    args = ap.parse_args()
    if not args.gguf:
        raise SystemExit("error: set HIPML_GGUF or pass --gguf")
    HIPML = _paths.hipml_exe(args.hipml)
    BASE = args.base.rstrip("/")
    os.makedirs(args.out, exist_ok=True)

    failures = 0
    fixture = []
    print(f"{'case':32s} {'bytes':>6s} {'ids':>5s}  result")
    for name, body in CASES.items():
        ok, ref = post("/apply-template", body)
        rejected = name.endswith("-rejected")
        hok, hip = hipml(args.gguf, body, os.path.join(args.out, name))
        if rejected:
            good = (not hok) and ("not supported" in hip)
            print(f"{name:32s} {'-':>6s} {'-':>5s}  "
                  + ("rejected both" if good else f"FAIL hipml={hip!r} llama_ok={ok}"))
            failures += 0 if good else 1
            continue
        if not ok:
            print(f"{name:32s}  llama rejected: {ref[:160]}")
            failures += 1
            continue
        prompt = ref["prompt"]
        tok = post("/tokenize", {"content": prompt, "add_special": False})[1]
        ids = tok["tokens"]
        if not hok:
            print(f"{name:32s} {'-':>6s} {'-':>5s}  FAIL hipml: {hip[:200]}")
            failures += 1
            continue
        text_ok = hip["text"] == prompt
        ids_ok = hip["ids"] == ids
        pos = next((i for i, (a, b) in enumerate(zip(hip["ids"], ids)) if a != b),
                   min(len(hip["ids"]), len(ids)))
        state = "match" if text_ok and ids_ok else f"DIFF at id {pos}"
        print(f"{name:32s} {len(prompt.encode()):6d} {len(ids):5d}  {state}")
        if not (text_ok and ids_ok):
            failures += 1
            if not text_ok:
                c = next((i for i, (a, b) in enumerate(zip(hip["text"], prompt))
                          if a != b), min(len(hip["text"]), len(prompt)))
                print(f"    text char {c}: hipml={hip['text'][c:c+60]!r}")
                print(f"                    llama={prompt[c:c+60]!r}")
            print(f"    hipml={hip['ids'][pos:pos+8]} llama={ids[pos:pos+8]}")
        if args.save:
            fixture.append((name, body, ids))
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            for name, body, ids in fixture:
                h = json.dumps(body, ensure_ascii=False,
                               separators=(",", ":")).encode().hex()
                f.write(f"{h}\t{name}\t{json.dumps(ids)}\n")
        print(f"wrote {args.save} ({len(fixture)} cases)")
    print(f"\n{len(CASES) - failures}/{len(CASES)} cases pass")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
