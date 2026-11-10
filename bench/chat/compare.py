#!/usr/bin/env python3
"""Cross-engine chat comparison: hipml (device, greedy) vs llama.cpp.

Starts llama-server on 8092, renders every conversation through llama's own
template (/apply-template + /tokenize), generates with the shared prompt ids,
stops the server, then runs hipml on the same ids and diffs the output token
streams and decoded text.

Usage (inside a GPU-lock hold):

    python bench/chat/compare.py --tokens 256

Needs HIPML_GGUF and LLAMA_SERVER (or --model / --llama-server). Outputs JSON
per conversation under bench/out/chat and a summary table.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "bench"))
import _paths  # noqa: E402

CHAT = os.path.join(ROOT, "bench", "chat")
OUT = os.path.join(ROOT, "bench", "out", "chat")
MODEL = None
HIPML = None
LLAMA = None
BASE = "http://127.0.0.1:8092"
CONVS = [
    ("01_single_turn_code", "01_single_turn_code.txt", None),
    ("02_multi_turn", "02_multi_turn.txt", None),
    ("03_system_tool", "03_system_tool.txt", "03_tools.jsonl"),
    ("04_longdoc_qa", "04_longdoc_qa.txt", None),
    ("05_copy", "05_copy.txt", None),
]


def post(path, obj, timeout=1800):
    data = json.dumps(obj).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as f:
            return json.load(f)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        raise RuntimeError(f"POST {path} -> {e.code}: {body[:2000]}")


def get(path, timeout=60):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as f:
        return json.load(f)


# ---------------------------------------------------------------- conversation
def parse_messages(path):
    """Same `--- role` format as crates/hipml-cli/src/chat_cmd.rs."""
    msgs = []
    role, lines = None, []
    for line in open(path, encoding="utf-8").read().splitlines():
        if line.startswith("--- "):
            if role is not None:
                msgs.append(build(role, lines))
            role, lines = line[4:].strip(), []
        else:
            if role is None:
                if line.strip():
                    raise SystemExit(f"{path}: content before a role line")
            else:
                lines.append(line)
    if role is not None:
        msgs.append(build(role, lines))
    return msgs


def build(role, lines):
    if lines and lines[-1] == "":
        lines = lines[:-1]
    content = "\n".join(lines)
    if role == "assistant" and content.startswith("::reasoning\n"):
        reasoning, _, content = content[len("::reasoning\n"):].partition("\n::content\n")
        return {"role": role, "content": content, "reasoning_content": reasoning}
    return {"role": role, "content": content}


def read_tools(path):
    if not path:
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def run_hipml_chat(conv_file, tools_file, out_prefix):
    cmd = [HIPML, "chat", "--gguf", MODEL, "--messages", conv_file,
           "--json", "--out", out_prefix + ".prompt.txt",
           "--ids-out", out_prefix + ".ids.json"]
    if tools_file:
        cmd += ["--tools", tools_file]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(f"hipml chat failed: {r.stderr}")
    return json.loads(r.stdout)


def build_longdoc(target=8192):
    """Pick the longest document prefix whose rendered prompt is <= target."""
    tmpl = os.path.join(CHAT, "04_longdoc_qa.tmpl")
    doc = open(os.path.join(ROOT, "bench", "prompts", "longdoc.txt"),
               encoding="utf-8").read()
    # The source document is ~7.7K tokens; repeat it so an 8K prompt is reachable.
    doc = doc * 4
    text = open(tmpl, encoding="utf-8").read()
    base = text.replace("@@DOC@@", "")
    # Token count is monotone in the prefix length; bisect characters.
    lo, hi = 0, len(doc)
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = text.replace("@@DOC@@", doc[:mid])
        path = os.path.join(OUT, "_longdoc_probe.txt")
        open(path, "w", encoding="utf-8").write(cand)
        info = run_hipml_chat(path, None, os.path.join(OUT, "_longdoc_probe"))
        if len(info["ids"]) <= target:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    final = text.replace("@@DOC@@", doc[:best])
    path = os.path.join(CHAT, "04_longdoc_qa.txt")
    open(path, "w", encoding="utf-8").write(final)
    info = run_hipml_chat(path, None, os.path.join(OUT, "04_longdoc_qa"))
    print(f"longdoc: {best} chars, prompt {len(info['ids'])} tokens (target {target})")
    return path


# ---------------------------------------------------------------------- llama
def start_llama(log_path, kv="q8_0", fa="off"):
    env = _paths.rocm_env()
    env["HIPML_ARCH"] = env.get("HIPML_ARCH", "gfx1201")
    log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [LLAMA, "-m", MODEL, "-c", "16384", "-ngl", "999",
         "-ctk", kv, "-ctv", kv, "-fa", fa, "--host", "127.0.0.1",
         "--port", "8092", "--jinja", "--reasoning-format", "none"],
        stdout=log, stderr=log, env=env)
    for _ in range(180):
        try:
            if get("/health", timeout=2).get("status") == "ok":
                return proc, log
        except Exception:
            pass
        time.sleep(2)
    stop_llama(proc, log)
    raise SystemExit("llama-server did not become healthy")


def stop_llama(proc, log):
    if proc is None:
        return
    if proc.poll() is None:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    try:
        log.close()
    except Exception:
        pass
    time.sleep(2)


def llama_phase(convs, tokens, n_probs):
    results = {}
    for name, message_file, tools_file in convs:
        messages = parse_messages(os.path.join(CHAT, message_file))
        tools = read_tools(os.path.join(CHAT, tools_file) if tools_file else None)
        req = {"messages": messages}
        if tools:
            req["tools"] = tools
        at = post("/apply-template", req)
        prompt = at["prompt"]
        tok = post("/tokenize", {"content": prompt, "add_special": False})
        ids = tok["tokens"]
        hip = run_hipml_chat(os.path.join(CHAT, message_file),
                             os.path.join(CHAT, tools_file) if tools_file else None,
                             os.path.join(OUT, name))
        if hip["text"] != prompt:
            first = next((i for i, (a, b) in enumerate(zip(hip["text"], prompt))
                          if a != b), min(len(hip["text"]), len(prompt)))
            print(f"  TEMPLATE TEXT DIFF at char {first}: "
                  f"hipml={hip['text'][first:first+40]!r} "
                  f"llama={prompt[first:first+40]!r}")
        if hip["ids"] != ids:
            first = next((i for i, (a, b) in enumerate(zip(hip["ids"], ids))
                          if a != b), min(len(hip["ids"]), len(ids)))
            print(f"  TEMPLATE ID DIFF at {first}: "
                  f"hipml={hip['ids'][first:first+8]} llama={ids[first:first+8]}")
        else:
            print(f"  template ids: {len(ids)} match")
        comp = post("/completion", {
            "prompt": ids, "n_predict": tokens, "temperature": 0.0,
            "top_k": 1, "top_p": 1.0, "min_p": 0.0, "repeat_penalty": 1.0,
            "ignore_eos": False, "cache_prompt": False, "n_probs": n_probs,
        })
        cp = comp.get("completion_probabilities") or []
        out_ids = [p["id"] for p in cp]
        results[name] = {
            "prompt_tokens": len(ids),
            "prompt_sha": ids[:16],
            "llama_ids": out_ids,
            "llama_text": comp.get("content", ""),
            "llama_stop": comp.get("stop"), "llama_stop_type": comp.get("stop_type"),
            "llama_tokens_predicted": comp.get("tokens_predicted"),
            "llama_tokens_evaluated": comp.get("tokens_evaluated"),
            "llama_probs": cp,
            "llama_eos": 248046 in out_ids,
            "llama_timings": comp.get("timings"),
        }
        json.dump(results[name], open(os.path.join(OUT, name + ".llama.json"), "w"), indent=1)
        print(f"  llama: {len(out_ids)} tokens, eos={results[name]['llama_eos']}, "
              f"stop_type={comp.get('stop_type')}")
    return results


# ---------------------------------------------------------------------- hipml
def hipml_phase(convs, tokens, kv, chunk):
    results = {}
    for name, _message, _tools in convs:
        ids_path = os.path.join(OUT, name + ".ids.json")
        info = json.load(open(ids_path))
        n_ctx = len(info["ids"]) + tokens + 32
        cmd = [HIPML, "generate", "--device", "--gguf", MODEL,
               "--ids-file", ids_path, "--tokens", str(tokens),
               "--ctx", str(n_ctx), "--kv", kv, "--stop-eos", "--json"]
        if chunk is not None:
            cmd += ["--chunk", str(chunk)]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        if r.returncode != 0:
            raise RuntimeError(f"hipml generate {name} failed: {r.stderr[-3000:]}")
        row = json.loads(r.stdout)
        rep = row["reps"][-1]
        results[name] = {
            "hipml_ids": rep["ids"],
            "hipml_text": row["text"],
            "hipml_stopped": rep["stopped"],
            "hipml_prefill_ms": rep["prefill_ms"],
            "hipml_decode_ms": rep["decode_ms"],
            "hipml_decode_tok_s": rep["decode_tok_s"],
            "hipml_timings": row["reps"],
            "prompt_tokens": row["prompt_tokens"],
        }
        json.dump(results[name], open(os.path.join(OUT, name + ".hipml.json"), "w"), indent=1)
        print(f"  hipml: {len(rep['ids'])} tokens, stopped={rep['stopped']}, "
              f"{rep['decode_tok_s']:.2f} tok/s")
    return results


# --------------------------------------------------------------------- compare
def token_text(tokenizer_ids, ids, upto):
    pass


def compare(convs, llama, hipml):
    print("\n=== cross-engine output comparison ===")
    same = 0
    summary = {}
    for name, _m, _t in convs:
        a = llama[name]["llama_ids"]
        b = hipml[name]["hipml_ids"]
        n = min(len(a), len(b))
        first = next((i for i in range(n) if a[i] != b[i]), None)
        if first is None and len(a) == len(b) and a == b:
            same += 1
            print(f"{name}: IDENTICAL {len(a)} tokens (eos={llama[name]['llama_eos']})")
            summary[name] = {"identical": True, "tokens": len(a),
                             "text_equal": llama[name]["llama_text"] == hipml[name]["hipml_text"]}
            continue
        pos = first if first is not None else n
        lo = a[pos] if pos < len(a) else None
        hi = b[pos] if pos < len(b) else None
        margin = None
        probs = llama[name]["llama_probs"]
        if pos < len(probs):
            tops = probs[pos].get("top_logprobs", [])
            by_id = {p["id"]: p["logprob"] for p in tops}
            if lo in by_id and hi in by_id:
                margin = by_id[lo] - by_id[hi]
        print(f"{name}: DIVERGE at {pos} (llama {lo} / hipml {hi}), "
              f"llama_len={len(a)} hipml_len={len(b)}, "
              f"margin={margin if margin is None else round(margin, 4)} nats")
        # Context around the divergence.
        print(f"  llama ctx: {a[max(0,pos-4):pos+4]}")
        print(f"  hipml ctx: {b[max(0,pos-4):pos+4]}")
        print(f"  text_equal={llama[name]['llama_text'] == hipml[name]['hipml_text']}")
        summary[name] = {"identical": False, "first_diff": pos,
                         "llama_id": lo, "hipml_id": hi, "margin": margin,
                         "llama_len": len(a), "hipml_len": len(b),
                         "text_equal": llama[name]["llama_text"] == hipml[name]["hipml_text"]}
        if pos < len(probs):
            tops = probs[pos].get("top_logprobs", [])[:6]
            print("  llama top: " +
                  " ".join(f"{p['id']}:{p['logprob']:.3f}" for p in tops))
    print(f"\n{same}/{len(convs)} conversations token-identical")
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    return same


def main():
    ap = argparse.ArgumentParser(
        description="Compare hipml and llama.cpp chat output token for token. "
                    "Paths come from HIPML_GGUF / LLAMA_SERVER.")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--kv", default="q8")
    ap.add_argument("--llama-kv", default="q8_0")
    ap.add_argument("--llama-fa", default="off")
    ap.add_argument("--hipml-chunk", type=int, default=None)
    ap.add_argument("--n-probs", type=int, default=32)
    ap.add_argument("--model", default=None,
                    help="target GGUF (default: $HIPML_GGUF)")
    ap.add_argument("--llama-server", default=None,
                    help="llama.cpp server binary (default: $LLAMA_SERVER)")
    ap.add_argument("--hipml", default=None,
                    help="hipml binary (default: $HIPML_EXE or target/release/hipml)")
    ap.add_argument("--out", default=None,
                    help="output directory (default: $HIPML_BENCH_OUT or "
                         "bench/out/chat)")
    args = ap.parse_args()
    global OUT, MODEL, HIPML, LLAMA
    OUT = _paths.out_dir("chat", args.out)
    MODEL = _paths.model(args.model)
    HIPML = _paths.hipml_exe(args.hipml)
    LLAMA = _paths.llama_server(args.llama_server)
    build_longdoc(8192)
    convs = CONVS
    proc = log = None
    try:
        print("starting llama-server on 8092")
        proc, log = start_llama(os.path.join(OUT, "llama.log"), args.llama_kv, args.llama_fa)
        llama = llama_phase(convs, args.tokens, args.n_probs)
    finally:
        stop_llama(proc, log)
    print("llama stopped; running hipml")
    hipml = hipml_phase(convs, args.tokens, args.kv, args.hipml_chunk)
    same = compare(convs, llama, hipml)
    return 0 if same >= 3 else 1


if __name__ == "__main__":
    sys.exit(main())
