"""REVIEW-4 server probes: exactness A/B, malformed input, and user stress.

Stdlib only. Starts `hipml serve` on 127.0.0.1:8093 (never 8080/8090/8091/
8092), runs one named plan against it, stops it, and appends every result
to a JSON-lines file. The caller holds the GPU lock.

    python bench/serve/review4_probe.py --hipml PATH --plan exact --drafts 3 \
        --ctx 512 --slots 2 --out out.jsonl --tag fixed_d3
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import _paths  # noqa: E402

PORT = 8093
MODEL = None
MTP = None
FIXTURE = os.path.join(os.path.dirname(HERE), "fixtures", "p2_math_169.ids")


def fixture_ids():
    with open(FIXTURE, encoding="utf-8") as f:
        return json.load(f)["ids"]


def ids_of_len(n, salt=0):
    """`n` ids built from the fixture, rotated by `salt` so two prompts of the
    same length do not share a prefix."""
    base = fixture_ids()
    out = []
    i = salt * 37
    while len(out) < n:
        out.append(base[i % len(base)])
        i += 1
    return out


# --- server lifecycle -------------------------------------------------------

def port_open():
    s = socket.socket()
    s.settimeout(0.5)
    try:
        return s.connect_ex(("127.0.0.1", PORT)) == 0
    finally:
        s.close()


def start(hipml, ctx, slots, drafts, log_path, extra_env=None):
    if port_open():
        raise RuntimeError(f"port {PORT} is already in use")
    env = _paths.rocm_env()
    env["HIPML_ARCH"] = env.get("HIPML_ARCH", "gfx1201")
    env.update(extra_env or {})
    cmd = [hipml, "serve", "--gguf", MODEL, "--port", str(PORT),
           "--ctx", str(ctx), "--slots", str(slots), "--chunk", "256",
           "--drafts", str(drafts)]
    if drafts > 0:
        cmd += ["--mtp", MTP]
    log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    t0 = time.time()
    while time.time() - t0 < 300:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited {proc.returncode} during load")
        if health():
            return proc, log, time.time() - t0
        time.sleep(1.0)
    proc.kill()
    raise RuntimeError("server not healthy after 300 s")


def stop(proc, log):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    log.close()


def health():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


# --- requests ---------------------------------------------------------------

def post(path, body, timeout=600):
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read()), time.time() - t0
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw), time.time() - t0
        except Exception:
            return e.code, raw.decode(errors="replace"), time.time() - t0
    except Exception as e:
        return None, repr(e), time.time() - t0


def raw(payload, read_timeout=30):
    """Send raw bytes; return (response bytes or error, server still healthy)."""
    s = socket.create_connection(("127.0.0.1", PORT), timeout=read_timeout)
    try:
        s.sendall(payload)
        got = b""
        while True:
            try:
                b = s.recv(65536)
            except Exception as e:
                got += f"<recv error {e!r}>".encode()
                break
            if not b:
                break
            got += b
    finally:
        s.close()
    time.sleep(1.0)
    return got[:300].decode(errors="replace"), health()


def http_post_bytes(path, body):
    return (f"POST {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n\r\n").encode() + body


def metrics():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=10) as r:
            text = r.read().decode()
    except Exception as e:
        return {"error": repr(e)}
    out = {}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            k, _, v = line.partition(" ")
            try:
                out[k] = float(v)
            except ValueError:
                pass
    return out


def completion(ids, n, **kw):
    body = {"prompt": ids, "n_predict": n, "ignore_eos": True, "cache_prompt": False}
    body.update(kw)
    return post("/completion", body)


def stream_open(body, path="/completion"):
    """Open a streaming request and return the socket after the headers."""
    data = json.dumps(dict(body, stream=True)).encode()
    s = socket.create_connection(("127.0.0.1", PORT), timeout=600)
    s.sendall(http_post_bytes(path, data))
    return s


def read_events(s, k, timeout=600):
    """Read until `k` SSE `data:` events have arrived (or EOF)."""
    s.settimeout(timeout)
    buf = b""
    while buf.count(b"data:") < k:
        b = s.recv(65536)
        if not b:
            break
        buf += b
    return buf.count(b"data:")


def wait_idle(timeout=60):
    """Seconds until the server reports no active slot and an empty queue."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        m = metrics()
        if m.get("hipml_active_slots", 1) == 0 and m.get("hipml_queue_depth", 1) == 0:
            return time.time() - t0
        time.sleep(0.25)
    return None


# --- plans ------------------------------------------------------------------

def plan_exact(a, rec):
    """Greedy ids for fixed prompts; compared across servers offline."""
    p = fixture_ids()
    st, js, dt = completion(p, 128)
    rec("exact_169x128", status=st, ids=js.get("tokens") if isinstance(js, dict) else js,
        secs=dt)
    # Fill the cache to the last row: prompt + n_predict == ctx.
    n = a.ctx - len(p)
    st, js, dt = completion(p, n)
    rec("exact_fill_ctx", status=st, n=n,
        ids=js.get("tokens") if isinstance(js, dict) else js, secs=dt, healthy=health())


def plan_crash(a, rec):
    """Malformed inputs that used to reach a panic or the device."""
    deep = b"[" * 200000
    resp, ok = raw(http_post_bytes("/completion", deep))
    rec("deep_json", response=resp, healthy_after=ok)
    if not ok:
        return
    st, js, _ = completion([10**7], 4)
    rec("id_past_vocab", status=st, body=js, healthy_after=health())
    st, js, _ = completion(fixture_ids(), 4, n_predict=a.ctx)
    rec("prompt_plus_max_over_ctx", status=st, body=str(js)[:300], healthy_after=health())
    st, js, _ = completion(ids_of_len(a.ctx + 10), 4)
    rec("prompt_over_ctx", status=st, body=str(js)[:300], healthy_after=health())
    resp, ok = raw(b"POST /completion HTTP/1.1\r\nContent-Length: 99999999999999999999\r\n\r\n")
    rec("huge_content_length", response=resp, healthy_after=ok)
    resp, ok = raw(b"GET /" + b"a" * 100000 + b" HTTP/1.1\r\n\r\n")
    rec("long_request_line", response=resp, healthy_after=ok)
    body = json.dumps({"prompt": "hi", "n_predict": 2}).encode()
    resp, ok = raw(b"POST /completion HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n"
                   + f"{len(body):x}\r\n".encode() + body + b"\r\n0\r\n\r\n")
    rec("chunked_body", response=resp, healthy_after=ok)
    resp, ok = raw(http_post_bytes("/completion", b'{"prompt":"\\ud800x"}'))
    rec("lone_surrogate", response=resp, healthy_after=ok)


def plan_stress(a, rec):
    p = fixture_ids()
    base = metrics()

    # max_tokens exactly, on both APIs, and 1.
    for n in (1, 17):
        st, js, _ = completion(p, n)
        rec(f"max_exact_completion_{n}", status=st,
            got=len(js.get("tokens", [])) if isinstance(js, dict) else js,
            stop=js.get("stop") if isinstance(js, dict) else None)
    st, js, _ = post("/v1/chat/completions", {
        "messages": [{"role": "user", "content": "Count from 1 to 100."}],
        "max_tokens": 9})
    rec("max_exact_chat_9", status=st,
        usage=js.get("usage") if isinstance(js, dict) else js,
        finish=js["choices"][0]["finish_reason"] if isinstance(js, dict) and "choices" in js else None)

    # Multi-token stop string: generate once, then stop on a later substring.
    st, js, _ = completion(p, 48)
    text = js.get("content", "") if isinstance(js, dict) else ""
    cut = text[len(text) // 2:][:12].strip()
    if cut:
        st2, js2, _ = completion(p, 48, stop=[cut])
        c2 = js2.get("content", "") if isinstance(js2, dict) else ""
        rec("stop_string", stop=cut, full_len=len(text), got=c2,
            ok=(c2 == text[:text.find(cut)]), stop_flag=js2.get("stop"))
    else:
        rec("stop_string", error="empty generation", text=text)

    # Client disconnects mid-stream.
    m0 = metrics()
    s = stream_open({"prompt": p, "n_predict": 3000, "ignore_eos": True})
    got = read_events(s, 5)
    s.close()
    t = wait_idle(30)
    m1 = metrics()
    rec("disconnect_mid_stream", events_before_close=got, idle_after_s=t,
        cancelled=m1.get("hipml_requests_cancelled_total", 0) - m0.get("hipml_requests_cancelled_total", 0))

    # Non-streaming client gives up after 2 s.
    m0 = metrics()
    s = socket.create_connection(("127.0.0.1", PORT))
    s.sendall(http_post_bytes("/completion", json.dumps(
        {"prompt": p, "n_predict": 3000, "ignore_eos": True}).encode()))
    time.sleep(2.0)
    s.close()
    t = wait_idle(30)
    m1 = metrics()
    rec("disconnect_non_stream", idle_after_s=t,
        cancelled=m1.get("hipml_requests_cancelled_total", 0) - m0.get("hipml_requests_cancelled_total", 0))

    # Cancel during a long prefill; a short request queued behind must not
    # wait for the abandoned prefill to finish.
    long_n = min(12000, a.ctx - 64)
    m0 = metrics()
    s = stream_open({"prompt": ids_of_len(long_n, 1), "n_predict": 8, "ignore_eos": True})
    time.sleep(2.0)
    s.close()
    st, js, dt = completion(p, 8)
    m1 = metrics()
    rec("cancel_long_prefill", long_tokens=long_n, short_status=st, short_latency_s=dt,
        cancelled=m1.get("hipml_requests_cancelled_total", 0) - m0.get("hipml_requests_cancelled_total", 0))
    wait_idle(60)

    # Solo references for the concurrent cells.
    lens = [120, 400, 900, 1500, 2200]
    solo = {}
    for i, n in enumerate(lens):
        st, js, dt = completion(ids_of_len(n, i + 2), 32)
        solo[n] = js.get("tokens") if isinstance(js, dict) else None

    # Two identical prompts at the same instant (cache on), then the same
    # prefix with different tails.
    shared_sys = "You are a careful reviewer. " * 120
    def chat(user):
        return {"messages": [{"role": "system", "content": shared_sys},
                             {"role": "user", "content": user}],
                "max_tokens": 24, "temperature": 0}
    out = {}
    def go(k, body):
        out[k] = post("/v1/chat/completions", body)
    for label, users in (("same_prompt", ["Say hello.", "Say hello."]),
                         ("same_prefix", ["Say hello.", "Name a colour."])):
        m0 = metrics()
        ts = [threading.Thread(target=go, args=(i, chat(u))) for i, u in enumerate(users)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        texts = [out[i][1]["choices"][0]["message"]["content"] if out[i][0] == 200 else str(out[i][1])[:200]
                 for i in range(2)]
        solo_texts = []
        for u in users:
            st, js, _ = post("/v1/chat/completions", chat(u))
            solo_texts.append(js["choices"][0]["message"]["content"] if st == 200 else str(js)[:200])
        m1 = metrics()
        rec(label, statuses=[out[i][0] for i in range(2)], match_solo=[texts[i] == solo_texts[i] for i in range(2)],
            texts=texts, solo=solo_texts,
            prefix_hits=m1.get("hipml_prefix_hits_total", 0) - m0.get("hipml_prefix_hits_total", 0))

    # Mixed lengths concurrently plus one very long prompt.
    res = {}
    def one(k, ids, n):
        res[k] = completion(ids, n)
    jobs = [(n, ids_of_len(n, i + 2), 32) for i, n in enumerate(lens)]
    jobs.append(("long", ids_of_len(long_n, 1), 32))
    ts = [threading.Thread(target=one, args=j) for j in jobs]
    t0 = time.time()
    [t.start() for t in ts]
    [t.join() for t in ts]
    cells = {}
    for k, _, _ in jobs:
        st, js, dt = res[k]
        ids = js.get("tokens") if isinstance(js, dict) else None
        cells[str(k)] = {"status": st, "secs": round(dt, 2), "n": len(ids or []),
                         "match_solo": (ids == solo.get(k)) if k in solo else None}
    rec("mixed_concurrent", wall_s=time.time() - t0, cells=cells, healthy_after=health())

    # Many-message conversation: CPU time spent cutting the prompt.
    for n_msgs in (20, 200):
        msgs = [{"role": "system", "content": "Be brief."}]
        for i in range(n_msgs):
            msgs.append({"role": "user" if i % 2 == 0 else "assistant",
                         "content": f"turn {i}: please consider item {i * 13} and reply."})
        if msgs[-1]["role"] != "user":
            msgs.append({"role": "user", "content": "done?"})
        st, js, dt = post("/v1/chat/completions", {"messages": msgs, "max_tokens": 1})
        rec(f"chat_{n_msgs}_messages", status=st, secs=dt,
            prompt_tokens=js.get("usage", {}).get("prompt_tokens") if isinstance(js, dict) else None)

    # The known split-scratch length past the 8K gate.
    for n in (8193, 8300):
        if n + 8 <= a.ctx:
            st, js, dt = completion(ids_of_len(n, 3), 4)
            rec(f"prompt_{n}", status=st, body=str(js)[:200], healthy_after=health())
            if not health():
                return

    end = metrics()
    rec("metrics_delta", **{k: end.get(k, 0) - base.get(k, 0) for k in end
                            if k.endswith("_total")})
    rec("metrics_end", **{k: v for k, v in end.items() if "active" in k or "queue" in k})


def plan_recheck(a, rec):
    """The three hold-3 regressions, after their fixes."""
    p = fixture_ids()
    completion(p, 8)  # warm: the first request pays one-time kernel setup
    m0 = metrics()
    s = socket.create_connection(("127.0.0.1", PORT))
    s.sendall(http_post_bytes("/completion", json.dumps(
        {"prompt": p, "n_predict": 3000, "ignore_eos": True}).encode()))
    time.sleep(2.0)
    s.close()
    t = wait_idle(30)
    m1 = metrics()
    rec("disconnect_non_stream", idle_after_s=t,
        cancelled=m1.get("hipml_requests_cancelled_total", 0) - m0.get("hipml_requests_cancelled_total", 0))

    shared_sys = "You are a careful reviewer. " * 120
    for user in ("Say hello.", "Name a colour."):
        m0 = metrics()
        st, js, dt = post("/v1/chat/completions", {
            "messages": [{"role": "system", "content": shared_sys},
                         {"role": "user", "content": user}], "max_tokens": 8})
        m1 = metrics()
        rec("shared_system_" + user.split()[0].lower(), status=st, secs=dt,
            hits=m1.get("hipml_prefix_hits_total", 0) - m0.get("hipml_prefix_hits_total", 0),
            hit_tokens=m1.get("hipml_prefix_hit_tokens_total", 0) - m0.get("hipml_prefix_hit_tokens_total", 0))

    for n_msgs in (20, 200):
        msgs = [{"role": "system", "content": "Be brief."}]
        for i in range(n_msgs):
            msgs.append({"role": "user" if i % 2 == 0 else "assistant",
                         "content": f"turn {i}: please consider item {i * 13} and reply."})
        if msgs[-1]["role"] != "user":
            msgs.append({"role": "user", "content": "done?"})
        m0 = metrics()
        st, js, dt = post("/v1/chat/completions", {"messages": msgs, "max_tokens": 1})
        m1 = metrics()
        rec(f"chat_{n_msgs}_messages", status=st, secs=dt,
            prompt_tokens=js.get("usage", {}).get("prompt_tokens") if isinstance(js, dict) else None,
            captures=m1.get("hipml_prefix_captures_total", 0) - m0.get("hipml_prefix_captures_total", 0))
    rec("metrics_end", **{k: v for k, v in metrics().items() if "prompt_tokens" in k or "requests" in k})


PLANS = {"exact": plan_exact, "crash": plan_crash, "stress": plan_stress,
         "recheck": plan_recheck}


def main():
    global MODEL, MTP
    ap = argparse.ArgumentParser(
        description="Server audit probes. Paths come from HIPML_GGUF / "
                    "HIPML_MTP_GGUF or the flags below.")
    ap.add_argument("--hipml", default=None,
                    help="hipml binary (default: $HIPML_EXE or target/release/hipml)")
    ap.add_argument("--model", default=None,
                    help="target GGUF (default: $HIPML_GGUF)")
    ap.add_argument("--mtp", default=None,
                    help="MTP drafter GGUF (default: $HIPML_MTP_GGUF)")
    ap.add_argument("--plan", required=True, choices=sorted(PLANS))
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--drafts", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE for the server")
    a = ap.parse_args()
    a.hipml = _paths.hipml_exe(a.hipml)
    MODEL = _paths.model(a.model)
    MTP = _paths.mtp_model(a.mtp)
    if a.drafts > 0 and not MTP:
        sys.exit("error: --drafts needs the MTP drafter; pass --mtp or set HIPML_MTP_GGUF")

    extra = dict(kv.split("=", 1) for kv in a.env)
    log_path = a.out + f".{a.tag}.server.log"
    out = open(a.out, "a", encoding="utf-8")

    def rec(name, **kw):
        row = {"tag": a.tag, "test": name, **kw}
        out.write(json.dumps(row) + "\n")
        out.flush()
        short = {k: (v if len(str(v)) < 160 else str(v)[:160] + "...") for k, v in row.items()}
        print(json.dumps(short), flush=True)

    proc, log, load_s = start(a.hipml, a.ctx, a.slots, a.drafts, log_path, extra)
    rec("load", secs=round(load_s, 1), ctx=a.ctx, slots=a.slots, drafts=a.drafts)
    try:
        PLANS[a.plan](a, rec)
    except Exception as e:
        rec("plan_error", error=repr(e), server_exit=proc.poll())
    finally:
        rec("server_exit", code=proc.poll())
        stop(proc, log)
        out.close()


if __name__ == "__main__":
    main()
