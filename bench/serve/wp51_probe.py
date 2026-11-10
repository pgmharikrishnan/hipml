"""WP51 probes: error isolation, backpressure, sampling honesty, exactness cost
and warm-up.

Stdlib only. Starts `hipml serve` on 127.0.0.1:8093 (never 8080/8090/8091/8092)
one or more times per plan, and appends every result to a JSON-lines file. The
caller holds the GPU lock.

    python bench/serve/wp51_probe.py --hipml PATH --plan isolation --out out.jsonl

Plans:

* `isolation`  4 concurrent streams, one slot's decode step injected to fail:
               the other three must match a no-fault run token for token and
               the server must keep serving.
* `backpressure` 40 concurrent against 16 slots, at two queue bounds: every
               request is a stream or a clean 503 with Retry-After, none hangs.
* `sampling`   temperature 0 works; temperature 1 and n=2 are a 400 that says
               the server is greedy-only; top_p and seed are accepted.
* `warmup`     first-request TTFT with and without the load-time warm-up.
* `exact`      the throughput cost of forcing the exact batched path
               (`HIPML_BATCH_LEGACY=1`) at 4 and 8 slots, plus the single-stream
               MTP rate for the no-regression check.
"""

import argparse
import http.client
import json
import os
import socket
import subprocess
import sys
import threading
import time

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
    """`n` ids from the fixture, rotated by `salt` so two prompts of the same
    length do not share a prefix."""
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


def health(timeout=3):
    try:
        c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=timeout)
        c.request("GET", "/health")
        r = c.getresponse()
        ok = r.status == 200
        r.read()
        c.close()
        return ok
    except Exception:
        return False


class Server:
    """One `hipml serve` process on the probe port."""

    def __init__(self, hipml, ctx=4096, slots=4, drafts=0, env=None, log="server.log",
                 extra=None):
        self.hipml, self.ctx, self.slots, self.drafts = hipml, ctx, slots, drafts
        self.env, self.log_path, self.extra = env or {}, log, extra or []
        self.proc = None
        self.log = None
        self.load_s = None

    def __enter__(self):
        if port_open():
            raise RuntimeError(f"port {PORT} is already in use")
        env = _paths.rocm_env()
        env["HIPML_ARCH"] = env.get("HIPML_ARCH", "gfx1201")
        env.update(self.env)
        cmd = [self.hipml, "serve", "--gguf", MODEL, "--port", str(PORT),
               "--ctx", str(self.ctx), "--slots", str(self.slots), "--chunk", "256",
               "--drafts", str(self.drafts)] + self.extra
        if self.drafts > 0:
            cmd += ["--mtp", MTP]
        self.log = open(self.log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, env=env)
        t0 = time.time()
        while time.time() - t0 < 300:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited {self.proc.returncode} during load")
            if health():
                self.load_s = time.time() - t0
                return self
            time.sleep(0.5)
        self.stop()
        raise RuntimeError("server not healthy after 300 s")

    def __exit__(self, *exc):
        self.stop()

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.log:
            self.log.close()


def metrics():
    try:
        c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=10)
        c.request("GET", "/metrics")
        r = c.getresponse()
        text = r.read().decode()
        c.close()
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


def wait_idle(timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        m = metrics()
        if m.get("hipml_active_slots", 1) == 0 and m.get("hipml_queue_depth", 1) == 0:
            return time.time() - t0
        time.sleep(0.2)
    return None


# --- requests ---------------------------------------------------------------

def post(path, body, timeout=600):
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=timeout)
    t0 = time.time()
    try:
        c.request("POST", path, body=json.dumps(body),
                  headers={"Content-Type": "application/json"})
        r = c.getresponse()
        raw = r.read()
        try:
            js = json.loads(raw)
        except Exception:
            js = raw.decode(errors="replace")
        return r.status, js, dict(r.getheaders()), time.time() - t0
    finally:
        c.close()


def completion(ids, npred, **kw):
    body = {"prompt": ids, "n_predict": npred, "ignore_eos": True, "cache_prompt": False}
    body.update(kw)
    return post("/completion", body)


def stream(ids, npred, timeout=600, **kw):
    """One streaming completion, read to EOF. Returns a dict with the ids that
    arrived, the TTFT and the decode rate."""
    body = {"prompt": ids, "n_predict": npred, "ignore_eos": True, "cache_prompt": False,
            "stream": True}
    body.update(kw)
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=timeout)
    t0 = time.time()
    try:
        c.request("POST", "/completion", body=json.dumps(body),
                  headers={"Content-Type": "application/json"})
        r = c.getresponse()
        out = {"status": r.status, "ids": [], "ttft_ms": None, "error": None,
               "headers": dict(r.getheaders())}
        if r.status != 200:
            raw = r.read()
            out["error"] = raw.decode(errors="replace")[:400]
            out["secs"] = time.time() - t0
            return out
        buf = b""
        stamps = []
        # One byte at a time: `HTTPResponse.read(n)` blocks until it has n
        # bytes, which would time the whole reply at the end. Every SSE frame
        # is its own HTTP chunk, so a 1-byte read returns as soon as a frame
        # arrives and the timestamps are the arrival times.
        while True:
            chunk = r.read(1)
            if not chunk:
                break
            buf += chunk
            while b"\n\n" in buf:
                frame, buf = buf.split(b"\n\n", 1)
                for line in frame.split(b"\n"):
                    if not line.startswith(b"data: "):
                        continue
                    payload = line[6:].strip()
                    now = time.time()
                    if payload == b"[DONE]":
                        continue
                    try:
                        ev = json.loads(payload)
                    except Exception:
                        continue
                    if "error" in ev:
                        out["error"] = str(ev["error"])[:400]
                        continue
                    toks = ev.get("tokens") or []
                    if toks:
                        if out["ttft_ms"] is None:
                            out["ttft_ms"] = (now - t0) * 1e3
                        out["ids"].extend(toks)
                        stamps.append(now)
        out["secs"] = time.time() - t0
        if len(stamps) > 1:
            out["decode_s"] = stamps[-1] - stamps[0]
            out["tok_s"] = (len(out["ids"]) - 1) / max(out["decode_s"], 1e-9)
        return out
    finally:
        c.close()


def concurrent(fn, jobs):
    """Run `fn(*args)` for every args tuple behind a barrier; return in order."""
    n = len(jobs)
    barrier = threading.Barrier(n)
    out = [None] * n

    def one(i, args):
        barrier.wait()
        try:
            out[i] = fn(*args)
        except Exception as e:
            out[i] = {"status": None, "error": repr(e)}

    ts = [threading.Thread(target=one, args=(i, a)) for i, a in enumerate(jobs)]
    t0 = time.time()
    [t.start() for t in ts]
    [t.join() for t in ts]
    return out, time.time() - t0


def aggregate(results):
    """Sum of per-stream decode rates, over the streams that produced tokens."""
    rates = [r["tok_s"] for r in results if r and r.get("tok_s")]
    ttfts = [r["ttft_ms"] for r in results if r and r.get("ttft_ms") is not None]
    return {
        "streams": len(rates),
        "aggregate_tok_s": round(sum(rates), 1),
        "per_stream_p50": round(sorted(rates)[len(rates) // 2], 1) if rates else None,
        "ttft_p50_ms": round(sorted(ttfts)[len(ttfts) // 2], 0) if ttfts else None,
    }


# --- plans ------------------------------------------------------------------

def plan_isolation(a, rec):
    """One slot's step fails; every other slot in the same batch must finish
    with exactly the ids a no-fault run produced."""
    prompts = [ids_of_len(120, i + 2) for i in range(4)]
    n = 24

    def batch_run(tag, env):
        with Server(a.hipml, ctx=2048, slots=4, drafts=0, env=env,
                    log=f"{a.out}.{tag}.log") as s:
            rec(f"{tag}_load", secs=round(s.load_s, 1))
            # Prime the prefix cache off, so the four prompts prefill in full.
            res, wall = concurrent(lambda p: stream(p, n), [(p,) for p in prompts])
            rec(f"{tag}_run", wall_s=round(wall, 2),
                statuses=[r["status"] for r in res],
                tokens=[len(r["ids"]) for r in res],
                errors=[r["error"] for r in res],
                metrics={k: v for k, v in metrics().items() if "fault" in k or "reset" in k
                         or "job_errors" in k or "failed" in k})
            return [r["ids"] for r in res]

    ref = batch_run("baseline", {})
    faulty = batch_run("fault", {"HIPML_FAULT_SLOT": "1", "HIPML_FAULT_STEP": "4"})

    failed = [i for i, r in enumerate(faulty) if len(r) < n]
    intact = [i for i in range(4) if i not in failed]
    rec("isolation",
        failed_requests=failed,
        intact_requests=intact,
        match_reference=[faulty[i] == ref[i] for i in intact],
        lengths=[len(r) for r in faulty],
        ok=(len(failed) == 1 and all(faulty[i] == ref[i] for i in intact)))

    # The engine keeps serving after the failure.
    with Server(a.hipml, ctx=2048, slots=4, drafts=0,
                env={"HIPML_FAULT_SLOT": "1", "HIPML_FAULT_STEP": "2"},
                log=f"{a.out}.after.log") as s:
        rec("after_load", secs=round(s.load_s, 1))
        first = stream(prompts[0], 8)
        res, _ = concurrent(lambda p: stream(p, 8), [(p,) for p in prompts])
        m = metrics()
        rec("survives_after_fault", statuses=[r["status"] for r in res],
            first_status=first["status"],
            slot_faults=m.get("hipml_slot_faults_total"),
            job_errors=m.get("hipml_job_errors_total"),
            batch_resets=m.get("hipml_batch_resets_total"),
            healthy=health())
        # A solo reference for the same prompt after the fault.
        solo = stream(prompts[0], 8)
        rec("solo_after_fault", status=solo["status"], tokens=len(solo["ids"]))


def plan_backpressure(a, rec):
    """40 concurrent against 16 slots: a stream or a clean 503, never a hang."""
    prompt = ids_of_len(64, 5)
    n = 16
    for tag, queue, expect_ok in (("bounded", 4, 20), ("default", None, 40)):
        extra = [] if queue is None else ["--max-queue", str(queue)]
        with Server(a.hipml, ctx=2048, slots=16, drafts=0, extra=extra,
                    log=f"{a.out}.{tag}.log") as s:
            rec(f"{tag}_load", secs=round(s.load_s, 1), max_queue=queue)
            t0 = time.time()
            res, wall = concurrent(lambda p: stream(p, n, timeout=180), [(prompt,)] * 40)
            ok = [r for r in res if r["status"] == 200 and len(r["ids"]) == n]
            busy = [r for r in res if r["status"] == 503]
            other = [r for r in res if r["status"] not in (200, 503)]
            retry = [r["headers"].get("Retry-After") for r in busy]
            m = metrics()
            rec(f"backpressure_{tag}",
                wall_s=round(wall, 2),
                streams=len(ok), busy_503=len(busy), other=len(other),
                all_answered=(len(res) == 40),
                retry_after_present=all(x is not None for x in retry),
                rejected_counter=m.get("hipml_requests_rejected_busy_total"),
                inflight_max=m.get("hipml_inflight_max"),
                inflight_now=m.get("hipml_inflight"),
                healthy=health(),
                ok=(len(other) == 0 and len(ok) + len(busy) == 40
                    and (len(busy) > 0 if queue is not None else len(ok) == 40)
                    and all(x is not None for x in retry)))
            wait_idle(30)


def plan_sampling(a, rec):
    ref = None
    with Server(a.hipml, ctx=2048, slots=2, drafts=0, log=f"{a.out}.sampling.log") as s:
        rec("load", secs=round(s.load_s, 1))
        p = fixture_ids()
        st, js, _, _ = completion(p, 24, temperature=0)
        ref = js.get("tokens") if isinstance(js, dict) else None
        rec("temperature_0", status=st, tokens=len(ref or []))
        st, js, _, _ = completion(p, 24)
        rec("temperature_absent", status=st,
            match_reference=(js.get("tokens") == ref) if isinstance(js, dict) else False)
        st, js, _, _ = completion(p, 24, temperature=1)
        rec("temperature_1", status=st, body=str(js)[:200],
            ok=(st == 400 and "greedy-only" in str(js)))
        st, js, _, _ = completion(p, 24, temperature=0.7)
        rec("temperature_0_7", status=st, ok=(st == 400))
        st, js, _, _ = completion(p, 24, n=2)
        rec("n_2", status=st, body=str(js)[:200], ok=(st == 400))
        st, js, _, _ = completion(p, 24, temperature=0, top_p=0.9, seed=42)
        rec("top_p_seed", status=st,
            match_reference=(js.get("tokens") == ref) if isinstance(js, dict) else False)
        st, js, _, _ = post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": "Say hi."}],
            "max_tokens": 4, "temperature": 0.7})
        rec("chat_temperature", status=st, body=str(js)[:200], ok=(st == 400))
        st, js, _, _ = post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": "Say hi."}],
            "max_tokens": 4, "temperature": 0})
        rec("chat_temperature_0", status=st,
            content=js["choices"][0]["message"]["content"] if isinstance(js, dict) and "choices" in js else str(js)[:120])


def plan_warmup(a, rec):
    """First-request TTFT on a cold process, with and without the warm-up.

    The first request is a 200-token prompt: short enough that the ~5 s
    first-touch dominates, which is the number the target is about. A 2000-token
    prompt follows, so the prefill-heavy case is on record too."""
    short = ids_of_len(200, 7)
    long = ids_of_len(2000, 7)
    for tag, env in (("warm", {}), ("cold", {"HIPML_NO_WARMUP": "1"})):
        with Server(a.hipml, ctx=8192, slots=4, drafts=0, env=env,
                    log=f"{a.out}.{tag}.log") as s:
            m0 = metrics()
            r = stream(short, 8, cache_prompt=False)
            r2 = stream(long, 8, cache_prompt=False)
            rec(f"first_request_{tag}", load_s=round(s.load_s, 1),
                ttft_200_ms=round(r["ttft_ms"], 0) if r["ttft_ms"] else None,
                ttft_2000_ms=round(r2["ttft_ms"], 0) if r2["ttft_ms"] else None,
                tokens=len(r["ids"]), status=r["status"],
                warmup_ms=round(m0.get("hipml_warmup_seconds", 0.0) * 1e3, 0))


def plan_exact(a, rec):
    """The cost of the exact batched path, and the MTP single-stream rate."""
    prompt_len, n = 400, 64
    for slots in (4, 8):
        for tag, env in (("tile", {}), ("exact", {"HIPML_BATCH_LEGACY": "1"})):
            prompts = [ids_of_len(prompt_len, i + 3) for i in range(slots)]
            with Server(a.hipml, ctx=4096, slots=slots, drafts=0, env=env,
                        log=f"{a.out}.{tag}{slots}.log") as s:
                # One warm wave (prefix cache off, so every prompt is prefilled).
                concurrent(lambda p: stream(p, 8), [(p,) for p in prompts])
                res, _ = concurrent(lambda p: stream(p, n), [(p,) for p in prompts])
                agg = aggregate(res)
                m = metrics()
                rec(f"exact_{tag}_{slots}slots", load_s=round(s.load_s, 1),
                    **agg,
                    batch_exact=m.get("hipml_batch_exact_path"),
                    tile_steps=m.get("hipml_tile_steps_total"),
                    errors=[r["error"] for r in res if r and r.get("error")])
    # Single-stream MTP: the 38 tok/s no-regression floor.
    with Server(a.hipml, ctx=8192, slots=1, drafts=1, log=f"{a.out}.mtp.log") as s:
        p = fixture_ids()
        stream(p, 8, cache_prompt=False)  # warm
        rates, ttfts = [], []
        for _ in range(3):
            r = stream(p, 64, cache_prompt=False)
            if r.get("tok_s"):
                rates.append(r["tok_s"])
            if r["ttft_ms"]:
                ttfts.append(r["ttft_ms"])
        rec("single_mtp", rates=[round(x, 1) for x in rates],
            p50=round(sorted(rates)[len(rates) // 2], 1) if rates else None,
            ttft_p50_ms=round(sorted(ttfts)[len(ttfts) // 2], 0) if ttfts else None)


def plan_scopes(a, rec):
    """The two failure scopes the isolation plan does not reach: a failed
    shared weight pass (batch reset, engine keeps serving) and a device that
    cannot finish a step at all (drain every stream, then exit 1)."""
    prompt = ids_of_len(80, 4)
    # One batch-wide failure with every slot active, then service resumes.
    # Step 30 is late enough that all four prompts are prefilled and decoding.
    with Server(a.hipml, ctx=2048, slots=4, drafts=0,
                env={"HIPML_FAULT_BATCH": "30"}, log=f"{a.out}.batch.log") as s:
        res, _ = concurrent(lambda p: stream(p, 48), [(prompt,)] * 4)
        m = metrics()
        rec("batch_fault", statuses=[r["status"] for r in res],
            errors=[bool(r["error"]) for r in res],
            resets=m.get("hipml_batch_resets_total"),
            job_errors=m.get("hipml_job_errors_total"),
            healthy=health())
        r = stream(prompt, 8)
        rec("after_batch_reset", status=r["status"], tokens=len(r["ids"]),
            healthy=health(), server_up=(s.proc.poll() is None))

    # Three failures with no successful step between them: drain, then exit 1.
    with Server(a.hipml, ctx=2048, slots=4, drafts=0,
                env={"HIPML_FAULT_BATCH_ALL": "1"}, log=f"{a.out}.device.log") as s:
        waves = []
        for _ in range(4):
            if s.proc.poll() is not None:
                waves.append({"statuses": "server already exited"})
                break
            res, _ = concurrent(lambda p: stream(p, 8, timeout=60), [(prompt,)] * 4)
            waves.append({
                "statuses": [r["status"] for r in res],
                "errors": [bool(r["error"]) for r in res],
                "tokens": [len(r["ids"]) for r in res],
            })
            time.sleep(0.5)
        try:
            code = s.proc.wait(timeout=60)
        except Exception:
            code = "still running"
        rec("device_drain", waves=waves, exit_code=code,
            resets=metrics().get("hipml_batch_resets_total"))


def plan_headers(a, rec):
    """The exactness signal a client can see: the header on a reply, and the
    gauges on /metrics, in both the default and the forced-exact modes."""
    p = ids_of_len(120, 3)
    with Server(a.hipml, ctx=2048, slots=4, drafts=0, log=f"{a.out}.hdr.log") as s:
        r = stream(p, 4)
        rec("stream_header", status=r["status"],
            exact=r["headers"].get("X-Hipml-Batch-Exact"),
            tile_min=r["headers"].get("X-Hipml-Batch-Tile-Min"))
        st, js, hdrs, _ = completion(p, 4)
        m = metrics()
        rec("json_header", status=st, exact=hdrs.get("X-Hipml-Batch-Exact"),
            metric_exact=m.get("hipml_batch_exact_path"),
            tile_steps=m.get("hipml_tile_steps_total"))
        # Four concurrent streams push the batch to B=4, where the tile is used.
        res, _ = concurrent(lambda q: stream(q, 8), [(ids_of_len(120, i + 3),) for i in range(4)])
        m = metrics()
        rec("tile_used_at_4", statuses=[r["status"] for r in res],
            tile_steps=m.get("hipml_tile_steps_total"),
            exact_header=[r["headers"].get("X-Hipml-Batch-Exact") for r in res])
    with Server(a.hipml, ctx=2048, slots=4, drafts=0,
                env={"HIPML_BATCH_LEGACY": "1"}, log=f"{a.out}.hdr_legacy.log") as s:
        st, js, hdrs, _ = completion(p, 4)
        m = metrics()
        rec("legacy_header", status=st, exact=hdrs.get("X-Hipml-Batch-Exact"),
            metric_exact=m.get("hipml_batch_exact_path"),
            tile_steps=m.get("hipml_tile_steps_total"))


PLANS = {"isolation": plan_isolation, "backpressure": plan_backpressure,
         "sampling": plan_sampling, "warmup": plan_warmup, "exact": plan_exact,
         "headers": plan_headers, "scopes": plan_scopes}


def main():
    global MODEL, MTP
    ap = argparse.ArgumentParser(
        description="Server hardening probes. Paths come from HIPML_GGUF / "
                    "HIPML_MTP_GGUF or the flags below.")
    ap.add_argument("--hipml", default=None,
                    help="hipml binary (default: $HIPML_EXE or target/release/hipml)")
    ap.add_argument("--model", default=None,
                    help="target GGUF (default: $HIPML_GGUF)")
    ap.add_argument("--mtp", default=None,
                    help="MTP drafter GGUF (default: $HIPML_MTP_GGUF)")
    ap.add_argument("--plan", required=True, choices=sorted(PLANS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    a.hipml = _paths.hipml_exe(a.hipml)
    MODEL = _paths.model(a.model)
    MTP = _paths.mtp_model(a.mtp)
    if not MTP:
        sys.exit("error: the server probes need the MTP drafter; "
                 "pass --mtp or set HIPML_MTP_GGUF")

    out = open(a.out, "a", encoding="utf-8")

    def rec(name, **kw):
        row = {"tag": a.tag, "test": name, **kw}
        out.write(json.dumps(row) + "\n")
        out.flush()
        short = {k: (v if len(str(v)) < 200 else str(v)[:200] + "...") for k, v in row.items()}
        print(json.dumps(short), flush=True)

    try:
        PLANS[a.plan](a, rec)
    except Exception as e:
        rec("plan_error", error=repr(e))
    finally:
        out.close()


if __name__ == "__main__":
    main()
