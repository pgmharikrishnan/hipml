#!/usr/bin/env python3
"""Shared path and environment resolution for the bench scripts.

Every machine-specific path is an environment variable or a CLI flag. Nothing
in this file contains a literal user name, drive letter or host name, so a
result recorded by any of these scripts is reproducible on another machine.

    env var              flag                 default if unset
    HIPML_GGUF           --model / --gguf     none: error
    HIPML_MTP_GGUF       --mtp                none: MTP cells are skipped
    HIPML_EXE            --hipml              <repo>/target/release/hipml[.exe]
    LLAMA_SERVER         --llama-server       none: llama cells are skipped
    HIP_PATH             (build only)         kernels are skipped at build time
    HIPML_BENCH_OUT      --out                <repo>/bench/out/<tool>
    HIPML_GPU_LOCK       --lock               unset: no locking
    HIPML_BENCH_HOST     --host               unset: run locally

Import this module instead of defining path constants in a script:

    from _paths import require, model, llama_server, rocm_env
"""

import os
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def env_or_flag(env_name, flag_value=None):
    """Return flag_value, else $env_name, else None."""
    if flag_value:
        return flag_value
    return os.environ.get(env_name) or None


def require(env_name, flag_value=None, what="value", flag="--model"):
    """Like env_or_flag, but exit with an actionable message when both are empty."""
    value = env_or_flag(env_name, flag_value)
    if not value:
        sys.exit(
            f"error: no {what} given\n"
            f"  pass {flag} <path> or set {env_name}\n"
            f"  see docs/usage.md for the environment table"
        )
    return value


def model(flag_value=None):
    """The target GGUF. Required."""
    return require("HIPML_GGUF", flag_value, "target model GGUF", "--model")


def mtp_model(flag_value=None):
    """The MTP drafter GGUF. Optional: callers skip MTP cells when it is None."""
    return env_or_flag("HIPML_MTP_GGUF", flag_value)


def llama_server(flag_value=None):
    """The llama.cpp server binary. Required only by scripts that run llama."""
    return require("LLAMA_SERVER", flag_value, "llama-server binary",
                   "--llama-server")


def hipml_exe(flag_value=None):
    """The hipml binary: flag, then $HIPML_EXE, then the release build."""
    value = env_or_flag("HIPML_EXE", flag_value)
    if value:
        return value
    name = "hipml.exe" if os.name == "nt" else "hipml"
    built = os.path.join(REPO_ROOT, "target", "release", name)
    if os.path.exists(built):
        return built
    found = shutil.which("hipml")
    return found or built


def out_dir(tool, flag_value=None):
    """Where a script writes raw output. Created if missing."""
    value = env_or_flag("HIPML_BENCH_OUT", flag_value)
    path = value if value else os.path.join(REPO_ROOT, "bench", "out", tool)
    os.makedirs(path, exist_ok=True)
    return path


def gpu_lock():
    """Optional lock script taking acquire/release/status. None means no locking."""
    return env_or_flag("HIPML_GPU_LOCK")


def base_url(flag_value=None, default="http://127.0.0.1:8092"):
    """A server base URL, for the scripts that talk HTTP."""
    return (flag_value or os.environ.get("HIPML_BASE_URL") or default).rstrip("/")


def rocm_env(env=None):
    """Return env with $HIP_PATH\\bin prepended to PATH when HIP_PATH is set.

    Never writes a literal ROCm path: the reader installs the SDK wherever they
    like and points HIP_PATH at it.
    """
    env = dict(os.environ if env is None else env)
    hip = env.get("HIP_PATH")
    if hip:
        bindir = os.path.join(hip, "bin")
        parts = [bindir] + [p for p in env.get("PATH", "").split(os.pathsep) if p]
        env["PATH"] = os.pathsep.join(parts)
    return env


def public_argv(cmd, replacements=()):
    """Return argv with machine-specific paths replaced by <PLACEHOLDER>.

    Recorded command lines are published with the results, so they must not
    carry somebody's drive layout. Numbers are never touched.
    """
    out = []
    for item in cmd:
        text = str(item)
        for value, placeholder in replacements:
            if value:
                text = text.replace(value, placeholder)
        out.append(text)
    return out


def add_common_args(ap, model_flag="--model", llama=False, mtp=False):
    """Add the shared flags with help text that names the environment variable."""
    ap.add_argument(model_flag, default=None,
                    help="target GGUF (default: $HIPML_GGUF)")
    if mtp:
        ap.add_argument("--mtp", default=None,
                        help="MTP drafter GGUF (default: $HIPML_MTP_GGUF; "
                             "MTP cells are skipped when unset)")
    if llama:
        ap.add_argument("--llama-server", default=None,
                        help="llama.cpp server binary (default: $LLAMA_SERVER)")
    ap.add_argument("--out", default=None,
                    help="output directory (default: $HIPML_BENCH_OUT or "
                         "bench/out/<tool>)")
    return ap
