# Security policy

## Reporting a vulnerability

Report privately through GitHub Security Advisories:

**https://github.com/pgmharikrishnan/hipml/security/advisories/new**

Use that form, not a public issue, a discussion or a pull request. There is no
security email address; the advisory form is the only private channel, and it
keeps the report, the fix and the disclosure note in one place.

Please include:

- the hipml commit or version (`git rev-parse HEAD`, or the release tag);
- what the bug is and what an attacker gains: crash, wrong output, out-of-bounds
  access, memory disclosure, denial of service;
- the exact command line and request that triggers it, and whether the server
  was started with a non-default `--host`;
- GPU, driver version, ROCm version and Windows build;
- the model file's name and SHA-256 if the bug needs a particular GGUF.

This is a single-maintainer project with no paid support and no response
guarantee. Reports are read and answered on a best-effort basis; a fix and a
`SECURITY.md`-linked advisory are issued together when the report is valid.

Please give a reasonable window before public disclosure. Credit in the advisory
is offered unless you ask otherwise.

## Supported versions

Only the latest `v0.1.x` release (or the tip of `main`) receives fixes. Older
commits and other branches are unsupported.

## What the server is, and is not

`hipml serve` is a local inference server for one model on one GPU. It is not
hardened and must not be exposed to an untrusted network. Its actual posture:

- **Binds `127.0.0.1` by default** (`--host`, port `8093`). `--host 0.0.0.0`
  puts an unauthenticated service on every interface.
- **No authentication and no authorisation.** There is no API key, no user
  model, no per-route access control, no TLS and no reverse-proxy awareness.
  Every client that can reach the port can run inference, read `/metrics` and
  `/props`, and consume GPU slots.
- **Greedy sampling only.** `temperature` other than 0 (or absent) and `n` other
  than 1 are refused with a 400; the server cannot be made to sample. This is a
  scope limit, not a safety property.
- **Not hardened for untrusted input.** The HTTP layer is hand-written HTTP/1.1:
  one request per connection, `Content-Length` bodies up to 32 MiB, a 30-second
  read timeout, at most 100 headers and 16 KiB per line. Admission is bounded by
  `--slots`, `--max-queue` (default 2 × slots) and `--max-conns` (64); past a
  bound the answer is 503 with `Retry-After`. There is no rate limiting, no
  request logging beyond `/metrics`, and no defence against a client that
  deliberately asks for long generations.
- **Trusted model, trusted GPU.** A GGUF is loaded with `mmap` and parsed
  in-process; there is no sandbox and no validation that a file is benign. Treat
  a model file as trusted input from its publisher.
- **Failure scopes.** A bad request fails one job; a slot or batch fault is
  contained; a device-level failure exits the process, and
  `scripts/hipml-serve.ps1` restarts it. The supervisor restarts a crashed
  server, not a wedged one, and it has been stress-tested for minutes, not
  hours.

If you need a shared or remote deployment, put an authenticating reverse proxy
in front of it, keep `--host 127.0.0.1` and tunnel to it.

## Out of scope

- Bugs in llama.cpp (report them to https://github.com/ggml-org/llama.cpp),
  in AMD ROCm / the HIP SDK or in the GPU driver (report them to AMD);
- anything that requires the server to be reachable from an untrusted network,
  since that deployment is unsupported;
- denial of service by an authenticated or local client asking for large
  generations, which is the documented behaviour of an unauthenticated local
  service;
- model output quality, model licensing, and the licence of any weights you
  download;
- crashes or wrong output caused by a GGUF that was not produced for this
  architecture, or by a card other than the one in the requirements.
