# Standalone WMMA probes

Small C++ programs that measure one thing on the GPU, outside the engine. They
are not part of the build. Each needs `hipcc` on `PATH` and `HIP_PATH` set to a
ROCm SDK root; compile one with, for example:

```powershell
hipcc -O3 --offload-arch=$env:HIPML_ARCH probe_wmma_gemm.cpp -o probe_wmma_gemm.exe
.\probe_wmma_gemm.exe
```

They contain no machine paths and are cited by `docs/notes/review-levers.md`.

| file | what it measures |
|---|---|
| `probe_wmma2.cpp` | WMMA throughput with two accumulators in flight |
| `probe_wmma_ceiling.cpp` | the achievable WMMA ceiling at one shape, to compare a kernel against |
| `probe_wmma_cmp.cpp` | the same GEMM through WMMA and through plain `v_mfma`/FMA paths |
| `probe_wmma_gemm.cpp` | a full quantized GEMV/GEMM tile through WMMA, end to end |
| `probe_wmma_layout.cpp` | fragment layout and bank-conflict behaviour of the WMMA load/store |
