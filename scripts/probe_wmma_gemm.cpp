// Standalone driver for the WMMA prefill GEMM: same kernel source, no engine.
//
//   hipcc -O3 --offload-arch=gfx1201 -I crates/hipml-kernels/kernels \
//         scripts/probe_wmma_gemm.cpp -o probe_gemm
//   probe_gemm [q4k|q6k|q8|half] [m] [k] [tokens] [reps]
//
// Geometry is the kernel's compile-time defaults, overridable with -D, so a
// shape sweep is a rebuild and no source edit. This exists to A/B one lever at
// a time against `probe_wmma_ceiling.cpp` (the register-only rate).
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "prefill_wmma.hip"

static size_t row_bytes(const char* kind, int k) {
    if (!strncmp(kind, "q4k", 3)) return (size_t)(k / 256) * 144;
    if (!strncmp(kind, "q6k", 3)) return (size_t)(k / 256) * 210;
    if (!strncmp(kind, "q8", 2)) return (size_t)(k / 32) * 34;
    return (size_t)k * 2;
}

int main(int argc, char** argv) {
    const char* kind = argc > 1 ? argv[1] : "q6k";
    const int m = argc > 2 ? atoi(argv[2]) : 17408;
    const int k = argc > 3 ? atoi(argv[3]) : 5120;
    const int tokens = argc > 4 ? atoi(argv[4]) : 512;
    const int reps = argc > 5 ? atoi(argv[5]) : 10;

    const size_t wb = row_bytes(kind, k) * (size_t)m;
    std::vector<unsigned char> w(wb);
    for (size_t i = 0; i < wb; ++i) w[i] = (unsigned char)(i * 37 + 11);
    // A trailing `x` in the kind means the activation tile is fp16 in global
    // memory (`prefill_*_wh`), the bytes-halving variant.
    const bool xhalf = kind[strlen(kind) - 1] == 'x';
    std::vector<float> x((size_t)tokens * k);
    for (size_t i = 0; i < x.size(); ++i) x[i] = (float)((int)(i % 97) - 48) * 0.01f;
    std::vector<_Float16> xh;
    if (xhalf) { xh.resize(x.size()); for (size_t i = 0; i < x.size(); ++i) xh[i] = (_Float16)x[i]; }
    std::vector<float> y((size_t)tokens * m);

    unsigned char* dw; void* dx; float* dy;
    hipMalloc(&dw, wb); hipMalloc(&dx, x.size() * (xhalf ? 2 : 4)); hipMalloc(&dy, y.size() * 4);
    hipMemcpy(dw, w.data(), wb, hipMemcpyHostToDevice);
    hipMemcpy(dx, xhalf ? (void*)xh.data() : (void*)x.data(),
              x.size() * (xhalf ? 2 : 4), hipMemcpyHostToDevice);
    hipMemset(dy, 0, y.size() * 4);

    const dim3 grid((m + PWW_RT - 1) / PWW_RT, (tokens + PWW_TT - 1) / PWW_TT);
    const dim3 block(PWW_THREADS);

    auto launch = [&]() {
        const float* xf = reinterpret_cast<const float*>(dx);
        if (!strncmp(kind, "q4k", 3) && !xhalf)
            hipLaunchKernelGGL(prefill_q4_k_w, grid, block, 0, 0, dw, xf, dy, m, k, tokens);
        else if (!strncmp(kind, "q6k", 3) && !xhalf)
            hipLaunchKernelGGL(prefill_q6_k_w, grid, block, 0, 0, dw, xf, dy, m, k, tokens);
        else if (!strncmp(kind, "q8", 2) && !xhalf)
            hipLaunchKernelGGL(prefill_q8_0_w, grid, block, 0, 0, dw, xf, dy, m, k, tokens);
        else if (!xhalf)
            hipLaunchKernelGGL(prefill_halfw_w, grid, block, 0, 0, dw, xf, dy, m, k, tokens);
        else if (!strncmp(kind, "q4k", 3))
            hipLaunchKernelGGL(prefill_q4_k_wh, grid, block, 0, 0, dw, dx, dy, m, k, tokens);
        else if (!strncmp(kind, "q6k", 3))
            hipLaunchKernelGGL(prefill_q6_k_wh, grid, block, 0, 0, dw, dx, dy, m, k, tokens);
        else if (!strncmp(kind, "q8", 2))
            hipLaunchKernelGGL(prefill_q8_0_wh, grid, block, 0, 0, dw, dx, dy, m, k, tokens);
        else
            hipLaunchKernelGGL(prefill_halfw_wh, grid, block, 0, 0, dw, dx, dy, m, k, tokens);
    };

    hipEvent_t t0, t1;
    hipEventCreate(&t0); hipEventCreate(&t1);
    launch();
    hipDeviceSynchronize();
    hipEventRecord(t0);
    for (int r = 0; r < reps; ++r) launch();
    hipEventRecord(t1);
    hipEventSynchronize(t1);
    float ms = 0.f;
    hipEventElapsedTime(&ms, t0, t1);

    const double per = ms / reps;
    const double flops = 2.0 * tokens * (double)m * k;
    printf("%s %dx%d M=%d tiles %dx%d: %.1f us  %.2f TFLOPS\n", kind, m, k, tokens,
           grid.x, grid.y, per * 1e3, flops / (per * 1e-3) / 1e12);
    return 0;
}
