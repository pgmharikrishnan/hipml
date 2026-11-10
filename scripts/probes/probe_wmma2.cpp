// SPDX-License-Identifier: Apache-2.0
// Standalone driver for the occupancy/pipeline prefill GEMM (`prefill_wmma2`).
//
//   hipcc -O3 --offload-arch=gfx1201 -I crates/hipml-kernels/kernels \
//         scripts/probe_wmma2.cpp -o probe2
//   probe2 [q4k|q6k|q8|half][x] [m] [k] [tokens] [reps]
//
// A trailing `x` selects the fp16-activation entry point (`_w2h`). Geometry and
// the pipeline switches are compile-time (`-DPWW2_*`) so a sweep is a rebuild,
// exactly like `probe_wmma_gemm.cpp` for the shipped kernel.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "prefill_wmma2.hip"

static size_t row_bytes(const char* kind, int k) {
    if (!strncmp(kind, "q4k", 3)) return (size_t)(k / 256) * PF_Q4K_BLOCK_BYTES;
    if (!strncmp(kind, "q6k", 3)) return (size_t)(k / 256) * PF_Q6K_BLOCK_BYTES;
    if (!strncmp(kind, "q8", 2)) return (size_t)(k / 32) * PF_Q8_0_BLOCK_BYTES;
    return (size_t)k * 2;
}

int main(int argc, char** argv) {
    const char* kind = argc > 1 ? argv[1] : "q6kx";
    const int m = argc > 2 ? atoi(argv[2]) : 17408;
    const int k = argc > 3 ? atoi(argv[3]) : 5120;
    const int tokens = argc > 4 ? atoi(argv[4]) : 512;
    const int reps = argc > 5 ? atoi(argv[5]) : 10;

    const size_t wb = row_bytes(kind, k) * (size_t)m;
    std::vector<unsigned char> w(wb);
    for (size_t i = 0; i < wb; ++i) w[i] = (unsigned char)(i * 37 + 11);
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

    const int rt = (m + PWW2_RT - 1) / PWW2_RT;
    const int tt = (tokens + PWW2_TT - 1) / PWW2_TT;
    const dim3 grid = PWW2_TOKFAST ? dim3(tt, rt) : dim3(rt, tt);
    const dim3 block(PWW2_THREADS);

    auto launch = [&]() {
        const float* xf = reinterpret_cast<const float*>(dx);
        if (!strncmp(kind, "q4k", 3) && !xhalf)
            hipLaunchKernelGGL(prefill_q4_k_w2, grid, block, 0, 0, dw, xf, dy, m, k, tokens);
        else if (!strncmp(kind, "q6k", 3) && !xhalf)
            hipLaunchKernelGGL(prefill_q6_k_w2, grid, block, 0, 0, dw, xf, dy, m, k, tokens);
        else if (!strncmp(kind, "q8", 2) && !xhalf)
            hipLaunchKernelGGL(prefill_q8_0_w2, grid, block, 0, 0, dw, xf, dy, m, k, tokens);
        else if (!xhalf)
            hipLaunchKernelGGL(prefill_halfw_w2, grid, block, 0, 0, dw, xf, dy, m, k, tokens);
        else if (!strncmp(kind, "q4k", 3))
            hipLaunchKernelGGL(prefill_q4_k_w2h, grid, block, 0, 0, dw, dx, dy, m, k, tokens);
        else if (!strncmp(kind, "q6k", 3))
            hipLaunchKernelGGL(prefill_q6_k_w2h, grid, block, 0, 0, dw, dx, dy, m, k, tokens);
        else if (!strncmp(kind, "q8", 2))
            hipLaunchKernelGGL(prefill_q8_0_w2h, grid, block, 0, 0, dw, dx, dy, m, k, tokens);
        else
            hipLaunchKernelGGL(prefill_halfw_w2h, grid, block, 0, 0, dw, dx, dy, m, k, tokens);
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
