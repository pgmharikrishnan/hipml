// SPDX-License-Identifier: Apache-2.0
// The decode's RMS norm on its own, at the decode's own shape.
//
//   hipcc -O3 --offload-arch=gfx1201 probe_norm.cpp -o probe_norm
//   probe_norm [n] [launches] [reps]
//
// `rmsnorm_f32` is the only norm on the decode path and it runs 129 times a
// token (64 attn_norm + 64 post_norm + 1 output_norm). It is one block of 256
// threads over one 5,120-float row: 20 KB in, 20 KB out, on one of 64 CUs. The
// decode ablation prices the whole set at ~1.3 ms a token, so this probe asks
// what one launch costs when nothing else is on the card, and where that cost
// goes.
//
// It compiles the shipped kernel source, so the A/B between two versions of the
// norm is "run this probe against each tree", not a copy of the kernel here.
//
// The second table re-launches the same kernel on a *chain* of rows (each launch
// reads what the previous one wrote) to show the launch-to-launch dependency the
// decode actually has; the first table has no cross-launch dependency and is the
// pure execution cost.
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "rmsnorm_f32.hip"

// Wall time, not CPU time: the launches are asynchronous, so a CPU clock reads
// zero for the whole pass.
static double now_ms()
{
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv)
{
    const int n = argc > 1 ? atoi(argv[1]) : 5120;
    const int launches = argc > 2 ? atoi(argv[2]) : 129;
    const int reps = argc > 3 ? atoi(argv[3]) : 20;
    const float eps = 1e-6f;

    int dev = 0;
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, dev);
    printf("device: %s (%d CUs)\n", prop.name, prop.multiProcessorCount);
    printf("n=%d  launches/pass=%d  reps=%d\n", n, launches, reps);

    std::vector<float> hx(n), hw(n);
    for (int i = 0; i < n; i++) {
        hx[i] = 0.001f * (float)((i * 37) % 1000) - 0.5f;
        hw[i] = 1.0f + 0.001f * (float)(i % 17);
    }
    float *dx = nullptr, *dy = nullptr, *dw = nullptr;
    hipMalloc(&dx, (size_t)n * sizeof(float) * 4);
    hipMalloc(&dy, (size_t)n * sizeof(float) * 4);
    hipMalloc(&dw, (size_t)n * sizeof(float));
    for (int r = 0; r < 4; r++) hipMemcpy(dx + (size_t)r * n, hx.data(), (size_t)n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(dw, hw.data(), (size_t)n * sizeof(float), hipMemcpyHostToDevice);

    // 1. no cross-launch dependency: 129 launches, one synchronize.
    double best = 1e30;
    for (int rep = 0; rep < reps; rep++) {
        hipDeviceSynchronize();
        double t0 = now_ms();
        for (int i = 0; i < launches; i++)
            hipLaunchKernelGGL(rmsnorm_f32, dim3(1), dim3(256), 0, 0, dx, dw, dy, n, eps);
        hipDeviceSynchronize();
        double ms = now_ms() - t0;
        if (ms < best) best = ms;
    }
    printf("independent: %.3f ms / %d launches = %.3f us/launch\n", best, launches, best * 1e3 / launches);

    // 2. the decode's shape: each launch reads the row the previous one wrote.
    best = 1e30;
    for (int rep = 0; rep < reps; rep++) {
        hipDeviceSynchronize();
        double t0 = now_ms();
        for (int i = 0; i < launches; i++)
            hipLaunchKernelGGL(rmsnorm_f32, dim3(1), dim3(256), 0, 0, dx, dw, dx, n, eps);
        hipDeviceSynchronize();
        double ms = now_ms() - t0;
        if (ms < best) best = ms;
    }
    printf("chained:     %.3f ms / %d launches = %.3f us/launch\n", best, launches, best * 1e3 / launches);

    // 3. one launch's arithmetic against the 40 KB it touches.
    const double bytes = 2.0 * (double)n * sizeof(float);
    const double us = best * 1e3 / launches;
    printf("bytes/launch: %.1f KB, %.1f GB/s at the chained rate\n", bytes / 1024.0,
           bytes / (us * 1e-6) * 1e-9);

    hipFree(dx);
    hipFree(dy);
    hipFree(dw);
    return 0;
}
