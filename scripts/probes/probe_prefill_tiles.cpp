// SPDX-License-Identifier: Apache-2.0
// Per-format throughput of the shipped prefill tile, outside the engine.
//
//   hipcc -O3 --offload-arch=gfx1201 probe_prefill_tiles.cpp -o probe_tiles
//   probe_tiles [tokens] [reps]
//
// The tile is the one `prefill_wmma2.hip` ships (256 tokens x 128 rows, 16
// accumulators, fp16 activation tile), compiled from the same source the build
// uses. Weights are random bytes of the right block layout: this measures the
// decoder's cost, not its arithmetic. `kind 3` is the pure fp16 GEMM (weights
// already half in memory), i.e. the ceiling the dequantising tiles compete
// against at the same shape.
//
// Compile with -DPWW2_PROBE_NO_STAGE / -DPWW2_PROBE_NO_COMPUTE /
// -DPWW2_PROBE_NO_FRAG to split the tile's cost, and -DPWW2_PF=0..3 or
// -DPWW2_RT/-DPWW2_TT to test another shape.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "prefill_wmma2.hip"

struct Shape {
    int m;
    int k;
    const char* label;
};

static size_t row_bytes(int kind, int k)
{
    switch (kind) {
    case 0: return (size_t)(k / 256) * PF_Q4K_BLOCK_BYTES;
    case 1: return (size_t)(k / 256) * PF_Q6K_BLOCK_BYTES;
    case 2: return (size_t)(k / 32) * PF_Q8_0_BLOCK_BYTES;
    case 3: return (size_t)k * 2;
    case 4: return (size_t)(k / 256) * PF_Q5K_BLOCK_BYTES;
    case 5: return (size_t)(k / 256) * PF_IQ4_XS_BLOCK_BYTES;
    case 7: return (size_t)(k / 256) * PF_Q3K_BLOCK_BYTES;
    case 8: return (size_t)(k / 256) * PF_IQ3S_BLOCK_BYTES;
    default: return (size_t)(k / 32) * PF_IQ4_NL_BLOCK_BYTES;
    }
}

static const char* kind_name(int kind)
{
    switch (kind) {
    case 0: return "Q4_K";
    case 1: return "Q6_K";
    case 2: return "Q8_0";
    case 3: return "fp16";
    case 4: return "Q5_K";
    case 5: return "IQ4_XS";
    case 6: return "IQ4_NL";
    case 7: return "Q3_K";
    case 8: return "IQ3_S";
    default: return "?";
    }
}

static void launch(int kind, dim3 grid, dim3 block, const unsigned char* w, const void* x,
                   float* y, int m, int k, int tokens)
{
    switch (kind) {
    case 0: hipLaunchKernelGGL(prefill_q4_k_w2h, grid, block, 0, 0, w, x, y, m, k, tokens); break;
    case 1: hipLaunchKernelGGL(prefill_q6_k_w2h, grid, block, 0, 0, w, x, y, m, k, tokens); break;
    case 2: hipLaunchKernelGGL(prefill_q8_0_w2h, grid, block, 0, 0, w, x, y, m, k, tokens); break;
    case 3: hipLaunchKernelGGL(prefill_halfw_w2h, grid, block, 0, 0, w, x, y, m, k, tokens); break;
    case 4: hipLaunchKernelGGL(prefill_q5_k_w2h, grid, block, 0, 0, w, x, y, m, k, tokens); break;
    case 5: hipLaunchKernelGGL(prefill_iq4_xs_w2h, grid, block, 0, 0, w, x, y, m, k, tokens); break;
    case 6: hipLaunchKernelGGL(prefill_iq4_nl_w2h, grid, block, 0, 0, w, x, y, m, k, tokens); break;
    default: break;
    }
}

int main(int argc, char** argv)
{
    const int tokens = argc > 1 ? atoi(argv[1]) : 2048;
    const int reps = argc > 2 ? atoi(argv[2]) : 3;
    const int only = argc > 3 ? atoi(argv[3]) : -1;

    const int kinds[] = {0, 4, 5, 6, 3};
    const Shape shapes[] = {
        {17408, 5120, "gate/up 17408x5120"},
        {5120, 17408, "down 5120x17408"},
    };

    printf("tile TT=%d RT=%d WT=%d WR=%d KW=%d threads=%d PF=%d\n",
           PWW2_TT, PWW2_RT, PWW2_WT, PWW2_WR, PWW2_KW, PWW2_THREADS, PWW2_PF);
    printf("%-8s %-20s %6s %8s %9s %8s %8s %9s\n", "kind", "shape", "tokens", "ms",
           "TFLOPS", "GB/s w", "GB/s x", "GB/s tot");

    for (int kind : kinds) {
        if (only >= 0 && kind != only) continue;
        for (const Shape& s : shapes) {
            const size_t rb = row_bytes(kind, s.k);
            const size_t wb = rb * (size_t)s.m;
            const size_t xb = (size_t)tokens * s.k * 2;
            std::vector<unsigned char> w(wb);
            for (size_t i = 0; i < wb; ++i) w[i] = (unsigned char)(i * 37u + 11u);
            std::vector<__half> x((size_t)tokens * s.k);
            for (size_t i = 0; i < x.size(); ++i) x[i] = __float2half((float)((i % 97) - 48) / 128.0f);
            std::vector<float> y((size_t)tokens * s.m);

            unsigned char* dw = nullptr;
            __half* dx = nullptr;
            float* dy = nullptr;
            hipMalloc(&dw, wb);
            hipMalloc(&dx, xb);
            hipMalloc(&dy, y.size() * sizeof(float));
            hipMemcpy(dw, w.data(), wb, hipMemcpyHostToDevice);
            hipMemcpy(dx, x.data(), xb, hipMemcpyHostToDevice);

            const dim3 grid((unsigned)((s.m + PWW2_RT - 1) / PWW2_RT),
                            (unsigned)((tokens + PWW2_TT - 1) / PWW2_TT), 1);
            const dim3 block(PWW2_THREADS, 1, 1);
            launch(kind, grid, block, dw, dx, dy, s.m, s.k, tokens);
            if (hipDeviceSynchronize() != hipSuccess) {
                printf("%-8s %-20s launch failed\n", kind_name(kind), s.label);
                continue;
            }
            hipEvent_t t0, t1;
            hipEventCreate(&t0);
            hipEventCreate(&t1);
            hipEventRecord(t0);
            for (int r = 0; r < reps; ++r)
                launch(kind, grid, block, dw, dx, dy, s.m, s.k, tokens);
            hipEventRecord(t1);
            hipEventSynchronize(t1);
            float ms = 0.0f;
            hipEventElapsedTime(&ms, t0, t1);
            ms /= reps;

            const double flops = 2.0 * tokens * (double)s.m * s.k;
            const double tf = flops / (ms * 1e-3) / 1e12;
            // x is re-read once per row tile; weights once.
            const double gbw = (double)wb * (double)((tokens + PWW2_TT - 1) / PWW2_TT) / (ms * 1e-3) / 1e9;
            const double gbx = (double)xb * (double)((s.m + PWW2_RT - 1) / PWW2_RT) / (ms * 1e-3) / 1e9;
            printf("%-8s %-20s %6d %8.3f %9.2f %8.1f %8.1f %9.1f\n", kind_name(kind), s.label,
                   tokens, ms, tf, gbw, gbx, gbw + gbx);
            hipFree(dw);
            hipFree(dx);
            hipFree(dy);
        }
    }
    return 0;
}
