// SPDX-License-Identifier: Apache-2.0
// Register-only throughput probe for the gfx12 matrix unit, the same shape as
// the `v_dot2_f32_f16` probe that measured 41.4 TFLOPS on this card.
//
//   hipcc -O3 --offload-arch=gfx1201 probe_wmma_ceiling.cpp -o probe_ceiling
//   probe_ceiling [blocks] [iters]
//
// Each warp holds 8 independent 16x16x16 accumulators and reuses two operand
// fragments, so there is no shared-memory traffic at all: what it measures is
// the instruction issue rate, not a GEMM. Measured on the R9700 (gfx1201):
// 171-188 TFLOPS, i.e. about twice the 95.7 TFLOPS the vendor quotes for dense
// fp16 — that figure is not this instruction's rate.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>

typedef _Float16 __attribute__((ext_vector_type(8))) half8;
typedef float __attribute__((ext_vector_type(8))) float8;

#define NACC 8

__global__ void probe(const half8* __restrict__ in, float* __restrict__ out, int iters) {
    half8 a = in[threadIdx.x & 1];
    half8 b = in[2 + (threadIdx.x & 1)];
    float8 acc[NACC];
#pragma unroll
    for (int j = 0; j < NACC; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[j][e] = 0.0f;
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int j = 0; j < NACC; ++j)
            acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc[j]);
        // Keeps the operand live and dependent without adding matrix work.
        a[0] = (_Float16)((float)a[0] * 1e-9f);
    }
    float8 s = acc[0];
#pragma unroll
    for (int j = 1; j < NACC; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) s[e] += acc[j][e];
    if (threadIdx.x == 0)
        for (int e = 0; e < 8; ++e) out[e] = s[e];
}

int main(int argc, char** argv) {
    const int blocks = argc > 1 ? atoi(argv[1]) : 1024;
    const int iters = argc > 2 ? atoi(argv[2]) : 4096;
    const int threads = argc > 3 ? atoi(argv[3]) : 256;
    half8 in[4];
    for (int i = 0; i < 4; ++i)
        for (int e = 0; e < 8; ++e) in[i][e] = (_Float16)0.001f;
    half8* din;
    float* dout;
    hipMalloc(&din, sizeof(in));
    hipMalloc(&dout, 8 * 4);
    hipMemcpy(din, in, sizeof(in), hipMemcpyHostToDevice);

    for (int rep = 0; rep < 3; ++rep) {
        hipEvent_t t0, t1;
        hipEventCreate(&t0);
        hipEventCreate(&t1);
        hipEventRecord(t0);
        hipLaunchKernelGGL(probe, dim3(blocks), dim3(threads), 0, 0, din, dout, iters);
        hipEventRecord(t1);
        hipEventSynchronize(t1);
        float ms = 0.0f;
        hipEventElapsedTime(&ms, t0, t1);
        // 8 accumulators of 16x16x16 per warp per iteration.
        const double flops = (double)blocks * (threads / 32.0) * (double)iters * NACC *
                             2.0 * 16.0 * 16.0 * 16.0;
        printf("blocks %d iters %d threads %d: %.3f ms  %.2f TFLOPS\n", blocks, iters, threads, ms,
               flops / (ms * 1e-3) / 1e12);
    }
    return 0;
}
