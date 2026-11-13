// SPDX-License-Identifier: Apache-2.0
// Register-only throughput of the two gfx1201 matrix instructions the prefill
// tile does not use: `v_wmma_i32_16x16x32_iu4` (packed 4-bit integer) and
// `v_wmma_f32_16x16x16_fp8_fp8` (OCP fp8). Same harness as
// `probe_wmma_ceiling.cpp`, so the fp16 row reproduces that probe's 171-188
// TFLOPS and the other two rows are directly comparable.
//
//   hipcc -O3 --offload-arch=gfx1201 probe_wmma_int4_fp8.cpp -o probe_iu4fp8
//   probe_iu4fp8 [blocks] [iters] [threads]
//
// Each warp holds NACC independent accumulators and reuses two operand
// fragments, so there is no shared-memory traffic at all: what it measures is
// the instruction issue rate, not a GEMM.
//
// FLOPs per instruction, which is why the three rows are not the same work:
//   fp16 16x16x16  2*16*16*16  =  8,192   A 16 B/lane, B 16 B/lane
//   fp8  16x16x16  2*16*16*16  =  8,192   A  8 B/lane, B  8 B/lane
//   iu4  16x16x32  2*16*16*32  = 16,384   A  8 B/lane, B  8 B/lane
// so fp8 halves the staged bytes at the fp16 instruction's work, and iu4
// quarters the bytes per FLOP.
//
// `iu4` and `fp8` both take *both* operands in the narrow type: there is no
// mixed i4 x f16 or fp8 x f16 WMMA on this target. The register-only rate says
// whether the instruction is worth building a tile on; the tile probe
// (`probe_prefill_iu4fp8.cpp`) answers the staging question.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>

typedef _Float16 __attribute__((ext_vector_type(8))) half8;
typedef float __attribute__((ext_vector_type(8))) float8;
typedef unsigned int __attribute__((ext_vector_type(2))) uint2v;
typedef int __attribute__((ext_vector_type(8))) int8v;

#define NACC 8

// fp16: the shipped instruction, the reference row.
__global__ void probe_f16(const half8* __restrict__ in, float* __restrict__ out, int iters) {
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

// fp8 (OCP E4M3): no builtin in this SDK, so the instruction is emitted
// directly. The encoding is the one `llvm-mc -mcpu=gfx1201` gives for
// `v_wmma_f32_16x16x16_fp8_fp8`.
__device__ __forceinline__ float8 wmma_fp8(float8 acc, uint2v a, uint2v b) {
    asm volatile("v_wmma_f32_16x16x16_fp8_fp8 %0, %1, %2, %0" : "+v"(acc) : "v"(a), "v"(b));
    return acc;
}

__global__ void probe_fp8(const uint2v* __restrict__ in, float* __restrict__ out, int iters) {
    uint2v a = in[threadIdx.x & 1];
    uint2v b = in[2 + (threadIdx.x & 1)];
    float8 acc[NACC];
#pragma unroll
    for (int j = 0; j < NACC; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[j][e] = 0.0f;
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int j = 0; j < NACC; ++j) acc[j] = wmma_fp8(acc[j], a, b);
        a.x = (a.x & 0x7F7F7F7Fu) | 0x01010101u;   // keeps the operand live
    }
    float8 s = acc[0];
#pragma unroll
    for (int j = 1; j < NACC; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) s[e] += acc[j][e];
    if (threadIdx.x == 0)
        for (int e = 0; e < 8; ++e) out[e] = s[e];
}

// int4 x int4, K=32. The builtin takes (negA, a, negB, b, acc, clamp).
__global__ void probe_iu4(const uint2v* __restrict__ in, float* __restrict__ out, int iters) {
    uint2v a = in[threadIdx.x & 1];
    uint2v b = in[2 + (threadIdx.x & 1)];
    int8v acc[NACC];
#pragma unroll
    for (int j = 0; j < NACC; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[j][e] = 0;
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int j = 0; j < NACC; ++j)
            acc[j] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(false, a, false, b, acc[j],
                                                                     false);
        a.x = (a.x & 0x77777777u) | 0x11111111u;   // keeps the operand live
    }
    int8v s = acc[0];
#pragma unroll
    for (int j = 1; j < NACC; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) s[e] += acc[j][e];
    if (threadIdx.x == 0)
        for (int e = 0; e < 8; ++e) out[e] = (float)s[e];
}

// int4 x int4, K=32, plus the per-block rescale the quantised format needs: the
// accumulator is int32 and its block scale changes every 32 k values, so each
// 16x16x32 result has to be converted and folded into an f32 accumulator before
// the next block. That is 8 converts + 8 FMAs per accumulator per k-block, and
// it needs the int32 and f32 accumulators live at once - the register cost this
// row measures.
__global__ void probe_iu4_rescale(const uint2v* __restrict__ in, float* __restrict__ out,
                                  int iters) {
    uint2v a = in[threadIdx.x & 1];
    uint2v b = in[2 + (threadIdx.x & 1)];
    int8v acc[NACC];
    float8 facc[NACC];
#pragma unroll
    for (int j = 0; j < NACC; ++j) {
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[j][e] = 0;
        for (int e = 0; e < 8; ++e) facc[j][e] = 0.0f;
    }
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int j = 0; j < NACC; ++j) {
            acc[j] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(false, a, false, b,
                                                                     acc[j], false);
#pragma unroll
            for (int e = 0; e < 8; ++e) facc[j][e] += (float)acc[j][e] * 0.0009765625f;
            acc[j] = (int8v)0;
        }
        a.x = (a.x & 0x77777777u) | 0x11111111u;
    }
    float8 s = facc[0];
#pragma unroll
    for (int j = 1; j < NACC; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) s[e] += facc[j][e];
    if (threadIdx.x == 0)
        for (int e = 0; e < 8; ++e) out[e] = s[e];
}

int main(int argc, char** argv) {
    const int blocks = argc > 1 ? atoi(argv[1]) : 1024;
    const int iters = argc > 2 ? atoi(argv[2]) : 4096;
    const int threads = argc > 3 ? atoi(argv[3]) : 256;

    half8 hin[4];
    uint2v uin[4];
    for (int i = 0; i < 4; ++i) {
        for (int e = 0; e < 8; ++e) hin[i][e] = (_Float16)0.001f;
        uin[i].x = 0x11111111u * (unsigned)(i + 1);
        uin[i].y = 0x22222222u * (unsigned)(i + 1);
    }
    half8* dh;
    uint2v* du;
    float* dout;
    hipMalloc(&dh, sizeof(hin));
    hipMalloc(&du, sizeof(uin));
    hipMalloc(&dout, 64);
    hipMemcpy(dh, hin, sizeof(hin), hipMemcpyHostToDevice);
    hipMemcpy(du, uin, sizeof(uin), hipMemcpyHostToDevice);

    // flops per warp-iteration for each instruction shape
    const double f16 = 2.0 * 16 * 16 * 16 * NACC;
    const double f8 = 2.0 * 16 * 16 * 16 * NACC;
    const double i4 = 2.0 * 16 * 16 * 32 * NACC;

    const char* names[4] = {"fp16 16x16x16", "fp8  16x16x16", "iu4  16x16x32",
                            "iu4  +rescale "};
    const double per[4] = {f16, f8, i4, i4};

    for (int rep = 0; rep < 3; ++rep) {
        for (int w = 0; w < 4; ++w) {
            hipEvent_t t0, t1;
            hipEventCreate(&t0);
            hipEventCreate(&t1);
            hipEventRecord(t0);
            if (w == 0)
                hipLaunchKernelGGL(probe_f16, dim3(blocks), dim3(threads), 0, 0, dh, dout, iters);
            else if (w == 1)
                hipLaunchKernelGGL(probe_fp8, dim3(blocks), dim3(threads), 0, 0, du, dout, iters);
            else if (w == 2)
                hipLaunchKernelGGL(probe_iu4, dim3(blocks), dim3(threads), 0, 0, du, dout, iters);
            else
                hipLaunchKernelGGL(probe_iu4_rescale, dim3(blocks), dim3(threads), 0, 0, du, dout,
                                   iters);
            hipEventRecord(t1);
            hipEventSynchronize(t1);
            float ms = 0.0f;
            hipEventElapsedTime(&ms, t0, t1);
            const double flops = (double)blocks * (threads / 32.0) * (double)iters * per[w];
            printf("%s blocks %d iters %d threads %d: %.3f ms  %.2f TFLOPS\n", names[w], blocks,
                   iters, threads, ms, flops / (ms * 1e-3) / 1e12);
            hipEventDestroy(t0);
            hipEventDestroy(t1);
        }
    }
    return 0;
}
