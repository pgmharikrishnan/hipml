// SPDX-License-Identifier: Apache-2.0
// Pure-read bandwidth probe: read size x vector width x outstanding loads x grid.
//
// The question it answers is why a 0.59 MB read runs at 125 GB/s when a 540 MB
// read runs at 571. A rate that rises monotonically with read size is the
// signature of too few bytes in flight, so the three levers that change
// in-flight bytes are separated here and nothing else is in the kernel: no
// dequant, no shared staging, no reduction, one XOR chain per accumulator.
//
//   _v4/_v8/_v16   four, eight and sixteen bytes per load. Sixteen is the
//                  widest global load the ISA has; a block format whose row
//                  stride is not a multiple of 16 is stuck at four.
//   _o1/_o2/_o4/_o8  independent loads issued per thread before the first is
//                  consumed. One accumulator per load, so the XOR chains are
//                  independent and the loads cannot be serialised by the
//                  arithmetic. `#pragma unroll 1` on the outer loop keeps the
//                  o1 case at one load in flight per iteration instead of
//                  letting the compiler invent the unroll the probe is trying
//                  to measure.
//   blocks/threads   the grid, from the host, so "how many workgroups" is a
//                  parameter rather than a conclusion.
//
// The row-tile entries keep the GEMV's geometry: `rpw` rows per warp, one warp
// per `rpw` rows, lane-strided in VEC-byte steps exactly like `q4k_tile`'s
// `for (c = lane * VPL; c < n; c += GROUP)` loop. With `rpw` and the block
// width both parameters, 512 rows can be issued as 512 one-row blocks or as 16
// blocks of 32 rows, which is the workgroup-count question asked directly.

#include <hip/hip_runtime.h>

template <int VEC>
struct BwVec;
template <>
struct BwVec<4> {
    using type = unsigned int;
    __device__ static unsigned fold(type v) { return v; }
};
template <>
struct BwVec<8> {
    using type = uint2;
    __device__ static unsigned fold(type v) { return v.x ^ v.y; }
};
template <>
struct BwVec<16> {
    using type = uint4;
    __device__ static unsigned fold(type v) { return (v.x ^ v.y) ^ (v.z ^ v.w); }
};

template <int VEC>
static __device__ __forceinline__ unsigned bw_xor(typename BwVec<VEC>::type v)
{
    return BwVec<VEC>::fold(v);
}

// Grid-stride read of `nvec` VEC-sized vectors starting at `base`, OUT loads in
// flight per thread.
template <int VEC, int OUT>
static __device__ __forceinline__ void bw_flat_body(const unsigned char* __restrict__ p,
                                                    unsigned long long nvec,
                                                    unsigned long long base,
                                                    float* __restrict__ out)
{
    using vec_t = typename BwVec<VEC>::type;
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    const unsigned long long end = base + nvec;
    unsigned long long i = base + (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;

    unsigned acc[OUT];
#pragma unroll
    for (int k = 0; k < OUT; ++k) acc[k] = 0;

#pragma unroll 1
    for (; i + (unsigned long long)(OUT - 1) * stride < end; i += (unsigned long long)OUT * stride) {
        vec_t v[OUT];
#pragma unroll
        for (int k = 0; k < OUT; ++k)
            v[k] = *reinterpret_cast<const vec_t*>(p + (i + (unsigned long long)k * stride) * VEC);
#pragma unroll
        for (int k = 0; k < OUT; ++k) acc[k] ^= bw_xor<VEC>(v[k]);
    }
    for (; i < end; i += stride) {
        const vec_t v = *reinterpret_cast<const vec_t*>(p + i * VEC);
        acc[0] ^= bw_xor<VEC>(v);
    }

    unsigned a = 0;
#pragma unroll
    for (int k = 0; k < OUT; ++k) a ^= acc[k];
    if (a == 0xFFFFFFFFu) out[blockIdx.x] = 1.0f;
}

#define BW_FLAT(VEC, OUT)                                                                          \
    extern "C" __global__ void bw_flat_v##VEC##_o##OUT(const unsigned char* __restrict__ p,         \
                                                       unsigned long long nvec,                    \
                                                       unsigned long long base,                    \
                                                       float* __restrict__ out)                    \
    {                                                                                              \
        bw_flat_body<VEC, OUT>(p, nvec, base, out);                                                \
    }

BW_FLAT(4, 1)
BW_FLAT(4, 2)
BW_FLAT(4, 4)
BW_FLAT(4, 8)
BW_FLAT(8, 1)
BW_FLAT(8, 2)
BW_FLAT(8, 4)
BW_FLAT(8, 8)
BW_FLAT(16, 1)
BW_FLAT(16, 2)
BW_FLAT(16, 4)
BW_FLAT(16, 8)

// The GEMV's geometry: `rpw` rows per warp, lane-strided in VEC-byte steps.
template <int VEC>
static __device__ __forceinline__ void bw_rows_body(const unsigned char* __restrict__ p,
                                                    int rows,
                                                    int row_bytes,
                                                    int rpw,
                                                    int row0,
                                                    float* __restrict__ out)
{
    using vec_t = typename BwVec<VEC>::type;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = (int)blockDim.x >> 5;
    const int per_block = warps * rpw;

    unsigned acc = 0;
    for (int rr = 0; rr < rpw; ++rr) {
        const int row = blockIdx.x * per_block + warp * rpw + rr;
        if (row >= rows) break;
        const unsigned char* rp = p + (size_t)(row + row0) * (size_t)row_bytes;
        for (int c = lane * VEC; c + VEC <= row_bytes; c += 32 * VEC) {
            const vec_t v = *reinterpret_cast<const vec_t*>(rp + c);
            acc ^= bw_xor<VEC>(v);
        }
    }
    if (acc == 0xFFFFFFFFu) out[blockIdx.x] = 1.0f;
}

#define BW_ROWS(VEC)                                                                               \
    extern "C" __global__ void bw_rows_v##VEC(const unsigned char* __restrict__ p,                  \
                                              int rows,                                            \
                                              int row_bytes,                                       \
                                              int rpw,                                             \
                                              int row0,                                            \
                                              float* __restrict__ out)                             \
    {                                                                                              \
        bw_rows_body<VEC>(p, rows, row_bytes, rpw, row0, out);                                     \
    }

BW_ROWS(4)
BW_ROWS(8)
BW_ROWS(16)

// The node cost on its own: one launch, no memory traffic, so a measured span
// minus N times this is the kernel and not the dispatch.
extern "C" __global__ void bw_empty(float* __restrict__ out)
{
    if (threadIdx.x == 1024) out[blockIdx.x] = 1.0f;
}
