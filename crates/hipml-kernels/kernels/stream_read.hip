// Pure streaming read: the memory side of a GEMV with no dequant math.
//
// These exist to separate two explanations for the decode stream running at
// ~420 GB/s while a cache-warm repeat of one tensor reaches ~640 GB/s:
// per-launch ramp/drain, or a per-byte limit of the access pattern. Both
// variants read the same model bytes; only the number of launches differs.
//
// Eight independent accumulators, held as scalars (an array would be
// address-taken and land in local memory). One dependent chain per thread
// measures latency rather than bandwidth, which is what a first version of
// this probe did.
//
// The accumulator is folded and stored so the loads cannot be eliminated.

#include <hip/hip_runtime.h>

#define SR_UNROLL 8

struct HipmlReadJob {
    const float4* p;
    unsigned long long nvec;
};

struct SrAcc {
    float a0, a1, a2, a3, a4, a5, a6, a7;
};

static __device__ __forceinline__ SrAcc sr_read(const float4* __restrict__ p,
                                                unsigned long long i,
                                                unsigned long long stride,
                                                unsigned long long nvec,
                                                SrAcc acc)
{
    const float4 v0 = p[i];
    acc.a0 += v0.x + v0.y + v0.z + v0.w;
    if (i + stride < nvec) {
        const float4 v = p[i + stride];
        acc.a1 += v.x + v.y + v.z + v.w;
    }
    if (i + 2 * stride < nvec) {
        const float4 v = p[i + 2 * stride];
        acc.a2 += v.x + v.y + v.z + v.w;
    }
    if (i + 3 * stride < nvec) {
        const float4 v = p[i + 3 * stride];
        acc.a3 += v.x + v.y + v.z + v.w;
    }
    if (i + 4 * stride < nvec) {
        const float4 v = p[i + 4 * stride];
        acc.a4 += v.x + v.y + v.z + v.w;
    }
    if (i + 5 * stride < nvec) {
        const float4 v = p[i + 5 * stride];
        acc.a5 += v.x + v.y + v.z + v.w;
    }
    if (i + 6 * stride < nvec) {
        const float4 v = p[i + 6 * stride];
        acc.a6 += v.x + v.y + v.z + v.w;
    }
    if (i + 7 * stride < nvec) {
        const float4 v = p[i + 7 * stride];
        acc.a7 += v.x + v.y + v.z + v.w;
    }
    return acc;
}

static __device__ __forceinline__ float sr_sum(SrAcc acc)
{
    return ((acc.a0 + acc.a1) + (acc.a2 + acc.a3))
         + ((acc.a4 + acc.a5) + (acc.a6 + acc.a7));
}

static __device__ __forceinline__ SrAcc sr_zero()
{
    return SrAcc{0, 0, 0, 0, 0, 0, 0, 0};
}

static __device__ __forceinline__ float block_sum(float acc, float* smem)
{
    const int tid = threadIdx.x;
    smem[tid] = acc;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) smem[tid] += smem[tid + off];
        __syncthreads();
    }
    return smem[0];
}

// One tensor, one launch: what the graph's weight stream does per GEMV.
extern "C" __global__ void stream_read(
    const float4* __restrict__ p,
    unsigned long long nvec,
    float* __restrict__ out)
{
    __shared__ float smem[256];
    SrAcc acc = sr_zero();
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + (unsigned long long)(SR_UNROLL - 1) * stride < nvec;
         i += (unsigned long long)SR_UNROLL * stride) {
        acc = sr_read(p, i, stride, nvec, acc);
    }
    for (; i < nvec; i += stride) {
        const float4 v = p[i];
        acc.a0 += v.x + v.y + v.z + v.w;
    }
    const float s = block_sum(sr_sum(acc), smem);
    if (threadIdx.x == 0) out[blockIdx.x] = s;
}

// Every tensor in one launch: a device-side job list walked by a fixed grid.
// No launch boundary between tensors, so if the loss is per-launch this is
// where it goes away.
extern "C" __global__ void stream_read_jobs(
    const HipmlReadJob* __restrict__ jobs,
    int n_jobs,
    float* __restrict__ out)
{
    __shared__ float smem[256];
    SrAcc acc = sr_zero();
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    const unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    for (int j = 0; j < n_jobs; ++j) {
        const float4* p = jobs[j].p;
        const unsigned long long nvec = jobs[j].nvec;
        unsigned long long i = tid;
        for (; i + (unsigned long long)(SR_UNROLL - 1) * stride < nvec;
             i += (unsigned long long)SR_UNROLL * stride) {
            acc = sr_read(p, i, stride, nvec, acc);
        }
        for (; i < nvec; i += stride) {
            const float4 v = p[i];
            acc.a0 += v.x + v.y + v.z + v.w;
        }
    }
    const float s = block_sum(sr_sum(acc), smem);
    if (threadIdx.x == 0) out[blockIdx.x] = s;
}
