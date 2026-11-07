// Flash-style decode attention over a *quantized* KV cache.
//
// Two kernels, because one block per query head cannot saturate the card at
// 100K depth:
//
//   attn_decode_kv_q8 / _q4   grid = n_kv * splits, one block per
//                             (kv head, contiguous slice of positions)
//   attn_decode_kv_reduce     grid = n_heads, merges the splits
//
// Block shape is 32 * (n_heads / n_kv) threads: one warp per query head of the
// KV group, so the six Qwen3.8-27B query heads that share a KV head sit in the same
// block and read the same cache row. The six reads of the row are the same
// addresses within a few hundred cycles, so L1/L2 serve five of them and DRAM
// sees one read. One block per *query* head would stream the cache 6x at 100K,
// where the cache is far larger than L2.
//
// Each lane owns 8 components of the query and of the accumulator (head_dim
// 256 / 32 lanes). A cache row is walked one 32-value block at a time, so a warp
// reads 32 consecutive payload bytes per block: fully coalesced, dequantized in
// registers, no shared-memory staging and no __syncthreads in the KV loop.
//
// Layout and the format itself: crates/hipml-core/src/kv.rs. Per-lane value:
//   q8: d * qs[lane]                     block = f16 d | 32 i8
//   q4: d * (nibble - 8)                 block = f16 d | 16 B, low = j, high = j+16
//
// Partials are fp32: part_acc[n_heads][splits][head_dim], part_ml[n_heads][splits][2]
// with (running max, running sum). The reduce recomputes the merge weights from
// the per-split maxima, so the merge is exact regardless of split order.
//
// gfx1201 is wave32; the reduction walks warpSize/2 so it is correct either way.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define KV_MAX_CHUNKS 8

// One block: f16 scale followed by 32 signed bytes.
struct KvQ8 {
    enum { BLOCK_BYTES = 34 };
    static __device__ __forceinline__ float at(const void* row, int chunk, int lane) {
        const char* b = (const char*)row + chunk * BLOCK_BYTES;
        const float d = __half2float(*(const __half*)b);
        return d * (float)((const signed char*)b)[2 + lane];
    }
};

// One block: f16 scale followed by 16 bytes, low nibble = value j, high = j+16.
struct KvQ4 {
    enum { BLOCK_BYTES = 18 };
    static __device__ __forceinline__ float at(const void* row, int chunk, int lane) {
        const char* b = (const char*)row + chunk * BLOCK_BYTES;
        const float d = __half2float(*(const __half*)b);
        const int byte = (int)((const unsigned char*)b)[2 + (lane & 15)];
        const int q = (lane < 16) ? (byte & 15) : (byte >> 4);
        return d * (float)(q - 8);
    }
};

// Sum across a warp, then broadcast so every lane sees the whole score.
__device__ __forceinline__ float warp_sum_all(float v) {
    for (int off = warpSize >> 1; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    return __shfl_sync(0xffffffffu, v, 0);
}

template <typename Loader>
__device__ __forceinline__ void kv_decode_body(
    const float* __restrict__ q,
    const void* __restrict__ k,
    const void* __restrict__ v,
    float* __restrict__ part_acc,
    float* __restrict__ part_ml,
    int n_heads,
    int n_kv,
    int head_dim,
    int seq,
    float scale,
    int splits)
{
    const int group = n_heads / n_kv;
    const int kvh = blockIdx.x / splits;
    const int split = blockIdx.x - kvh * splits;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= group) return;

    const int h = kvh * group + warp;
    const int nb = head_dim >> 5;       // 32-value blocks per row
    const int per = (seq + splits - 1) / splits;
    const int start = split * per;
    const int end = (start + per < seq) ? start + per : seq;

    // This lane's components: dims c*32 + lane for c in [0, nb).
    float qv[KV_MAX_CHUNKS];
    float acc[KV_MAX_CHUNKS];
    #pragma unroll
    for (int c = 0; c < KV_MAX_CHUNKS; ++c) {
        qv[c] = (c < nb) ? q[h * head_dim + c * 32 + lane] : 0.0f;
        acc[c] = 0.0f;
    }

    const char* kbase = (const char*)k;
    const char* vbase = (const char*)v;
    const size_t krow = (size_t)kvh * seq * nb;
    const size_t vrow = krow;

    float m = -INFINITY;
    float l = 0.0f;

    for (int pos = start; pos < end; ++pos) {
        const void* kr = kbase + (krow + (size_t)pos * nb) * Loader::BLOCK_BYTES;
        float dot = 0.0f;
        #pragma unroll
        for (int c = 0; c < KV_MAX_CHUNKS; ++c) {
            if (c < nb) dot += qv[c] * Loader::at(kr, c, lane);
        }
        const float s = warp_sum_all(dot) * scale;

        const float m_new = fmaxf(m, s);
        const float corr = expf(m - m_new);
        const float p = expf(s - m_new);
        l = l * corr + p;
        m = m_new;

        const void* vr = vbase + (vrow + (size_t)pos * nb) * Loader::BLOCK_BYTES;
        #pragma unroll
        for (int c = 0; c < KV_MAX_CHUNKS; ++c) {
            if (c < nb) acc[c] = acc[c] * corr + p * Loader::at(vr, c, lane);
        }
    }

    float* dst = part_acc + ((size_t)h * splits + split) * head_dim;
    #pragma unroll
    for (int c = 0; c < KV_MAX_CHUNKS; ++c) {
        if (c < nb) dst[c * 32 + lane] = acc[c];
    }
    if (lane == 0) {
        float* ml = part_ml + ((size_t)h * splits + split) * 2;
        ml[0] = m;
        ml[1] = l;
    }
}

extern "C" __global__ void attn_decode_kv_q8(
    const float* __restrict__ q,
    const void* __restrict__ k,
    const void* __restrict__ v,
    float* __restrict__ part_acc,
    float* __restrict__ part_ml,
    int n_heads, int n_kv, int head_dim, int seq, float scale, int splits)
{
    kv_decode_body<KvQ8>(q, k, v, part_acc, part_ml, n_heads, n_kv, head_dim, seq, scale, splits);
}

extern "C" __global__ void attn_decode_kv_q4(
    const float* __restrict__ q,
    const void* __restrict__ k,
    const void* __restrict__ v,
    float* __restrict__ part_acc,
    float* __restrict__ part_ml,
    int n_heads, int n_kv, int head_dim, int seq, float scale, int splits)
{
    kv_decode_body<KvQ4>(q, k, v, part_acc, part_ml, n_heads, n_kv, head_dim, seq, scale, splits);
}

// Merge per-split (acc, m, l) into out. One block per query head, one thread per
// head component; every thread walks the splits itself because splits is small
// (a few tens) and the per-split (m, l) pair is a broadcast load.
extern "C" __global__ void attn_decode_kv_reduce(
    const float* __restrict__ part_acc,
    const float* __restrict__ part_ml,
    float* __restrict__ out,
    int head_dim,
    int splits)
{
    const int h = blockIdx.x;
    const int t = threadIdx.x;
    if (t >= head_dim) return;

    const float* ml = part_ml + (size_t)h * splits * 2;
    float m = -INFINITY;
    for (int s = 0; s < splits; ++s) m = fmaxf(m, ml[s * 2]);

    float acc = 0.0f;
    float l = 0.0f;
    for (int s = 0; s < splits; ++s) {
        const float w = expf(ml[s * 2] - m);
        l += ml[s * 2 + 1] * w;
        acc += part_acc[((size_t)h * splits + s) * head_dim + t] * w;
    }
    out[(size_t)h * head_dim + t] = (l > 0.0f) ? acc / l : 0.0f;
}
