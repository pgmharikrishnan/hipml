// Device-side q8_0 / q4_0 KV encode: quantize one token's K and V rows straight
// into the cache at `pos`.
//
// Before this kernel the graph read the 2048-wide k/v rows back to the host,
// quantized them with `hipml_core::kv::quantize`, and uploaded 8 blocks per row
// per layer — two blocking copies and eight small uploads, sixteen times a
// token. That is the whole depth gap between 20K and 100K.
//
// Layout (identical to `crates/hipml-core/src/kv.rs`):
//
//   block (kvh, pos, chunk) at  ((kvh * seq + pos) * (head_dim/32) + chunk) * BB
//
// q8_0 block: f16 d | 32 x i8, d = amax/127, q = round(x/d) clamped to +-127.
// q4_0 block: f16 d | 16 packed bytes, d = -vmax/8 (sign of the largest
// magnitude element), value j in the low nibble of byte j and j+16 in the high.
//
// The arithmetic matches `kv::quantize_block` exactly, including `roundf`'s
// round-half-away-from-zero and `__float2half`'s round-to-nearest-even, so a
// cache written here is bit-identical to one written on the host.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define KVQ_BLOCK 32
#define KVQ_Q8_BYTES 34
#define KVQ_Q4_BYTES 18

// One block of `head_dim` threads, eight 32-value blocks per 256 threads: warp
// `w` owns chunk `w` of the head. Lane `l` owns value `l` of that chunk.
static __device__ __forceinline__ float kvq_amax(float x)
{
    float m = fabsf(x);
#pragma unroll
    for (int bit = 16; bit > 0; bit >>= 1) {
        m = fmaxf(m, __shfl_xor(m, bit));
    }
    return m;
}

static __device__ __forceinline__ float kvq_vmax(float x)
{
    // The value with the largest magnitude, sign included (ggml q4_0).
    float m = x;
    float a = fabsf(x);
#pragma unroll
    for (int bit = 16; bit > 0; bit >>= 1) {
        const float om = __shfl_xor(m, bit);
        const float oa = __shfl_xor(a, bit);
        if (oa > a) { m = om; a = oa; }
    }
    return m;
}

static __device__ __forceinline__ int kvq_q8_i8(float x, float id)
{
    return (int)fmaxf(-127.0f, fminf(127.0f, roundf(x * id)));
}

extern "C" __global__ void kv_encode_q8(
    const float* __restrict__ k,
    const float* __restrict__ v,
    unsigned char* __restrict__ kc,
    unsigned char* __restrict__ vc,
    int head_dim,
    int stride,
    const int* __restrict__ step)
{
    const int h = blockIdx.x;
    const int chunk = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nchunk = head_dim >> 5;
    if (chunk >= nchunk) return;

    const int i = h * head_dim + chunk * KVQ_BLOCK + lane;
    // `stride` is the cache's per-head row count (fixed); `step[0]` is the
    // position, in device memory so a captured graph is reusable.
    const size_t off = ((size_t)h * stride + step[0]) * nchunk * KVQ_Q8_BYTES
                     + (size_t)chunk * KVQ_Q8_BYTES;

    const float xk = k[i];
    const float amax_k = kvq_amax(xk);
    const float dk = amax_k / 127.0f;
    const float idk = dk > 0.0f ? 1.0f / dk : 0.0f;
    if (lane == 0) *reinterpret_cast<__half*>(kc + off) = __float2half(dk);
    kc[off + 2 + lane] = (unsigned char)(signed char)kvq_q8_i8(xk, idk);

    const float xv = v[i];
    const float amax_v = kvq_amax(xv);
    const float dv = amax_v / 127.0f;
    const float idv = dv > 0.0f ? 1.0f / dv : 0.0f;
    if (lane == 0) *reinterpret_cast<__half*>(vc + off) = __float2half(dv);
    vc[off + 2 + lane] = (unsigned char)(signed char)kvq_q8_i8(xv, idv);
}

// q4_0 packs value j and value j+16 into byte j, so a 32-value chunk needs the
// 32 quantized values before anything can be written. Each lane keeps its own
// `q` and reads the partner's with one shuffle.

// Bulk variant for the chunked-prefill handoff: the f32 chunk attention has
// just written `rows` positions of the position-major cache, and decode needs
// them in the head-major quantized cache. One block per (position, KV head).
extern "C" __global__ void kv_encode_bulk_q8(
    const float* __restrict__ k,
    const float* __restrict__ v,
    unsigned char* __restrict__ kc,
    unsigned char* __restrict__ vc,
    int n_kv,
    int head_dim,
    int stride,
    int start,
    int rows)
{
    const int r = blockIdx.x / n_kv;
    const int h = blockIdx.x - r * n_kv;
    const int pos = start + r;
    const int chunk = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nchunk = head_dim >> 5;
    if (chunk >= nchunk) return;

    const int i = (pos * n_kv + h) * head_dim + chunk * KVQ_BLOCK + lane;
    const size_t off = ((size_t)h * stride + pos) * nchunk * KVQ_Q8_BYTES
                     + (size_t)chunk * KVQ_Q8_BYTES;

    const float xk = k[i];
    const float amax_k = kvq_amax(xk);
    const float dk = amax_k / 127.0f;
    const float idk = dk > 0.0f ? 1.0f / dk : 0.0f;
    if (lane == 0) *reinterpret_cast<__half*>(kc + off) = __float2half(dk);
    kc[off + 2 + lane] = (unsigned char)(signed char)kvq_q8_i8(xk, idk);

    const float xv = v[i];
    const float amax_v = kvq_amax(xv);
    const float dv = amax_v / 127.0f;
    const float idv = dv > 0.0f ? 1.0f / dv : 0.0f;
    if (lane == 0) *reinterpret_cast<__half*>(vc + off) = __float2half(dv);
    vc[off + 2 + lane] = (unsigned char)(signed char)kvq_q8_i8(xv, idv);
}

extern "C" __global__ void kv_encode_bulk_q4(
    const float* __restrict__ k,
    const float* __restrict__ v,
    unsigned char* __restrict__ kc,
    unsigned char* __restrict__ vc,
    int n_kv,
    int head_dim,
    int stride,
    int start,
    int rows)
{
    const int r = blockIdx.x / n_kv;
    const int h = blockIdx.x - r * n_kv;
    const int pos = start + r;
    const int chunk = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nchunk = head_dim >> 5;
    if (chunk >= nchunk) return;

    const int i = (pos * n_kv + h) * head_dim + chunk * KVQ_BLOCK + lane;
    const size_t off = ((size_t)h * stride + pos) * nchunk * KVQ_Q4_BYTES
                     + (size_t)chunk * KVQ_Q4_BYTES;

    const float xk = k[i];
    const float vmax_k = kvq_vmax(xk);
    const float dk = -vmax_k / 8.0f;
    const float idk = dk != 0.0f ? 1.0f / dk : 0.0f;
    const unsigned qk = (unsigned)fmaxf(0.0f, fminf(15.0f, roundf(xk * idk) + 8.0f));

    const float xv = v[i];
    const float vmax_v = kvq_vmax(xv);
    const float dv = -vmax_v / 8.0f;
    const float idv = dv != 0.0f ? 1.0f / dv : 0.0f;
    const unsigned qv = (unsigned)fmaxf(0.0f, fminf(15.0f, roundf(xv * idv) + 8.0f));

    if (lane == 0) {
        *reinterpret_cast<__half*>(kc + off) = __float2half(dk);
        *reinterpret_cast<__half*>(vc + off) = __float2half(dv);
    }
    const unsigned hk = __shfl_sync(0xffffffffffffffffULL, qk, lane + 16);
    const unsigned hv = __shfl_sync(0xffffffffffffffffULL, qv, lane + 16);
    if (lane < 16) {
        kc[off + 2 + lane] = (unsigned char)(qk | (hk << 4));
        vc[off + 2 + lane] = (unsigned char)(qv | (hv << 4));
    }
}
extern "C" __global__ void kv_encode_q4(
    const float* __restrict__ k,
    const float* __restrict__ v,
    unsigned char* __restrict__ kc,
    unsigned char* __restrict__ vc,
    int head_dim,
    int stride,
    const int* __restrict__ step)
{
    const int h = blockIdx.x;
    const int chunk = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nchunk = head_dim >> 5;
    if (chunk >= nchunk) return;

    const int i = h * head_dim + chunk * KVQ_BLOCK + lane;
    const size_t off = ((size_t)h * stride + step[0]) * nchunk * KVQ_Q4_BYTES
                     + (size_t)chunk * KVQ_Q4_BYTES;

    const float xk = k[i];
    const float vmax_k = kvq_vmax(xk);
    const float dk = -vmax_k / 8.0f;
    const float idk = dk != 0.0f ? 1.0f / dk : 0.0f;
    const unsigned qk = (unsigned)fmaxf(0.0f, fminf(15.0f, roundf(xk * idk) + 8.0f));

    const float xv = v[i];
    const float vmax_v = kvq_vmax(xv);
    const float dv = -vmax_v / 8.0f;
    const float idv = dv != 0.0f ? 1.0f / dv : 0.0f;
    const unsigned qv = (unsigned)fmaxf(0.0f, fminf(15.0f, roundf(xv * idv) + 8.0f));

    if (lane == 0) {
        *reinterpret_cast<__half*>(kc + off) = __float2half(dk);
        *reinterpret_cast<__half*>(vc + off) = __float2half(dv);
    }
    const unsigned hk = __shfl_sync(0xffffffffffffffffULL, qk, lane + 16);
    const unsigned hv = __shfl_sync(0xffffffffffffffffULL, qv, lane + 16);
    if (lane < 16) {
        kc[off + 2 + lane] = (unsigned char)(qk | (hk << 4));
        vc[off + 2 + lane] = (unsigned char)(qv | (hv << 4));
    }
}
