// y[row] = dequant_q5_k(W[row,:]) . x
//
// Q5_K is the one format Qwen3.8-27B uses that had no GEMV: `blk.51/55.ffn_down` were
// dequantised to f32 once at graph build and served by `gemv_f32`, so the decode
// step read 4 bytes per weight value instead of 0.6875 — 712 MB a token at the
// f32 kernel's rate, the slowest bytes in the stream.
//
// Row tile, one row per warp: `gemv_q4_k_r1`'s geometry (8 rows per block,
// 2048-value shared x tile, 16 values per lane) with Q5_K's fifth bit. The
// dequant is the same shape as Q4_K's (`d*scale*q - dmin*min`) plus one bit
// test per value, so it costs about one extra instruction per 16 values.
//
// Q5_K super-block (176 bytes, 256 values):
//   d(2) dmin(2) scales(12) qh(32) qs(128)
// Value j: sub-block s = j/32, nibble from qs, fifth bit from qh[l] where
// l = j % 32 and the bit is (1 << (2*(j/64))) for the low half, doubled for
// the high half. Layout matches hipml_core::quant::dequant_q5_k.
//
// A lane owns 16 consecutive columns, which never crosses a 32-value sub-block
// or a 64-value qs group, so one scale/min pair, one 16-byte qs run, one 16-byte
// qh run and one bit test cover the whole chunk.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define Q5K_VALUES 256
#define Q5K_BLOCK_BYTES 176
#define Q5K_BLOCK 256
#define Q5K_WARPS (Q5K_BLOCK / 32)
#define Q5K_ROW_TILE Q5K_WARPS
#define Q5K_VPL 16
#define Q5K_GROUP (Q5K_VPL * 32)
#define Q5K_K_TILE 2048

// Mirrors the manifest's KernelKind::RowTile { rows = Q5K_ROW_TILE, block = Q5K_BLOCK }.

static __device__ __forceinline__ void q5k_scale_min(int j, const unsigned char* q,
                                                     int* sc, int* mn)
{
    if (j < 4) {
        *sc = q[j] & 63;
        *mn = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static __device__ __forceinline__ void q5k_stage_group(const float* __restrict__ x,
                                                   float* __restrict__ sx,
                                                   int base, int j)
{
    const int g = j >> 5;
    const int lane = j & 31;
    float* s = sx + g * Q5K_GROUP + lane;
    const float4* src = reinterpret_cast<const float4*>(x + base + Q5K_VPL * j);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const float4 v = src[q];
        s[(4 * q + 0) * 32] = v.x;
        s[(4 * q + 1) * 32] = v.y;
        s[(4 * q + 2) * 32] = v.z;
        s[(4 * q + 3) * 32] = v.w;
    }
}

// Sixteen-column contribution of one Q5_K row to `acc`.
static __device__ __forceinline__ void q5k_acc(const unsigned char* __restrict__ rp,
                                               int p, const float* xv, float* acc)
{
    const int blk = p >> 8;
    const int within = p & 255;
    const int sub = within >> 5;
    const int g = within >> 6;
    const bool high = (within & 32) != 0;

    const unsigned char* b = rp + (size_t)blk * Q5K_BLOCK_BYTES;
    const float d = __half2float(*reinterpret_cast<const __half*>(b));
    const float dmin = __half2float(*reinterpret_cast<const __half*>(b + 2));
    int sc, mn;
    q5k_scale_min(sub, b + 4, &sc, &mn);
    const float scale = d * (float)sc;
    const float minv = dmin * (float)mn;
    const unsigned int bit = high ? (2u << (2 * g)) : (1u << (2 * g));

    // 16 consecutive qs bytes and 16 consecutive qh bytes: four dwords each.
    // `within` is a multiple of 16, so `within & 31` selects which half of the
    // 32-value qs run this chunk starts in.
    const unsigned int* qsp = reinterpret_cast<const unsigned int*>(b + 48 + g * 32 + (within & 31));
    const unsigned int* qhp = reinterpret_cast<const unsigned int*>(b + 16 + (within & 31));
    union Q5kBytes16 {
        unsigned int w[4];
        unsigned char b[16];
    } qs, qh;
    qs.w[0] = qsp[0];
    qs.w[1] = qsp[1];
    qs.w[2] = qsp[2];
    qs.w[3] = qsp[3];
    qh.w[0] = qhp[0];
    qh.w[1] = qhp[1];
    qh.w[2] = qhp[2];
    qh.w[3] = qhp[3];

    float a = *acc;
#pragma unroll
    for (int i = 0; i < Q5K_VPL; ++i) {
        const unsigned int byte = qs.b[i];
        const unsigned int nib = high ? (byte >> 4) : (byte & 0xF);
        const float q = (float)nib + ((qh.b[i] & bit) ? 16.0f : 0.0f);
        a = fmaf(scale * q - minv, xv[i], a);
    }
    *acc = a;
}

// External linkage so gemv_jobs can call it; `tile` is blockIdx.x there.
__device__ __forceinline__ void q5k_rows(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    float* __restrict__ sx,
    int tile)
{
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nblocks = k / Q5K_VALUES;

    const int row = tile * Q5K_ROW_TILE + warp;
    const int r = min(row, m - 1);
    const unsigned char* rp = w + (size_t)r * (size_t)nblocks * Q5K_BLOCK_BYTES;
    float acc = 0.0f;

    for (int base = 0; base < k; base += Q5K_K_TILE) {
        const int n = min(Q5K_K_TILE, k - base);
        const int nchunks = n / Q5K_VPL;
        for (int j = tid; j < nchunks; j += Q5K_BLOCK) q5k_stage_group(x, sx, base, j);
        __syncthreads();

        for (int c = lane * Q5K_VPL; c < n; c += Q5K_GROUP) {
            const int g = c / Q5K_GROUP;
            const int s = g * Q5K_GROUP + lane;
            float xv[Q5K_VPL];
#pragma unroll
            for (int comp = 0; comp < Q5K_VPL; ++comp) xv[comp] = sx[s + comp * 32];
            q5k_acc(rp, base + c, xv, &acc);
        }
        __syncthreads();
    }

    float a = acc;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) a += __shfl_down(a, off);
    if (lane == 0 && row < m) y[row] = a;
}

extern "C" __global__ void gemv_q5_k(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k)
{
    __shared__ __align__(16) float sx[Q5K_K_TILE];
    q5k_rows(w, x, y, m, k, sx, blockIdx.x);
}

// Batched variant (see gemv_q4_k.hip): B x rows share one weight pass, and one
// shared x read feeds Q5KB_ROWS_PER_WARP rows. Each row keeps its own
// accumulator in the same k order (bit-identical to gemv_q5_k on x[b]).
// K_TILE is 512 (one group).
#define Q5KB_K_TILE 512
#define Q5KB_ROWS_PER_WARP 4

static __device__ __forceinline__ void q5k_load(const unsigned char* __restrict__ rp,
                                                int p, float* wv)
{
    const int blk = p >> 8;
    const int within = p & 255;
    const int sub = within >> 5;
    const int g = within >> 6;
    const bool high = (within & 32) != 0;

    const unsigned char* b = rp + (size_t)blk * Q5K_BLOCK_BYTES;
    const float d = __half2float(*reinterpret_cast<const __half*>(b));
    const float dmin = __half2float(*reinterpret_cast<const __half*>(b + 2));
    int sc, mn;
    q5k_scale_min(sub, b + 4, &sc, &mn);
    const float scale = d * (float)sc;
    const float minv = dmin * (float)mn;
    const unsigned int bit = high ? (2u << (2 * g)) : (1u << (2 * g));

    const unsigned int* qsp = reinterpret_cast<const unsigned int*>(b + 48 + g * 32 + (within & 31));
    const unsigned int* qhp = reinterpret_cast<const unsigned int*>(b + 16 + (within & 31));
    union Q5kBytesB {
        unsigned int w[4];
        unsigned char b[16];
    } qs, qh;
    qs.w[0] = qsp[0];
    qs.w[1] = qsp[1];
    qs.w[2] = qsp[2];
    qs.w[3] = qsp[3];
    qh.w[0] = qhp[0];
    qh.w[1] = qhp[1];
    qh.w[2] = qhp[2];
    qh.w[3] = qhp[3];

#pragma unroll
    for (int i = 0; i < Q5K_VPL; ++i) {
        const unsigned int byte = qs.b[i];
        const unsigned int nib = high ? (byte >> 4) : (byte & 0xF);
        const float q = (float)nib + ((qh.b[i] & bit) ? 16.0f : 0.0f);
        wv[i] = scale * q - minv;
    }
}

template <int B, int RPW>
static __device__ __forceinline__ void q5k_tile_b(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    float* __restrict__ sx,
    int tile)
{
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nblocks = k / Q5K_VALUES;
    const int row_tile = Q5K_WARPS * RPW;

    int row[RPW];
    int r[RPW];
    const unsigned char* rp[RPW];
    float acc[RPW][B];
#pragma unroll
    for (int rr = 0; rr < RPW; ++rr) {
        row[rr] = tile * row_tile + warp * RPW + rr;
        r[rr] = min(row[rr], m - 1);
        rp[rr] = w + (size_t)r[rr] * (size_t)nblocks * Q5K_BLOCK_BYTES;
#pragma unroll
        for (int bi = 0; bi < B; ++bi) acc[rr][bi] = 0.0f;
    }

    for (int base = 0; base < k; base += Q5KB_K_TILE) {
        const int n = min(Q5KB_K_TILE, k - base);
        const int nchunks = n / Q5K_VPL;
        for (int j = tid; j < nchunks; j += Q5K_BLOCK) {
#pragma unroll
            for (int bi = 0; bi < B; ++bi)
                q5k_stage_group(x + (size_t)bi * k, sx + bi * Q5KB_K_TILE, base, j);
        }
        __syncthreads();

        for (int c = lane * Q5K_VPL; c < n; c += Q5K_GROUP) {
            const int g = c / Q5K_GROUP;
            const int s = g * Q5K_GROUP + lane;
            const int p = base + c;
            float wv[RPW][Q5K_VPL];
#pragma unroll
            for (int rr = 0; rr < RPW; ++rr) q5k_load(rp[rr], p, wv[rr]);
#pragma unroll
            for (int bi = 0; bi < B; ++bi) {
                const float* src = sx + bi * Q5KB_K_TILE + s;
                float xv[Q5K_VPL];
#pragma unroll
                for (int i = 0; i < Q5K_VPL; ++i) xv[i] = src[i * 32];
#pragma unroll
                for (int rr = 0; rr < RPW; ++rr) {
                    float a = acc[rr][bi];
#pragma unroll
                    for (int i = 0; i < Q5K_VPL; ++i) a = fmaf(wv[rr][i], xv[i], a);
                    acc[rr][bi] = a;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int rr = 0; rr < RPW; ++rr) {
#pragma unroll
        for (int bi = 0; bi < B; ++bi) {
            float a = acc[rr][bi];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) a += __shfl_down(a, off);
            if (lane == 0 && row[rr] < m) y[(size_t)bi * m + row[rr]] = a;
        }
    }
}

template <int B>
static __device__ __forceinline__ void q5k_entry_b(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k)
{
    extern __shared__ __align__(16) float hipml_sx_b[];
    q5k_tile_b<B, Q5KB_ROWS_PER_WARP>(w, x, y, m, k, hipml_sx_b, blockIdx.x);
}

extern "C" __global__ void gemv_q5_k_b(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int batch)
{
    switch (batch) {
    case 1: q5k_entry_b<1>(w, x, y, m, k); break;
    case 2: q5k_entry_b<2>(w, x, y, m, k); break;
    case 3: q5k_entry_b<3>(w, x, y, m, k); break;
    case 4: q5k_entry_b<4>(w, x, y, m, k); break;
    case 5: q5k_entry_b<5>(w, x, y, m, k); break;
    case 6: q5k_entry_b<6>(w, x, y, m, k); break;
    case 7: q5k_entry_b<7>(w, x, y, m, k); break;
    default: q5k_entry_b<8>(w, x, y, m, k); break;
    }
}
