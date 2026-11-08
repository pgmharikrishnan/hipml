// y[row] = dequant_q6_k(W[row,:]) . x
//
// Row-tile GEMV: one block owns Q6K_ROW_TILE output rows and stages the x k-tile in
// shared memory once. Each warp owns Q6K_ROWS_PER_WARP rows and keeps one
// accumulator per row, so a value read from shared is applied to
// Q6K_ROWS_PER_WARP weight rows before it is discarded.
//
// The shared tile is de-interleaved by 256-value group: component `c` of a group
// lives at sx[g*256 + c*32 + lane], so a lane's eight scalar reads hit eight
// distinct banks and the warp has no bank conflicts.
//
// Each lane owns 8 consecutive columns per row: eight ql bytes and eight qh
// bytes (2-byte aligned, so four uint16 loads each) and one int8 scale. A chunk
// never crosses a 16-value scale sub-block.
//
// Q6_K super-block (210 bytes, 256 values):
//   ql(128) qh(64) scales(16) d(2)
// Math is unchanged: q = (ql nibble) | ((qh plane) << 4), y = d*scale*(q-32).
//
// Two barriers per k-tile; the only reduction is a warp shuffle tree per row.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define Q6K_VALUES 256
#define Q6K_BLOCK_BYTES 210
#define Q6K_BLOCK 256
#define Q6K_WARPS (Q6K_BLOCK / 32)
#define Q6K_ROWS_PER_WARP 2
#define Q6K_ROW_TILE (Q6K_WARPS * Q6K_ROWS_PER_WARP)
#define Q6K_K_TILE 2048

// Mirrors the manifest's KernelKind::RowTile { rows = Q6K_ROW_TILE, block = Q6K_BLOCK };
// change both together.

static __device__ __forceinline__ void stage_group6(const float* __restrict__ x,
                                                    float* __restrict__ sx,
                                                    int base, int j)
{
    const float4 a = reinterpret_cast<const float4*>(x + base)[2 * j];
    const float4 b = reinterpret_cast<const float4*>(x + base)[2 * j + 1];
    const int g = j >> 5;
    const int lane = j & 31;
    float* s = sx + g * 256 + lane;
    s[0 * 32] = a.x;
    s[1 * 32] = a.y;
    s[2 * 32] = a.z;
    s[3 * 32] = a.w;
    s[4 * 32] = b.x;
    s[5 * 32] = b.y;
    s[6 * 32] = b.z;
    s[7 * 32] = b.w;
}

static __device__ __forceinline__ unsigned int join16(unsigned short a,
                                                      unsigned short b)
{
    return (unsigned int)a | ((unsigned int)b << 16);
}

// Eight-column contribution of one Q6_K row to `acc`.
static __device__ __forceinline__ void q6k_acc8(const unsigned char* __restrict__ rp,
                                                int p, const float* xv, float* acc)
{
    const int blk = p >> 8;
    const int within = p & 255;
    const int half = within >> 7;
    const int ph = within & 127;
    const int group = ph >> 5;
    const int l = ph & 31;

    const unsigned char* b = rp + (size_t)blk * Q6K_BLOCK_BYTES;
    const float d = __half2float(*reinterpret_cast<const __half*>(b + 208));
    const int sub = group * 2 + (l >> 4);
    const float scale =
        d * (float)reinterpret_cast<const signed char*>(b + 192)[half * 8 + sub];

    const int ql_off = half * 64 + ((group & 1) ? 32 : 0) + l;
    const int qh_off = 128 + half * 32 + l;
    const unsigned int qlw0 = join16(
        *reinterpret_cast<const unsigned short*>(b + ql_off),
        *reinterpret_cast<const unsigned short*>(b + ql_off + 2));
    const unsigned int qlw1 = join16(
        *reinterpret_cast<const unsigned short*>(b + ql_off + 4),
        *reinterpret_cast<const unsigned short*>(b + ql_off + 6));
    const unsigned int qhw0 = join16(
        *reinterpret_cast<const unsigned short*>(b + qh_off),
        *reinterpret_cast<const unsigned short*>(b + qh_off + 2));
    const unsigned int qhw1 = join16(
        *reinterpret_cast<const unsigned short*>(b + qh_off + 4),
        *reinterpret_cast<const unsigned short*>(b + qh_off + 6));

    const int shift = 2 * group;
    const bool high = group >= 2;

    float a = *acc;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const unsigned int qw = (i < 4) ? qlw0 : qlw1;
        const unsigned int hw = (i < 4) ? qhw0 : qhw1;
        const unsigned int qb = (qw >> (8 * (i & 3))) & 0xFFu;
        const unsigned int hb = (hw >> (8 * (i & 3))) & 0xFFu;
        const unsigned int nib = high ? (qb >> 4) : (qb & 0xF);
        const unsigned int hi = (hb >> shift) & 3u;
        const int v = (int)(nib | (hi << 4)) - 32;
        a = fmaf(scale * (float)v, xv[i], a);
    }
    *acc = a;
}

// One row/warp when RPW is 1. Lane ownership and the reduction tree are
// unchanged, so this is bit-identical to the two-row entry with twice the grid.
// External linkage so gemv_jobs can call it; `tile` is blockIdx.x there.
template <int RPW>
__device__ __forceinline__ void q6k_tile(
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
    const int nblocks = k / Q6K_VALUES;
    const int row_tile = Q6K_WARPS * RPW;

    int row[RPW];
    int r[RPW];
    const unsigned char* rp[RPW];
    float acc[RPW];
#pragma unroll
    for (int rr = 0; rr < RPW; ++rr) {
        row[rr] = tile * row_tile + warp * RPW + rr;
        r[rr] = min(row[rr], m - 1);
        rp[rr] = w + (size_t)r[rr] * (size_t)nblocks * Q6K_BLOCK_BYTES;
        acc[rr] = 0.0f;
    }

    for (int base = 0; base < k; base += Q6K_K_TILE) {
        const int n = min(Q6K_K_TILE, k - base);
        const int nchunks = n >> 3;
        for (int j = tid; j < nchunks; j += Q6K_BLOCK) stage_group6(x, sx, base, j);
        __syncthreads();

        for (int c = lane * 8; c < n; c += 256) {
            const int g = c >> 8;
            const int s = g * 256 + lane;
            float xv[8];
#pragma unroll
            for (int comp = 0; comp < 8; ++comp) xv[comp] = sx[s + comp * 32];
            const int p = base + c;
#pragma unroll
            for (int rr = 0; rr < RPW; ++rr) q6k_acc8(rp[rr], p, xv, &acc[rr]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int rr = 0; rr < RPW; ++rr) {
        float a = acc[rr];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) a += __shfl_down(a, off);
        if (lane == 0 && row[rr] < m) y[row[rr]] = a;
    }
}

extern "C" __global__ void gemv_q6_k(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k)
{
    __shared__ __align__(16) float sx[Q6K_K_TILE];
    q6k_tile<Q6K_ROWS_PER_WARP>(w, x, y, m, k, sx, blockIdx.x);
}

extern "C" __global__ void gemv_q6_k_r1(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k)
{
    __shared__ __align__(16) float sx[Q6K_K_TILE];
    q6k_tile<1>(w, x, y, m, k, sx, blockIdx.x);
}

// ---------------------------------------------------------------------------
// Batched variant: B x rows share one weight pass, and one shared x read feeds
// Q6KB_ROWS_PER_WARP rows (see gemv_q4_k.hip for why both reuse axes matter).
// Each row keeps its own accumulator in the same k order, so row b is
// bit-identical to gemv_q6_k on x[b]. K_TILE is 512 (two 256-value groups).
// ---------------------------------------------------------------------------

#define Q6KB_K_TILE 512
#define Q6KB_ROWS_PER_WARP 2

static __device__ __forceinline__ void q6k_load(const unsigned char* __restrict__ rp,
                                                int p, float* wv)
{
    const int blk = p >> 8;
    const int within = p & 255;
    const int half = within >> 7;
    const int ph = within & 127;
    const int group = ph >> 5;
    const int l = ph & 31;

    const unsigned char* b = rp + (size_t)blk * Q6K_BLOCK_BYTES;
    const float d = __half2float(*reinterpret_cast<const __half*>(b + 208));
    const int sub = group * 2 + (l >> 4);
    const float scale =
        d * (float)reinterpret_cast<const signed char*>(b + 192)[half * 8 + sub];

    const int ql_off = half * 64 + ((group & 1) ? 32 : 0) + l;
    const int qh_off = 128 + half * 32 + l;
    const unsigned int qlw0 = join16(
        *reinterpret_cast<const unsigned short*>(b + ql_off),
        *reinterpret_cast<const unsigned short*>(b + ql_off + 2));
    const unsigned int qlw1 = join16(
        *reinterpret_cast<const unsigned short*>(b + ql_off + 4),
        *reinterpret_cast<const unsigned short*>(b + ql_off + 6));
    const unsigned int qhw0 = join16(
        *reinterpret_cast<const unsigned short*>(b + qh_off),
        *reinterpret_cast<const unsigned short*>(b + qh_off + 2));
    const unsigned int qhw1 = join16(
        *reinterpret_cast<const unsigned short*>(b + qh_off + 4),
        *reinterpret_cast<const unsigned short*>(b + qh_off + 6));

    const int shift = 2 * group;
    const bool high = group >= 2;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const unsigned int qw = (i < 4) ? qlw0 : qlw1;
        const unsigned int hw = (i < 4) ? qhw0 : qhw1;
        const unsigned int qb = (qw >> (8 * (i & 3))) & 0xFFu;
        const unsigned int hb = (hw >> (8 * (i & 3))) & 0xFFu;
        const unsigned int nib = high ? (qb >> 4) : (qb & 0xF);
        const unsigned int hi = (hb >> shift) & 3u;
        const int v = (int)(nib | (hi << 4)) - 32;
        wv[i] = scale * (float)v;
    }
}

template <int B, int RPW>
static __device__ __forceinline__ void q6k_tile_b(
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
    const int nblocks = k / Q6K_VALUES;
    const int row_tile = Q6K_WARPS * RPW;

    int row[RPW];
    int r[RPW];
    const unsigned char* rp[RPW];
    float acc[RPW][B];
#pragma unroll
    for (int rr = 0; rr < RPW; ++rr) {
        row[rr] = tile * row_tile + warp * RPW + rr;
        r[rr] = min(row[rr], m - 1);
        rp[rr] = w + (size_t)r[rr] * (size_t)nblocks * Q6K_BLOCK_BYTES;
#pragma unroll
        for (int bi = 0; bi < B; ++bi) acc[rr][bi] = 0.0f;
    }

    for (int base = 0; base < k; base += Q6KB_K_TILE) {
        const int n = min(Q6KB_K_TILE, k - base);
        const int nchunks = n >> 3;
        for (int j = tid; j < nchunks; j += Q6K_BLOCK) {
#pragma unroll
            for (int bi = 0; bi < B; ++bi)
                stage_group6(x + (size_t)bi * k, sx + bi * Q6KB_K_TILE, base, j);
        }
        __syncthreads();

        for (int c = lane * 8; c < n; c += 256) {
            const int g = c >> 8;
            const int s = g * 256 + lane;
            const int p = base + c;
            float wv[RPW][8];
#pragma unroll
            for (int rr = 0; rr < RPW; ++rr) q6k_load(rp[rr], p, wv[rr]);
#pragma unroll
            for (int bi = 0; bi < B; ++bi) {
                const float* src = sx + bi * Q6KB_K_TILE + s;
                float xv[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) xv[i] = src[i * 32];
#pragma unroll
                for (int rr = 0; rr < RPW; ++rr) {
                    float a = acc[rr][bi];
#pragma unroll
                    for (int i = 0; i < 8; ++i) a = fmaf(wv[rr][i], xv[i], a);
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
static __device__ __forceinline__ void q6k_entry_b(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k)
{
    extern __shared__ __align__(16) float hipml_sx_b[];
    q6k_tile_b<B, Q6KB_ROWS_PER_WARP>(w, x, y, m, k, hipml_sx_b, blockIdx.x);
}

extern "C" __global__ void gemv_q6_k_b(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int batch)
{
    switch (batch) {
    case 1: q6k_entry_b<1>(w, x, y, m, k); break;
    case 2: q6k_entry_b<2>(w, x, y, m, k); break;
    case 3: q6k_entry_b<3>(w, x, y, m, k); break;
    case 4: q6k_entry_b<4>(w, x, y, m, k); break;
    case 5: q6k_entry_b<5>(w, x, y, m, k); break;
    case 6: q6k_entry_b<6>(w, x, y, m, k); break;
    case 7: q6k_entry_b<7>(w, x, y, m, k); break;
    default: q6k_entry_b<8>(w, x, y, m, k); break;
    }
}
