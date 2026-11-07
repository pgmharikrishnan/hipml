// y[row] = dequant_q4_k(W[row,:]) . x
//
// Row-tile GEMV: one block owns ROW_TILE output rows and stages the x k-tile in
// shared memory once. Each warp owns ROWS_PER_WARP rows and keeps one
// accumulator per row, so a value read from shared is applied to
// ROWS_PER_WARP weight rows before it is discarded. With 4-bit weights x is 8x
// the weight bytes, so this reuse is what removes the shared/L2 ceiling.
//
// The shared tile is de-interleaved by 256-value group: component `c` of a group
// lives at sx[g*256 + c*32 + lane]. A lane reads eight consecutive columns as
// eight scalar shared loads, one bank per lane, so there are no bank conflicts.
//
// Each lane owns 8 consecutive columns per row; their 8 Q4_K bytes are two
// 32-bit words and one d/dmin/scale/min set covers them.
//
// Q4_K super-block (144 bytes, 256 values):
//   d(2) dmin(2) scales(12) qs(128)
// Math is unchanged: y = d*scale[j]*q - dmin*min[j] for sub-block j.
//
// Two barriers per k-tile; the only reduction is a warp shuffle tree per row.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define Q4K_VALUES 256
#define Q4K_BLOCK_BYTES 144
#define BLOCK 256
#define WARPS (BLOCK / 32)
#define ROWS_PER_WARP 2
#define ROW_TILE (WARPS * ROWS_PER_WARP)
#define K_TILE 4096

// Mirrors the manifest's KernelKind::RowTile { rows = ROW_TILE, block = BLOCK };
// change both together.

static __device__ __forceinline__ void q4k_scale_min(int j, const unsigned char* q,
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

static __device__ __forceinline__ void stage_group(const float* __restrict__ x,
                                                   float* __restrict__ sx,
                                                   int base, int j)
{
    const float4 a = reinterpret_cast<const float4*>(x + base)[2 * j];
    const float4 b = reinterpret_cast<const float4*>(x + base)[2 * j + 1];
    const int g = j >> 5; // 32 chunks of 8 values per 256-value group
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

// Eight-column contribution of one Q4_K row to `acc`.
static __device__ __forceinline__ void q4k_acc8(const unsigned char* __restrict__ rp,
                                                int p, const float* xv, float* acc)
{
    const int blk = p >> 8;
    const int within = p & 255;
    const int sub = within >> 5;
    const int part = (within & 31) >> 3;

    const unsigned char* b = rp + (size_t)blk * Q4K_BLOCK_BYTES;
    const float d = __half2float(*reinterpret_cast<const __half*>(b));
    const float dmin = __half2float(*reinterpret_cast<const __half*>(b + 2));
    int sc, mn;
    q4k_scale_min(sub, b + 4, &sc, &mn);
    const float scale = d * (float)sc;
    const float minv = dmin * (float)mn;

    const unsigned int* qp = reinterpret_cast<const unsigned int*>(
        b + 16 + (sub >> 1) * 32 + part * 8);
    union Q4Bytes {
        unsigned int w[2];
        unsigned char b[8];
    } q;
    q.w[0] = qp[0];
    q.w[1] = qp[1];
    const bool high = (sub & 1) != 0;

    float a = *acc;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const unsigned int byte = q.b[i];
        const int v = high ? (int)(byte >> 4) : (int)(byte & 0xF);
        a = fmaf(scale * (float)v - minv, xv[i], a);
    }
    *acc = a;
}

extern "C" __global__ void gemv_q4_k(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k)
{
    __shared__ __align__(16) float sx[K_TILE];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nblocks = k / Q4K_VALUES;

    int row[ROWS_PER_WARP];
    int r[ROWS_PER_WARP];
    const unsigned char* rp[ROWS_PER_WARP];
    float acc[ROWS_PER_WARP];
#pragma unroll
    for (int rr = 0; rr < ROWS_PER_WARP; ++rr) {
        row[rr] = blockIdx.x * ROW_TILE + warp * ROWS_PER_WARP + rr;
        r[rr] = min(row[rr], m - 1);
        rp[rr] = w + (size_t)r[rr] * (size_t)nblocks * Q4K_BLOCK_BYTES;
        acc[rr] = 0.0f;
    }

    for (int base = 0; base < k; base += K_TILE) {
        const int n = min(K_TILE, k - base);
        const int nchunks = n >> 3;
        for (int j = tid; j < nchunks; j += BLOCK) stage_group(x, sx, base, j);
        __syncthreads();

        for (int c = lane * 8; c < n; c += 256) {
            const int g = c >> 8;
            const int s = g * 256 + lane;
            float xv[8];
#pragma unroll
            for (int comp = 0; comp < 8; ++comp) xv[comp] = sx[s + comp * 32];
            const int p = base + c;
#pragma unroll
            for (int rr = 0; rr < ROWS_PER_WARP; ++rr) q4k_acc8(rp[rr], p, xv, &acc[rr]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int rr = 0; rr < ROWS_PER_WARP; ++rr) {
        float a = acc[rr];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) a += __shfl_down(a, off);
        if (lane == 0 && row[rr] < m) y[row[rr]] = a;
    }
}
