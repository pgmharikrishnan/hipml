// Batched per-slot work for the decode step.
//
// Every kernel here runs one block-grid axis over the slots (`blockIdx.y`) and
// the existing axis over the per-slot work, so `B` slots cost one launch
// instead of `B`. The per-slot buffers are contiguous: row `slot` starts at
// `slot * slot_stride` floats. The arithmetic of each path is copied from the
// single-slot kernel so a slot's result is unchanged.

#include <hip/hip_runtime.h>

// --- GDN gates -------------------------------------------------------------

extern "C" __global__ void ssm_gates_b(
    const float* __restrict__ alpha,
    const float* __restrict__ beta_raw,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a,
    float* __restrict__ g,
    float* __restrict__ beta,
    int n,
    int slot_stride)
{
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= n) return;
    const size_t off = (size_t)blockIdx.y * slot_stride + h;
    const float z = alpha[off] + dt_bias[h];
    const float sp = z > 20.0f ? z : logf(1.0f + expf(z));
    g[off] = sp * a[h];
    beta[off] = 1.0f / (1.0f + expf(-beta_raw[off]));
}

// --- causal depthwise conv + SiLU -----------------------------------------

extern "C" __global__ void ssm_conv1d_b(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ state,
    float* __restrict__ y,
    int channels,
    int d_conv,
    int x_stride,
    int state_stride)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const int slot = blockIdx.y;
    const int hist = d_conv - 1;
    const float* wc = w + (size_t)c * d_conv;
    const float* xs = x + (size_t)slot * x_stride;
    float* st = state + (size_t)slot * state_stride;

    float acc = 0.0f;
    for (int k = 0; k < hist; ++k) {
        acc += st[(size_t)k * channels + c] * wc[k];
    }
    acc += xs[c] * wc[hist];
    // The single-slot path runs ssm_silu_inplace_f32 next; fusing it here is
    // the same expression.
    y[(size_t)slot * x_stride + c] = acc / (1.0f + expf(-acc));

    for (int k = 0; k + 1 < hist; ++k) {
        st[(size_t)k * channels + c] = st[(size_t)(k + 1) * channels + c];
    }
    if (hist > 0) {
        st[(size_t)(hist - 1) * channels + c] = xs[c];
    }
}

// --- per-head L2 norm (q | k share one launch) -----------------------------

#define BLR_BLOCK 256

extern "C" __global__ void l2norm_b(
    const float* x,
    float* y,
    int heads,
    int dim,
    float eps,
    int slot_stride)
{
    const int h = blockIdx.x;
    const int slot = blockIdx.y;
    const int t = threadIdx.x;
    const float* src = x + (size_t)slot * slot_stride + (size_t)h * dim;
    float* dst = y + (size_t)slot * slot_stride + (size_t)h * dim;

    float acc = 0.0f;
    for (int i = t; i < dim; i += BLR_BLOCK) {
        acc += src[i] * src[i];
    }

    __shared__ float s[BLR_BLOCK];
    s[t] = acc;
    __syncthreads();
    for (int off = BLR_BLOCK / 2; off > 0; off >>= 1) {
        if (t < off) s[t] += s[t + off];
        __syncthreads();
    }

    __shared__ float scale;
    if (t == 0) scale = rsqrtf(s[0] + eps);
    __syncthreads();

    for (int i = t; i < dim; i += BLR_BLOCK) {
        dst[i] = src[i] * scale;
    }
}

// --- gated delta net, one block per (value head, slot) ---------------------

#define GDN_MAXV 4
#define GDN_LANES 32

static __device__ __forceinline__ float gdn_wave_sum(float v)
{
    for (int off = GDN_LANES / 2; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffffffffffULL, v, off, GDN_LANES);
    }
    return v;
}

extern "C" __global__ void gdn_ar_b(
    float* __restrict__ s,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ g,
    const float* __restrict__ beta,
    float* __restrict__ out,
    int n_v,
    int n_k,
    int sd,
    int qkv_stride,
    int vec_stride,
    int out_stride,
    int state_stride)
{
    const int h = blockIdx.x;
    const int slot = blockIdx.y;
    const int lane = threadIdx.x % GDN_LANES;
    const int wave = threadIdx.x / GDN_LANES;
    const int nwave = blockDim.x / GDN_LANES;

    const float* qs = q + (size_t)slot * qkv_stride;
    const float* ks = k + (size_t)slot * qkv_stride;
    const float* vs = v + (size_t)slot * qkv_stride;
    const float* gs = g + (size_t)slot * vec_stride;
    const float* bs = beta + (size_t)slot * vec_stride;
    float* os = out + (size_t)slot * out_stride;
    float* st = s + (size_t)slot * state_stride;

    const int hk = h % n_k;
    const float decay = expf(gs[h]);
    const float b = bs[h];
    const float scale = rsqrtf((float)sd);

    const float* qh = qs + (size_t)hk * sd;
    const float* kh = ks + (size_t)hk * sd;
    const float* vh = vs + (size_t)h * sd;
    float* head = st + (size_t)h * sd * sd;

    float qq[GDN_MAXV];
    float kk[GDN_MAXV];
    #pragma unroll
    for (int t = 0; t < GDN_MAXV; ++t) {
        const int i = lane + t * GDN_LANES;
        if (i < sd) {
            qq[t] = qh[i];
            kk[t] = kh[i];
        } else {
            qq[t] = 0.0f;
            kk[t] = 0.0f;
        }
    }

    for (int j = wave; j < sd; j += nwave) {
        float* row = head + (size_t)j * sd;
        float r[GDN_MAXV];
        float dot = 0.0f;
        #pragma unroll
        for (int t = 0; t < GDN_MAXV; ++t) {
            const int i = lane + t * GDN_LANES;
            r[t] = i < sd ? row[i] : 0.0f;
            dot += r[t] * decay * kk[t];
        }
        dot = gdn_wave_sum(dot);

        const float delta = (vh[j] - dot) * b;

        float acc = 0.0f;
        #pragma unroll
        for (int t = 0; t < GDN_MAXV; ++t) {
            const int i = lane + t * GDN_LANES;
            if (i < sd) {
                const float nv = r[t] * decay + kk[t] * delta;
                row[i] = nv;
                acc += nv * qq[t];
            }
        }
        acc = gdn_wave_sum(acc);

        if (lane == 0) {
            os[(size_t)h * sd + j] = acc * scale;
        }
    }
}

// --- gated RMS norm --------------------------------------------------------

#define BGN_BLOCK 256

extern "C" __global__ void ssm_gate_norm_b(
    const float* __restrict__ o,
    const float* __restrict__ norm,
    const float* __restrict__ z,
    float* __restrict__ y,
    int heads,
    int dim,
    float eps,
    int slot_stride)
{
    const int h = blockIdx.x;
    const int slot = blockIdx.y;
    const int t = threadIdx.x;
    const float* src = o + (size_t)slot * slot_stride + (size_t)h * dim;

    float acc = 0.0f;
    for (int i = t; i < dim; i += BGN_BLOCK) {
        acc += src[i] * src[i];
    }

    __shared__ float s[BGN_BLOCK];
    s[t] = acc;
    __syncthreads();
    for (int off = BGN_BLOCK / 2; off > 0; off >>= 1) {
        if (t < off) s[t] += s[t + off];
        __syncthreads();
    }

    __shared__ float scale;
    if (t == 0) scale = rsqrtf(s[0] / (float)dim + eps);
    __syncthreads();

    float* dst = y + (size_t)slot * slot_stride + (size_t)h * dim;
    const float* zz = z + (size_t)slot * slot_stride + (size_t)h * dim;
    for (int i = t; i < dim; i += BGN_BLOCK) {
        const float gv = zz[i];
        const float silu = gv / (1.0f + expf(-gv));
        dst[i] = src[i] * scale * norm[i] * silu;
    }
}
