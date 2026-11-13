// SPDX-License-Identifier: Apache-2.0
// What the fp8 staging tile costs in the numbers, against an exact f64 host
// reference.
//
//   hipcc -O3 --offload-arch=gfx1201 probe_prefill_fp8_acc.cpp -o acc_f16
//   hipcc -O3 --offload-arch=gfx1201 -DPWW2_W8=1 probe_prefill_fp8_acc.cpp -o acc_fp8
//   acc_f16 0 512 5120 8      # Q4_K blocks
//   acc_f16 3 512 5120 8      # fp16 weights: the same comparison with no decoder
//
// Both binaries run the shipped tile (`prefill_wmma2.hip`, the same source the
// build uses); the only difference is whether the LDS tiles are fp16 or fp8 and
// which WMMA multiplies them. The host reference is f64 over the *exact*
// dequantised weights and the exact fp16 activation values the kernel reads, so
// the residual is the tile's own arithmetic and staging, not the reference's.
//
// The fp16 build's residual is therefore the baseline the fp8 build has to be
// judged against, and `kind 3` (weights already half in memory) removes the
// decoder from both sides so the difference is purely the fp8 representation.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "prefill_wmma2.hip"

static size_t row_bytes(int kind, int k)
{
    switch (kind) {
    case 0: return (size_t)(k / 256) * PF_Q4K_BLOCK_BYTES;
    case 3: return (size_t)k * 2;
    default: return 0;
    }
}

static const char* kind_name(int kind) { return kind == 0 ? "Q4_K" : "fp16"; }

// Q4_K's 6-bit sub-block scale and min, from the 12 scale bytes.
static void q4k_scale_min(const unsigned char* sc, int sub, int* s, int* m)
{
    if (sub < 4) {
        *s = sc[sub] & 63;
        *m = sc[4 + sub] & 63;
    } else {
        const int lo = sc[sub - 4], mid = sc[sub], hi = sc[sub + 4];
        *s = (hi & 0xF) | ((lo & 0xC0) >> 2);
        *m = (hi >> 4) | ((mid & 0xC0) >> 2);
    }
}

// One Q4_K value in f64: d * scale[q] * nibble - dmin * min[q].
static double q4k_value(const unsigned char* blk, int v)
{
    const int within = v & 255, sub = within >> 5, part = (within & 31) >> 3;
    const __half d = *(const __half*)blk;
    const __half dmin = *((const __half*)blk + 1);
    const unsigned char* sc = blk + 4;
    int s, m;
    q4k_scale_min(sc, sub, &s, &m);
    const int byte = 16 + (sub >> 1) * 32 + part * 8 + (within & 7);
    const int q = (sub & 1) ? ((blk[byte] >> 4) & 0xF) : (blk[byte] & 0xF);
    return (double)__half2float(d) * (double)s * (double)q -
           (double)__half2float(dmin) * (double)m;
}

// A deterministic PRNG so both builds see the same bytes.
static unsigned rng_state = 0x9F130001u;
static unsigned rng()
{
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state >> 8;
}

// Valid Q4_K blocks: real fp16 scales/mins in the model's range, random
// nibbles. The decoder's arithmetic is what the tile pays for either way; what
// matters here is that the *values* land where a real model's do.
static void fill_q4_k(std::vector<unsigned char>& w, int k)
{
    const size_t blocks = w.size() / PF_Q4K_BLOCK_BYTES;
    for (size_t b = 0; b < blocks; ++b) {
        unsigned char* blk = w.data() + b * PF_Q4K_BLOCK_BYTES;
        const __half d = __float2half(0.001953125f * (float)(1 + (rng() % 8)));
        const __half dmin = __float2half(0.001953125f * (float)(1 + (rng() % 8)));
        memcpy(blk, &d, 2);
        memcpy(blk + 2, &dmin, 2);
        for (int i = 0; i < 12; ++i) blk[4 + i] = (unsigned char)(rng() & 0xFF);
        for (int i = 0; i < 128; ++i) blk[16 + i] = (unsigned char)(rng() & 0xFF);
    }
    (void)k;
}

static void launch(int kind, dim3 grid, dim3 block, const unsigned char* w, const void* x,
                   float* y, int m, int k, int tokens)
{
    if (kind == 0)
        hipLaunchKernelGGL(prefill_q4_k_w2h, grid, block, 0, 0, w, x, y, m, k, tokens);
    else
        hipLaunchKernelGGL(prefill_halfw_w2h, grid, block, 0, 0, w, x, y, m, k, tokens);
}

int main(int argc, char** argv)
{
    const int kind = argc > 1 ? atoi(argv[1]) : 0;
    const int m = argc > 2 ? atoi(argv[2]) : 512;
    const int k = argc > 3 ? atoi(argv[3]) : 5120;
    const int tokens = argc > 4 ? atoi(argv[4]) : 8;

    const size_t rb = row_bytes(kind, k);
    const size_t wb = rb * (size_t)m;
    std::vector<unsigned char> w(wb);
    if (kind == 0)
        fill_q4_k(w, k);
    else
        for (size_t i = 0; i < wb / 2; ++i) {
            const __half h = __float2half(((float)((int)(rng() % 2001) - 1000)) / 8192.0f);
            memcpy(w.data() + 2 * i, &h, 2);
        }

    std::vector<__half> x((size_t)tokens * k);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = __float2half(((float)((int)(rng() % 4001) - 2000)) / 2048.0f);

    std::vector<float> y((size_t)tokens * m);
    unsigned char* dw = nullptr;
    __half* dx = nullptr;
    float* dy = nullptr;
    hipMalloc(&dw, wb);
    hipMalloc(&dx, x.size() * 2);
    hipMalloc(&dy, y.size() * sizeof(float));
    hipMemcpy(dw, w.data(), wb, hipMemcpyHostToDevice);
    hipMemcpy(dx, x.data(), x.size() * 2, hipMemcpyHostToDevice);
    hipMemset(dy, 0, y.size() * sizeof(float));

    const dim3 grid((unsigned)((m + PWW2_RT - 1) / PWW2_RT),
                    (unsigned)((tokens + PWW2_TT - 1) / PWW2_TT), 1);
    launch(kind, grid, dim3(PWW2_THREADS, 1, 1), dw, dx, dy, m, k, tokens);
    if (hipDeviceSynchronize() != hipSuccess) {
        printf("launch failed\n");
        return 1;
    }
    hipMemcpy(y.data(), dy, y.size() * sizeof(float), hipMemcpyDeviceToHost);

    // Host reference: f64 over the exact staged weights and the exact fp16
    // activation values, in k order. Q4_K values are recomputed from the block
    // bytes; fp16 weights are read as the same halves the kernel reads.
    std::vector<double> ref((size_t)tokens * m, 0.0);
    for (int t = 0; t < tokens; ++t) {
        for (int r = 0; r < m; ++r) {
            const unsigned char* row = w.data() + (size_t)r * rb;
            double s = 0.0;
            for (int j = 0; j < k; ++j) {
                const double wv = kind == 0
                                      ? q4k_value(row + (size_t)(j / 256) * PF_Q4K_BLOCK_BYTES, j % 256)
                                      : (double)__half2float(((const __half*)row)[j]);
                s += wv * (double)__half2float(x[(size_t)t * k + j]);
            }
            ref[(size_t)t * m + r] = s;
        }
    }

    double max_abs = 0.0, max_ref = 0.0, sum_sq = 0.0;
    size_t n = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        const double d = (double)y[i] - ref[i];
        if (fabs(d) > max_abs) max_abs = fabs(d);
        if (fabs(ref[i]) > max_ref) max_ref = fabs(ref[i]);
        sum_sq += d * d;
        ++n;
    }
    const double rms = sqrt(sum_sq / (double)n);
    printf("kind %-5s W8=%d m=%d k=%d t=%d  max|gpu-ref| %.6g  rms %.6g  max|ref| %.6g  rel %.3g\n",
           kind_name(kind), PWW2_W8, m, k, tokens, max_abs, rms, max_ref,
           max_abs / (max_ref > 0 ? max_ref : 1.0));

    // What int4 would cost, which is the price of using
    // `v_wmma_i32_16x16x32_iu4` at all: that instruction takes both operands as
    // packed 4-bit, so the activation has to be int4 too. Simulated on the host
    // with per-32-block symmetric scales - the best case, since a real tile
    // would also carry the per-block int32 rescale and Q4_K's sum(x) term - and
    // measured against the same f64 reference, so it is directly comparable
    // with the fp8 column above.
    double i4_abs = 0.0, i4_sq = 0.0;
    for (int t = 0; t < tokens; ++t) {
        for (int r = 0; r < m; ++r) {
            const unsigned char* row = w.data() + (size_t)r * rb;
            double acc = 0.0;
            for (int b = 0; b < k / 32; ++b) {
                double wmax = 0.0, xmax = 0.0;
                for (int j = 0; j < 32; ++j) {
                    const int kk = b * 32 + j;
                    const double wv = kind == 0
                                          ? q4k_value(row + (size_t)(kk / 256) * PF_Q4K_BLOCK_BYTES,
                                                      kk % 256)
                                          : (double)__half2float(((const __half*)row)[kk]);
                    const double xv = (double)__half2float(x[(size_t)t * k + kk]);
                    if (fabs(wv) > wmax) wmax = fabs(wv);
                    if (fabs(xv) > xmax) xmax = fabs(xv);
                }
                const double sw = wmax > 0 ? wmax / 7.0 : 0.0;
                const double sx = xmax > 0 ? xmax / 7.0 : 0.0;
                if (sw == 0.0 || sx == 0.0) continue;   // whole block is zero
                for (int j = 0; j < 32; ++j) {
                    const int kk = b * 32 + j;
                    const double wv = kind == 0
                                          ? q4k_value(row + (size_t)(kk / 256) * PF_Q4K_BLOCK_BYTES,
                                                      kk % 256)
                                          : (double)__half2float(((const __half*)row)[kk]);
                    const double xv = (double)__half2float(x[(size_t)t * k + kk]);
                    acc += (round(wv / sw) * sw) * (round(xv / sx) * sx);
                }
            }
            const double d = acc - ref[(size_t)t * m + r];
            if (fabs(d) > i4_abs) i4_abs = fabs(d);
            i4_sq += d * d;
        }
    }
    printf("      int4 x int4, per-32-block scales, same reference: max|err| %.6g  rms %.6g  rel %.3g\n",
           i4_abs, sqrt(i4_sq / (double)((size_t)tokens * m)), i4_abs / (max_ref > 0 ? max_ref : 1.0));
    hipFree(dw);
    hipFree(dx);
    hipFree(dy);
    return 0;
}
