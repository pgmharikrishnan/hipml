// SPDX-License-Identifier: Apache-2.0
// Bit-identity check: the shipped WMMA prefill kernel against the occupancy
// shape (`prefill_wmma2`). Both are compiled from their own headers into this
// one translation unit and launched on the same weights and activations.
//
//   hipcc -O3 --offload-arch=gfx1201 -I crates/hipml-kernels/kernels \
//         scripts/probe_wmma_cmp.cpp -o probe_cmp
//   probe_cmp [q4k|q6k|q8|half] [m] [k] [tokens]
//
// The two kernels step k in the same 16-value order, so the expected result is
// bit identity; anything else is a layout or rounding bug, not a tolerance.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "prefill_wmma.hip"
#include "prefill_wmma2.hip"

static size_t row_bytes(const char* kind, int k) {
    if (!strncmp(kind, "q4k", 3)) return (size_t)(k / 256) * PF_Q4K_BLOCK_BYTES;
    if (!strncmp(kind, "q6k", 3)) return (size_t)(k / 256) * PF_Q6K_BLOCK_BYTES;
    if (!strncmp(kind, "q8", 2)) return (size_t)(k / 32) * PF_Q8_0_BLOCK_BYTES;
    return (size_t)k * 2;
}

// Pseudo-random payload with finite scale fields, so the comparison is over
// real numbers: the raw byte pattern used for timing decodes to NaN/inf halfs
// and would make every value compare equal through NaN.
static void fill_weights(const char* kind, std::vector<unsigned char>& w, int k, int m) {
    for (size_t i = 0; i < w.size(); ++i) w[i] = (unsigned char)(i * 37 + 11);
    const unsigned char one[2] = {0x00, 0x3C};  // _Float16 1.0
    if (!strncmp(kind, "half", 4)) {
        for (size_t i = 0; i + 1 < w.size(); i += 2) { w[i] = 0x00; w[i + 1] = 0x2C; }  // 0.0625
        return;
    }
    const size_t rb = row_bytes(kind, k);
    for (int r = 0; r < m; ++r) {
        unsigned char* row = w.data() + (size_t)r * rb;
        if (!strncmp(kind, "q4k", 3)) {
            for (int b = 0; b < k / 256; ++b) {
                unsigned char* blk = row + (size_t)b * PF_Q4K_BLOCK_BYTES;
                blk[0] = one[0]; blk[1] = one[1];
                blk[2] = 0; blk[3] = 0;
                for (int j = 4; j < 16; ++j) blk[j] = 1;
            }
        } else if (!strncmp(kind, "q6k", 3)) {
            for (int b = 0; b < k / 256; ++b) {
                unsigned char* blk = row + (size_t)b * PF_Q6K_BLOCK_BYTES;
                for (int j = 192; j < 208; ++j) blk[j] = 1;
                blk[208] = one[0]; blk[209] = one[1];
            }
        } else {
            for (int b = 0; b < k / 32; ++b) {
                unsigned char* blk = row + (size_t)b * PF_Q8_0_BLOCK_BYTES;
                blk[0] = one[0]; blk[1] = one[1];
            }
        }
    }
}

int main(int argc, char** argv) {
    const char* kind = argc > 1 ? argv[1] : "q6k";
    const int m = argc > 2 ? atoi(argv[2]) : 5120;
    const int k = argc > 3 ? atoi(argv[3]) : 5120;
    const int tokens = argc > 4 ? atoi(argv[4]) : 200;

    const size_t wb = row_bytes(kind, k) * (size_t)m;
    std::vector<unsigned char> w(wb);
    fill_weights(kind, w, k, m);
    std::vector<float> x((size_t)tokens * k);
    for (size_t i = 0; i < x.size(); ++i) x[i] = (float)((int)(i % 97) - 48) * 0.01f;
    std::vector<float> y1((size_t)tokens * m, 0.f), y2((size_t)tokens * m, 0.f);

    unsigned char* dw; float* dx; float* dy1; float* dy2;
    hipMalloc(&dw, wb); hipMalloc(&dx, x.size() * 4);
    hipMalloc(&dy1, y1.size() * 4); hipMalloc(&dy2, y2.size() * 4);
    hipMemcpy(dw, w.data(), wb, hipMemcpyHostToDevice);
    hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice);
    hipMemset(dy1, 0, y1.size() * 4); hipMemset(dy2, 0, y2.size() * 4);

    const dim3 g1((m + PWW_RT - 1) / PWW_RT, (tokens + PWW_TT - 1) / PWW_TT);
    const dim3 g2((m + PWW2_RT - 1) / PWW2_RT, (tokens + PWW2_TT - 1) / PWW2_TT);
#define ARGS dw, dx, dy1, m, k, tokens
    if (!strncmp(kind, "q4k", 3))
        hipLaunchKernelGGL(prefill_q4_k_w, g1, dim3(PWW_THREADS), 0, 0, ARGS);
    else if (!strncmp(kind, "q6k", 3))
        hipLaunchKernelGGL(prefill_q6_k_w, g1, dim3(PWW_THREADS), 0, 0, ARGS);
    else if (!strncmp(kind, "q8", 2))
        hipLaunchKernelGGL(prefill_q8_0_w, g1, dim3(PWW_THREADS), 0, 0, ARGS);
    else
        hipLaunchKernelGGL(prefill_halfw_w, g1, dim3(PWW_THREADS), 0, 0, ARGS);
    if (!strncmp(kind, "q4k", 3))
        hipLaunchKernelGGL(prefill_q4_k_w2, g2, dim3(PWW2_THREADS), 0, 0, dw, dx, dy2, m, k, tokens);
    else if (!strncmp(kind, "q6k", 3))
        hipLaunchKernelGGL(prefill_q6_k_w2, g2, dim3(PWW2_THREADS), 0, 0, dw, dx, dy2, m, k, tokens);
    else if (!strncmp(kind, "q8", 2))
        hipLaunchKernelGGL(prefill_q8_0_w2, g2, dim3(PWW2_THREADS), 0, 0, dw, dx, dy2, m, k, tokens);
    else
        hipLaunchKernelGGL(prefill_halfw_w2, g2, dim3(PWW2_THREADS), 0, 0, dw, dx, dy2, m, k, tokens);
#undef ARGS
    if (hipDeviceSynchronize() != hipSuccess) { printf("launch failed\n"); return 1; }

    hipMemcpy(y1.data(), dy1, y1.size() * 4, hipMemcpyDeviceToHost);
    hipMemcpy(y2.data(), dy2, y2.size() * 4, hipMemcpyDeviceToHost);

    double max_abs = 0.0, max_rel = 0.0;
    size_t diff = 0, finite = 0;
    for (size_t i = 0; i < y1.size(); ++i) {
        // Compare bit patterns: the probe's synthetic weight bytes decode to
        // inf/NaN scale fields in places, and those must match exactly too.
        unsigned ua, ub;
        memcpy(&ua, &y1[i], 4);
        memcpy(&ub, &y2[i], 4);
        if (ua != ub) ++diff;
        const double a = y1[i], b = y2[i];
        if (a == a && b == b && a != 1.0 / 0.0 && a != -1.0 / 0.0) {
            ++finite;
            const double d = (a > b ? a - b : b - a);
            if (d > max_abs) max_abs = d;
            const double den = (a < 0 ? -a : a) > 1e-12 ? (a < 0 ? -a : a) : 1e-12;
            if (d / den > max_rel) max_rel = d / den;
        }
    }
    printf("%s %dx%d t=%d: v1 vs v2  differing bits %zu/%zu (%zu finite)  max_abs %.3e  max_rel %.3e\n",
           kind, m, k, tokens, diff, y1.size(), finite, max_abs, max_rel);
    return diff == 0 ? 0 : 1;
}
