// Quantized-block decoding shared by the tiled prefill GEMMs.
//
// Q4_K, Q6_K and Q8_0 differ only in how 8 contiguous values of a row are
// unpacked, so every prefill kernel takes the block layout through this one
// templated reader: a wrong layout can only be wrong in one place, and the CPU
// oracle (`hipml-core/src/quant.rs`) is the reference for all of them.
//
// Included by `prefill_f16.hip` (packed-fp16 dot2 inner loop) and
// `prefill_wmma.hip` (matrix unit). Not a `.hip` file, so build.rs does not
// compile it on its own.

#define PF_Q4K_BLOCK_BYTES 144
#define PF_Q6K_BLOCK_BYTES 210
#define PF_Q8_0_BLOCK_BYTES 34

static __device__ __forceinline__ void pf_scale_min(int j, const unsigned char* q,
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

// Dequantise 8 contiguous values of one row, starting at value index `v`
// (a multiple of 8). KIND: 0 Q4_K, 1 Q6_K, 2 Q8_0, 3 already-half.
template <int KIND>
static __device__ __forceinline__ void pf_dequant8(const unsigned char* __restrict__ rp,
                                                   int v, float* out)
{
    if (KIND == 0) {
        const int blk = v >> 8;
        const int within = v & 255;
        const int sub = within >> 5;
        const int part = (within & 31) >> 3;
        const unsigned char* b = rp + (size_t)blk * PF_Q4K_BLOCK_BYTES;
        const float d = __half2float(*reinterpret_cast<const __half*>(b));
        const float dmin = __half2float(*reinterpret_cast<const __half*>(b + 2));
        int sc, mn;
        pf_scale_min(sub, b + 4, &sc, &mn);
        const float scale = d * (float)sc;
        const float minv = dmin * (float)mn;
        const bool high = (sub & 1) != 0;
        const unsigned char* qp = b + 16 + (sub >> 1) * 32 + part * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int q = high ? (qp[i] >> 4) : (qp[i] & 0xF);
            out[i] = scale * (float)q - minv;
        }
    } else if (KIND == 1) {
        const int blk = v >> 8;
        const int within = v & 255;
        const int j = within >> 4;
        const int l0 = within & 15;
        const unsigned char* b = rp + (size_t)blk * PF_Q6K_BLOCK_BYTES;
        const float d = __half2float(*reinterpret_cast<const __half*>(b + 208));
        const float scale =
            d * (float)reinterpret_cast<const signed char*>(b + 192)[j];
        const int half = j >> 3;
        const int s = j & 7;
        const int group = s >> 1;
        const int is = s & 1;
        const int ql_off = half * 64 + (group & 1) * 32 + is * 16 + l0;
        const int qh_off = 128 + half * 32 + is * 16 + l0;
        const int shift = 2 * group;
        const bool high = group >= 2;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int qb = b[ql_off + i];
            const int hb = (b[qh_off + i] >> shift) & 3;
            const int nib = high ? (qb >> 4) : (qb & 0xF);
            out[i] = scale * (float)((nib | (hb << 4)) - 32);
        }
    } else {
        const int blk = v >> 5;
        const int part = (v & 31) >> 3;
        const unsigned char* b = rp + (size_t)blk * PF_Q8_0_BLOCK_BYTES;
        const float d = __half2float(*reinterpret_cast<const __half*>(b));
        const signed char* qs = reinterpret_cast<const signed char*>(b + 2 + part * 8);
#pragma unroll
        for (int i = 0; i < 8; ++i) out[i] = d * (float)qs[i];
    }
}

// Bytes per row of a block-quantized weight, and the k-tile's byte offset
// within it. KIND == 3 rows are already half, so the "scale" is 2 bytes/value.
static __device__ __forceinline__ size_t pf_row_bytes(int k, int kind)
{
    return kind == 3 ? (size_t)k * 2
         : kind == 0 ? (size_t)(k / 256) * PF_Q4K_BLOCK_BYTES
         : kind == 1 ? (size_t)(k / 256) * PF_Q6K_BLOCK_BYTES
                     : (size_t)(k / 32) * PF_Q8_0_BLOCK_BYTES;
}
