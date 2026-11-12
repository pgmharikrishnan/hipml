// SPDX-License-Identifier: Apache-2.0
// Quantized-block decoding shared by the tiled prefill GEMMs.
//
// Q4_K, Q6_K and Q8_0 differ only in how 8 contiguous values of a row are
// unpacked, so every prefill kernel takes the block layout through this one
// templated reader: a wrong layout can only be wrong in one place, and the CPU
// oracle (`hipml-core/src/quant.rs`) is the reference for all of them.
//
// Included by `prefill_f16.hip` (packed-fp16 dot2 inner loop) and
// `prefill_wmma.hip` (matrix unit). Not a `.hip` file, so build.rs does not
// compile it on its own; the guard is what lets both include it from the one
// amalgamated translation unit.

#ifndef PF_PREFILL_COMMON_INC
#define PF_PREFILL_COMMON_INC

#define PF_Q4K_BLOCK_BYTES 144
#define PF_Q5K_BLOCK_BYTES 176
#define PF_Q6K_BLOCK_BYTES 210
#define PF_Q8_0_BLOCK_BYTES 34
#define PF_IQ4_XS_BLOCK_BYTES 136
#define PF_IQ4_NL_BLOCK_BYTES 18
#define PF_Q3K_BLOCK_BYTES 110
#define PF_IQ3S_BLOCK_BYTES 110

// ggml-common.h `kvalues_iq4nl`, shared by the two IQ4 kinds.
static __device__ __constant__ const signed char PF_IQ4NL_KV[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

// Weight bytes are streamed once per prompt tile and must not evict the
// activation working set, so they can be loaded non-temporally (gfx12:
// `global_load ... th:TH_LOAD_NT`, evict-first). Off by default; the prefill
// GEMM turns it on with `PF_NT=1`.
#ifndef PF_NT
#define PF_NT 0
#endif

static __device__ __forceinline__ unsigned char pf_ld8(const unsigned char* p) {
#if PF_NT
    return __builtin_nontemporal_load(p);
#else
    return *p;
#endif
}

static __device__ __forceinline__ __half pf_ldh(const __half* p) {
#if PF_NT
    // __builtin_nontemporal_load does not take __half; load the bits as
    // unsigned short and cast.
    return __ushort_as_half(
        __builtin_nontemporal_load(reinterpret_cast<const unsigned short*>(p)));
#else
    return *p;
#endif
}

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


// ggml-common.h `iq3s_grid`, four packed codebook values per index.
static __device__ __constant__ const unsigned int PF_IQ3S_GRID[512] = {
    0x01010101u, 0x01010103u, 0x01010105u, 0x0101010Bu, 0x0101010Fu, 0x01010301u, 0x01010303u, 0x01010305u,
    0x01010309u, 0x0101030Du, 0x01010501u, 0x01010503u, 0x0101050Bu, 0x01010707u, 0x01010901u, 0x01010905u,
    0x0101090Bu, 0x0101090Fu, 0x01010B03u, 0x01010B07u, 0x01010D01u, 0x01010D05u, 0x01010F03u, 0x01010F09u,
    0x01010F0Fu, 0x01030101u, 0x01030103u, 0x01030105u, 0x01030109u, 0x01030301u, 0x01030303u, 0x0103030Bu,
    0x01030501u, 0x01030507u, 0x0103050Fu, 0x01030703u, 0x0103070Bu, 0x01030909u, 0x01030D03u, 0x01030D0Bu,
    0x01030F05u, 0x01050101u, 0x01050103u, 0x0105010Bu, 0x0105010Fu, 0x01050301u, 0x01050307u, 0x0105030Du,
    0x01050503u, 0x0105050Bu, 0x01050701u, 0x01050709u, 0x01050905u, 0x0105090Bu, 0x0105090Fu, 0x01050B03u,
    0x01050B07u, 0x01050F01u, 0x01050F07u, 0x01070107u, 0x01070303u, 0x0107030Bu, 0x01070501u, 0x01070505u,
    0x01070703u, 0x01070707u, 0x0107070Du, 0x01070909u, 0x01070B01u, 0x01070B05u, 0x01070D0Fu, 0x01070F03u,
    0x01070F0Bu, 0x01090101u, 0x01090307u, 0x0109030Fu, 0x01090503u, 0x01090509u, 0x01090705u, 0x01090901u,
    0x01090907u, 0x01090B03u, 0x01090F01u, 0x010B0105u, 0x010B0109u, 0x010B0501u, 0x010B0505u, 0x010B050Du,
    0x010B0707u, 0x010B0903u, 0x010B090Bu, 0x010B090Fu, 0x010B0D0Du, 0x010B0F07u, 0x010D010Du, 0x010D0303u,
    0x010D0307u, 0x010D0703u, 0x010D0B05u, 0x010D0F03u, 0x010F0101u, 0x010F0105u, 0x010F0109u, 0x010F0501u,
    0x010F0505u, 0x010F050Du, 0x010F0707u, 0x010F0B01u, 0x010F0B09u, 0x03010101u, 0x03010103u, 0x03010105u,
    0x03010109u, 0x03010301u, 0x03010303u, 0x03010307u, 0x0301030Bu, 0x0301030Fu, 0x03010501u, 0x03010505u,
    0x03010703u, 0x03010709u, 0x0301070Du, 0x03010B09u, 0x03010B0Du, 0x03010D03u, 0x03010F05u, 0x03030101u,
    0x03030103u, 0x03030107u, 0x0303010Du, 0x03030301u, 0x03030309u, 0x03030503u, 0x03030701u, 0x03030707u,
    0x03030903u, 0x03030B01u, 0x03030B05u, 0x03030F01u, 0x03030F0Du, 0x03050101u, 0x03050305u, 0x0305030Bu,
    0x0305030Fu, 0x03050501u, 0x03050509u, 0x03050705u, 0x03050901u, 0x03050907u, 0x03050B0Bu, 0x03050D01u,
    0x03050F05u, 0x03070103u, 0x03070109u, 0x0307010Fu, 0x03070301u, 0x03070307u, 0x03070503u, 0x0307050Fu,
    0x03070701u, 0x03070709u, 0x03070903u, 0x03070D05u, 0x03070F01u, 0x03090107u, 0x0309010Bu, 0x03090305u,
    0x03090309u, 0x03090703u, 0x03090707u, 0x03090905u, 0x0309090Du, 0x03090B01u, 0x03090B09u, 0x030B0103u,
    0x030B0301u, 0x030B0307u, 0x030B0503u, 0x030B0701u, 0x030B0705u, 0x030B0B03u, 0x030D0501u, 0x030D0509u,
    0x030D050Fu, 0x030D0909u, 0x030D090Du, 0x030F0103u, 0x030F0107u, 0x030F0301u, 0x030F0305u, 0x030F0503u,
    0x030F070Bu, 0x030F0903u, 0x030F0D05u, 0x030F0F01u, 0x05010101u, 0x05010103u, 0x05010107u, 0x0501010Bu,
    0x0501010Fu, 0x05010301u, 0x05010305u, 0x05010309u, 0x0501030Du, 0x05010503u, 0x05010507u, 0x0501050Fu,
    0x05010701u, 0x05010705u, 0x05010903u, 0x05010907u, 0x0501090Bu, 0x05010B01u, 0x05010B05u, 0x05010D0Fu,
    0x05010F01u, 0x05010F07u, 0x05010F0Bu, 0x05030101u, 0x05030105u, 0x05030301u, 0x05030307u, 0x0503030Fu,
    0x05030505u, 0x0503050Bu, 0x05030703u, 0x05030709u, 0x05030905u, 0x05030B03u, 0x05050103u, 0x05050109u,
    0x0505010Fu, 0x05050503u, 0x05050507u, 0x05050701u, 0x0505070Fu, 0x05050903u, 0x05050B07u, 0x05050B0Fu,
    0x05050F03u, 0x05050F09u, 0x05070101u, 0x05070105u, 0x0507010Bu, 0x05070303u, 0x05070505u, 0x05070509u,
    0x05070703u, 0x05070707u, 0x05070905u, 0x05070B01u, 0x05070D0Du, 0x05090103u, 0x0509010Fu, 0x05090501u,
    0x05090507u, 0x05090705u, 0x0509070Bu, 0x05090903u, 0x05090F05u, 0x05090F0Bu, 0x050B0109u, 0x050B0303u,
    0x050B0505u, 0x050B070Fu, 0x050B0901u, 0x050B0B07u, 0x050B0F01u, 0x050D0101u, 0x050D0105u, 0x050D010Fu,
    0x050D0503u, 0x050D0B0Bu, 0x050D0D03u, 0x050F010Bu, 0x050F0303u, 0x050F050Du, 0x050F0701u, 0x050F0907u,
    0x050F0B01u, 0x07010105u, 0x07010303u, 0x07010307u, 0x0701030Bu, 0x0701030Fu, 0x07010505u, 0x07010703u,
    0x07010707u, 0x0701070Bu, 0x07010905u, 0x07010909u, 0x0701090Fu, 0x07010B03u, 0x07010D07u, 0x07010F03u,
    0x07030103u, 0x07030107u, 0x0703010Bu, 0x07030309u, 0x07030503u, 0x07030507u, 0x07030901u, 0x07030D01u,
    0x07030F05u, 0x07030F0Du, 0x07050101u, 0x07050305u, 0x07050501u, 0x07050705u, 0x07050709u, 0x07050B01u,
    0x07070103u, 0x07070301u, 0x07070309u, 0x07070503u, 0x07070507u, 0x0707050Fu, 0x07070701u, 0x07070903u,
    0x07070907u, 0x0707090Fu, 0x07070B0Bu, 0x07070F07u, 0x07090107u, 0x07090303u, 0x0709030Du, 0x07090505u,
    0x07090703u, 0x07090B05u, 0x07090D01u, 0x07090D09u, 0x070B0103u, 0x070B0301u, 0x070B0305u, 0x070B050Bu,
    0x070B0705u, 0x070B0909u, 0x070B0B0Du, 0x070B0F07u, 0x070D030Du, 0x070D0903u, 0x070F0103u, 0x070F0107u,
    0x070F0501u, 0x070F0505u, 0x070F070Bu, 0x09010101u, 0x09010109u, 0x09010305u, 0x09010501u, 0x09010509u,
    0x0901050Fu, 0x09010705u, 0x09010903u, 0x09010B01u, 0x09010F01u, 0x09030105u, 0x0903010Fu, 0x09030303u,
    0x09030307u, 0x09030505u, 0x09030701u, 0x0903070Bu, 0x09030907u, 0x09030B03u, 0x09030B0Bu, 0x09050103u,
    0x09050107u, 0x09050301u, 0x0905030Bu, 0x09050503u, 0x09050707u, 0x09050901u, 0x09050B0Fu, 0x09050D05u,
    0x09050F01u, 0x09070109u, 0x09070303u, 0x09070307u, 0x09070501u, 0x09070505u, 0x09070703u, 0x0907070Bu,
    0x09090101u, 0x09090105u, 0x09090509u, 0x0909070Fu, 0x09090901u, 0x09090F03u, 0x090B010Bu, 0x090B010Fu,
    0x090B0503u, 0x090B0D05u, 0x090D0307u, 0x090D0709u, 0x090D0D01u, 0x090F0301u, 0x090F030Bu, 0x090F0701u,
    0x090F0907u, 0x090F0B03u, 0x0B010105u, 0x0B010301u, 0x0B010309u, 0x0B010505u, 0x0B010901u, 0x0B010909u,
    0x0B01090Fu, 0x0B010B05u, 0x0B010D0Du, 0x0B010F09u, 0x0B030103u, 0x0B030107u, 0x0B03010Bu, 0x0B030305u,
    0x0B030503u, 0x0B030705u, 0x0B030F05u, 0x0B050101u, 0x0B050303u, 0x0B050507u, 0x0B050701u, 0x0B05070Du,
    0x0B050B07u, 0x0B070105u, 0x0B07010Fu, 0x0B070301u, 0x0B07050Fu, 0x0B070909u, 0x0B070B03u, 0x0B070D0Bu,
    0x0B070F07u, 0x0B090103u, 0x0B090109u, 0x0B090501u, 0x0B090705u, 0x0B09090Du, 0x0B0B0305u, 0x0B0B050Du,
    0x0B0B0B03u, 0x0B0B0B07u, 0x0B0D0905u, 0x0B0F0105u, 0x0B0F0109u, 0x0B0F0505u, 0x0D010303u, 0x0D010307u,
    0x0D01030Bu, 0x0D010703u, 0x0D010707u, 0x0D010D01u, 0x0D030101u, 0x0D030501u, 0x0D03050Fu, 0x0D030D09u,
    0x0D050305u, 0x0D050709u, 0x0D050905u, 0x0D050B0Bu, 0x0D050D05u, 0x0D050F01u, 0x0D070101u, 0x0D070309u,
    0x0D070503u, 0x0D070901u, 0x0D09050Bu, 0x0D090907u, 0x0D090D05u, 0x0D0B0101u, 0x0D0B0107u, 0x0D0B0709u,
    0x0D0B0D01u, 0x0D0D010Bu, 0x0D0D0901u, 0x0D0F0303u, 0x0D0F0307u, 0x0F010101u, 0x0F010109u, 0x0F01010Fu,
    0x0F010501u, 0x0F010505u, 0x0F01070Du, 0x0F010901u, 0x0F010B09u, 0x0F010D05u, 0x0F030105u, 0x0F030303u,
    0x0F030509u, 0x0F030907u, 0x0F03090Bu, 0x0F050103u, 0x0F050109u, 0x0F050301u, 0x0F05030Du, 0x0F050503u,
    0x0F050701u, 0x0F050B03u, 0x0F070105u, 0x0F070705u, 0x0F07070Bu, 0x0F070B07u, 0x0F090103u, 0x0F09010Bu,
    0x0F090307u, 0x0F090501u, 0x0F090B01u, 0x0F0B0505u, 0x0F0B0905u, 0x0F0D0105u, 0x0F0D0703u, 0x0F0F0101u,
};

// Q3_K's 12 packed scale bytes unpack to 16 signed six-bit scales; this is
// `dequantize_row_q3_K`'s shuffle, one selected byte at a time. A Q3_K block is
// 110 bytes, so every load is unaligned-safe (see `pww2_loadraw`).
static __device__ __forceinline__ unsigned int pf_q3k_ld32(const unsigned char* p) {
    unsigned int v;
    __builtin_memcpy(&v, p, 4);
    return v;
}

static __device__ __forceinline__ int pf_q3k_scale(const unsigned char* __restrict__ b,
                                                   int is) {
    const unsigned char* s = b + 96;
    const unsigned int KM1 = 0x03030303u;
    const unsigned int KM2 = 0x0f0f0f0fu;
    const int ai = is >> 2;
    unsigned int a;
    if (ai == 0) {
        a = (pf_q3k_ld32(s) & KM2) | ((pf_q3k_ld32(s + 8) & KM1) << 4);
    } else if (ai == 1) {
        a = (pf_q3k_ld32(s + 4) & KM2) | (((pf_q3k_ld32(s + 8) >> 2) & KM1) << 4);
    } else if (ai == 2) {
        a = ((pf_q3k_ld32(s) >> 4) & KM2) | (((pf_q3k_ld32(s + 8) >> 4) & KM1) << 4);
    } else {
        a = ((pf_q3k_ld32(s + 4) >> 4) & KM2) | (((pf_q3k_ld32(s + 8) >> 6) & KM1) << 4);
    }
    return (int)(signed char)((a >> (8 * (is & 3))) & 0xFFu) - 32;
}

// Dequantise 8 contiguous values of one row, starting at value index `v`
// (a multiple of 8). KIND: 0 Q4_K, 1 Q6_K, 2 Q8_0, 3 already-half, 4 Q5_K,
// 5 IQ4_XS, 6 IQ4_NL, 7 Q3_K, 8 IQ3_S.
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
        const float d = __half2float(pf_ldh(reinterpret_cast<const __half*>(b)));
        const float dmin = __half2float(pf_ldh(reinterpret_cast<const __half*>(b + 2)));
        int sc, mn;
        pf_scale_min(sub, b + 4, &sc, &mn);
        const float scale = d * (float)sc;
        const float minv = dmin * (float)mn;
        const bool high = (sub & 1) != 0;
        const unsigned char* qp = b + 16 + (sub >> 1) * 32 + part * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int q = high ? (pf_ld8(qp + i) >> 4) : (pf_ld8(qp + i) & 0xF);
            out[i] = scale * (float)q - minv;
        }
    } else if (KIND == 1) {
        const int blk = v >> 8;
        const int within = v & 255;
        const int j = within >> 4;
        const int l0 = within & 15;
        const unsigned char* b = rp + (size_t)blk * PF_Q6K_BLOCK_BYTES;
        const float d = __half2float(pf_ldh(reinterpret_cast<const __half*>(b + 208)));
        const float scale =
            d * (float)(signed char)pf_ld8(b + 192 + j);
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
            const int qb = pf_ld8(b + ql_off + i);
            const int hb = (pf_ld8(b + qh_off + i) >> shift) & 3;
            const int nib = high ? (qb >> 4) : (qb & 0xF);
            out[i] = scale * (float)((nib | (hb << 4)) - 32);
        }
    } else if (KIND == 2) {
        const int blk = v >> 5;
        const int part = (v & 31) >> 3;
        const unsigned char* b = rp + (size_t)blk * PF_Q8_0_BLOCK_BYTES;
        const float d = __half2float(pf_ldh(reinterpret_cast<const __half*>(b)));
        const signed char* qs = reinterpret_cast<const signed char*>(b + 2 + part * 8);
#pragma unroll
        for (int i = 0; i < 8; ++i) out[i] = d * (float)qs[i];
    } else if (KIND == 4) {
        // Q5_K: Q4_K's scale/min packing and qs nibbles plus the `qh` fifth
        // bit, `1 << 2g` for the low half of each 64-value group and `2 << 2g`
        // for the high half (dequantize_row_q5_K).
        const int blk = v >> 8;
        const int within = v & 255;
        const int sub = within >> 5;
        const int g = within >> 6;
        const int part = (within & 31) >> 3;
        const unsigned char* b = rp + (size_t)blk * PF_Q5K_BLOCK_BYTES;
        const float d = __half2float(pf_ldh(reinterpret_cast<const __half*>(b)));
        const float dmin = __half2float(pf_ldh(reinterpret_cast<const __half*>(b + 2)));
        int sc, mn;
        pf_scale_min(sub, b + 4, &sc, &mn);
        const float scale = d * (float)sc;
        const float minv = dmin * (float)mn;
        const bool high = (sub & 1) != 0;
        const unsigned int bit = high ? (2u << (2 * g)) : (1u << (2 * g));
        const unsigned char* qp = b + 48 + g * 32 + part * 8;
        const unsigned char* hp = b + 16 + part * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int nib = high ? (pf_ld8(qp + i) >> 4) : (pf_ld8(qp + i) & 0xF);
            const int q = nib + ((pf_ld8(hp + i) & bit) ? 16 : 0);
            out[i] = scale * (float)q - minv;
        }
    } else if (KIND == 5) {
        const int blk = v >> 8;
        const int within = v & 255;
        const int sub = within >> 5;
        const int part = (within & 31) >> 3;
        const unsigned char* b = rp + (size_t)blk * PF_IQ4_XS_BLOCK_BYTES;
        const float d = __half2float(pf_ldh(reinterpret_cast<const __half*>(b)));
        const unsigned short scales_h =
            *reinterpret_cast<const unsigned short*>(b + 2);
        const int ls = ((b[4 + (sub >> 1)] >> (4 * (sub & 1))) & 0xF) |
                       (((scales_h >> (2 * sub)) & 3) << 4);
        const float dl = d * (float)(ls - 32);
        const bool high = part >= 2;
        const unsigned char* qp = b + 8 + sub * 16 + (part & 1) * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int q = high ? (pf_ld8(qp + i) >> 4) : (pf_ld8(qp + i) & 0xF);
            out[i] = dl * (float)PF_IQ4NL_KV[q];
        }
    } else if (KIND == 7) {
        // Q3_K: one 8-value run shares its `n`/`j`/`half` triple, so one scale
        // and one hmask bit cover it. Mask bit m = 1 << (n*4 + j).
        const int blk = v >> 8;
        const int within = v & 255;
        const int n = within >> 7;
        const int j = (within >> 5) & 3;
        const int half = (within >> 4) & 1;
        const int l0 = within & 15;
        const unsigned char* b = rp + (size_t)blk * PF_Q3K_BLOCK_BYTES;
        const float d = __half2float(pf_ldh(reinterpret_cast<const __half*>(b + 108)));
        const float scale = d * (float)pf_q3k_scale(b, n * 8 + j * 2 + half);
        const unsigned char* qp = b + 32 + n * 32 + half * 16 + l0;
        const unsigned char* hp = b + half * 16 + l0;
        const int shift = 2 * j;
        const int bit = 1 << (n * 4 + j);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int q = (pf_ld8(qp + i) >> shift) & 3;
            const int sub = (pf_ld8(hp + i) & bit) ? 0 : 4;
            out[i] = scale * (float)(q - sub);
        }
    } else if (KIND == 8) {
        // IQ3_S: 8 values are one grid group pair, one sign byte and one scale
        // nibble. Two 9-bit indices, low 8 from qs and the ninth from qh.
        const int blk = v >> 8;
        const int within = v & 255;
        const int pair = within >> 6;
        const int half = (within >> 5) & 1;
        const int l = (within & 31) >> 3;
        const unsigned char* b = rp + (size_t)blk * PF_IQ3S_BLOCK_BYTES;
        const float d = __half2float(pf_ldh(reinterpret_cast<const __half*>(b)));
        const unsigned int sc = pf_ld8(b + 106 + pair);
        const int nib = half ? (int)(sc >> 4) : (int)(sc & 0xF);
        const float db = d * (float)(1 + 2 * nib);
        const unsigned int qh = pf_ld8(b + 66 + pair * 2 + half);
        const unsigned char* qp = b + 2 + pair * 16 + half * 8;
        const int i1 = (int)pf_ld8(qp + 2 * l) | (int)((qh << (8 - 2 * l)) & 256u);
        const int i2 = (int)pf_ld8(qp + 2 * l + 1) | (int)((qh << (7 - 2 * l)) & 256u);
        const unsigned int g1 = PF_IQ3S_GRID[i1];
        const unsigned int g2 = PF_IQ3S_GRID[i2];
        const unsigned int sign = pf_ld8(b + 74 + pair * 8 + half * 4 + l);
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const float s1 = (sign & (1u << t)) ? -1.0f : 1.0f;
            const float s2 = (sign & (1u << (t + 4))) ? -1.0f : 1.0f;
            out[t] = db * (float)((g1 >> (8 * t)) & 0xFFu) * s1;
            out[4 + t] = db * (float)((g2 >> (8 * t)) & 0xFFu) * s2;
        }
    } else {
        const int blk = v >> 5;
        const int part = (v & 31) >> 3;
        const unsigned char* b = rp + (size_t)blk * PF_IQ4_NL_BLOCK_BYTES;
        const float d = __half2float(pf_ldh(reinterpret_cast<const __half*>(b)));
        const bool high = part >= 2;
        const unsigned char* qp = b + 2 + (part & 1) * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int q = high ? (pf_ld8(qp + i) >> 4) : (pf_ld8(qp + i) & 0xF);
            out[i] = d * (float)PF_IQ4NL_KV[q];
        }
    }
}

// Bytes per row of a block-quantized weight, and the k-tile's byte offset
// within it. KIND == 3 rows are already half, so the "scale" is 2 bytes/value.
static __device__ __forceinline__ size_t pf_row_bytes(int k, int kind)
{
    return kind == 3 ? (size_t)k * 2
         : kind == 0 ? (size_t)(k / 256) * PF_Q4K_BLOCK_BYTES
         : kind == 1 ? (size_t)(k / 256) * PF_Q6K_BLOCK_BYTES
         : kind == 2 ? (size_t)(k / 32) * PF_Q8_0_BLOCK_BYTES
         : kind == 4 ? (size_t)(k / 256) * PF_Q5K_BLOCK_BYTES
         : kind == 5 ? (size_t)(k / 256) * PF_IQ4_XS_BLOCK_BYTES
         : kind == 7 ? (size_t)(k / 256) * PF_Q3K_BLOCK_BYTES
         : kind == 8 ? (size_t)(k / 256) * PF_IQ3S_BLOCK_BYTES
                     : (size_t)(k / 32) * PF_IQ4_NL_BLOCK_BYTES;
}

#endif // PF_PREFILL_COMMON_INC
