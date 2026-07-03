/**
 * @file gemm_kernels_q4k_q8k_vnni.c
 * @brief VNNI Q4_K x Q8_K matvec kernel (inference only)
 *
 * CK-ENGINE KERNEL RULES:
 * =======================
 * 1. NO malloc/free - memory via bump allocator, pointers passed in
 * 2. NO OpenMP - parallelization at orchestrator/codegen layer
 * 3. API must define: inputs, outputs, workspace, and memory layouts
 * 4. Pure computation - deterministic, no side effects
 *
 * After changes: make test && make llamacpp-parity-full
 *
 * Requires AVX512-VNNI for vpdpbusd instruction.
 *
 * Packed-meta status:
 * The packed-meta Q4_K helpers in this file are real kernel experiments, but
 * they are not the default production layout yet. The current v8 integration
 * is env-gated because the dispatcher still owns temporary packed-weight
 * lifetime and shape policy. The final production design should prepack Q4_K
 * weights at model-load/conversion time, record the extra memory in the model
 * layout, free it with the model runtime, and choose this kernel only through
 * hardware/shape dispatch after sweep data confirms it wins. Until then, keep
 * the canonical GGUF-layout Q4_K path as the parity fallback.
 */

#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "ckernel_engine.h"
#include "ck_threadpool.h"
#include "ckernel_quant.h"

#if (defined(__AVX512VNNI__) && defined(__AVX512VL__)) || defined(__AVX2__)
#include <immintrin.h>
#endif

void gemv_q4_k_q8_k_ref(float *y,
                        const void *W,
                        const void *x_q8,
                        int M, int K);

void gemv_q4_k_q8_k_avx2(float *y,
                         const void *W,
                         const void *x_q8,
                         int M, int K);

#if (defined(__AVX512VNNI__) && defined(__AVX512VL__)) || defined(__AVX2__)
static inline int32_t hsum256_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i sum = _mm_add_epi32(lo, hi);
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(sum);
}
#endif

#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
static inline int32_t dot_q4_k_q8_k_32_vnni(const uint8_t *q4_packed_32,
                                            const int8_t *q8_32,
                                            int high_nibble) {
    const __m256i q8_bytes = _mm256_loadu_si256((const __m256i *)q8_32);
    const __m256i packed = _mm256_loadu_si256((const __m256i *)q4_packed_32);
    const __m256i mask4 = _mm256_set1_epi8(0x0F);
    const __m256i q4_bytes = high_nibble
        ? _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask4)
        : _mm256_and_si256(packed, mask4);
    __m256i acc = _mm256_setzero_si256();
    acc = _mm256_dpbusd_epi32(acc, q4_bytes, q8_bytes);
    return hsum256_epi32(acc);
}

static inline int32_t dot_q4_k_q8_k_32_vnni_q8v(const uint8_t *q4_packed_32,
                                                 __m256i q8_bytes,
                                                 int high_nibble) {
    const __m256i packed = _mm256_loadu_si256((const __m256i *)q4_packed_32);
    const __m256i mask4 = _mm256_set1_epi8(0x0F);
    const __m256i q4_bytes = high_nibble
        ? _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask4)
        : _mm256_and_si256(packed, mask4);
    __m256i acc = _mm256_setzero_si256();
    acc = _mm256_dpbusd_epi32(acc, q4_bytes, q8_bytes);
    return hsum256_epi32(acc);
}

static inline __m256i q4_k_unpack_32_vnni_bytes(__m256i packed, int high_nibble) {
    const __m256i mask4 = _mm256_set1_epi8(0x0F);
    return high_nibble
        ? _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask4)
        : _mm256_and_si256(packed, mask4);
}

static inline int32_t dot_q4_k_q8_k_32_vnni_q4v_q8v(__m256i q4_bytes, __m256i q8_bytes) {
    __m256i acc = _mm256_setzero_si256();
    acc = _mm256_dpbusd_epi32(acc, q4_bytes, q8_bytes);
    return hsum256_epi32(acc);
}


static inline float dot_q4_k_q8_k_vnni_block(const block_q4_K *w,
                                             const block_q8_K *x) {
    uint8_t sc[8], m_val[8];
    unpack_q4_k_scales(w->scales, sc, m_val);

    const float d = CK_FP16_TO_FP32(w->d) * x->d;
    const float dmin = CK_FP16_TO_FP32(w->dmin) * x->d;
    float sumf = 0.0f;

    for (int j = 0, is = 0, q_offset = 0; j < QK_K; j += 64, is += 2, q_offset += 32) {
        const uint8_t *qs = &w->qs[q_offset];
        const int8_t *q8_lo = &x->qs[j];
        const int8_t *q8_hi = &x->qs[j + 32];
        if (j + 128 < QK_K) {
            __builtin_prefetch(&w->qs[q_offset + 64], 0, 1);
            __builtin_prefetch(&x->qs[j + 128], 0, 1);
        }

        const int32_t sum_lo = dot_q4_k_q8_k_32_vnni(qs, q8_lo, 0);
        const int32_t sum_hi = dot_q4_k_q8_k_32_vnni(qs, q8_hi, 1);
        const int32_t bsum_lo = (int32_t)x->bsums[j / 16] +
                                (int32_t)x->bsums[j / 16 + 1];
        const int32_t bsum_hi = (int32_t)x->bsums[(j + 32) / 16] +
                                (int32_t)x->bsums[(j + 32) / 16 + 1];

        sumf += d * (float)sc[is] * (float)sum_lo;
        sumf -= dmin * (float)m_val[is] * (float)bsum_lo;
        sumf += d * (float)sc[is + 1] * (float)sum_hi;
        sumf -= dmin * (float)m_val[is + 1] * (float)bsum_hi;
    }

    return sumf;
}
#endif

#if defined(__AVX2__) && !(defined(__AVX512VNNI__) && defined(__AVX512VL__))
static inline int32_t dot_q4_k_q8_k_32_avx2_q8v(const uint8_t *q4_packed_32,
                                                 __m256i q8_bytes,
                                                 int high_nibble) {
    const __m256i packed = _mm256_loadu_si256((const __m256i *)q4_packed_32);
    const __m256i mask4 = _mm256_set1_epi8(0x0F);
    const __m256i q4_bytes = high_nibble
        ? _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask4)
        : _mm256_and_si256(packed, mask4);
    const __m256i pair_sums_i16 = _mm256_maddubs_epi16(q4_bytes, q8_bytes);
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i sums_i32 = _mm256_madd_epi16(pair_sums_i16, ones);
    return hsum256_epi32(sums_i32);
}
#endif


typedef struct {
    ck_half d;
    ck_half dmin;
    uint8_t sc[8];
    uint8_t m[8];
    uint8_t qs[QK_K];
} block_q4_K_packed_u8;

typedef struct {
    ck_half d;
    ck_half dmin;
    uint8_t sc[8];
    uint8_t m[8];
    uint8_t qs[QK_K / 2];
} block_q4_K_packed_meta;

typedef struct {
    ck_half d[8];
    ck_half dmin[8];
    uint8_t sc[8][8];
    uint8_t m[8][8];
    uint8_t qs[8][QK_K / 2];
    uint8_t active;
    uint8_t reserved[7];
} block_q4_K_packed_meta_x8;

typedef struct {
    ck_half d[16];
    ck_half dmin[16];
    uint8_t sc[16][8];
    uint8_t m[16][8];
    uint8_t qs[16][QK_K / 2];
    uint8_t active;
    uint8_t reserved[15];
} block_q4_K_packed_meta_x16;

typedef struct {
    ck_half d[16];
    ck_half dmin[16];
    uint8_t sc[16][8];
    uint8_t m[16][8];
    uint8_t qs[16][QK_K];
    uint8_t active;
    uint8_t reserved[15];
} block_q4_K_packed_u8_x16;

size_t q4_k_packed_u8_block_size(void)
{
    return sizeof(block_q4_K_packed_u8);
}

size_t q4_k_packed_meta_block_size(void)
{
    return sizeof(block_q4_K_packed_meta);
}

size_t q4_k_packed_meta_x8_block_size(void)
{
    return sizeof(block_q4_K_packed_meta_x8);
}

size_t q4_k_packed_meta_x16_block_size(void)
{
    return sizeof(block_q4_K_packed_meta_x16);
}

size_t q4_k_packed_u8_x16_block_size(void)
{
    return sizeof(block_q4_K_packed_u8_x16);
}

void pack_q4_k_to_packed_u8(const void *src, void *dst, int N, int K)
{
    if (!src || !dst || N <= 0 || K <= 0 || (K % QK_K) != 0) {
        return;
    }
    const block_q4_K *in = (const block_q4_K *)src;
    block_q4_K_packed_u8 *out = (block_q4_K_packed_u8 *)dst;
    const int blocks_per_row = K / QK_K;
    for (int n = 0; n < N; ++n) {
        for (int b = 0; b < blocks_per_row; ++b) {
            const block_q4_K *sb = in + (size_t)n * (size_t)blocks_per_row + (size_t)b;
            block_q4_K_packed_u8 *pb = out + (size_t)n * (size_t)blocks_per_row + (size_t)b;
            pb->d = sb->d;
            pb->dmin = sb->dmin;
            unpack_q4_k_scales(sb->scales, pb->sc, pb->m);
            for (int j = 0, q_offset = 0; j < QK_K; j += 64, q_offset += 32) {
                const uint8_t *qs = &sb->qs[q_offset];
                for (int l = 0; l < 32; ++l) {
                    pb->qs[j + l] = qs[l] & 0x0F;
                    pb->qs[j + 32 + l] = qs[l] >> 4;
                }
            }
        }
    }
}

void pack_q4_k_to_packed_meta(const void *src, void *dst, int N, int K)
{
    if (!src || !dst || N <= 0 || K <= 0 || (K % QK_K) != 0) {
        return;
    }
    const block_q4_K *in = (const block_q4_K *)src;
    block_q4_K_packed_meta *out = (block_q4_K_packed_meta *)dst;
    const int blocks_per_row = K / QK_K;
    for (int n = 0; n < N; ++n) {
        for (int b = 0; b < blocks_per_row; ++b) {
            const block_q4_K *sb = in + (size_t)n * (size_t)blocks_per_row + (size_t)b;
            block_q4_K_packed_meta *pb = out + (size_t)n * (size_t)blocks_per_row + (size_t)b;
            pb->d = sb->d;
            pb->dmin = sb->dmin;
            unpack_q4_k_scales(sb->scales, pb->sc, pb->m);
            memcpy(pb->qs, sb->qs, sizeof(pb->qs));
        }
    }
}

void pack_q4_k_to_packed_meta_x8(const void *src, void *dst, int N, int K)
{
    if (!src || !dst || N <= 0 || K <= 0 || (K % QK_K) != 0) {
        return;
    }
    const block_q4_K *in = (const block_q4_K *)src;
    block_q4_K_packed_meta_x8 *out = (block_q4_K_packed_meta_x8 *)dst;
    const int blocks_per_row = K / QK_K;
    const int groups = (N + 7) / 8;
    memset(out, 0, (size_t)groups * (size_t)blocks_per_row * sizeof(*out));

    for (int g = 0; g < groups; ++g) {
        const int n0 = g * 8;
        const int active = (n0 + 8 <= N) ? 8 : (N - n0);
        for (int b = 0; b < blocks_per_row; ++b) {
            block_q4_K_packed_meta_x8 *pb = out + (size_t)g * (size_t)blocks_per_row + (size_t)b;
            pb->active = (uint8_t)active;
            for (int lane = 0; lane < active; ++lane) {
                const block_q4_K *sb = in + (size_t)(n0 + lane) * (size_t)blocks_per_row + (size_t)b;
                pb->d[lane] = sb->d;
                pb->dmin[lane] = sb->dmin;
                unpack_q4_k_scales(sb->scales, pb->sc[lane], pb->m[lane]);
                memcpy(pb->qs[lane], sb->qs, sizeof(pb->qs[lane]));
            }
        }
    }
}

void pack_q4_k_to_packed_meta_x16(const void *src, void *dst, int N, int K)
{
    if (!src || !dst || N <= 0 || K <= 0 || (K % QK_K) != 0) {
        return;
    }
    const block_q4_K *in = (const block_q4_K *)src;
    block_q4_K_packed_meta_x16 *out = (block_q4_K_packed_meta_x16 *)dst;
    const int blocks_per_row = K / QK_K;
    const int groups = (N + 15) / 16;
    memset(out, 0, (size_t)groups * (size_t)blocks_per_row * sizeof(*out));

    for (int g = 0; g < groups; ++g) {
        const int n0 = g * 16;
        const int active = (n0 + 16 <= N) ? 16 : (N - n0);
        for (int b = 0; b < blocks_per_row; ++b) {
            block_q4_K_packed_meta_x16 *pb = out + (size_t)g * (size_t)blocks_per_row + (size_t)b;
            pb->active = (uint8_t)active;
            for (int lane = 0; lane < active; ++lane) {
                const block_q4_K *sb = in + (size_t)(n0 + lane) * (size_t)blocks_per_row + (size_t)b;
                pb->d[lane] = sb->d;
                pb->dmin[lane] = sb->dmin;
                unpack_q4_k_scales(sb->scales, pb->sc[lane], pb->m[lane]);
                memcpy(pb->qs[lane], sb->qs, sizeof(pb->qs[lane]));
            }
        }
    }
}

void pack_q4_k_to_packed_u8_x16(const void *src, void *dst, int N, int K)
{
    if (!src || !dst || N <= 0 || K <= 0 || (K % QK_K) != 0) {
        return;
    }
    const block_q4_K *in = (const block_q4_K *)src;
    block_q4_K_packed_u8_x16 *out = (block_q4_K_packed_u8_x16 *)dst;
    const int blocks_per_row = K / QK_K;
    const int groups = (N + 15) / 16;
    memset(out, 0, (size_t)groups * (size_t)blocks_per_row * sizeof(*out));

    for (int g = 0; g < groups; ++g) {
        const int n0 = g * 16;
        const int active = (n0 + 16 <= N) ? 16 : (N - n0);
        for (int b = 0; b < blocks_per_row; ++b) {
            block_q4_K_packed_u8_x16 *pb = out + (size_t)g * (size_t)blocks_per_row + (size_t)b;
            pb->active = (uint8_t)active;
            for (int lane = 0; lane < active; ++lane) {
                const block_q4_K *sb = in + (size_t)(n0 + lane) * (size_t)blocks_per_row + (size_t)b;
                pb->d[lane] = sb->d;
                pb->dmin[lane] = sb->dmin;
                unpack_q4_k_scales(sb->scales, pb->sc[lane], pb->m[lane]);
                for (int j = 0, q_offset = 0; j < QK_K; j += 64, q_offset += 32) {
                    const uint8_t *qs = &sb->qs[q_offset];
                    for (int l = 0; l < 32; ++l) {
                        pb->qs[lane][j + l] = qs[l] & 0x0F;
                        pb->qs[lane][j + 32 + l] = qs[l] >> 4;
                    }
                }
            }
        }
    }
}

#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
static inline int32_t dot_q4_packed_u8_q8_32_vnni(const uint8_t *q4_32,
                                                   const int8_t *q8_32)
{
    const __m256i q4 = _mm256_loadu_si256((const __m256i *)q4_32);
    const __m256i q8 = _mm256_loadu_si256((const __m256i *)q8_32);
    __m256i acc = _mm256_setzero_si256();
    acc = _mm256_dpbusd_epi32(acc, q4, q8);
    return hsum256_epi32(acc);
}

static inline int32_t dot_q4_packed_u8_q8_32_vnni_q8v(const uint8_t *q4_32,
                                                       __m256i q8)
{
    const __m256i q4 = _mm256_loadu_si256((const __m256i *)q4_32);
    __m256i acc = _mm256_setzero_si256();
    acc = _mm256_dpbusd_epi32(acc, q4, q8);
    return hsum256_epi32(acc);
}
#endif

#if !(defined(__AVX512VNNI__) && defined(__AVX512VL__))
static inline int32_t dot_q4_packed_u8_q8_32_ref(const uint8_t *q4_32,
                                                  const int8_t *q8_32)
{
    int32_t acc = 0;
    for (int i = 0; i < 32; ++i) {
        acc += (int32_t)q4_32[i] * (int32_t)q8_32[i];
    }
    return acc;
}
#endif

static inline float dot_q4_k_packed_u8_q8_k_block(const block_q4_K_packed_u8 *w,
                                                   const block_q8_K *x)
{
    const float d = CK_FP16_TO_FP32(w->d) * x->d;
    const float dmin = CK_FP16_TO_FP32(w->dmin) * x->d;
    float sumf = 0.0f;
    for (int j = 0; j < QK_K; j += 32) {
        const int is = j / 32;
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
        const int32_t sumi = dot_q4_packed_u8_q8_32_vnni(&w->qs[j], &x->qs[j]);
#else
        const int32_t sumi = dot_q4_packed_u8_q8_32_ref(&w->qs[j], &x->qs[j]);
#endif
        const int32_t bsum = (int32_t)x->bsums[j / 16] + (int32_t)x->bsums[j / 16 + 1];
        sumf += d * (float)w->sc[is] * (float)sumi;
        sumf -= dmin * (float)w->m[is] * (float)bsum;
    }
    return sumf;
}

static inline float dot_q4_k_packed_meta_q8_k_block(const block_q4_K_packed_meta *w,
                                                     const block_q8_K *x)
{
    const float d = CK_FP16_TO_FP32(w->d) * x->d;
    const float dmin = CK_FP16_TO_FP32(w->dmin) * x->d;
    float sumf = 0.0f;
    for (int j = 0, is = 0, q_offset = 0; j < QK_K; j += 64, is += 2, q_offset += 32) {
        const uint8_t *qs = &w->qs[q_offset];
        const int8_t *q8_lo = &x->qs[j];
        const int8_t *q8_hi = &x->qs[j + 32];

#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
        const int32_t sum_lo = dot_q4_k_q8_k_32_vnni(qs, q8_lo, 0);
        const int32_t sum_hi = dot_q4_k_q8_k_32_vnni(qs, q8_hi, 1);
#else
        int32_t sum_lo = 0;
        int32_t sum_hi = 0;
        for (int l = 0; l < 32; ++l) {
            sum_lo += (int32_t)(qs[l] & 0x0F) * (int32_t)q8_lo[l];
            sum_hi += (int32_t)(qs[l] >> 4) * (int32_t)q8_hi[l];
        }
#endif
        const int32_t bsum_lo = (int32_t)x->bsums[j / 16] +
                                (int32_t)x->bsums[j / 16 + 1];
        const int32_t bsum_hi = (int32_t)x->bsums[(j + 32) / 16] +
                                (int32_t)x->bsums[(j + 32) / 16 + 1];

        sumf += d * (float)w->sc[is] * (float)sum_lo;
        sumf -= dmin * (float)w->m[is] * (float)bsum_lo;
        sumf += d * (float)w->sc[is + 1] * (float)sum_hi;
        sumf -= dmin * (float)w->m[is + 1] * (float)bsum_hi;
    }
    return sumf;
}

static inline void accum_q4_k_packed_meta_x8_q8_k_block(float acc[8],
                                                         const block_q4_K_packed_meta_x8 *w,
                                                         int active,
                                                         const block_q8_K *x)
{
    const float xd = x->d;
    for (int j = 0, is = 0, q_offset = 0; j < QK_K; j += 64, is += 2, q_offset += 32) {
        const int8_t *q8_lo_ptr = &x->qs[j];
        const int8_t *q8_hi_ptr = &x->qs[j + 32];
        const int32_t bsum_lo = (int32_t)x->bsums[j / 16] +
                                (int32_t)x->bsums[j / 16 + 1];
        const int32_t bsum_hi = (int32_t)x->bsums[(j + 32) / 16] +
                                (int32_t)x->bsums[(j + 32) / 16 + 1];

#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
        const __m256i q8_lo = _mm256_loadu_si256((const __m256i *)q8_lo_ptr);
        const __m256i q8_hi = _mm256_loadu_si256((const __m256i *)q8_hi_ptr);
#elif defined(__AVX2__)
        const __m256i q8_lo = _mm256_loadu_si256((const __m256i *)q8_lo_ptr);
        const __m256i q8_hi = _mm256_loadu_si256((const __m256i *)q8_hi_ptr);
#endif

        for (int lane = 0; lane < active; ++lane) {
            const uint8_t *qs = &w->qs[lane][q_offset];
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
            const int32_t sum_lo = dot_q4_k_q8_k_32_vnni_q8v(qs, q8_lo, 0);
            const int32_t sum_hi = dot_q4_k_q8_k_32_vnni_q8v(qs, q8_hi, 1);
#elif defined(__AVX2__)
            const int32_t sum_lo = dot_q4_k_q8_k_32_avx2_q8v(qs, q8_lo, 0);
            const int32_t sum_hi = dot_q4_k_q8_k_32_avx2_q8v(qs, q8_hi, 1);
#else
            int32_t sum_lo = 0;
            int32_t sum_hi = 0;
            for (int l = 0; l < 32; ++l) {
                sum_lo += (int32_t)(qs[l] & 0x0F) * (int32_t)q8_lo_ptr[l];
                sum_hi += (int32_t)(qs[l] >> 4) * (int32_t)q8_hi_ptr[l];
            }
#endif
            const float d = CK_FP16_TO_FP32(w->d[lane]) * xd;
            const float dmin = CK_FP16_TO_FP32(w->dmin[lane]) * xd;
            acc[lane] += d * (float)w->sc[lane][is] * (float)sum_lo;
            acc[lane] -= dmin * (float)w->m[lane][is] * (float)bsum_lo;
            acc[lane] += d * (float)w->sc[lane][is + 1] * (float)sum_hi;
            acc[lane] -= dmin * (float)w->m[lane][is + 1] * (float)bsum_hi;
        }
    }
}



static inline void accum_q4_k_packed_u8_x16_q8_k_block(float acc[16],
                                                        const block_q4_K_packed_u8_x16 *w,
                                                        int active,
                                                        const block_q8_K *x)
{
    const float xd = x->d;
    for (int j = 0; j < QK_K; j += 32) {
        const int is = j / 32;
        const int8_t *q8_ptr = &x->qs[j];
        const int32_t bsum = (int32_t)x->bsums[j / 16] + (int32_t)x->bsums[j / 16 + 1];
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
        const __m256i q8 = _mm256_loadu_si256((const __m256i *)q8_ptr);
#endif
        for (int lane = 0; lane < active; ++lane) {
            const uint8_t *q4 = &w->qs[lane][j];
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
            const int32_t sumi = dot_q4_packed_u8_q8_32_vnni_q8v(q4, q8);
#else
            const int32_t sumi = dot_q4_packed_u8_q8_32_ref(q4, q8_ptr);
#endif
            const float d = CK_FP16_TO_FP32(w->d[lane]) * xd;
            const float dmin = CK_FP16_TO_FP32(w->dmin[lane]) * xd;
            acc[lane] += d * (float)w->sc[lane][is] * (float)sumi;
            acc[lane] -= dmin * (float)w->m[lane][is] * (float)bsum;
        }
    }
}


static inline void accum_q4_k_packed_meta_x16_q8_k_block_mreuse(float acc[8][16],
                                                                 const block_q4_K_packed_meta_x16 *w,
                                                                 int active,
                                                                 const block_q8_K *A,
                                                                 int blocks_per_vec,
                                                                 int block_index,
                                                                 int m0,
                                                                 int m_count)
{
    for (int j = 0, is = 0, q_offset = 0; j < QK_K; j += 64, is += 2, q_offset += 32) {
        for (int lane = 0; lane < active; ++lane) {
            const uint8_t *qs = &w->qs[lane][q_offset];
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
            const __m256i packed = _mm256_loadu_si256((const __m256i *)qs);
            const __m256i q4_lo = q4_k_unpack_32_vnni_bytes(packed, 0);
            const __m256i q4_hi = q4_k_unpack_32_vnni_bytes(packed, 1);
#endif
            const float wd = CK_FP16_TO_FP32(w->d[lane]);
            const float wdmin = CK_FP16_TO_FP32(w->dmin[lane]);
            const float sc_lo = (float)w->sc[lane][is];
            const float sc_hi = (float)w->sc[lane][is + 1];
            const float min_lo = (float)w->m[lane][is];
            const float min_hi = (float)w->m[lane][is + 1];

            for (int mt = 0; mt < m_count; ++mt) {
                const block_q8_K *x = A + (size_t)(m0 + mt) * (size_t)blocks_per_vec + (size_t)block_index;
                const int8_t *q8_lo_ptr = &x->qs[j];
                const int8_t *q8_hi_ptr = &x->qs[j + 32];
                const int32_t bsum_lo = (int32_t)x->bsums[j / 16] +
                                        (int32_t)x->bsums[j / 16 + 1];
                const int32_t bsum_hi = (int32_t)x->bsums[(j + 32) / 16] +
                                        (int32_t)x->bsums[(j + 32) / 16 + 1];
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
                const __m256i q8_lo = _mm256_loadu_si256((const __m256i *)q8_lo_ptr);
                const __m256i q8_hi = _mm256_loadu_si256((const __m256i *)q8_hi_ptr);
                const int32_t sum_lo = dot_q4_k_q8_k_32_vnni_q4v_q8v(q4_lo, q8_lo);
                const int32_t sum_hi = dot_q4_k_q8_k_32_vnni_q4v_q8v(q4_hi, q8_hi);
#elif defined(__AVX2__)
                const __m256i q8_lo = _mm256_loadu_si256((const __m256i *)q8_lo_ptr);
                const __m256i q8_hi = _mm256_loadu_si256((const __m256i *)q8_hi_ptr);
                const int32_t sum_lo = dot_q4_k_q8_k_32_avx2_q8v(qs, q8_lo, 0);
                const int32_t sum_hi = dot_q4_k_q8_k_32_avx2_q8v(qs, q8_hi, 1);
#else
                int32_t sum_lo = 0;
                int32_t sum_hi = 0;
                for (int l = 0; l < 32; ++l) {
                    sum_lo += (int32_t)(qs[l] & 0x0F) * (int32_t)q8_lo_ptr[l];
                    sum_hi += (int32_t)(qs[l] >> 4) * (int32_t)q8_hi_ptr[l];
                }
#endif
                const float xd = x->d;
                const float d = wd * xd;
                const float dmin = wdmin * xd;
                acc[mt][lane] += d * sc_lo * (float)sum_lo;
                acc[mt][lane] -= dmin * min_lo * (float)bsum_lo;
                acc[mt][lane] += d * sc_hi * (float)sum_hi;
                acc[mt][lane] -= dmin * min_hi * (float)bsum_hi;
            }
        }
    }
}

static inline void accum_q4_k_packed_meta_x16_q8_k_block(float acc[16],
                                                          const block_q4_K_packed_meta_x16 *w,
                                                          int active,
                                                          const block_q8_K *x)
{
    const float xd = x->d;
    for (int j = 0, is = 0, q_offset = 0; j < QK_K; j += 64, is += 2, q_offset += 32) {
        const int8_t *q8_lo_ptr = &x->qs[j];
        const int8_t *q8_hi_ptr = &x->qs[j + 32];
        const int32_t bsum_lo = (int32_t)x->bsums[j / 16] +
                                (int32_t)x->bsums[j / 16 + 1];
        const int32_t bsum_hi = (int32_t)x->bsums[(j + 32) / 16] +
                                (int32_t)x->bsums[(j + 32) / 16 + 1];

#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
        const __m256i q8_lo = _mm256_loadu_si256((const __m256i *)q8_lo_ptr);
        const __m256i q8_hi = _mm256_loadu_si256((const __m256i *)q8_hi_ptr);
#elif defined(__AVX2__)
        const __m256i q8_lo = _mm256_loadu_si256((const __m256i *)q8_lo_ptr);
        const __m256i q8_hi = _mm256_loadu_si256((const __m256i *)q8_hi_ptr);
#endif

        for (int lane = 0; lane < active; ++lane) {
            const uint8_t *qs = &w->qs[lane][q_offset];
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
            const int32_t sum_lo = dot_q4_k_q8_k_32_vnni_q8v(qs, q8_lo, 0);
            const int32_t sum_hi = dot_q4_k_q8_k_32_vnni_q8v(qs, q8_hi, 1);
#elif defined(__AVX2__)
            const int32_t sum_lo = dot_q4_k_q8_k_32_avx2_q8v(qs, q8_lo, 0);
            const int32_t sum_hi = dot_q4_k_q8_k_32_avx2_q8v(qs, q8_hi, 1);
#else
            int32_t sum_lo = 0;
            int32_t sum_hi = 0;
            for (int l = 0; l < 32; ++l) {
                sum_lo += (int32_t)(qs[l] & 0x0F) * (int32_t)q8_lo_ptr[l];
                sum_hi += (int32_t)(qs[l] >> 4) * (int32_t)q8_hi_ptr[l];
            }
#endif
            const float d = CK_FP16_TO_FP32(w->d[lane]) * xd;
            const float dmin = CK_FP16_TO_FP32(w->dmin[lane]) * xd;
            acc[lane] += d * (float)w->sc[lane][is] * (float)sum_lo;
            acc[lane] -= dmin * (float)w->m[lane][is] * (float)bsum_lo;
            acc[lane] += d * (float)w->sc[lane][is + 1] * (float)sum_hi;
            acc[lane] -= dmin * (float)w->m[lane][is + 1] * (float)bsum_hi;
        }
    }
}

void gemm_nt_q4_k_packed_u8_q8_k(const void *A_q8,
                                  const void *B_packed,
                                  const float *bias,
                                  float *C,
                                  int M, int N, int K)
{
    if (!A_q8 || !B_packed || !C || M <= 0 || N <= 0 || K <= 0 || (K % QK_K) != 0) {
        return;
    }
    const block_q8_K *A = (const block_q8_K *)A_q8;
    const block_q4_K_packed_u8 *W = (const block_q4_K_packed_u8 *)B_packed;
    const int blocks_per_vec = K / QK_K;
    const int blocks_per_row = K / QK_K;
    for (int m = 0; m < M; ++m) {
        const block_q8_K *a_row = A + (size_t)m * (size_t)blocks_per_vec;
        float *c_row = C + (size_t)m * (size_t)N;
        for (int n = 0; n < N; ++n) {
            const block_q4_K_packed_u8 *w_row = W + (size_t)n * (size_t)blocks_per_row;
            float sum = bias ? bias[n] : 0.0f;
            for (int b = 0; b < blocks_per_row; ++b) {
                sum += dot_q4_k_packed_u8_q8_k_block(&w_row[b], &a_row[b]);
            }
            c_row[n] = sum;
        }
    }
}


void gemm_nt_q4_k_packed_meta_q8_k(const void *A_q8,
                                    const void *B_packed,
                                    const float *bias,
                                    float *C,
                                    int M, int N, int K)
{
    if (!A_q8 || !B_packed || !C || M <= 0 || N <= 0 || K <= 0 || (K % QK_K) != 0) {
        return;
    }
    const block_q8_K *A = (const block_q8_K *)A_q8;
    const block_q4_K_packed_meta *W = (const block_q4_K_packed_meta *)B_packed;
    const int blocks_per_vec = K / QK_K;
    const int blocks_per_row = K / QK_K;
    for (int m = 0; m < M; ++m) {
        const block_q8_K *a_row = A + (size_t)m * (size_t)blocks_per_vec;
        float *c_row = C + (size_t)m * (size_t)N;
        for (int n = 0; n < N; ++n) {
            const block_q4_K_packed_meta *w_row = W + (size_t)n * (size_t)blocks_per_row;
            float sum = bias ? bias[n] : 0.0f;
            for (int b = 0; b < blocks_per_row; ++b) {
                sum += dot_q4_k_packed_meta_q8_k_block(&w_row[b], &a_row[b]);
            }
            c_row[n] = sum;
        }
    }
}

void gemm_nt_q4_k_packed_meta_q8_k_tile(const void *A_q8,
                                         const void *B_packed,
                                         const float *bias,
                                         float *C,
                                         int M, int N, int K,
                                         int m0, int m1, int n0, int n1)
{
    if (!A_q8 || !B_packed || !C || M <= 0 || N <= 0 || K <= 0 || (K % QK_K) != 0) {
        return;
    }
    if (m0 < 0) m0 = 0;
    if (n0 < 0) n0 = 0;
    if (m1 > M) m1 = M;
    if (n1 > N) n1 = N;
    if (m0 >= m1 || n0 >= n1) {
        return;
    }

    const block_q8_K *A = (const block_q8_K *)A_q8;
    const block_q4_K_packed_meta *W = (const block_q4_K_packed_meta *)B_packed;
    const int blocks_per_vec = K / QK_K;
    const int blocks_per_row = K / QK_K;

    for (int m = m0; m < m1; ++m) {
        const block_q8_K *a_row = A + (size_t)m * (size_t)blocks_per_vec;
        float *c_row = C + (size_t)m * (size_t)N;
        for (int n = n0; n < n1; ++n) {
            const block_q4_K_packed_meta *w_row = W + (size_t)n * (size_t)blocks_per_row;
            float sum = bias ? bias[n] : 0.0f;
            for (int b = 0; b < blocks_per_row; ++b) {
                sum += dot_q4_k_packed_meta_q8_k_block(&w_row[b], &a_row[b]);
            }
            c_row[n] = sum;
        }
    }
}

void gemm_nt_q4_k_packed_meta_x8_q8_k(const void *A_q8,
                                       const void *B_packed_x8,
                                       const float *bias,
                                       float *C,
                                       int M, int N, int K)
{
    if (!A_q8 || !B_packed_x8 || !C || M <= 0 || N <= 0 || K <= 0 || (K % QK_K) != 0) {
        return;
    }
    const block_q8_K *A = (const block_q8_K *)A_q8;
    const block_q4_K_packed_meta_x8 *W = (const block_q4_K_packed_meta_x8 *)B_packed_x8;
    const int blocks_per_vec = K / QK_K;
    const int blocks_per_row = K / QK_K;
    const int groups = (N + 7) / 8;

    for (int m = 0; m < M; ++m) {
        const block_q8_K *a_row = A + (size_t)m * (size_t)blocks_per_vec;
        float *c_row = C + (size_t)m * (size_t)N;
        for (int g = 0; g < groups; ++g) {
            const int n0 = g * 8;
            const int active = (n0 + 8 <= N) ? 8 : (N - n0);
            float acc[8];
            for (int lane = 0; lane < active; ++lane) {
                acc[lane] = bias ? bias[n0 + lane] : 0.0f;
            }
            for (int b = 0; b < blocks_per_row; ++b) {
                const block_q4_K_packed_meta_x8 *w_group =
                    W + (size_t)g * (size_t)blocks_per_row + (size_t)b;
                accum_q4_k_packed_meta_x8_q8_k_block(acc, w_group, active, &a_row[b]);
            }
            for (int lane = 0; lane < active; ++lane) {
                c_row[n0 + lane] = acc[lane];
            }
        }
    }
}


typedef struct {
    const block_q8_K *A;
    const block_q4_K_packed_meta *W;
    const float *bias;
    float *C;
    int M;
    int N;
    int K;
    int blocks_per_vec;
    int blocks_per_row;
} gemm_q4_packed_meta_thread_work_t;

typedef struct {
    const block_q8_K *A;
    const block_q4_K_packed_meta_x8 *W;
    const float *bias;
    float *C;
    int M;
    int N;
    int K;
    int blocks_per_vec;
    int blocks_per_row;
    int groups;
    int tile_m;
    int jobs;
} gemm_q4_packed_meta_x8_thread_work_t;

typedef struct {
    const block_q8_K *A;
    const block_q4_K_packed_meta_x16 *W;
    const float *bias;
    float *C;
    int M;
    int N;
    int K;
    int blocks_per_vec;
    int blocks_per_row;
    int groups;
    int tile_m;
    int jobs;
} gemm_q4_packed_meta_x16_thread_work_t;

typedef struct {
    const block_q8_K *A;
    const block_q4_K_packed_u8_x16 *W;
    const float *bias;
    float *C;
    int M;
    int N;
    int K;
    int blocks_per_vec;
    int blocks_per_row;
    int groups;
    int tile_m;
    int jobs;
} gemm_q4_packed_u8_x16_thread_work_t;

static void gemm_q4_packed_meta_thread_fn(int ith, int nth, void *args)
{
    gemm_q4_packed_meta_thread_work_t *a = (gemm_q4_packed_meta_thread_work_t *)args;
    if (!a || ith < 0 || nth <= 0 || ith >= nth) {
        return;
    }
    const int dm = (a->M + nth - 1) / nth;
    const int m0 = dm * ith;
    const int m1 = (m0 + dm < a->M) ? (m0 + dm) : a->M;
    if (m0 >= a->M) {
        return;
    }
    for (int m = m0; m < m1; ++m) {
        const block_q8_K *a_row = a->A + (size_t)m * (size_t)a->blocks_per_vec;
        float *c_row = a->C + (size_t)m * (size_t)a->N;
        for (int n = 0; n < a->N; ++n) {
            const block_q4_K_packed_meta *w_row = a->W + (size_t)n * (size_t)a->blocks_per_row;
            float sum = a->bias ? a->bias[n] : 0.0f;
            for (int b = 0; b < a->blocks_per_row; ++b) {
                sum += dot_q4_k_packed_meta_q8_k_block(&w_row[b], &a_row[b]);
            }
            c_row[n] = sum;
        }
    }
}

void gemm_nt_q4_k_packed_meta_q8_k_threaded(const void *A_q8,
                                             const void *B_packed,
                                             const float *bias,
                                             float *C,
                                             int M, int N, int K,
                                             int active_threads)
{
    if (!A_q8 || !B_packed || !C || M <= 0 || N <= 0 || K <= 0 || (K % QK_K) != 0) {
        return;
    }
    ck_threadpool_t *pool = ck_threadpool_global();
    const int pool_threads = pool ? ck_threadpool_n_threads(pool) : 1;
    if (active_threads <= 0 || active_threads > pool_threads) {
        active_threads = pool_threads;
    }
    if (active_threads > M) {
        active_threads = M;
    }
    if (active_threads <= 1) {
        gemm_nt_q4_k_packed_meta_q8_k(A_q8, B_packed, bias, C, M, N, K);
        return;
    }
    gemm_q4_packed_meta_thread_work_t work = {
        .A = (const block_q8_K *)A_q8,
        .W = (const block_q4_K_packed_meta *)B_packed,
        .bias = bias,
        .C = C,
        .M = M,
        .N = N,
        .K = K,
        .blocks_per_vec = K / QK_K,
        .blocks_per_row = K / QK_K,
    };
    ck_threadpool_dispatch_n(pool, active_threads, gemm_q4_packed_meta_thread_fn, &work);
}


static void gemm_q4_packed_meta_nsplit_thread_fn(int ith, int nth, void *args)
{
    gemm_q4_packed_meta_thread_work_t *a = (gemm_q4_packed_meta_thread_work_t *)args;
    if (!a || ith < 0 || nth <= 0 || ith >= nth) {
        return;
    }
    const int dn = (a->N + nth - 1) / nth;
    const int n0 = dn * ith;
    const int n1 = (n0 + dn < a->N) ? (n0 + dn) : a->N;
    if (n0 >= a->N) {
        return;
    }
    for (int m = 0; m < a->M; ++m) {
        const block_q8_K *a_row = a->A + (size_t)m * (size_t)a->blocks_per_vec;
        float *c_row = a->C + (size_t)m * (size_t)a->N;
        for (int n = n0; n < n1; ++n) {
            const block_q4_K_packed_meta *w_row = a->W + (size_t)n * (size_t)a->blocks_per_row;
            float sum = a->bias ? a->bias[n] : 0.0f;
            for (int b = 0; b < a->blocks_per_row; ++b) {
                sum += dot_q4_k_packed_meta_q8_k_block(&w_row[b], &a_row[b]);
            }
            c_row[n] = sum;
        }
    }
}

void gemm_nt_q4_k_packed_meta_q8_k_threaded_nsplit(const void *A_q8,
                                                    const void *B_packed,
                                                    const float *bias,
                                                    float *C,
                                                    int M, int N, int K,
                                                    int active_threads)
{
    if (!A_q8 || !B_packed || !C || M <= 0 || N <= 0 || K <= 0 || (K % QK_K) != 0) {
        return;
    }
    ck_threadpool_t *pool = ck_threadpool_global();
    const int pool_threads = pool ? ck_threadpool_n_threads(pool) : 1;
    if (active_threads <= 0 || active_threads > pool_threads) {
        active_threads = pool_threads;
    }
    if (active_threads > N) {
        active_threads = N;
    }
    if (active_threads <= 1) {
        gemm_nt_q4_k_packed_meta_q8_k(A_q8, B_packed, bias, C, M, N, K);
        return;
    }
    gemm_q4_packed_meta_thread_work_t work = {
        .A = (const block_q8_K *)A_q8,
        .W = (const block_q4_K_packed_meta *)B_packed,
        .bias = bias,
        .C = C,
        .M = M,
        .N = N,
        .K = K,
        .blocks_per_vec = K / QK_K,
        .blocks_per_row = K / QK_K,
    };
    ck_threadpool_dispatch_n(pool, active_threads, gemm_q4_packed_meta_nsplit_thread_fn, &work);
}

static void gemm_q4_packed_meta_x8_nsplit_thread_fn(int ith, int nth, void *args)
{
    gemm_q4_packed_meta_x8_thread_work_t *a = (gemm_q4_packed_meta_x8_thread_work_t *)args;
    if (!a || ith < 0 || nth <= 0 || ith >= nth) {
        return;
    }
    const int dg = (a->groups + nth - 1) / nth;
    const int g0 = dg * ith;
    const int g1 = (g0 + dg < a->groups) ? (g0 + dg) : a->groups;
    if (g0 >= a->groups) {
        return;
    }

    for (int m = 0; m < a->M; ++m) {
        const block_q8_K *a_row = a->A + (size_t)m * (size_t)a->blocks_per_vec;
        float *c_row = a->C + (size_t)m * (size_t)a->N;
        for (int g = g0; g < g1; ++g) {
            const int n0 = g * 8;
            const int active = (n0 + 8 <= a->N) ? 8 : (a->N - n0);
            float acc[8];
            for (int lane = 0; lane < active; ++lane) {
                acc[lane] = a->bias ? a->bias[n0 + lane] : 0.0f;
            }
            for (int b = 0; b < a->blocks_per_row; ++b) {
                const block_q4_K_packed_meta_x8 *w_group =
                    a->W + (size_t)g * (size_t)a->blocks_per_row + (size_t)b;
                accum_q4_k_packed_meta_x8_q8_k_block(acc, w_group, active, &a_row[b]);
            }
            for (int lane = 0; lane < active; ++lane) {
                c_row[n0 + lane] = acc[lane];
            }
        }
    }
}

void gemm_nt_q4_k_packed_meta_x8_q8_k_threaded_nsplit(const void *A_q8,
                                                       const void *B_packed_x8,
                                                       const float *bias,
                                                       float *C,
                                                       int M, int N, int K,
                                                       int active_threads)
{
    if (!A_q8 || !B_packed_x8 || !C || M <= 0 || N <= 0 || K <= 0 || (K % QK_K) != 0) {
        return;
    }
    ck_threadpool_t *pool = ck_threadpool_global();
    const int pool_threads = pool ? ck_threadpool_n_threads(pool) : 1;
    const int groups = (N + 7) / 8;
    if (active_threads <= 0 || active_threads > pool_threads) {
        active_threads = pool_threads;
    }
    if (active_threads > groups) {
        active_threads = groups;
    }
    if (active_threads <= 1) {
        gemm_nt_q4_k_packed_meta_x8_q8_k(A_q8, B_packed_x8, bias, C, M, N, K);
        return;
    }
    gemm_q4_packed_meta_x8_thread_work_t work = {
        .A = (const block_q8_K *)A_q8,
        .W = (const block_q4_K_packed_meta_x8 *)B_packed_x8,
        .bias = bias,
        .C = C,
        .M = M,
        .N = N,
        .K = K,
        .blocks_per_vec = K / QK_K,
        .blocks_per_row = K / QK_K,
        .groups = groups,
    };
    ck_threadpool_dispatch_n(pool, active_threads, gemm_q4_packed_meta_x8_nsplit_thread_fn, &work);
}

static void gemm_q4_packed_meta_x8_mtile_thread_fn(int ith, int nth, void *args)
{
    gemm_q4_packed_meta_x8_thread_work_t *a = (gemm_q4_packed_meta_x8_thread_work_t *)args;
    if (!a || ith < 0 || nth <= 0 || ith >= nth) {
        return;
    }
    int tile_m = a->tile_m > 0 ? a->tile_m : 4;
    if (tile_m > 8) tile_m = 8;
    const int mt = (a->M + tile_m - 1) / tile_m;
    const int total = mt * a->groups;

    for (int job = ith; job < total; job += nth) {
        const int g = job / mt;
        const int tm = job - g * mt;
        const int m0 = tm * tile_m;
        const int m1 = (m0 + tile_m < a->M) ? (m0 + tile_m) : a->M;
        const int n0 = g * 8;
        const int active = (n0 + 8 <= a->N) ? 8 : (a->N - n0);
        if (m0 >= a->M || g >= a->groups) {
            continue;
        }

        float acc[8][8];
        for (int mt_lane = 0; mt_lane < m1 - m0; ++mt_lane) {
            for (int lane = 0; lane < active; ++lane) {
                acc[mt_lane][lane] = a->bias ? a->bias[n0 + lane] : 0.0f;
            }
        }

        for (int b = 0; b < a->blocks_per_row; ++b) {
            const block_q4_K_packed_meta_x8 *w_group =
                a->W + (size_t)g * (size_t)a->blocks_per_row + (size_t)b;
            for (int m = m0; m < m1; ++m) {
                const block_q8_K *a_row = a->A + (size_t)m * (size_t)a->blocks_per_vec;
                accum_q4_k_packed_meta_x8_q8_k_block(acc[m - m0], w_group, active, &a_row[b]);
            }
        }

        for (int m = m0; m < m1; ++m) {
            float *c_row = a->C + (size_t)m * (size_t)a->N;
            for (int lane = 0; lane < active; ++lane) {
                c_row[n0 + lane] = acc[m - m0][lane];
            }
        }
    }
}

void gemm_nt_q4_k_packed_meta_x8_q8_k_threaded_mtile(const void *A_q8,
                                                      const void *B_packed_x8,
                                                      const float *bias,
                                                      float *C,
                                                      int M, int N, int K,
                                                      int tile_m,
                                                      int active_threads)
{
    if (!A_q8 || !B_packed_x8 || !C || M <= 0 || N <= 0 || K <= 0 || (K % QK_K) != 0) {
        return;
    }
    ck_threadpool_t *pool = ck_threadpool_global();
    const int pool_threads = pool ? ck_threadpool_n_threads(pool) : 1;
    const int groups = (N + 7) / 8;
    int tm = tile_m > 0 ? tile_m : 4;
    if (tm > 8) tm = 8;
    const int mt = (M + tm - 1) / tm;
    const int jobs = mt * groups;
    if (active_threads <= 0 || active_threads > pool_threads) {
        active_threads = pool_threads;
    }
    if (active_threads > jobs) {
        active_threads = jobs;
    }
    if (active_threads <= 1) {
        gemm_nt_q4_k_packed_meta_x8_q8_k(A_q8, B_packed_x8, bias, C, M, N, K);
        return;
    }
    gemm_q4_packed_meta_x8_thread_work_t work = {
        .A = (const block_q8_K *)A_q8,
        .W = (const block_q4_K_packed_meta_x8 *)B_packed_x8,
        .bias = bias,
        .C = C,
        .M = M,
        .N = N,
        .K = K,
        .blocks_per_vec = K / QK_K,
        .blocks_per_row = K / QK_K,
        .groups = groups,
        .tile_m = tm,
        .jobs = jobs,
    };
    ck_threadpool_dispatch_n(pool, active_threads, gemm_q4_packed_meta_x8_mtile_thread_fn, &work);
}


static void gemm_q4_packed_meta_x16_mtile_thread_fn(int ith, int nth, void *args)
{
    gemm_q4_packed_meta_x16_thread_work_t *a = (gemm_q4_packed_meta_x16_thread_work_t *)args;
    if (!a || ith < 0 || nth <= 0 || ith >= nth) {
        return;
    }
    int tile_m = a->tile_m > 0 ? a->tile_m : 4;
    if (tile_m > 8) tile_m = 8;
    const int mt = (a->M + tile_m - 1) / tile_m;
    const int total = mt * a->groups;

    for (int job = ith; job < total; job += nth) {
        const int g = job / mt;
        const int tm = job - g * mt;
        const int m0 = tm * tile_m;
        const int m1 = (m0 + tile_m < a->M) ? (m0 + tile_m) : a->M;
        const int n0 = g * 16;
        const int active = (n0 + 16 <= a->N) ? 16 : (a->N - n0);
        if (m0 >= a->M || g >= a->groups) {
            continue;
        }

        float acc[8][16];
        for (int mt_lane = 0; mt_lane < m1 - m0; ++mt_lane) {
            for (int lane = 0; lane < active; ++lane) {
                acc[mt_lane][lane] = a->bias ? a->bias[n0 + lane] : 0.0f;
            }
        }

        for (int b = 0; b < a->blocks_per_row; ++b) {
            const block_q4_K_packed_meta_x16 *w_group =
                a->W + (size_t)g * (size_t)a->blocks_per_row + (size_t)b;
            for (int m = m0; m < m1; ++m) {
                const block_q8_K *a_row = a->A + (size_t)m * (size_t)a->blocks_per_vec;
                accum_q4_k_packed_meta_x16_q8_k_block(acc[m - m0], w_group, active, &a_row[b]);
            }
        }

        for (int m = m0; m < m1; ++m) {
            float *c_row = a->C + (size_t)m * (size_t)a->N;
            for (int lane = 0; lane < active; ++lane) {
                c_row[n0 + lane] = acc[m - m0][lane];
            }
        }
    }
}


static void gemm_q4_packed_meta_x16_mreuse_thread_fn(int ith, int nth, void *args)
{
    gemm_q4_packed_meta_x16_thread_work_t *a = (gemm_q4_packed_meta_x16_thread_work_t *)args;
    if (!a || ith < 0 || nth <= 0 || ith >= nth) {
        return;
    }
    int tile_m = a->tile_m > 0 ? a->tile_m : 4;
    if (tile_m > 8) tile_m = 8;
    const int mt = (a->M + tile_m - 1) / tile_m;
    const int total = mt * a->groups;

    for (int job = ith; job < total; job += nth) {
        const int g = job / mt;
        const int tm = job - g * mt;
        const int m0 = tm * tile_m;
        const int m1 = (m0 + tile_m < a->M) ? (m0 + tile_m) : a->M;
        const int m_count = m1 - m0;
        const int n0 = g * 16;
        const int active = (n0 + 16 <= a->N) ? 16 : (a->N - n0);
        if (m0 >= a->M || g >= a->groups || m_count <= 0) {
            continue;
        }

        float acc[8][16];
        for (int mt_lane = 0; mt_lane < m_count; ++mt_lane) {
            for (int lane = 0; lane < active; ++lane) {
                acc[mt_lane][lane] = a->bias ? a->bias[n0 + lane] : 0.0f;
            }
        }

        for (int b = 0; b < a->blocks_per_row; ++b) {
            const block_q4_K_packed_meta_x16 *w_group =
                a->W + (size_t)g * (size_t)a->blocks_per_row + (size_t)b;
            accum_q4_k_packed_meta_x16_q8_k_block_mreuse(acc, w_group, active, a->A,
                                                          a->blocks_per_vec, b, m0, m_count);
        }

        for (int m = m0; m < m1; ++m) {
            float *c_row = a->C + (size_t)m * (size_t)a->N;
            for (int lane = 0; lane < active; ++lane) {
                c_row[n0 + lane] = acc[m - m0][lane];
            }
        }
    }
}

void gemm_nt_q4_k_packed_meta_x16_q8_k_threaded_mreuse(const void *A_q8,
                                                        const void *B_packed_x16,
                                                        const float *bias,
                                                        float *C,
                                                        int M, int N, int K,
                                                        int tile_m,
                                                        int active_threads)
{
    if (!A_q8 || !B_packed_x16 || !C || M <= 0 || N <= 0 || K <= 0 || (K % QK_K) != 0) {
        return;
    }
    ck_threadpool_t *pool = ck_threadpool_global();
    const int pool_threads = pool ? ck_threadpool_n_threads(pool) : 1;
    const int groups = (N + 15) / 16;
    int tm = tile_m > 0 ? tile_m : 4;
    if (tm > 8) tm = 8;
    const int mt = (M + tm - 1) / tm;
    const int jobs = mt * groups;
    if (active_threads <= 0 || active_threads > pool_threads) {
        active_threads = pool_threads;
    }
    if (active_threads > jobs) {
        active_threads = jobs;
    }
    gemm_q4_packed_meta_x16_thread_work_t work = {
        .A = (const block_q8_K *)A_q8,
        .W = (const block_q4_K_packed_meta_x16 *)B_packed_x16,
        .bias = bias,
        .C = C,
        .M = M,
        .N = N,
        .K = K,
        .blocks_per_vec = K / QK_K,
        .blocks_per_row = K / QK_K,
        .groups = groups,
        .tile_m = tm,
        .jobs = jobs,
    };
    if (active_threads <= 1) {
        gemm_q4_packed_meta_x16_mreuse_thread_fn(0, 1, &work);
        return;
    }
    ck_threadpool_dispatch_n(pool, active_threads, gemm_q4_packed_meta_x16_mreuse_thread_fn, &work);
}

void gemm_nt_q4_k_packed_meta_x16_q8_k_threaded_mtile(const void *A_q8,
                                                       const void *B_packed_x16,
                                                       const float *bias,
                                                       float *C,
                                                       int M, int N, int K,
                                                       int tile_m,
                                                       int active_threads)
{
    if (!A_q8 || !B_packed_x16 || !C || M <= 0 || N <= 0 || K <= 0 || (K % QK_K) != 0) {
        return;
    }
    ck_threadpool_t *pool = ck_threadpool_global();
    const int pool_threads = pool ? ck_threadpool_n_threads(pool) : 1;
    const int groups = (N + 15) / 16;
    int tm = tile_m > 0 ? tile_m : 4;
    if (tm > 8) tm = 8;
    const int mt = (M + tm - 1) / tm;
    const int jobs = mt * groups;
    if (active_threads <= 0 || active_threads > pool_threads) {
        active_threads = pool_threads;
    }
    if (active_threads > jobs) {
        active_threads = jobs;
    }
    gemm_q4_packed_meta_x16_thread_work_t work = {
        .A = (const block_q8_K *)A_q8,
        .W = (const block_q4_K_packed_meta_x16 *)B_packed_x16,
        .bias = bias,
        .C = C,
        .M = M,
        .N = N,
        .K = K,
        .blocks_per_vec = K / QK_K,
        .blocks_per_row = K / QK_K,
        .groups = groups,
        .tile_m = tm,
        .jobs = jobs,
    };
    if (active_threads <= 1) {
        gemm_q4_packed_meta_x16_mtile_thread_fn(0, 1, &work);
        return;
    }
    ck_threadpool_dispatch_n(pool, active_threads, gemm_q4_packed_meta_x16_mtile_thread_fn, &work);
}


typedef struct {
    const block_q8_K *A;
    const block_q4_K_packed_meta_x16 *W;
    const float *bias;
    float *C;
    int M;
    int D;
    int K;
    int tile_m;
    int blocks_per_vec;
    int blocks_per_row;
    int groups_d;
    int jobs;
} gemm_q4_gateup_swiglu_x16_work_t;


static void gemm_q4_gateup_swiglu_x16_thread_fn(int ith, int nth, void *args)
{
    gemm_q4_gateup_swiglu_x16_work_t *a = (gemm_q4_gateup_swiglu_x16_work_t *)args;
    if (!a || ith < 0 || nth <= 0 || ith >= nth) {
        return;
    }

    int tile_m = a->tile_m > 0 ? a->tile_m : 4;
    if (tile_m > 8) tile_m = 8;
    const int mt = (a->M + tile_m - 1) / tile_m;

    for (int job = ith; job < a->jobs; job += nth) {
        const int g = job / mt;
        const int tm = job - g * mt;
        const int m0 = tm * tile_m;
        const int m1 = (m0 + tile_m < a->M) ? (m0 + tile_m) : a->M;
        const int m_count = m1 - m0;
        const int d0 = g * 16;
        const int active = (d0 + 16 <= a->D) ? 16 : (a->D - d0);
        if (m_count <= 0 || active <= 0 || g >= a->groups_d) {
            continue;
        }

        float acc_gate[8][16];
        float acc_up[8][16];
        for (int mt_lane = 0; mt_lane < m_count; ++mt_lane) {
            for (int lane = 0; lane < active; ++lane) {
                acc_gate[mt_lane][lane] = a->bias ? a->bias[d0 + lane] : 0.0f;
                acc_up[mt_lane][lane] = a->bias ? a->bias[a->D + d0 + lane] : 0.0f;
            }
        }

        for (int b = 0; b < a->blocks_per_row; ++b) {
            const block_q4_K_packed_meta_x16 *w_gate =
                a->W + (size_t)g * (size_t)a->blocks_per_row + (size_t)b;
            const block_q4_K_packed_meta_x16 *w_up =
                a->W + (size_t)(a->groups_d + g) * (size_t)a->blocks_per_row + (size_t)b;
            accum_q4_k_packed_meta_x16_q8_k_block_mreuse(acc_gate, w_gate, active, a->A,
                                                          a->blocks_per_vec, b, m0, m_count);
            accum_q4_k_packed_meta_x16_q8_k_block_mreuse(acc_up, w_up, active, a->A,
                                                          a->blocks_per_vec, b, m0, m_count);
        }

        for (int m = m0; m < m1; ++m) {
            float *c_row = a->C + (size_t)m * (size_t)a->D;
#if defined(__clang__) || defined(__INTEL_LLVM_COMPILER)
#pragma clang loop vectorize(enable) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
            for (int lane = 0; lane < active; ++lane) {
                const float gate = acc_gate[m - m0][lane];
                const float up = acc_up[m - m0][lane];
                c_row[d0 + lane] = (gate / (1.0f + expf(-gate))) * up;
            }
        }
    }
}

void gemm_nt_q4_k_packed_meta_x16_gateup_swiglu_fused_vnni(const void *A_q8,
                                                            const void *B_packed_x16,
                                                            const float *bias,
                                                            float *C,
                                                            int M, int D, int K,
                                                            int tile_m,
                                                            int active_threads)
{
    if (!A_q8 || !B_packed_x16 || !C || M <= 0 || D <= 0 || K <= 0 || (K % QK_K) != 0) {
        return;
    }
    ck_threadpool_t *pool = ck_threadpool_global();
    const int pool_threads = pool ? ck_threadpool_n_threads(pool) : 1;
    int tm = tile_m > 0 ? tile_m : 4;
    if (tm > 8) tm = 8;
    const int groups_d = (D + 15) / 16;
    const int mt = (M + tm - 1) / tm;
    const int jobs = mt * groups_d;
    if (active_threads <= 0 || active_threads > pool_threads) {
        active_threads = pool_threads;
    }
    if (active_threads > jobs) {
        active_threads = jobs;
    }
    if (active_threads < 1) {
        active_threads = 1;
    }

    gemm_q4_gateup_swiglu_x16_work_t work = {
        .A = (const block_q8_K *)A_q8,
        .W = (const block_q4_K_packed_meta_x16 *)B_packed_x16,
        .bias = bias,
        .C = C,
        .M = M,
        .D = D,
        .K = K,
        .tile_m = tm,
        .blocks_per_vec = K / QK_K,
        .blocks_per_row = K / QK_K,
        .groups_d = groups_d,
        .jobs = jobs,
    };
    if (!pool || active_threads <= 1) {
        gemm_q4_gateup_swiglu_x16_thread_fn(0, 1, &work);
        return;
    }
    ck_threadpool_dispatch_n(pool, active_threads, gemm_q4_gateup_swiglu_x16_thread_fn, &work);
}


static void gemm_q4_packed_u8_x16_mtile_thread_fn(int ith, int nth, void *args)
{
    gemm_q4_packed_u8_x16_thread_work_t *a = (gemm_q4_packed_u8_x16_thread_work_t *)args;
    if (!a || ith < 0 || nth <= 0 || ith >= nth) return;
    int tile_m = a->tile_m > 0 ? a->tile_m : 4;
    if (tile_m > 8) tile_m = 8;
    const int mt = (a->M + tile_m - 1) / tile_m;
    const int total = mt * a->groups;

    for (int job = ith; job < total; job += nth) {
        const int g = job / mt;
        const int tm = job - g * mt;
        const int m0 = tm * tile_m;
        const int m1 = (m0 + tile_m < a->M) ? (m0 + tile_m) : a->M;
        const int n0 = g * 16;
        const int active = (n0 + 16 <= a->N) ? 16 : (a->N - n0);
        if (m0 >= a->M || g >= a->groups) continue;

        float acc[8][16];
        for (int mt_lane = 0; mt_lane < m1 - m0; ++mt_lane) {
            for (int lane = 0; lane < active; ++lane) {
                acc[mt_lane][lane] = a->bias ? a->bias[n0 + lane] : 0.0f;
            }
        }

        for (int b = 0; b < a->blocks_per_row; ++b) {
            const block_q4_K_packed_u8_x16 *w_group =
                a->W + (size_t)g * (size_t)a->blocks_per_row + (size_t)b;
            for (int m = m0; m < m1; ++m) {
                const block_q8_K *a_row = a->A + (size_t)m * (size_t)a->blocks_per_vec;
                accum_q4_k_packed_u8_x16_q8_k_block(acc[m - m0], w_group, active, &a_row[b]);
            }
        }

        for (int m = m0; m < m1; ++m) {
            float *c_row = a->C + (size_t)m * (size_t)a->N;
            for (int lane = 0; lane < active; ++lane) {
                c_row[n0 + lane] = acc[m - m0][lane];
            }
        }
    }
}

void gemm_nt_q4_k_packed_u8_x16_q8_k_threaded_mtile(const void *A_q8,
                                                     const void *B_packed_u8_x16,
                                                     const float *bias,
                                                     float *C,
                                                     int M, int N, int K,
                                                     int tile_m,
                                                     int active_threads)
{
    if (!A_q8 || !B_packed_u8_x16 || !C || M <= 0 || N <= 0 || K <= 0 || (K % QK_K) != 0) return;
    ck_threadpool_t *pool = ck_threadpool_global();
    const int pool_threads = pool ? ck_threadpool_n_threads(pool) : 1;
    const int groups = (N + 15) / 16;
    int tm = tile_m > 0 ? tile_m : 4;
    if (tm > 8) tm = 8;
    const int mt = (M + tm - 1) / tm;
    const int jobs = mt * groups;
    if (active_threads <= 0 || active_threads > pool_threads) active_threads = pool_threads;
    if (active_threads > jobs) active_threads = jobs;
    gemm_q4_packed_u8_x16_thread_work_t work = {
        .A = (const block_q8_K *)A_q8,
        .W = (const block_q4_K_packed_u8_x16 *)B_packed_u8_x16,
        .bias = bias,
        .C = C,
        .M = M,
        .N = N,
        .K = K,
        .blocks_per_vec = K / QK_K,
        .blocks_per_row = K / QK_K,
        .groups = groups,
        .tile_m = tm,
        .jobs = jobs,
    };
    if (active_threads <= 1) {
        gemm_q4_packed_u8_x16_mtile_thread_fn(0, 1, &work);
        return;
    }
    ck_threadpool_dispatch_n(pool, active_threads, gemm_q4_packed_u8_x16_mtile_thread_fn, &work);
}


void gemv_q4_k_q8_k_vnni(float *y,
                         const void *W,
                         const void *x_q8,
                         int M, int K)
{
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    const char *fast_env = getenv("CK_ENABLE_Q4K_Q8K_VNNI_FAST");
    const int fast_disabled = fast_env && fast_env[0] && fast_env[0] == '0';
    if (!fast_disabled && !ck_strict_parity_enabled()) {
        if (!y || !W || !x_q8 || M <= 0 || K <= 0) {
            return;
        }

        const block_q4_K *blocks = (const block_q4_K *)W;
        const block_q8_K *x = (const block_q8_K *)x_q8;
        const int blocks_per_row = K / QK_K;

        for (int row = 0; row < M; ++row) {
            const block_q4_K *w_row = blocks + (size_t)row * (size_t)blocks_per_row;
            float sum = 0.0f;
            for (int b = 0; b < blocks_per_row; ++b) {
                sum += dot_q4_k_q8_k_vnni_block(&w_row[b], &x[b]);
            }
            y[row] = sum;
        }
        return;
    }
#endif

    /* Strict/debug parity keeps the llama-style scalar accumulation path.
     * Production AVX-512 hosts use VNNI by default; set
     * CK_ENABLE_Q4K_Q8K_VNNI_FAST=0 or CK_DEBUG_Q4K_Q8_REF=1 when attributing
     * borderline logit movement against scalar/reference behavior.
     */
    gemv_q4_k_q8_k_ref(y, W, x_q8, M, K);
}


static inline float ck_q4k_silu_f32(float x)
{
    return x / (1.0f + expf(-x));
}

typedef struct {
    const block_q8_K *A;
    const block_q4_K *W;
    const float *bias;
    float *C;
    int M;
    int D;
    int K;
    int blocks_per_vec;
    int blocks_per_row;
} gemm_q4_gateup_swiglu_work_t;

static void gemm_q4_gateup_swiglu_thread_fn(int ith, int nth, void *args)
{
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    gemm_q4_gateup_swiglu_work_t *a = (gemm_q4_gateup_swiglu_work_t *)args;
    if (!a || ith < 0 || nth <= 0 || ith >= nth) return;

    const int dd = (a->D + nth - 1) / nth;
    const int d0 = dd * ith;
    const int d1 = (d0 + dd < a->D) ? (d0 + dd) : a->D;
    if (d0 >= a->D) return;

    for (int d = d0; d < d1; ++d) {
        const block_q4_K *w_gate = a->W + (size_t)d * (size_t)a->blocks_per_row;
        const block_q4_K *w_up = a->W + (size_t)(a->D + d) * (size_t)a->blocks_per_row;
        const float b_gate = a->bias ? a->bias[d] : 0.0f;
        const float b_up = a->bias ? a->bias[a->D + d] : 0.0f;
        for (int m = 0; m < a->M; ++m) {
            const block_q8_K *x = a->A + (size_t)m * (size_t)a->blocks_per_vec;
            float gate = b_gate;
            float up = b_up;
            for (int b = 0; b < a->blocks_per_row; ++b) {
                gate += dot_q4_k_q8_k_vnni_block(&w_gate[b], &x[b]);
                up += dot_q4_k_q8_k_vnni_block(&w_up[b], &x[b]);
            }
            a->C[(size_t)m * (size_t)a->D + (size_t)d] = ck_q4k_silu_f32(gate) * up;
        }
    }
#else
    (void)ith;
    (void)nth;
    (void)args;
#endif
}

void gemm_nt_q4_k_q8_k_gateup_swiglu_fused_vnni(const void *A_q8,
                                                 const void *B_gate_up,
                                                 const float *bias,
                                                 float *C,
                                                 int M,
                                                 int D,
                                                 int K,
                                                 int threads)
{
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    if (!A_q8 || !B_gate_up || !C || M <= 0 || D <= 0 || K <= 0 || (K % QK_K) != 0) {
        return;
    }

    ck_threadpool_t *pool = ck_threadpool_global();
    int active = threads > 0 ? threads : (pool ? ck_threadpool_n_threads(pool) : 1);
    if (active < 1) active = 1;
    if (active > D) active = D;

    gemm_q4_gateup_swiglu_work_t work = {
        .A = (const block_q8_K *)A_q8,
        .W = (const block_q4_K *)B_gate_up,
        .bias = bias,
        .C = C,
        .M = M,
        .D = D,
        .K = K,
        .blocks_per_vec = K / QK_K,
        .blocks_per_row = K / QK_K,
    };

    if (active <= 1 || !pool) {
        gemm_q4_gateup_swiglu_thread_fn(0, 1, &work);
        return;
    }
    ck_threadpool_dispatch_n(pool, active, gemm_q4_gateup_swiglu_thread_fn, &work);
#else
    (void)A_q8;
    (void)B_gate_up;
    (void)bias;
    (void)C;
    (void)M;
    (void)D;
    (void)K;
    (void)threads;
#endif
}


void gemv_q4_k_q8_k_parallel_vnni(float *y,
                                  const void *W,
                                  const void *x_q8,
                                  int M, int K,
                                  int ith, int nth)
{
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    if (!y || !W || !x_q8 || M <= 0 || K <= 0) {
        return;
    }
    if (ith < 0 || nth <= 0 || ith >= nth) {
        return;
    }

    const int dr = (M + nth - 1) / nth;
    const int r0 = dr * ith;
    const int r1 = (r0 + dr < M) ? (r0 + dr) : M;
    if (r0 >= M) {
        return;
    }

    const block_q4_K *blocks = (const block_q4_K *)W;
    const block_q8_K *x = (const block_q8_K *)x_q8;
    const int blocks_per_row = K / QK_K;

    for (int row = r0; row < r1; ++row) {
        const block_q4_K *w_row = blocks + (size_t)row * (size_t)blocks_per_row;
        float sum = 0.0f;
        for (int b = 0; b < blocks_per_row; ++b) {
            sum += dot_q4_k_q8_k_vnni_block(&w_row[b], &x[b]);
        }
        y[row] = sum;
    }
#else
    (void)y;
    (void)W;
    (void)x_q8;
    (void)M;
    (void)K;
    (void)ith;
    (void)nth;
#endif
}
