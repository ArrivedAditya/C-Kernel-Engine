/**
 * @file gemm_kernels_q4k_q8k_avx2.c
 * @brief AVX2 Q4_K x Q8_K matvec kernel (inference only)
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
 * Requires AVX2 for 256-bit integer operations.
 */

#include <stddef.h>
#include <stdint.h>

#include "ckernel_quant.h"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

void gemv_q4_k_q8_k_ref(float *y,
                        const void *W,
                        const void *x_q8,
                        int M, int K);

#if defined(__AVX2__)
static inline int32_t hsum256_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i sum = _mm_add_epi32(lo, hi);
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(sum);
}

static inline int32_t dot_q4_q8_32_avx2(const uint8_t *q4,
                                        const int8_t *q8,
                                        int high_nibble) {
    const __m256i packed = _mm256_loadu_si256((const __m256i *)q4);
    const __m256i mask4 = _mm256_set1_epi8(0x0F);
    const __m256i q4_bytes = high_nibble
        ? _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask4)
        : _mm256_and_si256(packed, mask4);
    const __m256i q8_bytes = _mm256_loadu_si256((const __m256i *)q8);
    const __m256i pair_sums_i16 = _mm256_maddubs_epi16(q4_bytes, q8_bytes);
    const __m256i sums_i32 = _mm256_madd_epi16(pair_sums_i16, _mm256_set1_epi16(1));
    return hsum256_epi32(sums_i32);
}

static float dot_q4_k_q8_k_avx2(const block_q4_K *w,
                                const block_q8_K *x,
                                int k)
{
    const int nb = k / QK_K;
    float sumf = 0.0f;

    for (int i = 0; i < nb; ++i) {
        uint8_t sc[8], m_val[8];
        unpack_q4_k_scales(w[i].scales, sc, m_val);

        const float d = CK_FP16_TO_FP32(w[i].d) * x[i].d;
        const float dmin = CK_FP16_TO_FP32(w[i].dmin) * x[i].d;

        int sumi = 0;
        for (int j = 0; j < QK_K / 16; ++j) {
            sumi += (int)x[i].bsums[j] * (int)m_val[j / 2];
        }

        int32_t sumi_block = 0;
        int is = 0;
        int q_offset = 0;

        for (int j = 0; j < QK_K; j += 64) {
            const uint8_t *qs = &w[i].qs[q_offset];
            const int8_t *q8_lo = &x[i].qs[j];
            const int8_t *q8_hi = &x[i].qs[j + 32];

            const int32_t lo = dot_q4_q8_32_avx2(qs, q8_lo, 0);
            const int32_t hi = dot_q4_q8_32_avx2(qs, q8_hi, 1);
            sumi_block += (int32_t)sc[is] * lo + (int32_t)sc[is + 1] * hi;

            q_offset += 32;
            is += 2;
        }

        sumf += d * (float)sumi_block - dmin * (float)sumi;
    }

    return sumf;
}
#endif

void gemv_q4_k_q8_k_avx2(float *y,
                         const void *W,
                         const void *x_q8,
                         int M, int K)
{
#if defined(__AVX2__)
    if (!y || !W || !x_q8 || M <= 0 || K <= 0) {
        return;
    }

    const block_q4_K *blocks = (const block_q4_K *)W;
    const block_q8_K *x = (const block_q8_K *)x_q8;
    const int blocks_per_row = K / QK_K;

    for (int row = 0; row < M; ++row) {
        const block_q4_K *w_row = blocks + (size_t)row * (size_t)blocks_per_row;
        y[row] = dot_q4_k_q8_k_avx2(w_row, x, K);
    }
#else
    gemv_q4_k_q8_k_ref(y, W, x_q8, M, K);
#endif
}
