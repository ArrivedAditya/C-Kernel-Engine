/**
 * @file gemm_kernels_q5_1_q8_1.c
 * @brief Q5_1 x Q8_1 contract kernels used for ggml parity (Gemma-sensitive path)
 */

#include <stdint.h>
#include <string.h>
#include <math.h>

#include "ckernel_quant.h"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define CK_Q51_STACK_Q8_BLOCKS 256

/* Q8_1 is used only by this contract kernel path. Keep a local definition so
 * this file does not depend on global header churn.
 * Layout matches ggml: fp16 d, fp16 s, 32 int8 quants. */
#ifndef QK8_1
#define QK8_1 32
#endif

typedef struct {
    ck_half d;
    ck_half s;
    int8_t qs[QK8_1];
} block_q8_1;

#if defined(__AVX2__)
static const uint32_t ck_q51_high_nibble_lut[16] = {
    0x00000000U, 0x00000010U, 0x00001000U, 0x00001010U,
    0x00100000U, 0x00100010U, 0x00101000U, 0x00101010U,
    0x10000000U, 0x10000010U, 0x10001000U, 0x10001010U,
    0x10100000U, 0x10100010U, 0x10101000U, 0x10101010U,
};

static inline uint32_t ck_q51_high_nibble_word(uint32_t bits4) {
    return ck_q51_high_nibble_lut[bits4 & 0x0fU];
}

static inline __m128i ck_q51_high_bits_16_avx2(uint32_t bits16) {
    const uint32_t w0 = ck_q51_high_nibble_word(bits16);
    const uint32_t w1 = ck_q51_high_nibble_word(bits16 >> 4);
    const uint32_t w2 = ck_q51_high_nibble_word(bits16 >> 8);
    const uint32_t w3 = ck_q51_high_nibble_word(bits16 >> 12);
    return _mm_set_epi32((int)w3, (int)w2, (int)w1, (int)w0);
}

static inline int ck_q51_hsum256_epi32(__m256i v) {
    const __m128i lo = _mm256_castsi256_si128(v);
    const __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i sum = _mm_add_epi32(lo, hi);
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(sum);
}

#endif

/* Quantize one FP32 row to Q8_1 blocks (ggml-compatible scalar path). */
static void quantize_row_q8_1_scalar(const float *x, block_q8_1 *y, int k) {
    const int nb = k / QK8_1;
    for (int b = 0; b < nb; ++b) {
        const float *xb = x + (size_t)b * QK8_1;
        float amax = 0.0f;
        for (int j = 0; j < QK8_1; ++j) {
            float av = xb[j] >= 0.0f ? xb[j] : -xb[j];
            if (av > amax) amax = av;
        }

        const float d = amax / 127.0f;
        const float id = (d != 0.0f) ? (1.0f / d) : 0.0f;
        y[b].d = CK_FP32_TO_FP16(d);

        int sum = 0;
        for (int j = 0; j < QK8_1; ++j) {
            int q = (int)roundf(xb[j] * id);
            y[b].qs[j] = (int8_t)q;
            sum += q;
        }
        y[b].s = CK_FP32_TO_FP16((float)sum * d);
    }
}

#if defined(__AVX2__)
static inline __m256i dot_q5_1_q8_1_block_sumi_avx2(const block_q5_1 *w,
                                                     const block_q8_1 *x) {
    uint32_t qh;
    memcpy(&qh, w->qh, sizeof(qh));

    const __m128i qpacked = _mm_loadu_si128((const __m128i *)(const void *)w->qs);
    const __m128i low_mask = _mm_set1_epi8(0x0f);
    const __m128i qlo = _mm_or_si128(_mm_and_si128(qpacked, low_mask),
                                     ck_q51_high_bits_16_avx2(qh));
    const __m128i qhi = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(qpacked, 4), low_mask),
                                     ck_q51_high_bits_16_avx2(qh >> 16));
    const __m256i q5 = _mm256_inserti128_si256(_mm256_castsi128_si256(qlo), qhi, 1);
    const __m256i q8 = _mm256_loadu_si256((const __m256i *)(const void *)x->qs);
#if defined(__AVXVNNI__)
    return _mm256_dpbusd_epi32(_mm256_setzero_si256(), q5, q8);
#else
    const __m256i prod16 = _mm256_maddubs_epi16(q5, q8);
    return _mm256_madd_epi16(prod16, _mm256_set1_epi16(1));
#endif
}

/* One 32-element block dot: Q5_1(weights) x Q8_1(activations), AVX2.
 * The high-bit placement intentionally mirrors dot_q5_1_q8_1_block().
 */
static float dot_q5_1_q8_1_block_avx2(const block_q5_1 *w, const block_q8_1 *x) {
    const __m256i sumi = dot_q5_1_q8_1_block_sumi_avx2(w, x);

    const float wd = CK_FP16_TO_FP32(w->d);
    const float wm = CK_FP16_TO_FP32(w->m);
    const float xd = CK_FP16_TO_FP32(x->d);
    const float xs = CK_FP16_TO_FP32(x->s);
    return (wd * xd) * (float)ck_q51_hsum256_epi32(sumi) + wm * xs;
}

#endif

/* One 32-element block dot: Q5_1(weights) x Q8_1(activations). */
static float dot_q5_1_q8_1_block(const block_q5_1 *w, const block_q8_1 *x) {
#if defined(__AVX2__)
    return dot_q5_1_q8_1_block_avx2(w, x);
#else
    uint32_t qh;
    memcpy(&qh, w->qh, sizeof(qh));

    int sumi0 = 0;
    int sumi1 = 0;
    for (int j = 0; j < QK5_1 / 2; ++j) {
        const uint8_t xh0 = (uint8_t)(((qh >> (j + 0)) << 4) & 0x10);
        const uint8_t xh1 = (uint8_t)(((qh >> (j + 12))     ) & 0x10);
        const int32_t q0 = (int32_t)((w->qs[j] & 0x0F) | xh0);
        const int32_t q1 = (int32_t)((w->qs[j] >> 4) | xh1);
        sumi0 += q0 * (int32_t)x->qs[j];
        sumi1 += q1 * (int32_t)x->qs[j + QK5_1 / 2];
    }

    const float wd = CK_FP16_TO_FP32(w->d);
    const float wm = CK_FP16_TO_FP32(w->m);
    const float xd = CK_FP16_TO_FP32(x->d);
    const float xs = CK_FP16_TO_FP32(x->s);
    return (wd * xd) * (float)(sumi0 + sumi1) + wm * xs;
#endif
}

void gemv_q5_1_q8_1_ref(float *y,
                        const void *W,
                        const void *x_q8,
                        int M, int K)
{
    if (!y || !W || !x_q8 || M <= 0 || K <= 0 || (K % QK5_1) != 0) {
        return;
    }

    const block_q5_1 *blocks = (const block_q5_1 *)W;
    const block_q8_1 *x = (const block_q8_1 *)x_q8;
    const int blocks_per_row = K / QK5_1;

    for (int row = 0; row < M; ++row) {
        const block_q5_1 *w_row = &blocks[row * blocks_per_row];
        float sum = 0.0f;
        for (int b = 0; b < blocks_per_row; ++b) {
            sum += dot_q5_1_q8_1_block(&w_row[b], &x[b]);
        }
        y[row] = sum;
    }
}

void gemm_nt_q5_1_q8_1_ref(const void *A_q8,
                           const void *B,
                           const float *bias,
                           float *C,
                           int M, int N, int K)
{
    if (!A_q8 || !B || !C || M <= 0 || N <= 0 || K <= 0 || (K % QK5_1) != 0) {
        return;
    }

    const block_q8_1 *A = (const block_q8_1 *)A_q8;
    const block_q5_1 *W = (const block_q5_1 *)B;
    const int blocks_per_row = K / QK5_1;

    for (int m = 0; m < M; ++m) {
        const block_q8_1 *a_row = &A[m * blocks_per_row];
        for (int n = 0; n < N; ++n) {
            const block_q5_1 *w_row = &W[n * blocks_per_row];
            float sum = 0.0f;
            for (int b = 0; b < blocks_per_row; ++b) {
                sum += dot_q5_1_q8_1_block(&w_row[b], &a_row[b]);
            }
            C[m * N + n] = sum + (bias ? bias[n] : 0.0f);
        }
    }
}

void gemv_q5_1_q8_1(float *y,
                    const void *W,
                    const float *x,
                    int M,
                    int K)
{
    if (!y || !W || !x || M <= 0 || K <= 0 || (K % QK5_1) != 0) {
        return;
    }

    const int blocks_per_row = K / QK5_1;
    if (blocks_per_row > CK_Q51_STACK_Q8_BLOCKS) {
        return;
    }

    block_q8_1 x_q8[CK_Q51_STACK_Q8_BLOCKS];
    quantize_row_q8_1_scalar(x, x_q8, K);
    gemv_q5_1_q8_1_ref(y, W, x_q8, M, K);
}

void gemm_nt_q5_1_q8_1(const float *A,
                       const void *B,
                       const float *bias,
                       float *C,
                       int M,
                       int N,
                       int K)
{
    if (!A || !B || !C || M <= 0 || N <= 0 || K <= 0 || (K % QK5_1) != 0) {
        return;
    }

    const int blocks_per_row = K / QK5_1;
    if (blocks_per_row > CK_Q51_STACK_Q8_BLOCKS) {
        return;
    }

    const block_q5_1 *W = (const block_q5_1 *)B;

    for (int m = 0; m < M; ++m) {
        block_q8_1 a_q8[CK_Q51_STACK_Q8_BLOCKS];
        quantize_row_q8_1_scalar(&A[m * K], a_q8, K);
        float *c_row = &C[(size_t)m * (size_t)N];

        int n = 0;
        for (; n + 7 < N; n += 8) {
            const block_q5_1 *w0 = &W[(size_t)(n + 0) * (size_t)blocks_per_row];
            const block_q5_1 *w1 = &W[(size_t)(n + 1) * (size_t)blocks_per_row];
            const block_q5_1 *w2 = &W[(size_t)(n + 2) * (size_t)blocks_per_row];
            const block_q5_1 *w3 = &W[(size_t)(n + 3) * (size_t)blocks_per_row];
            const block_q5_1 *w4 = &W[(size_t)(n + 4) * (size_t)blocks_per_row];
            const block_q5_1 *w5 = &W[(size_t)(n + 5) * (size_t)blocks_per_row];
            const block_q5_1 *w6 = &W[(size_t)(n + 6) * (size_t)blocks_per_row];
            const block_q5_1 *w7 = &W[(size_t)(n + 7) * (size_t)blocks_per_row];
            float s0 = 0.0f;
            float s1 = 0.0f;
            float s2 = 0.0f;
            float s3 = 0.0f;
            float s4 = 0.0f;
            float s5 = 0.0f;
            float s6 = 0.0f;
            float s7 = 0.0f;

            for (int b = 0; b < blocks_per_row; ++b) {
                const block_q8_1 *x = &a_q8[b];
                s0 += dot_q5_1_q8_1_block(&w0[b], x);
                s1 += dot_q5_1_q8_1_block(&w1[b], x);
                s2 += dot_q5_1_q8_1_block(&w2[b], x);
                s3 += dot_q5_1_q8_1_block(&w3[b], x);
                s4 += dot_q5_1_q8_1_block(&w4[b], x);
                s5 += dot_q5_1_q8_1_block(&w5[b], x);
                s6 += dot_q5_1_q8_1_block(&w6[b], x);
                s7 += dot_q5_1_q8_1_block(&w7[b], x);
            }

            c_row[n + 0] = s0 + (bias ? bias[n + 0] : 0.0f);
            c_row[n + 1] = s1 + (bias ? bias[n + 1] : 0.0f);
            c_row[n + 2] = s2 + (bias ? bias[n + 2] : 0.0f);
            c_row[n + 3] = s3 + (bias ? bias[n + 3] : 0.0f);
            c_row[n + 4] = s4 + (bias ? bias[n + 4] : 0.0f);
            c_row[n + 5] = s5 + (bias ? bias[n + 5] : 0.0f);
            c_row[n + 6] = s6 + (bias ? bias[n + 6] : 0.0f);
            c_row[n + 7] = s7 + (bias ? bias[n + 7] : 0.0f);
        }

        for (; n < N; ++n) {
            const block_q5_1 *w_row = &W[(size_t)n * (size_t)blocks_per_row];
            float sum = 0.0f;
            for (int b = 0; b < blocks_per_row; ++b) {
                sum += dot_q5_1_q8_1_block(&w_row[b], &a_q8[b]);
            }
            c_row[n] = sum + (bias ? bias[n] : 0.0f);
        }
    }
}
