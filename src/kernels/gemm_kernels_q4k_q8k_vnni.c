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
 */

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "ckernel_engine.h"
#include "ck_threadpool.h"
#include "ckernel_quant.h"

#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
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

#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
static inline int32_t hsum256_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i sum = _mm_add_epi32(lo, hi);
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(sum);
}

static inline int32_t dot_q4_k_q8_k_32_vnni(const uint8_t *q4_packed_32,
                                            const int8_t *q8_32,
                                            int high_nibble) {
    const __m256i packed = _mm256_loadu_si256((const __m256i *)q4_packed_32);
    const __m256i mask4 = _mm256_set1_epi8(0x0F);
    const __m256i q4_bytes = high_nibble
        ? _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask4)
        : _mm256_and_si256(packed, mask4);
    const __m256i q8_bytes = _mm256_loadu_si256((const __m256i *)q8_32);
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

size_t q4_k_packed_u8_block_size(void)
{
    return sizeof(block_q4_K_packed_u8);
}

size_t q4_k_packed_meta_block_size(void)
{
    return sizeof(block_q4_K_packed_meta);
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
