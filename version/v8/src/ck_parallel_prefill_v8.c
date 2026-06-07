/**
 * @file ck_parallel_prefill.c
 * @brief Thread-pool-parallel GEMM dispatch for v7 prefill
 *
 * Wraps each GEMM kernel call in a threadpool dispatch, splitting work
 * across the M (tokens) dimension. Each thread processes rows [r0, r1):
 *
 *   int dr = (M + nth - 1) / nth;
 *   int r0 = dr * ith;
 *   int r1 = min(r0 + dr, M);
 *   serial_gemm(A + r0 * A_row_bytes, B, bias, C + r0 * N, r1-r0, N, K);
 *
 * B (weights) and bias are shared read-only across all threads.
 *
 * Fast path: if M <= 1 or pool has <= 1 thread, calls serial directly.
 *
 * Reuses the same global thread pool as decode (ck_threadpool_global()).
 */

#include "ck_parallel_prefill_v8.h"
#include "ck_threadpool.h"
#include "ckernel_quant.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* Serial GEMM kernels (defined in src/kernels/) */
extern void gemm_nt_q5_0_q8_0(const void *A, const void *B, const float *bias,
                                float *C, int M, int N, int K);
extern void gemm_nt_q8_0_q8_0(const void *A, const void *B, const float *bias,
                                float *C, int M, int N, int K);
extern void gemm_nt_q6_k_q8_k(const void *A, const void *B, const float *bias,
                                float *C, int M, int N, int K);
extern void gemm_nt_q5_1_q8_1(const float *A, const void *B, const float *bias,
                                float *C, int M, int N, int K);
extern void gemm_nt_q5_k(const float *A, const void *B, const float *bias,
                          float *C, int M, int N, int K);

/* ============================================================================
 * Lifecycle
 * ============================================================================ */

void ck_parallel_prefill_init(void)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    if (pool) {
        fprintf(stderr, "[CK parallel prefill] Initialized with %d threads\n",
                ck_threadpool_n_threads(pool));
    }
}

void ck_parallel_prefill_shutdown(void)
{
    /* Pool is shared with decode — decode shutdown handles actual destroy.
     * This is a no-op to avoid double-free. The global pool is destroyed
     * once by whichever shutdown path runs last (or by ck_model_free). */
}

/* ============================================================================
 * Argument Packing Struct
 * ============================================================================ */

typedef struct {
    const void  *A;           /* Input activations (quantized) */
    const void  *B;           /* Weight matrix (quantized, read-only) */
    const float *bias;        /* Optional bias vector (read-only) */
    float       *C;           /* Output matrix [M x N] */
    int          M;           /* Number of tokens (rows to split) */
    int          N;           /* Output dimension */
    int          K;           /* Input dimension */
    size_t       A_row_bytes; /* Bytes per row of A (for pointer arithmetic) */
} gemm_args_t;

static int ck_min_int(int a, int b) { return a < b ? a : b; }

static int ck_env_enabled(const char *name)
{
    const char *v = getenv(name);
    return v && v[0] && strcmp(v, "0") != 0;
}

static int ck_env_int_or2(const char *primary, const char *secondary, int fallback)
{
    const char *v = getenv(primary);
    if ((!v || !v[0]) && secondary) v = getenv(secondary);
    if (!v || !v[0]) return fallback;
    int parsed = atoi(v);
    return parsed > 0 ? parsed : fallback;
}

static int ck_shape_aware_enabled(const ck_threadpool_t *pool)
{
    const int pool_threads = ck_threadpool_n_threads(pool);
    if (ck_env_enabled("CK_DISABLE_SHAPE_AWARE_THREADPOOL")) return 0;
    if (getenv("CK_GEMM_THREAD_CAP") || getenv("CK_GEMV_THREAD_CAP")) return 1;
    return pool_threads > 16;
}

static int ck_select_gemm_active_threads(const ck_threadpool_t *pool, int M, int N, int K)
{
    const int pool_threads = ck_threadpool_n_threads(pool);
    if (pool_threads <= 1 || M <= 1 || N <= 0 || K <= 0) return 1;
    if (!ck_shape_aware_enabled(pool)) return pool_threads;
    if (N >= 4096 || K >= 4096) return pool_threads;
    if (M >= 512) return ck_min_int(pool_threads, ck_env_int_or2("CK_GEMM_THREAD_CAP", "CK_GEMV_THREAD_CAP", 24));
    return pool_threads;
}

static int ck_should_run_gemm_serial(const ck_threadpool_t *pool, int M, int N, int K)
{
    if (!ck_shape_aware_enabled(pool)) return 0;
    const int threshold = ck_env_int_or2("CK_GEMM_SMALL_SERIAL_THRESHOLD", NULL, 16);
    return M < threshold && N < 512 && K < 512;
}

/* ============================================================================
 * Work Functions (called on each thread)
 *
 * Each computes rows [r0, r1) of the output by calling the serial GEMM
 * on a sub-range of A and C.
 * ============================================================================ */

static void work_gemm_nt_q5_0_q8_0(int ith, int nth, void *args)
{
    const gemm_args_t *a = (const gemm_args_t *)args;
    int dr = (a->M + nth - 1) / nth;
    int r0 = dr * ith;
    int r1 = (r0 + dr < a->M) ? (r0 + dr) : a->M;
    if (r0 >= a->M) return;

    gemm_nt_q5_0_q8_0(
        (const char *)a->A + (size_t)r0 * a->A_row_bytes,
        a->B,
        a->bias,
        a->C + (size_t)r0 * a->N,
        r1 - r0, a->N, a->K
    );
}

static void work_gemm_nt_q8_0_q8_0(int ith, int nth, void *args)
{
    const gemm_args_t *a = (const gemm_args_t *)args;
    int dr = (a->M + nth - 1) / nth;
    int r0 = dr * ith;
    int r1 = (r0 + dr < a->M) ? (r0 + dr) : a->M;
    if (r0 >= a->M) return;

    gemm_nt_q8_0_q8_0(
        (const char *)a->A + (size_t)r0 * a->A_row_bytes,
        a->B,
        a->bias,
        a->C + (size_t)r0 * a->N,
        r1 - r0, a->N, a->K
    );
}

static void work_gemm_nt_q6_k_q8_k(int ith, int nth, void *args)
{
    const gemm_args_t *a = (const gemm_args_t *)args;
    int dr = (a->M + nth - 1) / nth;
    int r0 = dr * ith;
    int r1 = (r0 + dr < a->M) ? (r0 + dr) : a->M;
    if (r0 >= a->M) return;

    gemm_nt_q6_k_q8_k(
        (const char *)a->A + (size_t)r0 * a->A_row_bytes,
        a->B,
        a->bias,
        a->C + (size_t)r0 * a->N,
        r1 - r0, a->N, a->K
    );
}

static void work_gemm_nt_q5_1_q8_1(int ith, int nth, void *args)
{
    const gemm_args_t *a = (const gemm_args_t *)args;
    int dr = (a->M + nth - 1) / nth;
    int r0 = dr * ith;
    int r1 = (r0 + dr < a->M) ? (r0 + dr) : a->M;
    if (r0 >= a->M) return;

    gemm_nt_q5_1_q8_1(
        (const float *)((const char *)a->A + (size_t)r0 * a->A_row_bytes),
        a->B,
        a->bias,
        a->C + (size_t)r0 * a->N,
        r1 - r0, a->N, a->K
    );
}

static void work_gemm_nt_q5_k(int ith, int nth, void *args)
{
    const gemm_args_t *a = (const gemm_args_t *)args;
    int dr = (a->M + nth - 1) / nth;
    int r0 = dr * ith;
    int r1 = (r0 + dr < a->M) ? (r0 + dr) : a->M;
    if (r0 >= a->M) return;

    gemm_nt_q5_k(
        (const float *)((const char *)a->A + (size_t)r0 * a->A_row_bytes),
        a->B,
        a->bias,
        a->C + (size_t)r0 * a->N,
        r1 - r0, a->N, a->K
    );
}

/* ============================================================================
 * Parallel Dispatch Wrappers
 *
 * Same signature as serial GEMM functions. Pack args, dispatch to pool.
 * Fast path: M <= 1 or single thread -> call serial directly.
 * ============================================================================ */

void gemm_nt_q5_0_q8_0_parallel_dispatch(
    const void *A, const void *B, const float *bias, float *C,
    int M, int N, int K)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    if (!pool || ck_threadpool_n_threads(pool) <= 1 || M <= 1 || ck_should_run_gemm_serial(pool, M, N, K)) {
        gemm_nt_q5_0_q8_0(A, B, bias, C, M, N, K);
        return;
    }

    /* A is Q8_0: row_bytes = (K / QK8_0) * sizeof(block_q8_0) */
    size_t A_row_bytes = (size_t)(K / QK8_0) * sizeof(block_q8_0);

    gemm_args_t args = {
        .A = A, .B = B, .bias = bias, .C = C,
        .M = M, .N = N, .K = K,
        .A_row_bytes = A_row_bytes
    };
    ck_threadpool_dispatch_n(pool, ck_select_gemm_active_threads(pool, M, N, K), work_gemm_nt_q5_0_q8_0, &args);
}

void gemm_nt_q8_0_q8_0_parallel_dispatch(
    const void *A, const void *B, const float *bias, float *C,
    int M, int N, int K)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    if (!pool || ck_threadpool_n_threads(pool) <= 1 || M <= 1 || ck_should_run_gemm_serial(pool, M, N, K)) {
        gemm_nt_q8_0_q8_0(A, B, bias, C, M, N, K);
        return;
    }

    /* A is Q8_0: row_bytes = (K / QK8_0) * sizeof(block_q8_0) */
    size_t A_row_bytes = (size_t)(K / QK8_0) * sizeof(block_q8_0);

    gemm_args_t args = {
        .A = A, .B = B, .bias = bias, .C = C,
        .M = M, .N = N, .K = K,
        .A_row_bytes = A_row_bytes
    };
    ck_threadpool_dispatch_n(pool, ck_select_gemm_active_threads(pool, M, N, K), work_gemm_nt_q8_0_q8_0, &args);
}

void gemm_nt_q6_k_q8_k_parallel_dispatch(
    const void *A, const void *B, const float *bias, float *C,
    int M, int N, int K)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    if (!pool || ck_threadpool_n_threads(pool) <= 1 || M <= 1 || ck_should_run_gemm_serial(pool, M, N, K)) {
        gemm_nt_q6_k_q8_k(A, B, bias, C, M, N, K);
        return;
    }

    /* A is Q8_K: row_bytes = (K / QK_K) * sizeof(block_q8_K) */
    size_t A_row_bytes = (size_t)(K / QK_K) * sizeof(block_q8_K);

    gemm_args_t args = {
        .A = A, .B = B, .bias = bias, .C = C,
        .M = M, .N = N, .K = K,
        .A_row_bytes = A_row_bytes
    };
    ck_threadpool_dispatch_n(pool, ck_select_gemm_active_threads(pool, M, N, K), work_gemm_nt_q6_k_q8_k, &args);
}

void gemm_nt_q5_1_q8_1_parallel_dispatch(
    const float *A, const void *B, const float *bias, float *C,
    int M, int N, int K)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    if (!pool || ck_threadpool_n_threads(pool) <= 1 || M <= 1 || ck_should_run_gemm_serial(pool, M, N, K)) {
        gemm_nt_q5_1_q8_1(A, B, bias, C, M, N, K);
        return;
    }

    /* A is FP32 token-major [M, K] */
    size_t A_row_bytes = (size_t)K * sizeof(float);

    gemm_args_t args = {
        .A = A, .B = B, .bias = bias, .C = C,
        .M = M, .N = N, .K = K,
        .A_row_bytes = A_row_bytes
    };
    ck_threadpool_dispatch_n(pool, ck_select_gemm_active_threads(pool, M, N, K), work_gemm_nt_q5_1_q8_1, &args);
}

void gemm_nt_q5_k_parallel_dispatch(
    const float *A, const void *B, const float *bias, float *C,
    int M, int N, int K)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    if (!pool || ck_threadpool_n_threads(pool) <= 1 || M <= 1 || ck_should_run_gemm_serial(pool, M, N, K)) {
        gemm_nt_q5_k(A, B, bias, C, M, N, K);
        return;
    }

    /* A is FP32 token-major [M, K] */
    size_t A_row_bytes = (size_t)K * sizeof(float);

    gemm_args_t args = {
        .A = A, .B = B, .bias = bias, .C = C,
        .M = M, .N = N, .K = K,
        .A_row_bytes = A_row_bytes
    };
    ck_threadpool_dispatch_n(pool, ck_select_gemm_active_threads(pool, M, N, K), work_gemm_nt_q5_k, &args);
}
