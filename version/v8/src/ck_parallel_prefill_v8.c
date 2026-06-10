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
 * Q6_K x Q8_K prefill scheduling note:
 * ------------------------------------
 * The optional 2D scheduler is a load-balancing tool, not a universally faster
 * Q6 kernel. Row splitting reuses each Q8_K activation row across the full N
 * output dimension. Splitting N into tiles creates more independent jobs, but
 * rereads the same activation tile once per N tile and adds scheduler work.
 *
 * Local i7 roofline-style sweeps on 2026-06-09 showed:
 *   - Qwen2-like MLP-down (N=896, K=4864): 2D was slower through M=256 and
 *     only +1.4% at M=512.
 *   - Nanbeige/large-Q6-like MLP-down (N=2560, K=10240): 2D was faster from
 *     M=16 onward, +5% to +23% depending on M.
 *
 * Therefore CK_ENABLE_Q6K_Q8K_2D_PREFILL is still opt-in, and the dispatcher
 * additionally gates it to wide Q6 shapes by default. Use
 * CK_FORCE_Q6K_Q8K_2D_PREFILL=1 only for benchmarking raw 2D behavior.
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
extern void gemm_nt_q6_k_q8_k_tile(const void *A, const void *B, const float *bias,
                                    float *C, int M, int N, int K,
                                    int m0, int m1, int n0, int n1);
extern void gemm_nt_q6_k_q8_k_tiled(const void *A, const void *B, const float *bias,
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
    int          tile_m;      /* 2D scheduler token tile height */
    int          tile_n;      /* 2D scheduler output tile width */
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

static int ck_ceil_div_int(int a, int b)
{
    return (a + b - 1) / b;
}

static int ck_select_gemm_active_threads(const ck_threadpool_t *pool, int M, int N, int K);

static int ck_q6k_q8k_2d_prefill_forced(void)
{
    return ck_env_enabled("CK_FORCE_Q6K_Q8K_2D_PREFILL");
}

static int ck_should_use_q6k_q8k_2d_prefill(const ck_threadpool_t *pool,
                                             int M, int N, int K,
                                             int tile_m, int tile_n)
{
    if (!ck_env_enabled("CK_ENABLE_Q6K_Q8K_2D_PREFILL")) return 0;
    if (ck_q6k_q8k_2d_prefill_forced()) return 1;
    if (!pool || ck_threadpool_n_threads(pool) <= 1) return 0;
    if (M <= 1 || N <= 0 || K <= 0) return 0;
    if (K % QK_K != 0) return 0;

    const int tm = tile_m > 0 ? tile_m : 16;
    const int tn = tile_n > 0 ? tile_n : 256;
    const int mt = ck_ceil_div_int(M, tm);
    const int nt = ck_ceil_div_int(N, tn);
    const int jobs = mt * nt;
    const int active = ck_select_gemm_active_threads(pool, M, N, K);

    if (jobs < active * 2) return 0;

    /* 2D tiling pays off only when the output dimension is wide enough that
     * N-side job balance offsets the extra activation rereads. Narrow MLP-down
     * shapes such as Qwen2/Gemma/Qwen3.5 remain faster with row splitting. */
    if (N < 2048 || K < 8192) return 0;

    return 1;
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

    if (getenv("CK_GEMM_THREAD_CAP") || getenv("CK_GEMV_THREAD_CAP")) {
        return ck_min_int(pool_threads,
                          ck_env_int_or2("CK_GEMM_THREAD_CAP", "CK_GEMV_THREAD_CAP", pool_threads));
    }

    if (N >= 4096 || K >= 4096) return pool_threads;
    if (M >= 512) return ck_min_int(pool_threads, 24);
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

    if (ck_env_enabled("CK_ENABLE_Q6K_Q8K_TILED_PREFILL")) {
        gemm_nt_q6_k_q8_k_tiled(
            (const char *)a->A + (size_t)r0 * a->A_row_bytes,
            a->B,
            a->bias,
            a->C + (size_t)r0 * a->N,
            r1 - r0, a->N, a->K
        );
    } else {
        gemm_nt_q6_k_q8_k(
            (const char *)a->A + (size_t)r0 * a->A_row_bytes,
            a->B,
            a->bias,
            a->C + (size_t)r0 * a->N,
            r1 - r0, a->N, a->K
        );
    }
}

static void work_gemm_nt_q6_k_q8_k_2d(int ith, int nth, void *args)
{
    const gemm_args_t *a = (const gemm_args_t *)args;
    if (!a || ith < 0 || nth <= 0 || ith >= nth) return;

    const int tile_m = a->tile_m > 0 ? a->tile_m : 16;
    const int tile_n = a->tile_n > 0 ? a->tile_n : 256;
    const int mt = ck_ceil_div_int(a->M, tile_m);
    const int nt = ck_ceil_div_int(a->N, tile_n);
    const int total = mt * nt;

    for (int job = ith; job < total; job += nth) {
        const int jm = job % mt;
        const int jn = job / mt;
        const int m0 = jm * tile_m;
        const int m1 = ck_min_int(m0 + tile_m, a->M);
        const int n0 = jn * tile_n;
        const int n1 = ck_min_int(n0 + tile_n, a->N);
        gemm_nt_q6_k_q8_k_tile(a->A, a->B, a->bias, a->C,
                               a->M, a->N, a->K, m0, m1, n0, n1);
    }
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
        .A_row_bytes = A_row_bytes,
        .tile_m = ck_env_int_or2("CK_PREFILL_TILE_M", NULL, 16),
        .tile_n = ck_env_int_or2("CK_PREFILL_TILE_N", NULL, 256)
    };
    const int active = ck_select_gemm_active_threads(pool, M, N, K);
    if (ck_should_use_q6k_q8k_2d_prefill(pool, M, N, K, args.tile_m, args.tile_n)) {
        ck_threadpool_dispatch_n(pool, active, work_gemm_nt_q6_k_q8_k_2d, &args);
    } else {
        ck_threadpool_dispatch_n(pool, active, work_gemm_nt_q6_k_q8_k, &args);
    }
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
