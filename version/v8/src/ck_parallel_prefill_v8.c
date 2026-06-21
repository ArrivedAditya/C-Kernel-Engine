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
#include <pthread.h>

/* Serial GEMM kernels (defined in src/kernels/) */
extern void gemm_nt_q5_0_q8_0(const void *A, const void *B, const float *bias,
                                float *C, int M, int N, int K);
extern void gemm_nt_q8_0_q8_0(const void *A, const void *B, const float *bias,
                                float *C, int M, int N, int K);
extern void gemm_nt_q4_k_q8_k(const void *A, const void *B, const float *bias,
                                float *C, int M, int N, int K);
extern void gemv_q4_k_q8_k(float *y, const void *W, const void *x_q8, int M, int K);
extern size_t q4_k_packed_meta_block_size(void);
extern void pack_q4_k_to_packed_meta(const void *src, void *dst, int N, int K);
extern void gemm_nt_q4_k_packed_meta_q8_k_threaded(const void *A_q8,
                                                    const void *B_packed,
                                                    const float *bias,
                                                    float *C,
                                                    int M, int N, int K,
                                                    int threads);
extern void gemm_nt_q4_k_packed_meta_q8_k_threaded_nsplit(const void *A_q8,
                                                          const void *B_packed,
                                                          const float *bias,
                                                          float *C,
                                                          int M, int N, int K,
                                                          int threads);
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

typedef struct ck_q4k_packed_meta_cache_entry {
    const void *src;
    int N;
    int K;
    void *packed;
    struct ck_q4k_packed_meta_cache_entry *next;
} ck_q4k_packed_meta_cache_entry_t;

static pthread_mutex_t ck_q4k_packed_meta_cache_mu = PTHREAD_MUTEX_INITIALIZER;
static ck_q4k_packed_meta_cache_entry_t *ck_q4k_packed_meta_cache_head = NULL;

/* Q4_K packed-meta prefill experiment
 * -----------------------------------
 * This is intentionally kept in the v8 prefill dispatcher for now instead of
 * being promoted to a default src/kernels entry point. The pure kernel pieces
 * live in gemm_kernels_q4k_q8k_vnni.c, but this code owns runtime policy:
 * shape gating, thread selection, and temporary packed-weight lifetime.
 *
 * What we tested locally on the i7 laptop (2026-06-11):
 *   - Standalone Qwen3.5-like Q4_K x Q8_K shapes (N=3584,K=1024) improved
 *     about 1.2x-1.4x across M=128..1024, depending on thread count.
 *   - Nanbeige-like large Q4_K shapes (N=10496,K=2560) usually improved, but
 *     N-split ownership regressed at some early 12-thread M=512 experiments,
 *     but the later dispatch matrix showed it was the best measured candidate
 *     for the Qwen3.5-like shapes used by the v8 prompt path.
 *   - Full-model Qwen3.5 prefill improved at prompt 128 and 256 tokens
 *     (roughly 1.10x-1.13x), while prompt 512 was only a small full-model win
 *     because other operators became the bottleneck.
 *   - Kernel-level parity stayed tight in the standalone benchmark
 *     (max abs around 6e-05 to 1.2e-04), and v8/threadpool quick gates passed.
 *
 * What is still missing before promotion:
 *   - Packed weights should be produced at model-load/conversion time or stored
 *     in the model runtime layout. This lazy cache is acceptable for profiling,
 *     but it is not the final ownership model and currently leaks until process
 *     exit.
 *   - The dispatch rule must be hardware-swept on Xeon/AVX-512 and lower-core
 *     AVX2 machines. The current i7 data supports Qwen3.5-like M-split, not a
 *     universal Q4_K policy.
 *   - N-split should remain a profiling option until it is consistently faster
 *     on a target platform. On this laptop it was mixed.
 *   - Model-level profiling must show that the full prompt path benefits after
 *     other bottlenecks such as Q5_K recurrent projection, SSM conv, attention,
 *     and Q6/Q4 down projection are accounted for.
 *
 * Promotion criteria:
 *   1. Add a real packed-weight layout to the v8 model/load path, with explicit
 *      memory accounting and cleanup.
 *   2. Keep canonical GGUF-layout Q4_K kernels as the parity fallback.
 *   3. Select packed-meta only through shape/hardware dispatch after benchmark
 *      sweeps pass for the target CPU class.
 *   4. Add nightly sweep coverage so CK can learn/verify which Q4_K prefill
 *      policy is best per hardware/model shape.
 */
static void *ck_get_q4k_packed_meta_cached(const void *B, int N, int K)
{
    if (!B || N <= 0 || K <= 0 || (K % QK_K) != 0) return NULL;

    pthread_mutex_lock(&ck_q4k_packed_meta_cache_mu);
    for (ck_q4k_packed_meta_cache_entry_t *e = ck_q4k_packed_meta_cache_head; e; e = e->next) {
        if (e->src == B && e->N == N && e->K == K) {
            void *packed = e->packed;
            pthread_mutex_unlock(&ck_q4k_packed_meta_cache_mu);
            return packed;
        }
    }

    const size_t blocks = (size_t)N * (size_t)(K / QK_K);
    const size_t bytes = blocks * q4_k_packed_meta_block_size();
    void *packed = malloc(bytes);
    ck_q4k_packed_meta_cache_entry_t *entry =
        (ck_q4k_packed_meta_cache_entry_t *)malloc(sizeof(*entry));
    if (!packed || !entry) {
        free(packed);
        free(entry);
        pthread_mutex_unlock(&ck_q4k_packed_meta_cache_mu);
        return NULL;
    }

    pack_q4_k_to_packed_meta(B, packed, N, K);
    entry->src = B;
    entry->N = N;
    entry->K = K;
    entry->packed = packed;
    entry->next = ck_q4k_packed_meta_cache_head;
    ck_q4k_packed_meta_cache_head = entry;
    pthread_mutex_unlock(&ck_q4k_packed_meta_cache_mu);
    return packed;
}

static int ck_should_use_q4k_packed_meta_prefill(int M, int N, int K)
{
    if (ck_env_enabled("CK_DISABLE_Q4K_PACKED_META_PREFILL")) return 0;
    if (M <= 1 || N <= 0 || K <= 0 || (K % QK_K) != 0) return 0;

    const int min_m = ck_env_int_or2("CK_Q4K_PACKED_META_MIN_M", NULL, 32);
    if (M < min_m) return 0;

    if (getenv("CK_FORCE_Q4K_PACKED_META_PREFILL")) return 1;

    /* The dispatch matrix benchmark tracks canonical serial, canonical pool,
     * packed M-split, packed N-split, and a llama.cpp shim. Local AVX2 data
     * shows packed N-split is the best measured Q4_K prefill candidate for
     * Qwen-like shapes:
     *   - M=32,N=1024,K=1024:  ~1.36x over canonical
     *   - M=32,N=896,K=4864:   ~1.25x over canonical
     *   - M=128,N=896,K=4864:  ~1.33x over canonical
     *
     * Keep decode on GEMV. This gate is prefill-only because M > 1 and is
     * still shape-gated; use CK_DISABLE_Q4K_PACKED_META_PREFILL=1 to force
     * canonical Q4_K while validating a new CPU. */
    if (N >= 768 && K >= 1024) return 1;
    if (K <= 2048 && N >= 1024 && N <= 8192) return 1;
    return 0;
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

    if (N >= 4096 || K >= 4096) return ck_min_int(pool_threads, ck_env_int_or2("CK_GEMM_THREAD_CAP", "CK_GEMV_THREAD_CAP", 24));
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

static void work_gemm_nt_q4_k_q8_k(int ith, int nth, void *args)
{
    const gemm_args_t *a = (const gemm_args_t *)args;
    int dr = (a->M + nth - 1) / nth;
    int r0 = dr * ith;
    int r1 = (r0 + dr < a->M) ? (r0 + dr) : a->M;
    if (r0 >= a->M) return;

    /* Do not call gemm_nt_q4_k_q8_k() here: that raw implementation can
     * start its own internal output-row threadpool for large Q4_K shapes.
     * This worker already runs inside the v8 prefill pool, so nesting the
     * same pool can corrupt scheduling and parity. Use the one-token GEMV
     * primitive directly for each assigned token row. */
    for (int m = r0; m < r1; ++m) {
        const void *x_row = (const char *)a->A + (size_t)m * a->A_row_bytes;
        float *c_row = a->C + (size_t)m * (size_t)a->N;
        gemv_q4_k_q8_k(c_row, a->B, x_row, a->N, a->K);
        if (a->bias) {
            for (int n = 0; n < a->N; ++n) {
                c_row[n] += a->bias[n];
            }
        }
    }
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

void gemm_nt_q4_k_q8_k_parallel_dispatch(
    const void *A, const void *B, const float *bias, float *C,
    int M, int N, int K)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    if (pool && ck_should_use_q4k_packed_meta_prefill(M, N, K)) {
        void *packed = ck_get_q4k_packed_meta_cached(B, N, K);
        if (packed) {
            gemm_nt_q4_k_packed_meta_q8_k_threaded_nsplit(
                A, packed, bias, C, M, N, K,
                ck_select_gemm_active_threads(pool, M, N, K)
            );
            return;
        }
    }

    /* Q4_K prefill row-splitting is currently a benchmark path, not a default
     * production path. On local i7 AVX2 testing it was parity-clean but slower
     * for the Qwen/Qwen3.5 shapes measured by test_threadpool_parity and the
     * v8 decoder matrix. Keep it opt-in until hardware sweeps show a stable win.
     */
    const char *enable_q4_pool = getenv("CK_ENABLE_Q4K_Q8K_PREFILL_POOL");
    if (!enable_q4_pool || enable_q4_pool[0] == '\0' || enable_q4_pool[0] == '0') {
        gemm_nt_q4_k_q8_k(A, B, bias, C, M, N, K);
        return;
    }

    if (!pool || ck_threadpool_n_threads(pool) <= 1 || M <= 1 || ck_should_run_gemm_serial(pool, M, N, K)) {
        gemm_nt_q4_k_q8_k(A, B, bias, C, M, N, K);
        return;
    }

    /* A is Q8_K: row_bytes = (K / QK_K) * sizeof(block_q8_K) */
    size_t A_row_bytes = (size_t)(K / QK_K) * sizeof(block_q8_K);

    gemm_args_t args = {
        .A = A, .B = B, .bias = bias, .C = C,
        .M = M, .N = N, .K = K,
        .A_row_bytes = A_row_bytes
    };
    ck_threadpool_dispatch_n(pool, ck_select_gemm_active_threads(pool, M, N, K), work_gemm_nt_q4_k_q8_k, &args);
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
