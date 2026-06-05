/**
 * @file gemm_kernels_bf16.c
 * @brief Optimized BF16 GEMM Kernels for AVX-512
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
 * Layout:
 *   A: [M x K] row-major (BF16)
 *   B: [N x K] row-major, stored as [out x in] (BF16)
 *   C: [M x N] row-major (BF16 or FP32)
 *
 * Key optimizations:
 *   1. AVX-512 BF16 instructions (VDPBF16PS) when available
 *   2. Cache blocking for L1/L2 efficiency
 *   3. Vectorized BF16<->FP32 conversion
 *   4. OpenMP parallelization
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#if defined(__linux__) && defined(__AMX_TILE__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include "bf16_utils.h"
#include "ckernel_engine.h"

/* Block sizes tuned for typical L1/L2 cache */
#define BLK_M 64
#define BLK_N 64
#define BLK_K 256

static inline int ck_min_i(int a, int b) { return a < b ? a : b; }

/* ==========================================================================
 * Reference Implementation (scalar, for correctness testing)
 * Kept for debugging/validation but not called in normal operation.
 * ========================================================================== */
__attribute__((unused))
static void gemm_bf16_scalar(const uint16_t *A,
                             const uint16_t *B,
                             const uint16_t *bias,
                             uint16_t *C,
                             int M, int N, int K)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = bias ? bf16_to_float(bias[j]) : 0.0f;
            const size_t a_row = (size_t)i * (size_t)K;
            const size_t b_row = (size_t)j * (size_t)K;
            for (int k = 0; k < K; ++k) {
                sum += bf16_to_float(A[a_row + k]) * bf16_to_float(B[b_row + k]);
            }
            C[(size_t)i * (size_t)N + j] = float_to_bf16(sum);
        }
    }
}
#if defined(__AVX512F__)

/* ==========================================================================
 * AVX-512F: Vectorized BF16 conversion + FMA
 * Works on all AVX-512 CPUs (no BF16 instruction required)
 *
 * BF16 conversion functions (bf16x16_to_fp32, fp32x16_to_bf16) are now
 * provided by bf16_utils.h for consistency across all kernels.
 * ========================================================================== */

/* BF16 dot product: 16 pairs, accumulate to FP32 */
static inline __m512 bf16_dot16(__m256i a_bf16, __m256i b_bf16, __m512 acc)
{
    __m512 a_fp32 = bf16x16_to_fp32(a_bf16);
    __m512 b_fp32 = bf16x16_to_fp32(b_bf16);
    return _mm512_fmadd_ps(a_fp32, b_fp32, acc);
}

/* ==========================================================================
 * AVX-512 Vectorized GEMM (using AVX-512F, works everywhere)
 * C[M,N] = A[M,K] @ B[N,K].T
 * ========================================================================== */
static void gemm_bf16_avx512(const uint16_t *A,
                             const uint16_t *B,
                             const uint16_t *bias,
                             uint16_t *C,
                             int M, int N, int K)
{
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < M; ++i) {
        const uint16_t *a_row = A + (size_t)i * K;

        for (int j = 0; j < N; ++j) {
            const uint16_t *b_row = B + (size_t)j * K;

            /* Initialize accumulator */
            __m512 sum_vec = _mm512_setzero_ps();

            /* Vectorized inner loop: process 16 elements at a time */
            int k = 0;
            for (; k <= K - 16; k += 16) {
                __m256i a_bf16 = _mm256_loadu_si256((const __m256i *)(a_row + k));
                __m256i b_bf16 = _mm256_loadu_si256((const __m256i *)(b_row + k));
                sum_vec = bf16_dot16(a_bf16, b_bf16, sum_vec);
            }

            /* Horizontal sum */
            float sum = _mm512_reduce_add_ps(sum_vec);

            /* Scalar tail */
            for (; k < K; ++k) {
                sum += bf16_to_float(a_row[k]) * bf16_to_float(b_row[k]);
            }

            /* Add bias */
            if (bias) {
                sum += bf16_to_float(bias[j]);
            }

            C[(size_t)i * N + j] = float_to_bf16(sum);
        }
    }
}

/* ==========================================================================
 * Cache-Blocked AVX-512 GEMM
 * Better memory access pattern for large matrices
 * ========================================================================== */
static void gemm_bf16_blocked_avx512(const uint16_t *A,
                                      const uint16_t *B,
                                      const uint16_t *bias,
                                      uint16_t *C,
                                      int M, int N, int K)
{
    /* Initialize C with bias */
    #pragma omp parallel for
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float b = bias ? bf16_to_float(bias[j]) : 0.0f;
            C[(size_t)i * N + j] = float_to_bf16(b);
        }
    }

    /* Blocked GEMM */
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int ii = 0; ii < M; ii += BLK_M) {
        for (int jj = 0; jj < N; jj += BLK_N) {
            int i_end = ck_min_i(ii + BLK_M, M);
            int j_end = ck_min_i(jj + BLK_N, N);

            /* Local FP32 accumulator for this block */
            float acc[BLK_M][BLK_N];
            for (int i = 0; i < BLK_M; ++i) {
                for (int j = 0; j < BLK_N; ++j) {
                    acc[i][j] = 0.0f;
                }
            }

            /* K-dimension blocking */
            for (int kk = 0; kk < K; kk += BLK_K) {
                int k_end = ck_min_i(kk + BLK_K, K);

                for (int i = ii; i < i_end; ++i) {
                    const uint16_t *a_row = A + (size_t)i * K;
                    int local_i = i - ii;

                    for (int j = jj; j < j_end; ++j) {
                        const uint16_t *b_row = B + (size_t)j * K;
                        int local_j = j - jj;

                        __m512 sum_vec = _mm512_setzero_ps();

                        int k = kk;
                        for (; k <= k_end - 16; k += 16) {
                            __m256i a_bf16 = _mm256_loadu_si256((const __m256i *)(a_row + k));
                            __m256i b_bf16 = _mm256_loadu_si256((const __m256i *)(b_row + k));
                            sum_vec = bf16_dot16(a_bf16, b_bf16, sum_vec);
                        }

                        float partial = _mm512_reduce_add_ps(sum_vec);
                        for (; k < k_end; ++k) {
                            partial += bf16_to_float(a_row[k]) * bf16_to_float(b_row[k]);
                        }

                        acc[local_i][local_j] += partial;
                    }
                }
            }

            /* Write accumulated results back */
            for (int i = ii; i < i_end; ++i) {
                for (int j = jj; j < j_end; ++j) {
                    float old_val = bf16_to_float(C[(size_t)i * N + j]);
                    float new_val = old_val + acc[i - ii][j - jj];
                    C[(size_t)i * N + j] = float_to_bf16(new_val);
                }
            }
        }
    }
}

/*
 * Native AVX-512 BF16 support (VDPBF16PS instruction)
 * Only compiles on Ice Lake / Sapphire Rapids or newer
 * Compile with: -mavx512bf16 (gcc/clang) or /arch:AVX512 (MSVC with recent SDK)
 */
#if defined(__AVX512BF16__) && defined(__AVX512VL__)

/* Load 32 BF16 values into __m512bh */
static inline __m512bh load_bf16x32(const uint16_t *ptr)
{
    return (__m512bh)_mm512_loadu_si512((const __m512i *)ptr);
}

#if defined(__AMX_TILE__) && defined(__AMX_BF16__)

#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#ifndef XFEATURE_XTILE_DATA
#define XFEATURE_XTILE_DATA 18
#endif

typedef struct ck_amx_tile_config {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved_0[14];
    uint16_t colsb[16];
    uint8_t rows[16];
} ck_amx_tile_config;

static int ck_amx_request_xtile_data(void)
{
#if defined(__linux__)
    static int state = 0;
    if (state == 1) {
        return 1;
    }
    if (state == -1) {
        return 0;
    }
    long rc = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILE_DATA);
    state = (rc == 0) ? 1 : -1;
    return state == 1;
#else
    return 0;
#endif
}

static void ck_amx_config_bf16_16x16x32(void)
{
    ck_amx_tile_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.palette_id = 1;

    cfg.rows[0] = 16;       /* A: 16 rows x 32 BF16 */
    cfg.colsb[0] = 64;
    cfg.rows[1] = 16;       /* B: 16 K-pair rows x 16 BF16-pair columns */
    cfg.colsb[1] = 64;
    cfg.rows[2] = 16;       /* C: 16 rows x 16 FP32 */
    cfg.colsb[2] = 64;

    _tile_loadconfig(&cfg);
}

static void ck_pack_bf16_ktile_pairs_16x16(uint16_t *dst,
                                             const uint16_t *B,
                                             int K,
                                             int j,
                                             int k)
{
    for (int kp = 0; kp < 16; ++kp) {
        const int k0 = k + kp * 2;
        for (int nn = 0; nn < 16; ++nn) {
            dst[(size_t)kp * 32u + (size_t)nn * 2u + 0u] =
                B[(size_t)(j + nn) * (size_t)K + (size_t)k0];
            dst[(size_t)kp * 32u + (size_t)nn * 2u + 1u] =
                B[(size_t)(j + nn) * (size_t)K + (size_t)(k0 + 1)];
        }
    }
}

static void gemm_bf16_fp32out_amx(const uint16_t *A,
                                  const uint16_t *B,
                                  const float *bias,
                                  float *C,
                                  int M, int N, int K)
{
    ck_amx_config_bf16_16x16x32();

    uint16_t b_tile[16 * 32];

    for (int i = 0; i < M; i += 16) {
        for (int j = 0; j < N; j += 16) {
            _tile_zero(2);

            for (int k = 0; k < K; k += 32) {
                ck_pack_bf16_ktile_pairs_16x16(b_tile, B, K, j, k);
                _tile_loadd(0, A + (size_t)i * (size_t)K + (size_t)k, K * (int)sizeof(uint16_t));
                _tile_loadd(1, b_tile, 32 * (int)sizeof(uint16_t));
                _tile_dpbf16ps(2, 0, 1);
            }

            _tile_stored(2, C + (size_t)i * (size_t)N + (size_t)j, N * (int)sizeof(float));

            if (bias) {
                for (int ii = 0; ii < 16; ++ii) {
                    float *c_row = C + (size_t)(i + ii) * (size_t)N + (size_t)j;
                    for (int jj = 0; jj < 16; ++jj) {
                        c_row[jj] += bias[j + jj];
                    }
                }
            }
        }
    }

    _tile_release();
}

#define HAVE_AMX_BF16 1
#else
#define HAVE_AMX_BF16 0
#endif /* __AMX_TILE__ && __AMX_BF16__ */

static void gemm_bf16_native(const uint16_t *A,
                              const uint16_t *B,
                              const uint16_t *bias,
                              uint16_t *C,
                              int M, int N, int K)
{
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            /* Initialize accumulator */
            __m512 sum_vec = _mm512_setzero_ps();

            /* Native BF16 dot product: 32 pairs per instruction! */
            int k = 0;
            for (; k <= K - 32; k += 32) {
                __m512bh a_vec = load_bf16x32(A + (size_t)i * K + k);
                __m512bh b_vec = load_bf16x32(B + (size_t)j * K + k);
                sum_vec = _mm512_dpbf16_ps(sum_vec, a_vec, b_vec);
            }

            float sum = _mm512_reduce_add_ps(sum_vec);

            /* Scalar tail */
            for (; k < K; ++k) {
                sum += bf16_to_float(A[(size_t)i * K + k]) *
                       bf16_to_float(B[(size_t)j * K + k]);
            }

            if (bias) {
                sum += bf16_to_float(bias[j]);
            }

            C[(size_t)i * N + j] = float_to_bf16(sum);
        }
    }
}

#define HAVE_NATIVE_BF16 1
#else
#define HAVE_NATIVE_BF16 0
#endif /* __AVX512BF16__ && __AVX512VL__ */

#endif /* __AVX512F__ */

/* ==========================================================================
 * Public API: Auto-dispatch to best available implementation
 * ========================================================================== */
void gemm_blocked_serial_bf16(const uint16_t *A,
                              const uint16_t *B,
                              const uint16_t *bias,
                              uint16_t *C,
                              int M, int N, int K)
{
    if (!A || !B || !C || M <= 0 || N <= 0 || K <= 0) {
        return;
    }

#if HAVE_NATIVE_BF16
    /* Native BF16 instructions available (Ice Lake / Sapphire Rapids+) */
    gemm_bf16_native(A, B, bias, C, M, N, K);
#elif defined(__AVX512F__)
    /* Use AVX-512F with software BF16 conversion */
    if (M * N > 4096) {
        gemm_bf16_blocked_avx512(A, B, bias, C, M, N, K);
    } else {
        gemm_bf16_avx512(A, B, bias, C, M, N, K);
    }
#else
    /* Scalar fallback */
    gemm_bf16_scalar(A, B, bias, C, M, N, K);
#endif
}

/* ==========================================================================
 * GEMM with FP32 output (useful for intermediate computations)
 * ========================================================================== */
void gemm_bf16_fp32out(const uint16_t *A,
                       const uint16_t *B,
                       const float *bias,
                       float *C,
                       int M, int N, int K)
{
    if (!A || !B || !C || M <= 0 || N <= 0 || K <= 0) {
        return;
    }

#if HAVE_NATIVE_BF16
    const char *amx_env = getenv("CK_BF16_AMX");
    if (amx_env && amx_env[0] == '1' && HAVE_AMX_BF16 &&
        (M % 16) == 0 && (N % 16) == 0 && (K % 32) == 0 &&
        M >= 16 && N >= 16 && K >= 32 && ck_amx_request_xtile_data()) {
        gemm_bf16_fp32out_amx(A, B, bias, C, M, N, K);
        return;
    }

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < M; ++i) {
        const uint16_t *a_row = A + (size_t)i * K;
        int j = 0;

        for (; j + 4 <= N; j += 4) {
            const uint16_t *b0 = B + (size_t)(j + 0) * K;
            const uint16_t *b1 = B + (size_t)(j + 1) * K;
            const uint16_t *b2 = B + (size_t)(j + 2) * K;
            const uint16_t *b3 = B + (size_t)(j + 3) * K;
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            int k = 0;
            for (; k <= K - 32; k += 32) {
                const __m512bh a_vec = load_bf16x32(a_row + k);
                acc0 = _mm512_dpbf16_ps(acc0, a_vec, load_bf16x32(b0 + k));
                acc1 = _mm512_dpbf16_ps(acc1, a_vec, load_bf16x32(b1 + k));
                acc2 = _mm512_dpbf16_ps(acc2, a_vec, load_bf16x32(b2 + k));
                acc3 = _mm512_dpbf16_ps(acc3, a_vec, load_bf16x32(b3 + k));
            }

            float s0 = _mm512_reduce_add_ps(acc0);
            float s1 = _mm512_reduce_add_ps(acc1);
            float s2 = _mm512_reduce_add_ps(acc2);
            float s3 = _mm512_reduce_add_ps(acc3);
            for (; k < K; ++k) {
                const float a = bf16_to_float(a_row[k]);
                s0 += a * bf16_to_float(b0[k]);
                s1 += a * bf16_to_float(b1[k]);
                s2 += a * bf16_to_float(b2[k]);
                s3 += a * bf16_to_float(b3[k]);
            }
            if (bias) {
                s0 += bias[j + 0];
                s1 += bias[j + 1];
                s2 += bias[j + 2];
                s3 += bias[j + 3];
            }
            C[(size_t)i * N + (j + 0)] = s0;
            C[(size_t)i * N + (j + 1)] = s1;
            C[(size_t)i * N + (j + 2)] = s2;
            C[(size_t)i * N + (j + 3)] = s3;
        }

        for (; j < N; ++j) {
            const uint16_t *b_row = B + (size_t)j * K;
            __m512 sum_vec = _mm512_setzero_ps();

            int k = 0;
            for (; k <= K - 32; k += 32) {
                const __m512bh a_vec = load_bf16x32(a_row + k);
                const __m512bh b_vec = load_bf16x32(b_row + k);
                sum_vec = _mm512_dpbf16_ps(sum_vec, a_vec, b_vec);
            }

            float sum = _mm512_reduce_add_ps(sum_vec);
            for (; k < K; ++k) {
                sum += bf16_to_float(a_row[k]) * bf16_to_float(b_row[k]);
            }
            if (bias) {
                sum += bias[j];
            }
            C[(size_t)i * N + j] = sum;
        }
    }
#elif defined(__AVX512F__)
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < M; ++i) {
        const uint16_t *a_row = A + (size_t)i * K;

        for (int j = 0; j < N; ++j) {
            const uint16_t *b_row = B + (size_t)j * K;

            __m512 sum_vec = _mm512_setzero_ps();

            int k = 0;
            for (; k <= K - 16; k += 16) {
                __m256i a_bf16 = _mm256_loadu_si256((const __m256i *)(a_row + k));
                __m256i b_bf16 = _mm256_loadu_si256((const __m256i *)(b_row + k));
                sum_vec = bf16_dot16(a_bf16, b_bf16, sum_vec);
            }

            float sum = _mm512_reduce_add_ps(sum_vec);

            for (; k < K; ++k) {
                sum += bf16_to_float(a_row[k]) * bf16_to_float(b_row[k]);
            }

            if (bias) {
                sum += bias[j];
            }

            C[(size_t)i * N + j] = sum;
        }
    }
#else
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = bias ? bias[j] : 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += bf16_to_float(A[(size_t)i * K + k]) *
                       bf16_to_float(B[(size_t)j * K + k]);
            }
            C[(size_t)i * N + j] = sum;
        }
    }
#endif
}

/* ==========================================================================
 * Backward kernels for training
 * ========================================================================== */

/* gemm_nn_bf16: C = A @ B (no transpose), for dL/dX computation */
void gemm_nn_bf16(const uint16_t *A,
                  const uint16_t *B,
                  const uint16_t *bias,
                  uint16_t *C,
                  int M, int N, int K)
{
    if (!A || !B || !C || M <= 0 || N <= 0 || K <= 0) {
        return;
    }

#if defined(__AVX512F__)
    #pragma omp parallel for
    for (int i = 0; i < M; ++i) {
        /* Initialize row with bias */
        int j = 0;
        for (; j <= N - 16; j += 16) {
            __m512 b_vec = bias ? bf16x16_to_fp32(_mm256_loadu_si256((const __m256i *)(bias + j)))
                                : _mm512_setzero_ps();
            __m256i out = fp32x16_to_bf16(b_vec);
            _mm256_storeu_si256((__m256i *)(C + (size_t)i * N + j), out);
        }
        for (; j < N; ++j) {
            float b = bias ? bf16_to_float(bias[j]) : 0.0f;
            C[(size_t)i * N + j] = float_to_bf16(b);
        }

        /* Accumulate: C[i,:] += A[i,k] * B[k,:] */
        for (int k = 0; k < K; ++k) {
            float a_val = bf16_to_float(A[(size_t)i * K + k]);
            __m512 a_broadcast = _mm512_set1_ps(a_val);

            j = 0;
            for (; j <= N - 16; j += 16) {
                __m256i b_bf16 = _mm256_loadu_si256((const __m256i *)(B + (size_t)k * N + j));
                __m512 b_fp32 = bf16x16_to_fp32(b_bf16);

                __m256i c_bf16 = _mm256_loadu_si256((const __m256i *)(C + (size_t)i * N + j));
                __m512 c_fp32 = bf16x16_to_fp32(c_bf16);

                c_fp32 = _mm512_fmadd_ps(a_broadcast, b_fp32, c_fp32);

                __m256i c_out = fp32x16_to_bf16(c_fp32);
                _mm256_storeu_si256((__m256i *)(C + (size_t)i * N + j), c_out);
            }
            for (; j < N; ++j) {
                float c_val = bf16_to_float(C[(size_t)i * N + j]);
                c_val += a_val * bf16_to_float(B[(size_t)k * N + j]);
                C[(size_t)i * N + j] = float_to_bf16(c_val);
            }
        }
    }
#else
    /* Scalar fallback */
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = bias ? bf16_to_float(bias[j]) : 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += bf16_to_float(A[(size_t)i * K + k]) *
                       bf16_to_float(B[(size_t)k * N + j]);
            }
            C[(size_t)i * N + j] = float_to_bf16(sum);
        }
    }
#endif
}

/* gemm_tn_bf16: C = A.T @ B, for dL/dW computation */
void gemm_tn_bf16(const uint16_t *A,
                  const uint16_t *B,
                  const uint16_t *bias,
                  uint16_t *C,
                  int M, int N, int K)
{
    if (!A || !B || !C || M <= 0 || N <= 0 || K <= 0) {
        return;
    }

    /* A is [K x M], we want A.T which is [M x K] */
    /* B is [K x N] */
    /* C is [M x N] */

#if defined(__AVX512F__)
    /* Initialize C with bias */
    #pragma omp parallel for
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float b = bias ? bf16_to_float(bias[j]) : 0.0f;
            C[(size_t)i * N + j] = float_to_bf16(b);
        }
    }

    /* Accumulate: C[i,j] += sum_k A[k,i] * B[k,j] */
    #pragma omp parallel for
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            __m512 sum_vec = _mm512_setzero_ps();

            int k = 0;
            for (; k <= K - 16; k += 16) {
                /* Gather A[k:k+16, i] - strided access */
                __m512 a_fp32 = _mm512_setzero_ps();
                for (int kk = 0; kk < 16; ++kk) {
                    float val = bf16_to_float(A[(size_t)(k + kk) * M + i]);
                    a_fp32 = _mm512_mask_mov_ps(a_fp32, 1 << kk, _mm512_set1_ps(val));
                }

                /* Note: B has stride N, so we need to gather element by element */
                __m512 b_fp32 = _mm512_setzero_ps();
                for (int kk = 0; kk < 16; ++kk) {
                    float val = bf16_to_float(B[(size_t)(k + kk) * N + j]);
                    b_fp32 = _mm512_mask_mov_ps(b_fp32, 1 << kk, _mm512_set1_ps(val));
                }

                sum_vec = _mm512_fmadd_ps(a_fp32, b_fp32, sum_vec);
            }

            float sum = _mm512_reduce_add_ps(sum_vec);

            for (; k < K; ++k) {
                sum += bf16_to_float(A[(size_t)k * M + i]) *
                       bf16_to_float(B[(size_t)k * N + j]);
            }

            float old_val = bf16_to_float(C[(size_t)i * N + j]);
            C[(size_t)i * N + j] = float_to_bf16(old_val + sum);
        }
    }
#else
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = bias ? bf16_to_float(bias[j]) : 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += bf16_to_float(A[(size_t)k * M + i]) *
                       bf16_to_float(B[(size_t)k * N + j]);
            }
            C[(size_t)i * N + j] = float_to_bf16(sum);
        }
    }
#endif
}

/*
 * Mixed-precision BF16 linear backward for training.
 *
 * Forward contract:
 *   Y[t, o] = dot(input[t, :], weight[o, :]) + bias[o]
 *
 * Inputs are BF16 storage, math and gradients are FP32. This mirrors the
 * standard mixed-precision training contract where activations/weights may be
 * BF16 but gradient accumulation remains FP32.
 */
void gemm_backward_bf16_mixed(const uint16_t *d_output,
                              const uint16_t *input,
                              const uint16_t *weight,
                              float *d_input,
                              float *d_weight,
                              float *d_bias,
                              int tokens,
                              int in_dim,
                              int out_dim)
{
    if (!d_output || !input || !weight || tokens <= 0 || in_dim <= 0 || out_dim <= 0) {
        return;
    }

    if (d_input) {
        for (int t = 0; t < tokens; ++t) {
            for (int i = 0; i < in_dim; ++i) {
                float sum = 0.0f;
                for (int o = 0; o < out_dim; ++o) {
                    const float dy = bf16_to_float(d_output[(size_t)t * (size_t)out_dim + (size_t)o]);
                    const float w = bf16_to_float(weight[(size_t)o * (size_t)in_dim + (size_t)i]);
                    sum += dy * w;
                }
                d_input[(size_t)t * (size_t)in_dim + (size_t)i] = sum;
            }
        }
    }

    if (d_weight) {
        for (int o = 0; o < out_dim; ++o) {
            for (int i = 0; i < in_dim; ++i) {
                float sum = 0.0f;
                for (int t = 0; t < tokens; ++t) {
                    const float dy = bf16_to_float(d_output[(size_t)t * (size_t)out_dim + (size_t)o]);
                    const float x = bf16_to_float(input[(size_t)t * (size_t)in_dim + (size_t)i]);
                    sum += dy * x;
                }
                d_weight[(size_t)o * (size_t)in_dim + (size_t)i] = sum;
            }
        }
    }

    if (d_bias) {
        for (int o = 0; o < out_dim; ++o) {
            float sum = 0.0f;
            for (int t = 0; t < tokens; ++t) {
                sum += bf16_to_float(d_output[(size_t)t * (size_t)out_dim + (size_t)o]);
            }
            d_bias[o] = sum;
        }
    }
}
