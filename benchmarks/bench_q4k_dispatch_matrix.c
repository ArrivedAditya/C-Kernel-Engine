/*
 * bench_q4k_dispatch_matrix.c
 *
 * Q4_K x Q8_K prefill dispatch matrix:
 *   - CK canonical serial
 *   - CK canonical v8 threadpool dispatch
 *   - CK packed-meta M-split threadpool
 *   - CK packed-meta N-split threadpool
 *   - CK experimental packed-meta x8 N-tile threadpool
 *   - llama.cpp test shim, when llama.cpp/libggml_kernel_test.so is available
 *
 * The llama.cpp shim currently accepts FP32 activations and quantizes them to
 * Q8_K internally. CK timings use pre-quantized Q8_K activations, matching the
 * current v8 prefill contract. Treat the llama column as a useful reference,
 * not a perfect apples-to-apples prequantized-kernel measurement.
 */

#include "ck_threadpool.h"
#include "ckernel_quant.h"

#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern void quantize_row_q8_k(const float *x, void *vy, int k);
extern void gemm_nt_q4_k_q8_k(const void *A, const void *B, const float *bias,
                              float *C, int M, int N, int K);
extern void gemm_nt_q4_k_q8_k_parallel_dispatch(const void *A, const void *B,
                                                const float *bias, float *C,
                                                int M, int N, int K);
extern size_t q4_k_packed_meta_block_size(void);
extern size_t q4_k_packed_meta_x8_block_size(void);
extern void pack_q4_k_to_packed_meta(const void *src, void *dst, int N, int K);
extern void pack_q4_k_to_packed_meta_x8(const void *src, void *dst, int N, int K);
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
extern void gemm_nt_q4_k_packed_meta_x8_q8_k_threaded_nsplit(const void *A_q8,
                                                             const void *B_packed_x8,
                                                             const float *bias,
                                                             float *C,
                                                             int M, int N, int K,
                                                             int threads);

typedef void (*llama_gemm_q4_k_fn)(const void *, const float *, float *, int, int, int);

typedef struct {
    const char *name;
    int M;
    int N;
    int K;
    const char *comment;
} shape_t;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static uint32_t rng_state = 0x12345678u;

static uint32_t rng_u32(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}

static float rng_f32(float scale) {
    const uint32_t v = rng_u32() >> 8;
    const float u = (float)v / (float)0x00ffffffu;
    return (u * 2.0f - 1.0f) * scale;
}

static void fill_f32(float *p, size_t n, float scale) {
    for (size_t i = 0; i < n; ++i) {
        p[i] = rng_f32(scale);
    }
}

static void fill_q4k(uint8_t *dst, int N, int K) {
    const int blocks = N * (K / QK_K);
    for (int b = 0; b < blocks; ++b) {
        block_q4_K *blk = (block_q4_K *)(void *)(dst + (size_t)b * sizeof(block_q4_K));
        blk->d = GGML_FP32_TO_FP16(0.03125f);
        blk->dmin = GGML_FP32_TO_FP16(0.00390625f);
        for (size_t i = 0; i < sizeof(blk->scales); ++i) {
            blk->scales[i] = (uint8_t)(rng_u32() & 0x3fu);
        }
        for (size_t i = 0; i < sizeof(blk->qs); ++i) {
            blk->qs[i] = (uint8_t)(rng_u32() & 0xffu);
        }
    }
}

static void quantize_acts_q8k(const float *A, uint8_t *A_q8, int M, int K) {
    const int blocks_per_row = K / QK_K;
    for (int m = 0; m < M; ++m) {
        quantize_row_q8_k(A + (size_t)m * K,
                          A_q8 + (size_t)m * blocks_per_row * sizeof(block_q8_K),
                          K);
    }
}

static double bench_ms(void (*fn)(void *), void *ctx, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) {
        fn(ctx);
    }
    const double t0 = now_ms();
    for (int i = 0; i < iters; ++i) {
        fn(ctx);
    }
    return (now_ms() - t0) / (double)iters;
}

static float max_abs_diff(const float *a, const float *b, size_t n) {
    float maxv = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float d = fabsf(a[i] - b[i]);
        if (d > maxv) maxv = d;
    }
    return maxv;
}

typedef struct {
    const uint8_t *A_q8;
    const uint8_t *W_q4;
    const uint8_t *W_packed;
    const uint8_t *W_packed_x8;
    const float *A_fp32;
    const float *bias;
    float *C;
    int M;
    int N;
    int K;
    int threads;
    llama_gemm_q4_k_fn llama_fn;
} bench_ctx_t;

static void call_ck_serial(void *p) {
    bench_ctx_t *c = (bench_ctx_t *)p;
    gemm_nt_q4_k_q8_k(c->A_q8, c->W_q4, c->bias, c->C, c->M, c->N, c->K);
}

static void call_ck_threadpool(void *p) {
    bench_ctx_t *c = (bench_ctx_t *)p;
    gemm_nt_q4_k_q8_k_parallel_dispatch(c->A_q8, c->W_q4, c->bias, c->C, c->M, c->N, c->K);
}

static void call_ck_packed_msplit(void *p) {
    bench_ctx_t *c = (bench_ctx_t *)p;
    gemm_nt_q4_k_packed_meta_q8_k_threaded(c->A_q8, c->W_packed, c->bias, c->C, c->M, c->N, c->K, c->threads);
}

static void call_ck_packed_nsplit(void *p) {
    bench_ctx_t *c = (bench_ctx_t *)p;
    gemm_nt_q4_k_packed_meta_q8_k_threaded_nsplit(c->A_q8, c->W_packed, c->bias, c->C, c->M, c->N, c->K, c->threads);
}

static void call_ck_packed_x8_nsplit(void *p) {
    bench_ctx_t *c = (bench_ctx_t *)p;
    gemm_nt_q4_k_packed_meta_x8_q8_k_threaded_nsplit(c->A_q8, c->W_packed_x8, c->bias, c->C, c->M, c->N, c->K, c->threads);
}

static void call_llama(void *p) {
    bench_ctx_t *c = (bench_ctx_t *)p;
    c->llama_fn(c->W_q4, c->A_fp32, c->C, c->N, c->K, c->M);
}

static llama_gemm_q4_k_fn load_llama_gemm(void) {
    const char *paths[] = {
        "./llama.cpp/libggml_kernel_test.so",
        "./llama.cpp/build/libggml_kernel_test.so",
        "./llama.cpp/build/bin/libggml_kernel_test.so",
        NULL,
    };
    for (int i = 0; paths[i]; ++i) {
        void *handle = dlopen(paths[i], RTLD_LAZY | RTLD_LOCAL);
        if (!handle) continue;
        void *sym = dlsym(handle, "test_gemm_q4_k");
        if (sym) {
            return (llama_gemm_q4_k_fn)sym;
        }
    }
    return NULL;
}

static int parse_threads(void) {
    const char *v = getenv("CK_NUM_THREADS");
    if (!v || !v[0]) return 0;
    const int n = atoi(v);
    return n > 0 ? n : 0;
}

int main(int argc, char **argv) {
    int quick = 0;
    int iters = 20;
    int warmup = 3;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--quick") == 0) {
            quick = 1;
            iters = 8;
            warmup = 2;
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = atoi(argv[++i]);
        }
    }

    ck_threadpool_t *pool = ck_threadpool_global();
    const int pool_threads = pool ? ck_threadpool_n_threads(pool) : 1;
    const int threads = parse_threads() > 0 ? parse_threads() : pool_threads;
    llama_gemm_q4_k_fn llama_fn = load_llama_gemm();

    const shape_t quick_shapes[] = {
        /* Tiny smoke shape: validates dispatch overhead and tail handling. */
        {"small", 8, 256, 256, "tiny smoke"},
        /* Qwen3.5/Nemotron-style compact MLP-down projection: smaller N, large K. */
        {"qwen35_down", 32, 896, 4864, "compact down"},
        /* Medium/wide prefill projection: enough N to expose output-tile layout wins. */
        {"wide", 64, 1024, 2560, "medium wide"},
        {NULL, 0, 0, 0, NULL},
    };
    const shape_t full_shapes[] = {
        /* Qwen3.5 attention projection scale: square-ish, bandwidth sensitive. */
        {"qwen35_qkv", 32, 1024, 1024, "attention proj"},
        /* Qwen3.5/Nemotron compact down projection: hot prefill MLP shape. */
        {"qwen35_down", 32, 896, 4864, "compact down"},
        /* Longer-prompt version of the compact down shape. */
        {"prefill128", 128, 896, 4864, "long compact down"},
        /* Wide MLP shape: stresses output tiling and thread occupancy. */
        {"wide_mlp", 64, 2560, 10240, "wide mlp"},
        {NULL, 0, 0, 0, NULL},
    };
    const shape_t *shapes = quick ? quick_shapes : full_shapes;

    printf("Q4_K x Q8_K prefill dispatch matrix; lower ms is better\n");
    printf("threads=%d warmup=%d iters=%d llama=%s\n", threads, warmup, iters, llama_fn ? "yes" : "no");
    printf("%-14s %5s %6s %6s %10s %10s %10s %10s %10s %10s %8s %8s %8s %9s %9s %9s %9s\n",
           "shape", "M", "N", "K", "serial", "pool", "packed-M", "packed-N", "packed-x8", "llama*",
           "pool/x", "packN/x", "x8/x", "d_pool", "d_packN", "d_x8", "d_llama");

    for (int s = 0; shapes[s].name; ++s) {
        const int M = shapes[s].M;
        const int N = shapes[s].N;
        const int K = shapes[s].K;
        if (K % QK_K != 0) continue;

        const size_t out_elems = (size_t)M * (size_t)N;
        const size_t q4_bytes = (size_t)N * (size_t)(K / QK_K) * sizeof(block_q4_K);
        const size_t q8_bytes = (size_t)M * (size_t)(K / QK_K) * sizeof(block_q8_K);
        const size_t packed_bytes = (size_t)N * (size_t)(K / QK_K) * q4_k_packed_meta_block_size();
        const size_t packed_x8_bytes = (size_t)((N + 7) / 8) * (size_t)(K / QK_K) * q4_k_packed_meta_x8_block_size();

        float *A = (float *)malloc((size_t)M * (size_t)K * sizeof(float));
        uint8_t *A_q8 = (uint8_t *)malloc(q8_bytes);
        uint8_t *W = (uint8_t *)malloc(q4_bytes);
        uint8_t *W_packed = (uint8_t *)malloc(packed_bytes);
        uint8_t *W_packed_x8 = (uint8_t *)malloc(packed_x8_bytes);
        float *bias = (float *)malloc((size_t)N * sizeof(float));
        float *C_ref = (float *)calloc(out_elems, sizeof(float));
        float *C = (float *)calloc(out_elems, sizeof(float));
        float *C_llama = (float *)calloc(out_elems, sizeof(float));
        if (!A || !A_q8 || !W || !W_packed || !W_packed_x8 || !bias || !C_ref || !C || !C_llama) {
            fprintf(stderr, "allocation failed for %s\n", shapes[s].name);
            return 2;
        }

        rng_state = 0x12345678u + (uint32_t)s * 7919u;
        fill_f32(A, (size_t)M * (size_t)K, 1.0f);
        fill_f32(bias, (size_t)N, 0.01f);
        fill_q4k(W, N, K);
        quantize_acts_q8k(A, A_q8, M, K);
        pack_q4_k_to_packed_meta(W, W_packed, N, K);
        pack_q4_k_to_packed_meta_x8(W, W_packed_x8, N, K);

        bench_ctx_t ctx = {
            .A_q8 = A_q8,
            .W_q4 = W,
            .W_packed = W_packed,
            .W_packed_x8 = W_packed_x8,
            .A_fp32 = A,
            .bias = bias,
            .C = C,
            .M = M,
            .N = N,
            .K = K,
            .threads = threads,
            .llama_fn = llama_fn,
        };

        gemm_nt_q4_k_q8_k(A_q8, W, bias, C_ref, M, N, K);

        ctx.C = C;
        const double t_serial = bench_ms(call_ck_serial, &ctx, warmup, iters);
        const float d_serial = max_abs_diff(C_ref, C, out_elems);

        memset(C, 0, out_elems * sizeof(float));
        const double t_pool = bench_ms(call_ck_threadpool, &ctx, warmup, iters);
        const float d_pool = max_abs_diff(C_ref, C, out_elems);

        memset(C, 0, out_elems * sizeof(float));
        const double t_packed_m = bench_ms(call_ck_packed_msplit, &ctx, warmup, iters);
        const float d_packed_m = max_abs_diff(C_ref, C, out_elems);

        memset(C, 0, out_elems * sizeof(float));
        const double t_packed_n = bench_ms(call_ck_packed_nsplit, &ctx, warmup, iters);
        const float d_packed_n = max_abs_diff(C_ref, C, out_elems);

        memset(C, 0, out_elems * sizeof(float));
        const double t_packed_x8 = bench_ms(call_ck_packed_x8_nsplit, &ctx, warmup, iters);
        const float d_packed_x8 = max_abs_diff(C_ref, C, out_elems);

        double t_llama = 0.0;
        float d_llama = 0.0f;
        if (llama_fn) {
            ctx.C = C_llama;
            t_llama = bench_ms(call_llama, &ctx, warmup, iters);
            d_llama = max_abs_diff(C_ref, C_llama, out_elems);
        }

        printf("%-14s %5d %6d %6d %10.3f %10.3f %10.3f %10.3f %10.3f ",
               shapes[s].name, M, N, K, t_serial, t_pool, t_packed_m, t_packed_n, t_packed_x8);
        if (llama_fn) {
            printf("%10.3f ", t_llama);
        } else {
            printf("%10s ", "n/a");
        }
        printf("%8.2fx %8.2fx %8.2fx %9.2g %9.2g %9.2g %9.2g\n",
               t_serial / t_pool,
               t_serial / t_packed_n,
               t_serial / t_packed_x8,
               d_pool,
               d_packed_n,
               d_packed_x8,
               d_llama);

        free(A);
        free(A_q8);
        free(W);
        free(W_packed);
        free(W_packed_x8);
        free(bias);
        free(C_ref);
        free(C);
        free(C_llama);
    }

    ck_threadpool_global_destroy();
    printf("\n* llama column uses the local llama.cpp parity shim and includes FP32->Q8_K activation quantization.\n");
    printf("* shapes: small=tiny overhead smoke; qwen35_qkv=attention projection; qwen35_down/prefill128=compact MLP-down; wide/wide_mlp=wide output-tile stress.\n");
    return 0;
}
