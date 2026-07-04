
/*
 * bench_q4k_gateup_swiglu.c
 *
 * Focused Qwen3-VL OCR MLP gate/up benchmark.
 * Compares the current unfused path:
 *   gemm_nt_q4_k_q8_k(A, W_gate_up) -> swiglu_forward_exact
 * against the experimental fused VNNI path:
 *   Q4_K gate/up row pair + Q8_K activation rows -> SiLU output
 */

#include "ck_threadpool.h"
#include "ckernel_quant.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern void quantize_row_q8_k(const float *x, void *vy, int k);
extern void gemm_nt_q4_k_q8_k(const void *A, const void *B, const float *bias,
                              float *C, int M, int N, int K);
extern void swiglu_forward_exact(const float *input, float *output, int tokens, int dim);
extern void gemm_nt_q4_k_q8_k_gateup_swiglu_fused_vnni(const void *A_q8,
                                                        const void *B_gate_up,
                                                        const float *bias,
                                                        float *C,
                                                        int M,
                                                        int D,
                                                        int K,
                                                        int threads);
extern size_t q4_k_packed_meta_x16_block_size(void);
extern void pack_q4_k_to_packed_meta_x16(const void *src, void *dst, int N, int K);
extern void gemm_nt_q4_k_packed_meta_x16_gateup_swiglu_fused_vnni(const void *A_q8,
                                                                   const void *B_packed_x16,
                                                                   const float *bias,
                                                                   float *C,
                                                                   int M,
                                                                   int D,
                                                                   int K,
                                                                   int tile_m,
                                                                   int active_threads);

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static uint32_t rng_state = 0x456789abu;
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
    for (size_t i = 0; i < n; ++i) p[i] = rng_f32(scale);
}
static void fill_q4k(uint8_t *dst, int N, int K) {
    const int blocks = N * (K / QK_K);
    for (int b = 0; b < blocks; ++b) {
        block_q4_K *blk = (block_q4_K *)(void *)(dst + (size_t)b * sizeof(block_q4_K));
        blk->d = 0x3c00;     /* 1.0 */
        blk->dmin = 0x3800;  /* 0.5 */
        for (int i = 0; i < (int)sizeof(blk->scales); ++i) blk->scales[i] = (uint8_t)(rng_u32() & 0xffu);
        for (int i = 0; i < (int)sizeof(blk->qs); ++i) blk->qs[i] = (uint8_t)(rng_u32() & 0xffu);
    }
}
static void quantize_acts_q8k(const float *src, uint8_t *dst, int M, int K) {
    const size_t row_bytes = (size_t)(K / QK_K) * sizeof(block_q8_K);
    for (int m = 0; m < M; ++m) {
        quantize_row_q8_k(src + (size_t)m * (size_t)K, dst + (size_t)m * row_bytes, K);
    }
}
static float max_abs_diff(const float *a, const float *b, size_t n) {
    float mx = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}
static float max_abs_val(const float *a, size_t n) {
    float mx = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float v = fabsf(a[i]);
        if (v > mx) mx = v;
    }
    return mx;
}
static double cosine_sim(const float *a, const float *b, size_t n) {
    double dot = 0.0, aa = 0.0, bb = 0.0;
    for (size_t i = 0; i < n; ++i) {
        dot += (double)a[i] * (double)b[i];
        aa += (double)a[i] * (double)a[i];
        bb += (double)b[i] * (double)b[i];
    }
    return (aa > 0.0 && bb > 0.0) ? dot / sqrt(aa * bb) : 0.0;
}

typedef void (*bench_fn)(void *);
static double bench_ms(bench_fn fn, void *ctx, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) fn(ctx);
    const double t0 = now_ms();
    for (int i = 0; i < iters; ++i) fn(ctx);
    const double t1 = now_ms();
    return (t1 - t0) / (double)iters;
}

typedef struct {
    const uint8_t *A_q8;
    const uint8_t *W;
    const void *W_x16;
    const float *bias;
    float *gate_up;
    float *out;
    int M;
    int D;
    int K;
    int threads;
    int tile_m;
} ctx_t;

static void run_unfused(void *p) {
    ctx_t *c = (ctx_t *)p;
    gemm_nt_q4_k_q8_k(c->A_q8, c->W, c->bias, c->gate_up, c->M, 2 * c->D, c->K);
    swiglu_forward_exact(c->gate_up, c->out, c->M, c->D);
}
static void run_fused(void *p) {
    ctx_t *c = (ctx_t *)p;
    gemm_nt_q4_k_q8_k_gateup_swiglu_fused_vnni(c->A_q8, c->W, c->bias, c->out, c->M, c->D, c->K, c->threads);
}
static void run_x16(void *p) {
    ctx_t *c = (ctx_t *)p;
    gemm_nt_q4_k_packed_meta_x16_gateup_swiglu_fused_vnni(
        c->A_q8, c->W_x16, c->bias, c->out, c->M, c->D, c->K, c->tile_m, c->threads);
}

int main(int argc, char **argv) {
    int M = 79;
    int D = 12288;
    int K = 4096;
    int iters = 8;
    int warmup = 2;
    int mode = 0; /* 0=both, 1=unfused, 2=fused, 3=x16 */
    int tile_m = 8;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "--M") == 0 && i + 1 < argc) M = atoi(argv[++i]);
        else if (strcmp(argv[i], "--D") == 0 && i + 1 < argc) D = atoi(argv[++i]);
        else if (strcmp(argv[i], "--K") == 0 && i + 1 < argc) K = atoi(argv[++i]);
        else if (strcmp(argv[i], "--tile-m") == 0 && i + 1 < argc) tile_m = atoi(argv[++i]);
        else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            const char *m = argv[++i];
            if (strcmp(m, "unfused") == 0) mode = 1;
            else if (strcmp(m, "fused") == 0) mode = 2;
            else if (strcmp(m, "x16") == 0) mode = 3;
            else mode = 0;
        }
    }
    const char *th_env = getenv("CK_NUM_THREADS");
    int threads = th_env && th_env[0] ? atoi(th_env) : 1;
    if (threads <= 0) threads = 1;

    const size_t q8_bytes = (size_t)M * (size_t)(K / QK_K) * sizeof(block_q8_K);
    const size_t q4_bytes = (size_t)(2 * D) * (size_t)(K / QK_K) * sizeof(block_q4_K);
    const size_t gateup_elems = (size_t)M * (size_t)(2 * D);
    const size_t out_elems = (size_t)M * (size_t)D;
    const size_t x16_blocks = (size_t)((2 * D + 15) / 16) * (size_t)(K / QK_K);
    const size_t x16_bytes = x16_blocks * q4_k_packed_meta_x16_block_size();

    float *A = (float *)malloc((size_t)M * (size_t)K * sizeof(float));
    uint8_t *A_q8 = (uint8_t *)malloc(q8_bytes);
    uint8_t *W = (uint8_t *)malloc(q4_bytes);
    void *W_x16 = aligned_alloc(64, (x16_bytes + 63u) & ~63u);
    float *bias = (float *)malloc((size_t)(2 * D) * sizeof(float));
    float *gate_up = (float *)malloc(gateup_elems * sizeof(float));
    float *out_ref = (float *)malloc(out_elems * sizeof(float));
    float *out = (float *)malloc(out_elems * sizeof(float));
    if (!A || !A_q8 || !W || !W_x16 || !bias || !gate_up || !out_ref || !out) {
        fprintf(stderr, "allocation failed\n");
        return 2;
    }

    fill_f32(A, (size_t)M * (size_t)K, 0.05f);
    fill_f32(bias, (size_t)(2 * D), 0.01f);
    fill_q4k(W, 2 * D, K);
    quantize_acts_q8k(A, A_q8, M, K);
    const double pack_t0 = now_ms();
    pack_q4_k_to_packed_meta_x16(W, W_x16, 2 * D, K);
    const double pack_ms = now_ms() - pack_t0;

    ctx_t ctx = {.A_q8 = A_q8, .W = W, .W_x16 = W_x16, .bias = bias, .gate_up = gate_up, .out = out_ref,
                 .M = M, .D = D, .K = K, .threads = threads, .tile_m = tile_m};
    run_unfused(&ctx);

    ctx.out = out;
    memset(out, 0, out_elems * sizeof(float));
    if (mode == 3) run_x16(&ctx);
    else run_fused(&ctx);
    const float diff = max_abs_diff(out_ref, out, out_elems);
    const float max_ref = max_abs_val(out_ref, out_elems);
    const double cos = cosine_sim(out_ref, out, out_elems);

    double unfused = 0.0;
    double fused = 0.0;
    double x16 = 0.0;
    if (mode == 0 || mode == 1 || mode == 3) {
        ctx.out = out_ref;
        unfused = bench_ms(run_unfused, &ctx, warmup, iters);
    }
    if (mode == 0 || mode == 2) {
        ctx.out = out;
        fused = bench_ms(run_fused, &ctx, warmup, iters);
    }
    if (mode == 3) {
        ctx.out = out;
        x16 = bench_ms(run_x16, &ctx, warmup, iters);
    }

    const double w_mib = (double)q4_bytes / (1024.0 * 1024.0);
    const double scratch_mib = (double)(gateup_elems * sizeof(float)) / (1024.0 * 1024.0);
    printf("Q4_K gate_up+SwiGLU M=%d D=%d K=%d threads=%d warmup=%d iters=%d mode=%s tile_m=%d\n",
           M, D, K, threads, warmup, iters,
           mode == 1 ? "unfused" : (mode == 2 ? "fused" : (mode == 3 ? "x16" : "both")), tile_m);
    printf("weights=%.1f MiB packed_x16=%.1f MiB pack_ms=%.3f gate_up_scratch=%.1f MiB output=%.1f MiB\n",
           w_mib, (double)x16_bytes / (1024.0 * 1024.0), pack_ms, scratch_mib,
           (double)(out_elems * sizeof(float)) / (1024.0 * 1024.0));
    printf("unfused_ms %.3f\n", unfused);
    printf("fused_ms   %.3f\n", fused);
    printf("x16_ms     %.3f\n", x16);
    if (unfused > 0.0 && fused > 0.0) printf("speedup_fused %.3fx\n", unfused / fused);
    if (unfused > 0.0 && x16 > 0.0) printf("speedup_x16   %.3fx\n", unfused / x16);
    printf("max_diff   %.6g\n", diff);
    printf("max_ref    %.6g\n", max_ref);
    printf("rel_diff   %.6g\n", max_ref > 0.0f ? diff / max_ref : 0.0f);
    printf("cosine     %.9f\n", cos);

    free(A); free(A_q8); free(W); free(W_x16); free(bias); free(gate_up); free(out_ref); free(out);
    ck_threadpool_global_destroy();
    return (diff < 1e-3f || (max_ref > 0.0f && diff / max_ref < 1e-5f) || cos > 0.999999) ? 0 : 1;
}
