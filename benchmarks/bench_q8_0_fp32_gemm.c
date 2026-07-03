/*
 * bench_q8_0_fp32_gemm.c
 *
 * Focused fp32 activation x Q8_0 weight GEMM benchmark for Qwen3-VL
 * vision encoder branch/projector FC shapes.
 */

#include "ckernel_quant.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern void gemm_nt_q8_0(const float *A, const void *B, const float *bias, float *C, int M, int N, int K);
extern void gemv_q8_0(float *y, const void *W, const float *x, int M, int K);

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static uint32_t rng_state = 0x91e10da5u;
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
static void fill_q8_0(uint8_t *dst, int N, int K) {
    const int nb = K / QK8_0;
    for (int n = 0; n < N; ++n) {
        for (int b = 0; b < nb; ++b) {
            block_q8_0 *blk = (block_q8_0 *)(void *)(dst + ((size_t)n * nb + b) * sizeof(block_q8_0));
            blk->d = 0x3800; /* 0.5 */
            for (int i = 0; i < QK8_0; ++i) blk->qs[i] = (int8_t)((int)(rng_u32() % 255u) - 127);
        }
    }
}
static void gemm_nt_q8_0_reference(const float *A, const void *B, const float *bias, float *C, int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        gemv_q8_0(C + (size_t)m * N, B, A + (size_t)m * K, N, K);
        if (bias) {
            for (int n = 0; n < N; ++n) C[(size_t)m * N + n] += bias[n];
        }
    }
}
static float max_abs_diff(const float *a, const float *b, size_t n) {
    float mx = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}
static float max_abs_val(const float *a, size_t n) {
    float mx = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float v = fabsf(a[i]);
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

typedef void (*bench_fn)(const float *, const void *, const float *, float *, int, int, int);
static double bench_ms(bench_fn fn, const float *A, const void *B, const float *bias, float *C,
                       int M, int N, int K, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) fn(A, B, bias, C, M, N, K);
    const double t0 = now_ms();
    for (int i = 0; i < iters; ++i) fn(A, B, bias, C, M, N, K);
    return (now_ms() - t0) / (double)iters;
}

int main(int argc, char **argv) {
    int M = 1028;
    int N = 4096;
    int K = 4096;
    int iters = 5;
    int warmup = 2;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--M") == 0 && i + 1 < argc) M = atoi(argv[++i]);
        else if (strcmp(argv[i], "--N") == 0 && i + 1 < argc) N = atoi(argv[++i]);
        else if (strcmp(argv[i], "--K") == 0 && i + 1 < argc) K = atoi(argv[++i]);
        else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = atoi(argv[++i]);
    }
    if (K % QK8_0 != 0) {
        fprintf(stderr, "K must be divisible by %d\n", QK8_0);
        return 2;
    }

    const size_t a_elems = (size_t)M * K;
    const size_t c_elems = (size_t)M * N;
    const size_t w_bytes = (size_t)N * (size_t)(K / QK8_0) * sizeof(block_q8_0);
    float *A = (float *)malloc(a_elems * sizeof(float));
    uint8_t *B = (uint8_t *)malloc(w_bytes);
    float *bias = (float *)malloc((size_t)N * sizeof(float));
    float *C_ref = (float *)malloc(c_elems * sizeof(float));
    float *C = (float *)malloc(c_elems * sizeof(float));
    if (!A || !B || !bias || !C_ref || !C) {
        fprintf(stderr, "allocation failed\n");
        return 2;
    }
    fill_f32(A, a_elems, 0.05f);
    fill_f32(bias, (size_t)N, 0.01f);
    fill_q8_0(B, N, K);

    gemm_nt_q8_0_reference(A, B, bias, C_ref, M, N, K);
    gemm_nt_q8_0(A, B, bias, C, M, N, K);
    const float diff = max_abs_diff(C_ref, C, c_elems);
    const float max_ref = max_abs_val(C_ref, c_elems);
    const double cos = cosine_sim(C_ref, C, c_elems);

    const double ref_ms = bench_ms(gemm_nt_q8_0_reference, A, B, bias, C_ref, M, N, K, warmup, iters);
    const double opt_ms = bench_ms(gemm_nt_q8_0, A, B, bias, C, M, N, K, warmup, iters);

    printf("fp32 x Q8_0 GEMM M=%d N=%d K=%d warmup=%d iters=%d\n", M, N, K, warmup, iters);
    printf("weights=%.1f MiB A=%.1f MiB C=%.1f MiB\n",
           (double)w_bytes / (1024.0 * 1024.0),
           (double)(a_elems * sizeof(float)) / (1024.0 * 1024.0),
           (double)(c_elems * sizeof(float)) / (1024.0 * 1024.0));
    printf("ref_ms     %.3f\n", ref_ms);
    printf("opt_ms     %.3f\n", opt_ms);
    if (opt_ms > 0.0) printf("speedup    %.3fx\n", ref_ms / opt_ms);
    printf("max_diff   %.6g\n", diff);
    printf("max_ref    %.6g\n", max_ref);
    printf("rel_diff   %.6g\n", max_ref > 0.0f ? diff / max_ref : 0.0f);
    printf("cosine     %.9f\n", cos);

    free(A); free(B); free(bias); free(C_ref); free(C);
    return (diff < 1e-3f || (max_ref > 0.0f && diff / max_ref < 1e-5f) || cos > 0.999999) ? 0 : 1;
}
