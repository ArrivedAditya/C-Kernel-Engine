
/*
 * Standalone OpenMP Q4_K x Q8_K gate_up + SwiGLU benchmark.
 *
 * This file is intentionally not wired into CK runtime. It is a CPU-tuning
 * harness for the Qwen3-VL OCR MLP gate_up shape: M=79, D=12288, K=4096.
 * It compares CK's current unfused reference path against an OpenMP-owned
 * x16 packed-panel fused prototype.
 */

#include "ckernel_quant.h"
#include "ck_threadpool.h"

#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern void quantize_row_q8_k(const float *x, void *vy, int k);
extern void gemm_nt_q4_k_q8_k(const void *A, const void *B, const float *bias,
                              float *C, int M, int N, int K);
extern void swiglu_forward_exact(const float *input, float *output, int tokens, int dim);

typedef struct {
    ck_half d[16];
    ck_half dmin[16];
    uint8_t sc[16][8];
    uint8_t m[16][8];
    uint8_t qs[16][QK_K / 2];
    uint8_t active;
    uint8_t reserved[15];
} q4k_x16_panel_t;

static inline float fp16_to_fp32(uint16_t h) {
    const uint32_t s = ((uint32_t)h & 0x8000u) << 16;
    uint32_t e = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t f = (uint32_t)h & 0x03ffu;
    uint32_t out;
    if (e == 0) {
        if (f == 0) out = s;
        else {
            e = 1;
            while ((f & 0x0400u) == 0) { f <<= 1; --e; }
            f &= 0x03ffu;
            out = s | ((e + (127 - 15)) << 23) | (f << 13);
        }
    } else if (e == 31) {
        out = s | 0x7f800000u | (f << 13);
    } else {
        out = s | ((e + (127 - 15)) << 23) | (f << 13);
    }
    float v;
    memcpy(&v, &out, sizeof(v));
    return v;
}

static inline void unpack_q4_k_scales_local(const uint8_t *scales, uint8_t *sc, uint8_t *m) {
    for (int i = 0; i < 4; ++i) sc[i] = scales[i] & 0x3f;
    for (int i = 0; i < 4; ++i) m[i] = scales[i + 4] & 0x3f;
    for (int i = 0; i < 4; ++i) {
        sc[i + 4] = (uint8_t)(((scales[i] >> 6) & 0x03) | ((scales[i + 8] & 0x0f) << 2));
        m[i + 4] = (uint8_t)(((scales[i + 4] >> 6) & 0x03) | ((scales[i + 8] >> 4) << 2));
    }
}

static void pack_x16(const block_q4_K *src, q4k_x16_panel_t *dst, int N, int K) {
    const int groups = (N + 15) / 16;
    const int blocks_per_row = K / QK_K;
    memset(dst, 0, (size_t)groups * (size_t)blocks_per_row * sizeof(*dst));
    for (int g = 0; g < groups; ++g) {
        const int n0 = g * 16;
        const int active = (n0 + 16 <= N) ? 16 : (N - n0);
        for (int b = 0; b < blocks_per_row; ++b) {
            q4k_x16_panel_t *panel = dst + (size_t)g * (size_t)blocks_per_row + (size_t)b;
            panel->active = (uint8_t)active;
            for (int lane = 0; lane < active; ++lane) {
                const block_q4_K *w = src + (size_t)(n0 + lane) * (size_t)blocks_per_row + (size_t)b;
                panel->d[lane] = w->d;
                panel->dmin[lane] = w->dmin;
                unpack_q4_k_scales(w->scales, panel->sc[lane], panel->m[lane]);
                memcpy(panel->qs[lane], w->qs, QK_K / 2);
            }
        }
    }
}

static inline int hsum256_epi32_local(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4e));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xb1));
    return _mm_cvtsi128_si32(s);
}

static inline __m256i unpack_q4_32(__m256i packed, int high) {
    const __m256i mask = _mm256_set1_epi8(0x0f);
    return high ? _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask)
                : _mm256_and_si256(packed, mask);
}

static inline int32_t dot32_vnni(__m256i q4, __m256i q8) {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    __m256i acc = _mm256_setzero_si256();
    acc = _mm256_dpbusd_epi32(acc, q4, q8);
    return hsum256_epi32_local(acc);
#else
    const __m256i prod16 = _mm256_maddubs_epi16(q4, q8);
    const __m256i ones = _mm256_set1_epi16(1);
    return hsum256_epi32_local(_mm256_madd_epi16(prod16, ones));
#endif
}

static inline void accum_panel(float acc[8][16], const q4k_x16_panel_t *w, int active,
                               const block_q8_K *A, int blocks_per_vec, int block_index,
                               int m0, int m_count) {
    for (int j = 0, is = 0, qoff = 0; j < QK_K; j += 64, is += 2, qoff += 32) {
        for (int lane = 0; lane < active; ++lane) {
            const uint8_t *qs = &w->qs[lane][qoff];
            const __m256i packed = _mm256_loadu_si256((const __m256i *)qs);
            const __m256i q4lo = unpack_q4_32(packed, 0);
            const __m256i q4hi = unpack_q4_32(packed, 1);
            const float wd = CK_FP16_TO_FP32(w->d[lane]);
            const float wdmin = CK_FP16_TO_FP32(w->dmin[lane]);
            const float sc_lo = (float)w->sc[lane][is];
            const float sc_hi = (float)w->sc[lane][is + 1];
            const float min_lo = (float)w->m[lane][is];
            const float min_hi = (float)w->m[lane][is + 1];
            for (int mt = 0; mt < m_count; ++mt) {
                const block_q8_K *x = A + (size_t)(m0 + mt) * (size_t)blocks_per_vec + (size_t)block_index;
                const __m256i q8lo = _mm256_loadu_si256((const __m256i *)&x->qs[j]);
                const __m256i q8hi = _mm256_loadu_si256((const __m256i *)&x->qs[j + 32]);
                const int32_t sum_lo = dot32_vnni(q4lo, q8lo);
                const int32_t sum_hi = dot32_vnni(q4hi, q8hi);
                const int32_t bsum_lo = (int32_t)x->bsums[j / 16] + (int32_t)x->bsums[j / 16 + 1];
                const int32_t bsum_hi = (int32_t)x->bsums[(j + 32) / 16] + (int32_t)x->bsums[(j + 32) / 16 + 1];
                const float xd = x->d;
                acc[mt][lane] += (wd * xd) * sc_lo * (float)sum_lo;
                acc[mt][lane] -= (wdmin * xd) * min_lo * (float)bsum_lo;
                acc[mt][lane] += (wd * xd) * sc_hi * (float)sum_hi;
                acc[mt][lane] -= (wdmin * xd) * min_hi * (float)bsum_hi;
            }
        }
    }
}

static inline float silu_f32(float x) { return x / (1.0f + expf(-x)); }


typedef struct {
    const block_q8_K *A;
    const q4k_x16_panel_t *W;
    const float *bias;
    float *C;
    int M;
    int D;
    int K;
    int tile_m;
    int blocks_per_vec;
    int blocks_per_row;
    int groups_d;
    int total_jobs;
} ck_x16_work_t;

static void run_x16_job(const ck_x16_work_t *a, int job) {
    const int tm_size = a->tile_m > 0 ? (a->tile_m > 8 ? 8 : a->tile_m) : 4;
    const int mt = (a->M + tm_size - 1) / tm_size;
    const int g = job / mt;
    const int tm = job - g * mt;
    const int m0 = tm * tm_size;
    const int m1 = (m0 + tm_size < a->M) ? (m0 + tm_size) : a->M;
    const int m_count = m1 - m0;
    const int d0 = g * 16;
    const int active = (d0 + 16 <= a->D) ? 16 : (a->D - d0);
    if (m_count <= 0 || active <= 0) return;

    float acc_gate[8][16];
    float acc_up[8][16];
    for (int mm = 0; mm < m_count; ++mm) {
        for (int lane = 0; lane < active; ++lane) {
            acc_gate[mm][lane] = a->bias ? a->bias[d0 + lane] : 0.0f;
            acc_up[mm][lane] = a->bias ? a->bias[a->D + d0 + lane] : 0.0f;
        }
    }
    for (int b = 0; b < a->blocks_per_row; ++b) {
        const q4k_x16_panel_t *wg = a->W + (size_t)g * (size_t)a->blocks_per_row + (size_t)b;
        const q4k_x16_panel_t *wu = a->W + (size_t)(a->groups_d + g) * (size_t)a->blocks_per_row + (size_t)b;
        accum_panel(acc_gate, wg, active, a->A, a->blocks_per_vec, b, m0, m_count);
        accum_panel(acc_up, wu, active, a->A, a->blocks_per_vec, b, m0, m_count);
    }
    for (int mm = 0; mm < m_count; ++mm) {
        float *c = a->C + (size_t)(m0 + mm) * (size_t)a->D;
        for (int lane = 0; lane < active; ++lane) c[d0 + lane] = silu_f32(acc_gate[mm][lane]) * acc_up[mm][lane];
    }
}

static void ck_x16_worker(int ith, int nth, void *args) {
    const ck_x16_work_t *a = (const ck_x16_work_t *)args;
    for (int job = ith; job < a->total_jobs; job += nth) run_x16_job(a, job);
}

static void ckpool_x16_fused(const block_q8_K *A, const q4k_x16_panel_t *W, const float *bias,
                             float *C, int M, int D, int K, int tile_m, int threads) {
    ck_threadpool_t *pool = ck_threadpool_global();
    const int tm_size = tile_m > 0 ? (tile_m > 8 ? 8 : tile_m) : 4;
    ck_x16_work_t work = {
        .A = A,
        .W = W,
        .bias = bias,
        .C = C,
        .M = M,
        .D = D,
        .K = K,
        .tile_m = tm_size,
        .blocks_per_vec = K / QK_K,
        .blocks_per_row = K / QK_K,
        .groups_d = (D + 15) / 16,
    };
    const int mt = (M + tm_size - 1) / tm_size;
    work.total_jobs = mt * work.groups_d;
    int active = threads > 0 ? threads : (pool ? ck_threadpool_n_threads(pool) : 1);
    if (active < 1) active = 1;
    if (active > work.total_jobs) active = work.total_jobs;
    if (!pool || active <= 1) ck_x16_worker(0, 1, &work);
    else ck_threadpool_dispatch_n(pool, active, ck_x16_worker, &work);
}

static void omp_x16_fused(const block_q8_K *A, const q4k_x16_panel_t *W, const float *bias,
                          float *C, int M, int D, int K, int tile_m, int threads, int schedule_kind) {
    const int blocks_per_vec = K / QK_K;
    const int blocks_per_row = K / QK_K;
    const int groups_d = (D + 15) / 16;
    const int tm_size = tile_m > 0 ? (tile_m > 8 ? 8 : tile_m) : 4;
    const int mt = (M + tm_size - 1) / tm_size;
    const int total = mt * groups_d;
    omp_set_num_threads(threads);

#pragma omp parallel for schedule(static)
    for (int job = 0; job < total; ++job) {
        (void)schedule_kind;
        const int g = job / mt;
        const int tm = job - g * mt;
        const int m0 = tm * tm_size;
        const int m1 = (m0 + tm_size < M) ? (m0 + tm_size) : M;
        const int m_count = m1 - m0;
        const int d0 = g * 16;
        const int active = (d0 + 16 <= D) ? 16 : (D - d0);
        float acc_gate[8][16];
        float acc_up[8][16];
        for (int mm = 0; mm < m_count; ++mm) {
            for (int lane = 0; lane < active; ++lane) {
                acc_gate[mm][lane] = bias ? bias[d0 + lane] : 0.0f;
                acc_up[mm][lane] = bias ? bias[D + d0 + lane] : 0.0f;
            }
        }
        for (int b = 0; b < blocks_per_row; ++b) {
            const q4k_x16_panel_t *wg = W + (size_t)g * (size_t)blocks_per_row + (size_t)b;
            const q4k_x16_panel_t *wu = W + (size_t)(groups_d + g) * (size_t)blocks_per_row + (size_t)b;
            accum_panel(acc_gate, wg, active, A, blocks_per_vec, b, m0, m_count);
            accum_panel(acc_up, wu, active, A, blocks_per_vec, b, m0, m_count);
        }
        for (int mm = 0; mm < m_count; ++mm) {
            float *c = C + (size_t)(m0 + mm) * (size_t)D;
            for (int lane = 0; lane < active; ++lane) c[d0 + lane] = silu_f32(acc_gate[mm][lane]) * acc_up[mm][lane];
        }
    }
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}
static uint32_t rng_state = 0x456789abu;
static uint32_t rng_u32(void) { rng_state = rng_state * 1664525u + 1013904223u; return rng_state; }
static float rng_f32(float scale) { return (((float)(rng_u32() >> 8) / (float)0x00ffffffu) * 2.0f - 1.0f) * scale; }
static void fill_f32(float *p, size_t n, float scale) { for (size_t i = 0; i < n; ++i) p[i] = rng_f32(scale); }
static void fill_q4k(uint8_t *dst, int N, int K) {
    const int blocks = N * (K / QK_K);
    for (int b = 0; b < blocks; ++b) {
        block_q4_K *blk = (block_q4_K *)(void *)(dst + (size_t)b * sizeof(block_q4_K));
        blk->d = 0x3c00;
        blk->dmin = 0x3800;
        for (int i = 0; i < (int)sizeof(blk->scales); ++i) blk->scales[i] = (uint8_t)(rng_u32() & 0xffu);
        for (int i = 0; i < (int)sizeof(blk->qs); ++i) blk->qs[i] = (uint8_t)(rng_u32() & 0xffu);
    }
}
static void quantize_acts_q8k(const float *src, uint8_t *dst, int M, int K) {
    const size_t row_bytes = (size_t)(K / QK_K) * sizeof(block_q8_K);
    for (int m = 0; m < M; ++m) quantize_row_q8_k(src + (size_t)m * (size_t)K, dst + (size_t)m * row_bytes, K);
}
static float max_abs_diff(const float *a, const float *b, size_t n) {
    float mx = 0.0f;
    for (size_t i = 0; i < n; ++i) { float d = fabsf(a[i] - b[i]); if (d > mx) mx = d; }
    return mx;
}
static float max_abs_val(const float *a, size_t n) {
    float mx = 0.0f;
    for (size_t i = 0; i < n; ++i) { float v = fabsf(a[i]); if (v > mx) mx = v; }
    return mx;
}
static double cosine_sim(const float *a, const float *b, size_t n) {
    double dot = 0.0, aa = 0.0, bb = 0.0;
    for (size_t i = 0; i < n; ++i) { dot += (double)a[i] * b[i]; aa += (double)a[i] * a[i]; bb += (double)b[i] * b[i]; }
    return (aa > 0.0 && bb > 0.0) ? dot / sqrt(aa * bb) : 0.0;
}

int main(int argc, char **argv) {
    int M = 79, D = 12288, K = 4096, iters = 8, warmup = 2, threads = 16, tile_m = 4;
    const char *mode = "both";
    const char *scheduler = "omp";
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--M") && i + 1 < argc) M = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--D") && i + 1 < argc) D = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--K") && i + 1 < argc) K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tile-m") && i + 1 < argc) tile_m = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mode") && i + 1 < argc) mode = argv[++i];
        else if (!strcmp(argv[i], "--scheduler") && i + 1 < argc) scheduler = argv[++i];
    }

    const size_t q8_bytes = (size_t)M * (size_t)(K / QK_K) * sizeof(block_q8_K);
    const size_t q4_bytes = (size_t)(2 * D) * (size_t)(K / QK_K) * sizeof(block_q4_K);
    const size_t x16_bytes = (size_t)((2 * D + 15) / 16) * (size_t)(K / QK_K) * sizeof(q4k_x16_panel_t);
    const size_t gateup_elems = (size_t)M * (size_t)(2 * D);
    const size_t out_elems = (size_t)M * (size_t)D;

    float *A = (float *)malloc((size_t)M * (size_t)K * sizeof(float));
    uint8_t *A_q8 = (uint8_t *)malloc(q8_bytes);
    uint8_t *W = (uint8_t *)malloc(q4_bytes);
    q4k_x16_panel_t *W_x16 = (q4k_x16_panel_t *)malloc(x16_bytes);
    float *bias = (float *)malloc((size_t)(2 * D) * sizeof(float));
    float *gate_up = (float *)malloc(gateup_elems * sizeof(float));
    float *out_ref = (float *)malloc(out_elems * sizeof(float));
    float *out_x16 = (float *)malloc(out_elems * sizeof(float));
    if (!A || !A_q8 || !W || !W_x16 || !bias || !gate_up || !out_ref || !out_x16) { fprintf(stderr, "alloc failed\n"); return 2; }

    fill_f32(A, (size_t)M * (size_t)K, 0.05f);
    fill_f32(bias, (size_t)(2 * D), 0.01f);
    fill_q4k(W, 2 * D, K);
    quantize_acts_q8k(A, A_q8, M, K);
    const double p0 = now_ms();
    pack_x16((const block_q4_K *)W, W_x16, 2 * D, K);
    const double pack_ms = now_ms() - p0;

    gemm_nt_q4_k_q8_k(A_q8, W, bias, gate_up, M, 2 * D, K);
    swiglu_forward_exact(gate_up, out_ref, M, D);
    if (!strcmp(scheduler, "ck")) ckpool_x16_fused((const block_q8_K *)A_q8, W_x16, bias, out_x16, M, D, K, tile_m, threads);
    else omp_x16_fused((const block_q8_K *)A_q8, W_x16, bias, out_x16, M, D, K, tile_m, threads, 0);
    const float diff = max_abs_diff(out_ref, out_x16, out_elems);
    const float max_ref = max_abs_val(out_ref, out_elems);
    const double cos = cosine_sim(out_ref, out_x16, out_elems);

    double ref_ms = 0.0, x16_ms = 0.0;
    if (strcmp(mode, "x16") != 0) {
        for (int i = 0; i < warmup; ++i) { gemm_nt_q4_k_q8_k(A_q8, W, bias, gate_up, M, 2 * D, K); swiglu_forward_exact(gate_up, out_ref, M, D); }
        const double t0 = now_ms();
        for (int i = 0; i < iters; ++i) { gemm_nt_q4_k_q8_k(A_q8, W, bias, gate_up, M, 2 * D, K); swiglu_forward_exact(gate_up, out_ref, M, D); }
        ref_ms = (now_ms() - t0) / (double)iters;
    }
    if (strcmp(mode, "ref") != 0) {
        for (int i = 0; i < warmup; ++i) {
            if (!strcmp(scheduler, "ck")) ckpool_x16_fused((const block_q8_K *)A_q8, W_x16, bias, out_x16, M, D, K, tile_m, threads);
            else omp_x16_fused((const block_q8_K *)A_q8, W_x16, bias, out_x16, M, D, K, tile_m, threads, 0);
        }
        const double t0 = now_ms();
        for (int i = 0; i < iters; ++i) {
            if (!strcmp(scheduler, "ck")) ckpool_x16_fused((const block_q8_K *)A_q8, W_x16, bias, out_x16, M, D, K, tile_m, threads);
            else omp_x16_fused((const block_q8_K *)A_q8, W_x16, bias, out_x16, M, D, K, tile_m, threads, 0);
        }
        x16_ms = (now_ms() - t0) / (double)iters;
    }

    printf("standalone_q4k_gateup_swiglu M=%d D=%d K=%d threads=%d tile_m=%d iters=%d mode=%s scheduler=%s\n", M, D, K, threads, tile_m, iters, mode, scheduler);
    printf("weights=%.1fMiB packed_x16=%.1fMiB scratch=%.1fMiB output=%.1fMiB pack_ms=%.3f\n",
           (double)q4_bytes / 1048576.0, (double)x16_bytes / 1048576.0,
           (double)(gateup_elems * sizeof(float)) / 1048576.0,
           (double)(out_elems * sizeof(float)) / 1048576.0, pack_ms);
    printf("ref_ms %.3f\n", ref_ms);
    printf("x16_ms %.3f\n", x16_ms);
    if (ref_ms > 0.0 && x16_ms > 0.0) printf("speedup %.3fx\n", ref_ms / x16_ms);
    printf("max_diff %.6g\n", diff);
    printf("max_ref %.6g\n", max_ref);
    printf("rel_diff %.6g\n", max_ref > 0.0f ? diff / max_ref : 0.0f);
    printf("cosine %.9f\n", cos);

    free(A); free(A_q8); free(W); free(W_x16); free(bias); free(gate_up); free(out_ref); free(out_x16);
    return (diff < 1e-3f || (max_ref > 0.0f && diff / max_ref < 1e-5f) || cos > 0.999999) ? 0 : 1;
}
