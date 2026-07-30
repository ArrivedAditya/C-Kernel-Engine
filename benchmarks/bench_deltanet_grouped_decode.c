/*
 * Production-shape Qwen3.6 grouped DeltaNet decode benchmark.
 *
 * Compares the exact serial provider with the head-parallel runtime wrapper.
 */

#include "ck_threadpool.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern void gated_deltanet_llama_avx2_forward(
    const float *, const float *, const float *, const float *, const float *,
    const float *, float *, float *, int, int, int, float);
extern void gated_deltanet_llama_avx2_parallel_forward(
    const float *, const float *, const float *, const float *, const float *,
    const float *, float *, float *, int, int, int, float);

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static float fixture(size_t i, uint32_t salt)
{
    uint32_t x = (uint32_t)i ^ salt;
    x = x * 1664525u + 1013904223u;
    return ((float)(x & 0xffffu) / 32768.0f - 1.0f) * 0.08f;
}

typedef void (*decode_fn)(
    const float *, const float *, const float *, const float *, const float *,
    const float *, float *, float *, int, int, int, float);

static double bench(
    decode_fn fn,
    const float *q, const float *k, const float *v,
    const float *g, const float *beta, const float *state,
    float *state_out, float *out,
    int heads, int groups, int dim, int warmup, int iters)
{
    for (int i = 0; i < warmup; ++i) {
        fn(q, k, v, g, beta, state, state_out, out,
           heads, groups, dim, 1e-6f);
    }
    const double start = now_ms();
    for (int i = 0; i < iters; ++i) {
        fn(q, k, v, g, beta, state, state_out, out,
           heads, groups, dim, 1e-6f);
    }
    return (now_ms() - start) / (double)iters;
}

int main(int argc, char **argv)
{
    const int quick = argc > 1 && strcmp(argv[1], "--quick") == 0;
    const int heads = 48;
    const int groups = 16;
    const int dim = 128;
    const int warmup = quick ? 2 : 8;
    const int iters = quick ? 8 : 40;
    const size_t qk_count = (size_t)groups * (size_t)dim;
    const size_t value_count = (size_t)heads * (size_t)dim;
    const size_t state_count = value_count * (size_t)dim;

    float *q = malloc(qk_count * sizeof(float));
    float *k = malloc(qk_count * sizeof(float));
    float *v = malloc(value_count * sizeof(float));
    float *g = malloc((size_t)heads * sizeof(float));
    float *beta = malloc((size_t)heads * sizeof(float));
    float *state = malloc(state_count * sizeof(float));
    float *serial_state = malloc(state_count * sizeof(float));
    float *parallel_state = malloc(state_count * sizeof(float));
    float *serial_out = malloc(value_count * sizeof(float));
    float *parallel_out = malloc(value_count * sizeof(float));
    if (!q || !k || !v || !g || !beta || !state || !serial_state ||
        !parallel_state || !serial_out || !parallel_out) {
        fprintf(stderr, "allocation failed\n");
        return 2;
    }
    for (size_t i = 0; i < qk_count; ++i) {
        q[i] = fixture(i, 1u);
        k[i] = fixture(i, 2u);
    }
    for (size_t i = 0; i < value_count; ++i) v[i] = fixture(i, 3u);
    for (int i = 0; i < heads; ++i) {
        g[i] = -0.04f - (float)i * 0.0001f;
        beta[i] = fixture((size_t)i, 4u);
    }
    for (size_t i = 0; i < state_count; ++i) state[i] = fixture(i, 5u);

    gated_deltanet_llama_avx2_forward(
        q, k, v, g, beta, state, serial_state, serial_out,
        heads, groups, dim, 1e-6f);
    gated_deltanet_llama_avx2_parallel_forward(
        q, k, v, g, beta, state, parallel_state, parallel_out,
        heads, groups, dim, 1e-6f);
    if (memcmp(serial_out, parallel_out, value_count * sizeof(float)) != 0 ||
        memcmp(serial_state, parallel_state, state_count * sizeof(float)) != 0) {
        fprintf(stderr, "parallel provider is not bit-exact to serial\n");
        return 3;
    }

    const double serial_ms = bench(
        gated_deltanet_llama_avx2_forward,
        q, k, v, g, beta, state, serial_state, serial_out,
        heads, groups, dim, warmup, iters);
    const double parallel_ms = bench(
        gated_deltanet_llama_avx2_parallel_forward,
        q, k, v, g, beta, state, parallel_state, parallel_out,
        heads, groups, dim, warmup, iters);
    printf(
        "qwen36_grouped_decode heads=%d groups=%d dim=%d threads=%d "
        "serial_ms=%.4f parallel_ms=%.4f speedup=%.3fx exact=true\n",
        heads, groups, dim,
        ck_threadpool_n_threads(ck_threadpool_global()),
        serial_ms, parallel_ms, serial_ms / parallel_ms);

    free(q);
    free(k);
    free(v);
    free(g);
    free(beta);
    free(state);
    free(serial_state);
    free(parallel_state);
    free(serial_out);
    free(parallel_out);
    ck_threadpool_global_destroy();
    return 0;
}
