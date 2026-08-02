/*
 * Isolate Q4_K/Q6_K unpacking, dot-product, and heterogeneous scheduling cost.
 *
 * Threaded cases use CKE's global persistent threadpool. The dynamic case
 * changes only job assignment inside the callback; it creates no second pool.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "ck_parity_api.h"
#include "ck_threadpool.h"
#include "ckernel_quant.h"

#include <dlfcn.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CACHE_LINE 64

typedef void (*llama_dequant_fn)(const void *, float *, int);
typedef void (*llama_dot_fn)(const void *, const void *, float *, int);
typedef void (*llama_vec_dot_fn)(int, float *, size_t, const void *, size_t,
                                 const void *, size_t, int);

typedef enum { QUANT_Q4, QUANT_Q6 } quant_kind_t;

typedef struct {
    _Alignas(CACHE_LINE) uint64_t elapsed_ns;
    uint64_t finished_ns;
    int jobs;
    int cpu_start;
    int cpu_end;
    char pad[CACHE_LINE - 2 * sizeof(uint64_t) - 3 * sizeof(int)];
} worker_stat_t;

typedef struct {
    quant_kind_t kind;
    const void *weight;
    const void *activation;
    float *output;
    int jobs;
    int k;
    int dynamic;
    int chunk;
    atomic_int next_job;
    worker_stat_t stats[CK_THREADPOOL_MAX_THREADS];
} pool_work_t;

static volatile float checksum_sink;
static uint32_t rng_state = UINT32_C(0x12345678);
static llama_vec_dot_fn llama_vec_q4;
static llama_vec_dot_fn llama_vec_q6;

extern void gemv_q4_k_q8_k_avx2(float *, const void *, const void *, int, int);
extern void gemv_q6_k_q8_k_avx2(float *, const void *, const void *, int, int);

static void cke_dot_q4_direct(const void *weight, const void *activation,
                              float *output, int k)
{
    gemv_q4_k_q8_k_avx2(output, weight, activation, 1, k);
}

static void cke_dot_q6_direct(const void *weight, const void *activation,
                              float *output, int k)
{
    gemv_q6_k_q8_k_avx2(output, weight, activation, 1, k);
}

static void llama_dot_q4_direct(const void *weight, const void *activation,
                                float *output, int k)
{
    *output = 0.0f;
    llama_vec_q4(k, output, sizeof(*output), weight, sizeof(block_q4_K),
                 activation, sizeof(block_q8_K), 1);
}

static void llama_dot_q6_direct(const void *weight, const void *activation,
                                float *output, int k)
{
    *output = 0.0f;
    llama_vec_q6(k, output, sizeof(*output), weight, sizeof(block_q6_K),
                 activation, sizeof(block_q8_K), 1);
}

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}

static void *aligned_calloc(size_t alignment, size_t size)
{
    void *ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) return NULL;
    memset(ptr, 0, size);
    return ptr;
}

static uint32_t rng_u32(void)
{
    rng_state = rng_state * UINT32_C(1664525) + UINT32_C(1013904223);
    return rng_state;
}

static void fill_q4(block_q4_K *blocks, int count)
{
    for (int b = 0; b < count; ++b) {
        blocks[b].d = GGML_FP32_TO_FP16(0.03125f);
        blocks[b].dmin = GGML_FP32_TO_FP16(0.00390625f);
        for (size_t i = 0; i < sizeof(blocks[b].scales); ++i) {
            blocks[b].scales[i] = (uint8_t)(rng_u32() & 0x3fu);
        }
        for (size_t i = 0; i < sizeof(blocks[b].qs); ++i) {
            blocks[b].qs[i] = (uint8_t)rng_u32();
        }
    }
}

static void fill_q6(block_q6_K *blocks, int count)
{
    for (int b = 0; b < count; ++b) {
        blocks[b].d = GGML_FP32_TO_FP16(0.03125f);
        for (size_t i = 0; i < sizeof(blocks[b].ql); ++i) {
            blocks[b].ql[i] = (uint8_t)rng_u32();
        }
        for (size_t i = 0; i < sizeof(blocks[b].qh); ++i) {
            blocks[b].qh[i] = (uint8_t)rng_u32();
        }
        for (size_t i = 0; i < sizeof(blocks[b].scales); ++i) {
            blocks[b].scales[i] = (int8_t)((int)(rng_u32() % 31u) - 15);
        }
    }
}

static void fill_q8(block_q8_K *blocks, int count)
{
    for (int b = 0; b < count; ++b) {
        int sums[QK_K / 16] = {0};
        blocks[b].d = 0.015625f;
        for (int i = 0; i < QK_K; ++i) {
            const int value = (int)(rng_u32() % 31u) - 15;
            blocks[b].qs[i] = (int8_t)value;
            sums[i / 16] += value;
        }
        for (int i = 0; i < QK_K / 16; ++i) {
            blocks[b].bsums[i] = (int16_t)sums[i];
        }
    }
}

static void *load_llama_shim(void)
{
    const char *explicit_path = getenv("CK_LLAMA_KERNEL_TEST_LIB");
    const char *paths[] = {explicit_path, "./llama.cpp/libggml_kernel_test.so", NULL};
    for (int i = 0; paths[i] != NULL || i == 0; ++i) {
        if (!paths[i] || !paths[i][0]) continue;
        void *handle = dlopen(paths[i], RTLD_NOW | RTLD_LOCAL);
        if (handle) return handle;
    }
    return NULL;
}

static double bench_dequant(const char *name,
                            void (*fn)(const void *, float *, int),
                            const void *weight,
                            float *output,
                            int k,
                            int warmup,
                            int iters)
{
    for (int i = 0; i < warmup; ++i) fn(weight, output, k);
    const uint64_t start = now_ns();
    for (int i = 0; i < iters; ++i) fn(weight, output, k);
    const uint64_t elapsed = now_ns() - start;
    checksum_sink += output[(iters * 131) % k];
    const double ns_call = (double)elapsed / (double)iters;
    printf("leaf stage=dequant provider=%s ns_call=%.2f ns_block=%.2f blocks_per_s=%.3fM\n",
           name, ns_call, ns_call / (double)(k / QK_K),
           (double)(k / QK_K) * 1000.0 / ns_call);
    return ns_call;
}

static double bench_dot(const char *name,
                        void (*fn)(const void *, const void *, float *, int),
                        const void *weight,
                        const void *activation,
                        int k,
                        int warmup,
                        int iters,
                        float *last)
{
    float output = 0.0f;
    for (int i = 0; i < warmup; ++i) fn(weight, activation, &output, k);
    const uint64_t start = now_ns();
    for (int i = 0; i < iters; ++i) fn(weight, activation, &output, k);
    const uint64_t elapsed = now_ns() - start;
    checksum_sink += output;
    *last = output;
    const double ns_call = (double)elapsed / (double)iters;
    printf("leaf stage=dot provider=%s ns_call=%.2f ns_block=%.2f blocks_per_s=%.3fM\n",
           name, ns_call, ns_call / (double)(k / QK_K),
           (double)(k / QK_K) * 1000.0 / ns_call);
    return ns_call;
}

static void run_leaf_job(pool_work_t *work, int job)
{
    float value = 0.0f;
    if (work->kind == QUANT_Q4) {
        ck_test_vec_dot_q4_k_q8_k(work->weight, work->activation, &value, work->k);
    } else {
        ck_test_vec_dot_q6_k_q8_k(work->weight, work->activation, &value, work->k);
    }
    work->output[job] = value;
}

static void pool_worker(int ith, int nth, void *opaque)
{
    pool_work_t *work = (pool_work_t *)opaque;
    worker_stat_t *stat = &work->stats[ith];
    stat->cpu_start = sched_getcpu();
    const uint64_t start = now_ns();
    int completed = 0;

    if (work->dynamic) {
        for (;;) {
            const int begin = atomic_fetch_add_explicit(
                &work->next_job, work->chunk, memory_order_relaxed);
            if (begin >= work->jobs) break;
            int end = begin + work->chunk;
            if (end > work->jobs) end = work->jobs;
            for (int job = begin; job < end; ++job) {
                run_leaf_job(work, job);
                ++completed;
            }
        }
    } else {
        for (int job = ith; job < work->jobs; job += nth) {
            run_leaf_job(work, job);
            ++completed;
        }
    }

    stat->elapsed_ns = now_ns() - start;
    stat->finished_ns = now_ns();
    stat->jobs = completed;
    stat->cpu_end = sched_getcpu();
}

static double bench_pool(quant_kind_t kind,
                         const char *kind_name,
                         const void *weight,
                         const void *activation,
                         int k,
                         int jobs,
                         int dynamic,
                         int chunk,
                         int iters,
                         float *reference)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    const int threads = pool ? ck_threadpool_n_threads(pool) : 1;
    float *output = calloc((size_t)jobs, sizeof(*output));
    if (!output) return -1.0;
    pool_work_t work = {
        .kind = kind, .weight = weight, .activation = activation,
        .output = output, .jobs = jobs, .k = k, .dynamic = dynamic, .chunk = chunk,
    };
    worker_stat_t best_stats[CK_THREADPOOL_MAX_THREADS] = {{0}};
    uint64_t best_dispatch_end = 0;

    uint64_t best = UINT64_MAX;
    for (int iteration = 0; iteration < iters + 1; ++iteration) {
        memset(work.stats, 0, sizeof(work.stats));
        atomic_store_explicit(&work.next_job, 0, memory_order_relaxed);
        const uint64_t start = now_ns();
        ck_threadpool_dispatch(pool, pool_worker, &work);
        const uint64_t dispatch_end = now_ns();
        const uint64_t elapsed = dispatch_end - start;
        if (iteration > 0 && elapsed < best) {
            best = elapsed;
            best_dispatch_end = dispatch_end;
            memcpy(best_stats, work.stats, sizeof(best_stats));
        }
    }

    uint64_t min_worker = UINT64_MAX;
    uint64_t max_worker = 0;
    int total_jobs = 0;
    for (int i = 0; i < threads; ++i) {
        const worker_stat_t *stat = &best_stats[i];
        if (stat->elapsed_ns < min_worker) min_worker = stat->elapsed_ns;
        if (stat->elapsed_ns > max_worker) max_worker = stat->elapsed_ns;
        total_jobs += stat->jobs;
        const uint64_t barrier_wait = best_dispatch_end > stat->finished_ns
            ? best_dispatch_end - stat->finished_ns : 0;
        printf("worker quant=%s schedule=%s ith=%d cpu=%d->%d jobs=%d "
               "active_ms=%.3f barrier_wait_ms=%.3f\n",
               kind_name, dynamic ? "dynamic" : "static", i,
               stat->cpu_start, stat->cpu_end, stat->jobs,
               (double)stat->elapsed_ns / 1.0e6, (double)barrier_wait / 1.0e6);
    }

    int exact = total_jobs == jobs;
    for (int i = 0; i < jobs && exact; ++i) {
        exact = memcmp(&output[i], reference, sizeof(float)) == 0;
    }
    checksum_sink += output[jobs / 2];
    printf("pool quant=%s schedule=%s threads=%d jobs=%d chunk=%d best_ms=%.3f "
           "jobs_per_s=%.3fM worker_spread_ms=%.3f exact=%s\n",
           kind_name, dynamic ? "dynamic" : "static", threads, jobs,
           dynamic ? chunk : 0, (double)best / 1.0e6,
           (double)jobs * 1000.0 / (double)best,
           (double)(max_worker - min_worker) / 1.0e6,
           exact ? "PASS" : "FAIL");
    free(output);
    return exact ? (double)best / 1.0e6 : -1.0;
}

static int parse_int_arg(int argc, char **argv, const char *name, int fallback)
{
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], name) == 0) return atoi(argv[i + 1]);
    }
    return fallback;
}

static const char *parse_string_arg(int argc, char **argv,
                                    const char *name, const char *fallback)
{
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], name) == 0) return argv[i + 1];
    }
    return fallback;
}

static int selected(const char *only, const char *name)
{
    return strcmp(only, "all") == 0 || strcmp(only, name) == 0;
}

int main(int argc, char **argv)
{
    const int k = parse_int_arg(argc, argv, "--k", 4096);
    const int leaf_iters = parse_int_arg(argc, argv, "--leaf-iters", 20000);
    const int jobs = parse_int_arg(argc, argv, "--jobs", 4096);
    const int pool_iters = parse_int_arg(argc, argv, "--pool-iters", 3);
    const int chunk = parse_int_arg(argc, argv, "--chunk", 4);
    const char *only = parse_string_arg(argc, argv, "--only", "all");
    const char *schedule = parse_string_arg(argc, argv, "--schedule", "both");
    const int run_static = strcmp(schedule, "both") == 0 || strcmp(schedule, "static") == 0;
    const int run_dynamic = strcmp(schedule, "both") == 0 || strcmp(schedule, "dynamic") == 0;
    if (k <= 0 || k % QK_K != 0 || leaf_iters <= 0 || jobs <= 0 || pool_iters <= 0) {
        fprintf(stderr, "invalid arguments: K must be positive and divisible by %d\n", QK_K);
        return 2;
    }
    if (!run_static && !run_dynamic) {
        fprintf(stderr, "invalid --schedule: expected static, dynamic, or both\n");
        return 2;
    }

    const int blocks = k / QK_K;
    block_q4_K *q4 = aligned_calloc(CACHE_LINE, (size_t)blocks * sizeof(*q4));
    block_q6_K *q6 = aligned_calloc(CACHE_LINE, (size_t)blocks * sizeof(*q6));
    block_q8_K *q8 = aligned_calloc(CACHE_LINE, (size_t)blocks * sizeof(*q8));
    float *ck_out = aligned_calloc(CACHE_LINE, (size_t)k * sizeof(*ck_out));
    float *llama_out = aligned_calloc(CACHE_LINE, (size_t)k * sizeof(*llama_out));
    if (!q4 || !q6 || !q8 || !ck_out || !llama_out) {
        fprintf(stderr, "allocation failed\n");
        return 2;
    }
    fill_q4(q4, blocks);
    fill_q6(q6, blocks);
    fill_q8(q8, blocks);

    void *llama = load_llama_shim();
    void *llama_cpu = NULL;
    const char *llama_cpu_path = getenv("CK_LLAMA_GGML_CPU_LIB");
    if (llama_cpu_path && llama_cpu_path[0]) {
        llama_cpu = dlopen(llama_cpu_path, RTLD_NOW | RTLD_LOCAL);
    }
    llama_dequant_fn llama_dequant_q4 = NULL;
    llama_dequant_fn llama_dequant_q6 = NULL;
    llama_dot_fn llama_dot_q4 = NULL;
    llama_dot_fn llama_dot_q6 = NULL;
    if (llama) {
        void *cpu_symbols = llama_cpu ? llama_cpu : llama;
        void (*llama_cpu_init)(void) = (void (*)(void))dlsym(cpu_symbols, "ggml_cpu_init");
        if (llama_cpu_init) llama_cpu_init();
        llama_dequant_q4 = (llama_dequant_fn)dlsym(llama, "test_dequant_q4_k");
        llama_dequant_q6 = (llama_dequant_fn)dlsym(llama, "test_dequant_q6_k");
        llama_vec_q4 = (llama_vec_dot_fn)dlsym(cpu_symbols, "ggml_vec_dot_q4_K_q8_K");
        llama_vec_q6 = (llama_vec_dot_fn)dlsym(cpu_symbols, "ggml_vec_dot_q6_K_q8_K");
        if (llama_vec_q4) llama_dot_q4 = llama_dot_q4_direct;
        if (llama_vec_q6) llama_dot_q6 = llama_dot_q6_direct;
    }

    ck_threadpool_t *pool = ck_threadpool_global();
    printf("config k=%d blocks=%d leaf_iters=%d jobs=%d pool_threads=%d affinity_cpu=%d llama=%s\n",
           k, blocks, leaf_iters, jobs,
           pool ? ck_threadpool_n_threads(pool) : 1, sched_getcpu(),
           llama_dot_q4 && llama_dot_q6 ? "yes" : "no");

    if (selected(only, "cke-q4-dequant")) {
        bench_dequant("cke_q4", ck_test_dequant_q4_k, q4, ck_out, k, 100, leaf_iters);
    }
    if (llama_dequant_q4 && selected(only, "llama-q4-dequant")) {
        llama_dequant_q4(q4, llama_out, k);
        if (strcmp(only, "all") == 0) {
            printf("exact stage=dequant quant=q4 result=%s\n",
                   memcmp(ck_out, llama_out, (size_t)k * sizeof(float)) == 0 ? "PASS" : "FAIL");
        }
        bench_dequant("llama_q4", llama_dequant_q4, q4, llama_out, k, 100, leaf_iters);
    }
    if (selected(only, "cke-q6-dequant")) {
        bench_dequant("cke_q6", ck_test_dequant_q6_k, q6, ck_out, k, 100, leaf_iters);
    }
    if (llama_dequant_q6 && selected(only, "llama-q6-dequant")) {
        llama_dequant_q6(q6, llama_out, k);
        if (strcmp(only, "all") == 0) {
            printf("exact stage=dequant quant=q6 result=%s\n",
                   memcmp(ck_out, llama_out, (size_t)k * sizeof(float)) == 0 ? "PASS" : "FAIL");
        }
        bench_dequant("llama_q6", llama_dequant_q6, q6, llama_out, k, 100, leaf_iters);
    }

    float ck_q4 = 0.0f;
    float ck_q6 = 0.0f;
    if (selected(only, "cke-q4-dot")) {
        bench_dot("cke_q4", ck_test_vec_dot_q4_k_q8_k, q4, q8, k, 100, leaf_iters, &ck_q4);
    }
    if (selected(only, "cke-q4-dot-direct")) {
        bench_dot("cke_q4_direct", cke_dot_q4_direct, q4, q8, k, 100, leaf_iters, &ck_q4);
    }
    if (llama_dot_q4 && selected(only, "llama-q4-dot")) {
        float llama_q4 = 0.0f;
        bench_dot("llama_q4", llama_dot_q4, q4, q8, k, 100, leaf_iters, &llama_q4);
        if (strcmp(only, "all") == 0) {
            printf("exact stage=dot quant=q4 result=%s ck=%a llama=%a\n",
                   memcmp(&ck_q4, &llama_q4, sizeof(float)) == 0 ? "PASS" : "FAIL",
                   ck_q4, llama_q4);
        }
    }
    if (selected(only, "cke-q6-dot")) {
        bench_dot("cke_q6", ck_test_vec_dot_q6_k_q8_k, q6, q8, k, 100, leaf_iters, &ck_q6);
    }
    if (selected(only, "cke-q6-dot-direct")) {
        bench_dot("cke_q6_direct", cke_dot_q6_direct, q6, q8, k, 100, leaf_iters, &ck_q6);
    }
    if (llama_dot_q6 && selected(only, "llama-q6-dot")) {
        float llama_q6 = 0.0f;
        bench_dot("llama_q6", llama_dot_q6, q6, q8, k, 100, leaf_iters, &llama_q6);
        if (strcmp(only, "all") == 0) {
            printf("exact stage=dot quant=q6 result=%s ck=%a llama=%a\n",
                   memcmp(&ck_q6, &llama_q6, sizeof(float)) == 0 ? "PASS" : "FAIL",
                   ck_q6, llama_q6);
        }
    }

    const int run_pool_q4 = strcmp(only, "all") == 0 || strcmp(only, "pool-q4") == 0;
    const int run_pool_q6 = strcmp(only, "all") == 0 || strcmp(only, "pool-q6") == 0;
    if (!run_pool_q4 && !run_pool_q6) {
        ck_threadpool_global_destroy();
        if (llama_cpu) dlclose(llama_cpu);
        if (llama) dlclose(llama);
        free(q4); free(q6); free(q8); free(ck_out); free(llama_out);
        printf("summary only=%s checksum=%g\n", only, checksum_sink);
        return 0;
    }

    double q4_static = 1.0, q4_dynamic = 1.0;
    double q6_static = 1.0, q6_dynamic = 1.0;
    if (run_pool_q4) {
        ck_test_vec_dot_q4_k_q8_k(q4, q8, &ck_q4, k);
        if (run_static) q4_static = bench_pool(
            QUANT_Q4, "q4", q4, q8, k, jobs, 0, chunk, pool_iters, &ck_q4);
        if (run_dynamic) q4_dynamic = bench_pool(
            QUANT_Q4, "q4", q4, q8, k, jobs, 1, chunk, pool_iters, &ck_q4);
        if (run_static && run_dynamic) {
            printf("summary quant=q4 chunk=%d dynamic_speedup=%.4f\n",
                   chunk, q4_static / q4_dynamic);
        }
    }
    if (run_pool_q6) {
        ck_test_vec_dot_q6_k_q8_k(q6, q8, &ck_q6, k);
        if (run_static) q6_static = bench_pool(
            QUANT_Q6, "q6", q6, q8, k, jobs, 0, chunk, pool_iters, &ck_q6);
        if (run_dynamic) q6_dynamic = bench_pool(
            QUANT_Q6, "q6", q6, q8, k, jobs, 1, chunk, pool_iters, &ck_q6);
        if (run_static && run_dynamic) {
            printf("summary quant=q6 chunk=%d dynamic_speedup=%.4f\n",
                   chunk, q6_static / q6_dynamic);
        }
    }
    printf("summary checksum=%g\n", checksum_sink);

    ck_threadpool_global_destroy();
    if (llama_cpu) dlclose(llama_cpu);
    if (llama) dlclose(llama);
    free(q4);
    free(q6);
    free(q8);
    free(ck_out);
    free(llama_out);
    return q4_static > 0.0 && q4_dynamic > 0.0 && q6_static > 0.0 && q6_dynamic > 0.0 ? 0 : 1;
}
