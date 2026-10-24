#include "ggml.h"
#include "ggml-cpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int read_exact(const char *path, void *dst, size_t bytes) {
    FILE *file = fopen(path, "rb");
    if (!file) return 0;
    const int ok = fread(dst, 1, bytes, file) == bytes && fgetc(file) == EOF;
    fclose(file);
    return ok;
}

static int parse_int(const char *text, int *value) {
    char *end = NULL;
    const long parsed = strtol(text, &end, 10);
    if (end == text || *end || parsed < 1 || parsed > 32768) return 0;
    *value = (int) parsed;
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 12) {
        fprintf(stderr, "usage: replay Q.f32 K.f16 V.f16 OUT.f32 queries keys heads kv_heads dim scale threads\n");
        return 2;
    }
    int queries, keys, heads, kv_heads, dim, threads;
    if (!parse_int(argv[5], &queries) || !parse_int(argv[6], &keys) ||
        !parse_int(argv[7], &heads) || !parse_int(argv[8], &kv_heads) ||
        !parse_int(argv[9], &dim) || !parse_int(argv[11], &threads) ||
        queries > 32768 || heads > 128 || kv_heads > heads || heads % kv_heads ||
        dim > 256 || threads > 256) {
        fprintf(stderr, "unsupported flash attention geometry\n");
        return 2;
    }
    char *end = NULL;
    const float scale = strtof(argv[10], &end);
    if (end == argv[10] || *end || !isfinite(scale) || scale <= 0.0f) {
        fprintf(stderr, "invalid flash attention scale\n");
        return 2;
    }
    const size_t q_bytes = (size_t) queries * heads * dim * sizeof(float);
    const size_t kv_bytes = (size_t) keys * kv_heads * dim * sizeof(ggml_fp16_t);
    if (q_bytes + 2 * kv_bytes > 256u * 1024u * 1024u) {
        fprintf(stderr, "flash attention operands exceed replay capacity\n");
        return 2;
    }

    ggml_cpu_init();
    struct ggml_init_params params = {
        .mem_size = q_bytes + 2 * kv_bytes + 64u * 1024u * 1024u,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    struct ggml_context *ctx = ggml_init(params);
    if (!ctx) return 3;
    struct ggml_tensor *q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, heads, queries);
    struct ggml_tensor *k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, dim, kv_heads, keys);
    struct ggml_tensor *v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, dim, kv_heads, keys);
    if (!q || !k || !v || !read_exact(argv[1], q->data, q_bytes) ||
        !read_exact(argv[2], k->data, kv_bytes) || !read_exact(argv[3], v->data, kv_bytes)) {
        fprintf(stderr, "missing or incorrectly sized flash attention operand\n");
        ggml_free(ctx);
        return 2;
    }

    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);
    struct ggml_tensor *result = ggml_flash_attn_ext(ctx, q, k, v, NULL, scale, 0.0f, 0.0f);
    ggml_prec_set_acc(result, GGML_PREC_F32);
    result = ggml_cont_2d(ctx, result, dim * heads, queries);
    struct ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    if (ggml_graph_compute_with_ctx(ctx, graph, threads) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "GGML flash attention graph failed\n");
        ggml_free(ctx);
        return 3;
    }

    FILE *file = fopen(argv[4], "wb");
    int ok = file && fwrite(result->data, 1, q_bytes, file) == q_bytes;
    if (file && fclose(file) != 0) ok = 0;
    if (!ok) fprintf(stderr, "cannot write complete flash attention result\n");
    ggml_free(ctx);
    return ok ? 0 : 3;
}
