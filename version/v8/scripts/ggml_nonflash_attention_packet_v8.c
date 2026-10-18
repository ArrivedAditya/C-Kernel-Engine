#include "ggml.h"
#include "ggml-cpu.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_exact(const char *path, void *dst, size_t bytes) {
    FILE *file = fopen(path, "rb");
    if (!file) return 0;
    const int ok = fread(dst, 1, bytes, file) == bytes && fgetc(file) == EOF;
    fclose(file);
    return ok;
}

int main(int argc, char **argv) {
    if (argc != 9) {
        fprintf(stderr, "usage: replay Q.f32 K.f16 V.f16 OUT.f32 tokens heads kv_heads dim\n");
        return 2;
    }
    const int tokens = atoi(argv[5]);
    const int heads = atoi(argv[6]);
    const int kv_heads = atoi(argv[7]);
    const int dim = atoi(argv[8]);
    if (tokens < 1 || tokens > 4096 || heads < 1 || heads > 128 ||
        kv_heads < 1 || kv_heads > heads || heads % kv_heads || dim < 1 || dim > 1024) {
        fprintf(stderr, "unsupported attention geometry\n");
        return 2;
    }
    const size_t q_count = (size_t) tokens * heads * dim;
    const size_t kv_count = (size_t) tokens * kv_heads * dim;
    float *q = malloc(q_count * sizeof(*q));
    ggml_fp16_t *k = malloc(kv_count * sizeof(*k));
    ggml_fp16_t *v = malloc(kv_count * sizeof(*v));
    float *out = malloc(q_count * sizeof(*out));
    if (!q || !k || !v || !out || !read_exact(argv[1], q, q_count * sizeof(*q)) ||
        !read_exact(argv[2], k, kv_count * sizeof(*k)) ||
        !read_exact(argv[3], v, kv_count * sizeof(*v))) {
        fprintf(stderr, "missing or incorrectly sized operand\n");
        free(q); free(k); free(v); free(out);
        return 2;
    }

    ggml_cpu_init();
    for (int t = 0; t < tokens; ++t) {
        struct ggml_init_params params = { .mem_size = 64 * 1024 * 1024, .mem_buffer = NULL, .no_alloc = false };
        struct ggml_context *ctx = ggml_init(params);
        if (!ctx) return 3;
        struct ggml_tensor *qt = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, heads, 1);
        struct ggml_tensor *kt = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, dim, kv_heads, t + 1);
        struct ggml_tensor *vt = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, dim, kv_heads, t + 1);
        memcpy(qt->data, q + (size_t) t * heads * dim, (size_t) heads * dim * sizeof(*q));
        memcpy(kt->data, k, (size_t) (t + 1) * kv_heads * dim * sizeof(*k));
        memcpy(vt->data, v, (size_t) (t + 1) * kv_heads * dim * sizeof(*v));

        // Mirror build_attn_mha's non-flash branch with only valid causal keys in view.
        struct ggml_tensor *qp = ggml_permute(ctx, qt, 0, 2, 1, 3);
        struct ggml_tensor *kp = ggml_permute(ctx, kt, 0, 2, 1, 3);
        struct ggml_tensor *vp = ggml_permute(ctx, vt, 0, 2, 1, 3);
        struct ggml_tensor *scores = ggml_mul_mat(ctx, kp, qp);
        ggml_prec_set_acc(scores, GGML_PREC_F32);
        struct ggml_tensor *prob = ggml_soft_max_ext(ctx, scores, NULL, 1.0f, 0.0f);
        struct ggml_tensor *vc = ggml_cont(ctx, ggml_transpose(ctx, vp));
        struct ggml_tensor *value = ggml_mul_mat(ctx, vc, prob);
        struct ggml_tensor *result = ggml_permute(ctx, value, 0, 2, 1, 3);
        result = ggml_cont_2d(ctx, result, dim * heads, 1);
        struct ggml_cgraph *graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, result);
        if (ggml_graph_compute_with_ctx(ctx, graph, 1) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "GGML graph failed at token %d\n", t);
            ggml_free(ctx);
            return 3;
        }
        memcpy(out + (size_t) t * heads * dim, result->data, (size_t) heads * dim * sizeof(*out));
        ggml_free(ctx);
    }
    FILE *file = fopen(argv[4], "wb");
    int ok = file && fwrite(out, sizeof(*out), q_count, file) == q_count;
    if (file && fclose(file) != 0) ok = 0;
    if (!ok) fprintf(stderr, "cannot write complete result\n");
    free(q); free(k); free(v); free(out);
    return ok ? 0 : 3;
}
