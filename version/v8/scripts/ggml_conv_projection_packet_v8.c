#include "ggml.h"
#include "ggml-cpu.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_exact(const char *path, void *buffer, size_t size) {
    FILE *file = fopen(path, "rb");
    if (!file) return 0;
    const int ok = fread(buffer, 1, size, file) == size && fgetc(file) == EOF;
    fclose(file);
    return ok;
}

static int parse_positive(const char *value, int limit) {
    char *end = NULL;
    errno = 0;
    const long parsed = strtol(value, &end, 10);
    return errno || !end || *end || parsed < 1 || parsed > limit ? 0 : (int) parsed;
}

int main(int argc, char **argv) {
    if (argc != 9) {
        fprintf(stderr, "usage: replay WEIGHTS PATCHES OUTPUT f16|f32 rows channels reduction threads\n");
        return 2;
    }
    const int rows = parse_positive(argv[5], 8192);
    const int channels = parse_positive(argv[6], 8192);
    const int reduction = parse_positive(argv[7], 16384);
    const int threads = parse_positive(argv[8], 256);
    const enum ggml_type weight_type = strcmp(argv[4], "f16") == 0 ? GGML_TYPE_F16 : GGML_TYPE_F32;
    if ((strcmp(argv[4], "f16") && strcmp(argv[4], "f32")) ||
        !rows || !channels || !reduction || !threads ||
        (uint64_t) rows * channels > (64u << 20) / sizeof(float) ||
        (uint64_t) channels * reduction > (256u << 20) / sizeof(float) ||
        (uint64_t) rows * reduction > (256u << 20) / sizeof(ggml_fp16_t)) {
        fprintf(stderr, "unsupported projection geometry or storage\n");
        return 2;
    }

    ggml_cpu_init();
    const size_t memory = (size_t) channels * reduction * ggml_type_size(weight_type) +
                          (size_t) rows * reduction * sizeof(ggml_fp16_t) +
                          (size_t) rows * channels * sizeof(float) + (16u << 20);
    struct ggml_init_params params = { .mem_size = memory, .mem_buffer = NULL, .no_alloc = false };
    struct ggml_context *ctx = ggml_init(params);
    if (!ctx) return 3;
    struct ggml_tensor *weights = ggml_new_tensor_2d(ctx, weight_type, reduction, channels);
    struct ggml_tensor *patches = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, reduction, rows);
    // The captured conv graph uses im2col as src[0] and weights as src[1].
    struct ggml_tensor *result = ggml_mul_mat(ctx, patches, weights);
    struct ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);

    const size_t weight_bytes = ggml_nbytes(weights);
    const size_t patch_bytes = ggml_nbytes(patches);
    const size_t result_bytes = ggml_nbytes(result);
    if (!read_exact(argv[1], weights->data, weight_bytes) ||
        !read_exact(argv[2], patches->data, patch_bytes)) {
        fprintf(stderr, "missing or incorrectly sized projection operands\n");
        ggml_free(ctx);
        return 2;
    }
    if (ggml_graph_compute_with_ctx(ctx, graph, threads) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "GGML projection replay failed\n");
        ggml_free(ctx);
        return 3;
    }
    FILE *file = fopen(argv[3], "wb");
    const int ok = file && fwrite(result->data, 1, result_bytes, file) == result_bytes;
    if (file && fclose(file) != 0) {
        ggml_free(ctx);
        return 3;
    }
    if (!ok) fprintf(stderr, "cannot write complete projection result\n");
    ggml_free(ctx);
    return ok ? 0 : 3;
}
