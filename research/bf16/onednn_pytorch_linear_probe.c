#include <dnnl.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static dnnl_engine_t engine;
static dnnl_stream_t stream;

static int init_runtime(void)
{
    if (engine) return 0;
    if (dnnl_engine_create(&engine, dnnl_cpu, 0) != dnnl_success) return -1;
    if (dnnl_stream_create(&stream, engine, dnnl_stream_default_flags) != dnnl_success) return -1;
    return 0;
}

int onednn_linear_bf16(const uint16_t *src,
                       const uint16_t *weights_nk,
                       const uint16_t *bias,
                       uint16_t *dst,
                       int m, int n, int k)
{
    if (!src || !weights_nk || !dst || m <= 0 || n <= 0 || k <= 0) return -1;
    if (init_runtime() != 0) return -1;

    dnnl_memory_desc_t src_md = NULL, weights_md = NULL, dst_md = NULL;
    dnnl_primitive_attr_t attr = NULL;
    dnnl_post_ops_t post_ops = NULL;
    dnnl_primitive_desc_t pd = NULL;
    dnnl_primitive_t primitive = NULL;
    dnnl_memory_t src_mem = NULL, weights_mem = NULL, dst_mem = NULL;
    dnnl_dims_t src_dims = {m, k};
    dnnl_dims_t weights_dims = {k, n};
    dnnl_dims_t dst_dims = {m, n};
    dnnl_dims_t src_strides = {k, 1};
    dnnl_dims_t weights_strides = {1, k};
    dnnl_dims_t dst_strides = {n, 1};
    int rc = -1;

#define CHECK(call) do { if ((call) != dnnl_success) { rc = -__LINE__; goto cleanup; } } while (0)
    CHECK(dnnl_memory_desc_create_with_strides(&src_md, 2, src_dims, dnnl_bf16, src_strides));
    CHECK(dnnl_memory_desc_create_with_strides(
        &weights_md, 2, weights_dims, dnnl_bf16, weights_strides));
    CHECK(dnnl_memory_desc_create_with_strides(&dst_md, 2, dst_dims, dnnl_bf16, dst_strides));
    if (bias) {
        for (int i = 0; i < m; ++i) {
            memcpy(dst + (size_t)i * (size_t)n, bias, (size_t)n * sizeof(*dst));
        }
        CHECK(dnnl_primitive_attr_create(&attr));
        CHECK(dnnl_post_ops_create(&post_ops));
        CHECK(dnnl_post_ops_append_sum(post_ops, 1.0f, 0, dnnl_bf16));
        CHECK(dnnl_primitive_attr_set_post_ops(attr, post_ops));
    }
    CHECK(dnnl_matmul_primitive_desc_create(
        &pd, engine, src_md, weights_md, NULL, dst_md, attr));
    CHECK(dnnl_primitive_create(&primitive, pd));
    CHECK(dnnl_memory_create(&src_mem, src_md, engine, (void *)src));
    CHECK(dnnl_memory_create(&weights_mem, weights_md, engine, (void *)weights_nk));
    CHECK(dnnl_memory_create(&dst_mem, dst_md, engine, dst));

    dnnl_exec_arg_t args[] = {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_WEIGHTS, weights_mem},
        {DNNL_ARG_DST, dst_mem},
    };
    CHECK(dnnl_primitive_execute(
        primitive, stream, (int)(sizeof(args) / sizeof(args[0])), args));
    CHECK(dnnl_stream_wait(stream));
    rc = 0;

cleanup:
    if (dst_mem) dnnl_memory_destroy(dst_mem);
    if (weights_mem) dnnl_memory_destroy(weights_mem);
    if (src_mem) dnnl_memory_destroy(src_mem);
    if (primitive) dnnl_primitive_destroy(primitive);
    if (pd) dnnl_primitive_desc_destroy(pd);
    if (dst_md) dnnl_memory_desc_destroy(dst_md);
    if (weights_md) dnnl_memory_desc_destroy(weights_md);
    if (src_md) dnnl_memory_desc_destroy(src_md);
    if (post_ops) dnnl_post_ops_destroy(post_ops);
    if (attr) dnnl_primitive_attr_destroy(attr);
    return rc;
#undef CHECK
}
