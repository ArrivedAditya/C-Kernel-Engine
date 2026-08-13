/**
 * @file ssm_kernels.c
 * @brief FP32 SSM causal depthwise convolution kernels for qwen3next/Qwen3.5.
 *
 * CK-ENGINE KERNEL RULES:
 * =======================
 * 1. NO malloc/free - memory via bump allocator, pointers passed in
 * 2. NO OpenMP - parallelization at orchestrator/codegen layer
 * 3. API must define: inputs, outputs, workspace, and memory layouts
 * 4. Pure computation - deterministic, no side effects
 *
 * After changes: make test-ssm-conv && make test-kernels
 *
 * This file implements the GGML_OP_SSM_CONV semantics used by qwen3next before
 * the recurrent DeltaNet update:
 *   out[seq, token, ch] = dot(conv_x[seq, ch, token:token+kernel], kernel[ch, :])
 *
 * Memory layouts:
 *   conv_x   : [num_seqs, num_channels, kernel_size - 1 + num_tokens]
 *   kernel   : [num_channels, kernel_size]
 *   out      : [num_seqs, num_tokens, num_channels]
 *   d_out    : same as out
 *   d_conv_x : same as conv_x
 *   d_kernel : same as kernel
 */

#include "bf16_utils.h"
#include "ckernel_engine.h"
#include "ck_threadpool.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#if defined(CK_TARGET_X86)
#include <immintrin.h>
#endif

void ssm_conv1d_forward_ref(const float *conv_x,
                            const float *kernel,
                            float *out,
                            int kernel_size,
                            int num_channels,
                            int num_tokens,
                            int num_seqs)
{
    if (!conv_x || !kernel || !out) {
        return;
    }
    if (kernel_size <= 0 || num_channels <= 0 || num_tokens < 0 || num_seqs <= 0) {
        return;
    }

    const size_t seq_width = (size_t)kernel_size - 1u + (size_t)num_tokens;
    const size_t conv_seq_stride = (size_t)num_channels * seq_width;
    const size_t out_seq_stride = (size_t)num_tokens * (size_t)num_channels;

    for (int seq = 0; seq < num_seqs; ++seq) {
        const float *conv_seq = conv_x + (size_t)seq * conv_seq_stride;
        float *out_seq = out + (size_t)seq * out_seq_stride;

        for (int tok = 0; tok < num_tokens; ++tok) {
            float *out_tok = out_seq + (size_t)tok * (size_t)num_channels;

            for (int ch = 0; ch < num_channels; ++ch) {
                const float *conv_row = conv_seq + (size_t)ch * seq_width + (size_t)tok;
                const float *kernel_row = kernel + (size_t)ch * (size_t)kernel_size;
                float sumf = 0.0f;
                for (int k = 0; k < kernel_size; ++k) {
                    sumf += conv_row[k] * kernel_row[k];
                }
                out_tok[ch] = sumf;
            }
        }
    }
}

void ssm_conv1d_backward_ref(const float *d_out,
                             const float *conv_x,
                             const float *kernel,
                             float *d_conv_x,
                             float *d_kernel,
                             int kernel_size,
                             int num_channels,
                             int num_tokens,
                             int num_seqs)
{
    if (!d_out || !conv_x || !kernel || !d_conv_x || !d_kernel) {
        return;
    }
    if (kernel_size <= 0 || num_channels <= 0 || num_tokens < 0 || num_seqs <= 0) {
        return;
    }

    const size_t seq_width = (size_t)kernel_size - 1u + (size_t)num_tokens;
    const size_t conv_total = (size_t)num_seqs * (size_t)num_channels * seq_width;
    const size_t kernel_total = (size_t)num_channels * (size_t)kernel_size;
    const size_t conv_seq_stride = (size_t)num_channels * seq_width;
    const size_t out_seq_stride = (size_t)num_tokens * (size_t)num_channels;

    memset(d_conv_x, 0, conv_total * sizeof(float));
    memset(d_kernel, 0, kernel_total * sizeof(float));

    for (int seq = 0; seq < num_seqs; ++seq) {
        const float *d_out_seq = d_out + (size_t)seq * out_seq_stride;
        const float *conv_seq = conv_x + (size_t)seq * conv_seq_stride;
        float *d_conv_seq = d_conv_x + (size_t)seq * conv_seq_stride;

        for (int tok = 0; tok < num_tokens; ++tok) {
            const float *d_out_tok = d_out_seq + (size_t)tok * (size_t)num_channels;

            for (int ch = 0; ch < num_channels; ++ch) {
                const float grad = d_out_tok[ch];
                const float *conv_row = conv_seq + (size_t)ch * seq_width + (size_t)tok;
                float *d_conv_row = d_conv_seq + (size_t)ch * seq_width + (size_t)tok;
                const float *kernel_row = kernel + (size_t)ch * (size_t)kernel_size;
                float *d_kernel_row = d_kernel + (size_t)ch * (size_t)kernel_size;

                for (int k = 0; k < kernel_size; ++k) {
                    d_kernel_row[k] += grad * conv_row[k];
                    d_conv_row[k] += grad * kernel_row[k];
                }
            }
        }
    }
}

void ssm_conv1d_forward(const float *conv_x,
                        const float *kernel,
                        float *out,
                        int kernel_size,
                        int num_channels,
                        int num_tokens,
                        int num_seqs)
{
    ssm_conv1d_forward_ref(conv_x, kernel, out, kernel_size, num_channels, num_tokens, num_seqs);
}

/*
 * llama.cpp's production GGML_OP_SSM_CONV CPU implementation accumulates
 * every channel/token dot product as an ascending scalar multiply/add chain.
 * Preserve the rounded product before each addition: contraction or
 * vectorization changes FP32 rounding boundaries.
 */
typedef struct {
    const float *conv_x;
    const float *kernel;
    float *out;
    int kernel_size;
    int num_channels;
    int num_tokens;
    int num_seqs;
} ck_ssm_conv1d_llama_args_t;

static void ck_ssm_conv1d_llama_channel_range(int begin, int end, void *opaque)
{
    const ck_ssm_conv1d_llama_args_t *args =
        (const ck_ssm_conv1d_llama_args_t *)opaque;
    const int kernel_size = args->kernel_size;
    const int num_channels = args->num_channels;
    const int num_tokens = args->num_tokens;
    const size_t seq_width = (size_t)kernel_size - 1u + (size_t)num_tokens;
    const size_t conv_seq_stride = (size_t)num_channels * seq_width;
    const size_t out_seq_stride = (size_t)num_tokens * (size_t)num_channels;

    for (int seq = 0; seq < args->num_seqs; ++seq) {
        const float *conv_seq = args->conv_x + (size_t)seq * conv_seq_stride;
        float *out_seq = args->out + (size_t)seq * out_seq_stride;

        for (int ch = begin; ch < end; ++ch) {
            const float *kernel_row =
                args->kernel + (size_t)ch * (size_t)kernel_size;
            const float *conv_channel = conv_seq + (size_t)ch * seq_width;
            for (int tok = 0; tok < num_tokens; ++tok) {
                const float *conv_row =
                    conv_channel + (size_t)tok;
                float sum = 0.0f;
                for (int k = 0; k < kernel_size; ++k) {
                    volatile float product = conv_row[k] * kernel_row[k];
                    volatile float next = sum + product;
                    sum = next;
                }
                out_seq[(size_t)tok * (size_t)num_channels + (size_t)ch] = sum;
            }
        }
    }
}

void ssm_conv1d_forward_llama_production_serial(const float *conv_x,
                                                const float *kernel,
                                                float *out,
                                                int kernel_size,
                                                int num_channels,
                                                int num_tokens,
                                                int num_seqs)
{
    if (!conv_x || !kernel || !out || kernel_size <= 0 || num_channels <= 0 ||
        num_tokens < 0 || num_seqs <= 0) {
        return;
    }
    ck_ssm_conv1d_llama_args_t args = {
        conv_x, kernel, out, kernel_size, num_channels, num_tokens, num_seqs,
    };
    ck_ssm_conv1d_llama_channel_range(0, num_channels, &args);
}

void ssm_conv1d_forward_llama_production(const float *conv_x,
                                         const float *kernel,
                                         float *out,
                                         int kernel_size,
                                         int num_channels,
                                         int num_tokens,
                                         int num_seqs)
{
    if (!conv_x || !kernel || !out || kernel_size <= 0 || num_channels <= 0 ||
        num_tokens < 0 || num_seqs <= 0) {
        return;
    }
    ck_ssm_conv1d_llama_args_t args = {
        conv_x, kernel, out, kernel_size, num_channels, num_tokens, num_seqs,
    };
    ck_threadpool_t *pool = ck_threadpool_global();
    const int workers = pool ? ck_threadpool_n_threads(pool) : 1;
    const int active = workers < num_channels ? workers : num_channels;
    if (active > 1 && num_tokens > 1) {
        ck_threadpool_parallel_for_n(
            pool, active, 0, num_channels, 32,
            ck_ssm_conv1d_llama_channel_range, &args);
    } else {
        ck_ssm_conv1d_llama_channel_range(0, num_channels, &args);
    }
}

/*
 * Explicit contracted counterpart to the separated multiply/add provider.
 * Some llama.cpp x86 builds compile GGML_OP_SSM_CONV's source expression to
 * this arithmetic. Keep it distinct so flags cannot silently change the
 * numerical contract.
 */
static void ck_ssm_conv1d_llama_fma_channel_range(
    int begin, int end, void *opaque)
{
    const ck_ssm_conv1d_llama_args_t *args =
        (const ck_ssm_conv1d_llama_args_t *)opaque;
    const int kernel_size = args->kernel_size;
    const int num_channels = args->num_channels;
    const int num_tokens = args->num_tokens;
    const size_t seq_width = (size_t)kernel_size - 1u + (size_t)num_tokens;
    const size_t conv_seq_stride = (size_t)num_channels * seq_width;
    const size_t out_seq_stride = (size_t)num_tokens * (size_t)num_channels;

    for (int seq = 0; seq < args->num_seqs; ++seq) {
        const float *conv_seq = args->conv_x + (size_t)seq * conv_seq_stride;
        float *out_seq = args->out + (size_t)seq * out_seq_stride;

        for (int ch = begin; ch < end; ++ch) {
            const float *kernel_row =
                args->kernel + (size_t)ch * (size_t)kernel_size;
            const float *conv_channel = conv_seq + (size_t)ch * seq_width;
            for (int tok = 0; tok < num_tokens; ++tok) {
                const float *conv_row = conv_channel + (size_t)tok;
                float sum = 0.0f;
                for (int k = 0; k < kernel_size; ++k) {
                    sum = fmaf(conv_row[k], kernel_row[k], sum);
                }
                out_seq[(size_t)tok * (size_t)num_channels + (size_t)ch] = sum;
            }
        }
    }
}

void ssm_conv1d_forward_llama_fma(const float *conv_x,
                                  const float *kernel,
                                  float *out,
                                  int kernel_size,
                                  int num_channels,
                                  int num_tokens,
                                  int num_seqs)
{
    if (!conv_x || !kernel || !out || kernel_size <= 0 || num_channels <= 0 ||
        num_tokens < 0 || num_seqs <= 0) {
        return;
    }
    ck_ssm_conv1d_llama_args_t args = {
        conv_x, kernel, out, kernel_size, num_channels, num_tokens, num_seqs,
    };
    ck_threadpool_t *pool = ck_threadpool_global();
    const int workers = pool ? ck_threadpool_n_threads(pool) : 1;
    const int active = workers < num_channels ? workers : num_channels;
    if (active > 1 && num_tokens > 1) {
        ck_threadpool_parallel_for_n(
            pool, active, 0, num_channels, 32,
            ck_ssm_conv1d_llama_fma_channel_range, &args);
    } else {
        ck_ssm_conv1d_llama_fma_channel_range(0, num_channels, &args);
    }
}

void ssm_conv1d_forward_pytorch_bf16_storage(const float *conv_x,
                                              const float *kernel,
                                              float *out,
                                              int kernel_size,
                                              int num_channels,
                                              int num_tokens,
                                              int num_seqs)
{
    ssm_conv1d_forward_ref(
        conv_x, kernel, out, kernel_size, num_channels, num_tokens, num_seqs);
    const size_t count =
        (size_t)num_seqs * (size_t)num_tokens * (size_t)num_channels;
    for (size_t i = 0; i < count; ++i) {
        out[i] = bf16_to_float(float_to_bf16(out[i]));
    }
}

void ssm_conv1d_backward(const float *d_out,
                         const float *conv_x,
                         const float *kernel,
                         float *d_conv_x,
                         float *d_kernel,
                         int kernel_size,
                         int num_channels,
                         int num_tokens,
                         int num_seqs)
{
    ssm_conv1d_backward_ref(d_out, conv_x, kernel, d_conv_x, d_kernel, kernel_size, num_channels, num_tokens, num_seqs);
}
