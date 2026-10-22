#include "ckernel_transformer_rows_checked.h"
#include "checked_float_buffer.h"
#include "ckernel_engine.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <limits.h>
#include <string.h>

int rmsnorm_rows_checked_f32(const float *input, size_t input_elements,
                            size_t input_stride, const float *gamma,
                            size_t gamma_elements, float *output,
                            size_t output_elements, size_t output_stride,
                            size_t rows, size_t width, float eps)
{
    size_t ni, no;
    if (!ck_checked_span(rows, width, input_stride, &ni) ||
        !ck_checked_span(rows, width, output_stride, &no) ||
        input_elements < ni || output_elements < no || gamma_elements < width)
        return -2;
    if (!isfinite(eps) || eps <= 0 ||
        !ck_checked_region(input, ni) || !ck_checked_region(gamma, width) ||
        !ck_checked_region(output, no) ||
        ck_checked_overlap(input, ni, output, no) ||
        ck_checked_overlap(gamma, width, output, no)) return -1;
    if (!ck_checked_finite(gamma, width)) return -3;
    for (size_t row = 0; row < rows; ++row) {
        const float *x = input + row * input_stride;
        if (!ck_checked_finite(x, width)) return -3;
        float sum = 0;
        for (size_t col = 0; col < width; ++col) sum += x[col] * x[col];
        const float scale = 1.0f / sqrtf(sum / (float)width + eps);
        if (!isfinite(scale)) return -3;
        for (size_t col = 0; col < width; ++col)
            if (!isfinite(x[col] * scale * gamma[col])) return -3;
    }
    for (size_t row = 0; row < rows; ++row) {
        const float *x = input + row * input_stride;
        float *y = output + row * output_stride;
        float sum = 0;
        for (size_t col = 0; col < width; ++col) sum += x[col] * x[col];
        const float scale = 1.0f / sqrtf(sum / (float)width + eps);
        for (size_t col = 0; col < width; ++col)
            y[col] = x[col] * scale * gamma[col];
    }
    return 0;
}

int token_head_layout_checked_f32(const float *input, size_t input_elements,
                                  float *output, size_t output_elements,
                                  size_t tokens, size_t heads, size_t head_dim,
                                  int head_to_token)
{
    if (!tokens || !heads || !head_dim ||
        tokens > SIZE_MAX / heads || tokens * heads > SIZE_MAX / head_dim)
        return -2;
    const size_t elements = tokens * heads * head_dim;
    if (input_elements < elements || output_elements < elements) return -2;
    if ((head_to_token != 0 && head_to_token != 1) ||
        !ck_checked_region(input, elements) ||
        !ck_checked_region(output, elements) ||
        ck_checked_overlap(input, elements, output, elements) ||
        !ck_checked_finite(input, elements)) return -1;
    for (size_t token = 0; token < tokens; ++token)
        for (size_t head = 0; head < heads; ++head)
            for (size_t channel = 0; channel < head_dim; ++channel) {
                const size_t token_offset = (token * heads + head) * head_dim + channel;
                const size_t head_offset = (head * tokens + token) * head_dim + channel;
                if (head_to_token)
                    output[token_offset] = input[head_offset];
                else
                    output[head_offset] = input[token_offset];
            }
    return 0;
}

int add_rows_checked_f32(const float *left, size_t left_elements,
                         const float *right, size_t right_elements,
                         float *output, size_t output_elements,
                         size_t elements)
{
    if (!elements || left_elements < elements || right_elements < elements ||
        output_elements < elements) return -2;
    if (!ck_checked_region(left, elements) ||
        !ck_checked_region(right, elements) ||
        !ck_checked_region(output, elements) ||
        ck_checked_overlap(left, elements, output, elements) ||
        ck_checked_overlap(right, elements, output, elements)) return -1;
    for (size_t i = 0; i < elements; ++i)
        if (!isfinite(left[i]) || !isfinite(right[i]) ||
            !isfinite(left[i] + right[i])) return -3;
    for (size_t i = 0; i < elements; ++i)
        output[i] = left[i] + right[i];
    return 0;
}

int silu_mul_rows_checked_f32(const float *gate, size_t gate_elements,
                              const float *up, size_t up_elements,
                              float *output, size_t output_elements,
                              size_t elements)
{
    if (!elements || gate_elements < elements || up_elements < elements ||
        output_elements < elements) return -2;
    if (!ck_checked_region(gate, elements) ||
        !ck_checked_region(up, elements) ||
        !ck_checked_region(output, elements) ||
        ck_checked_overlap(gate, elements, output, elements) ||
        ck_checked_overlap(up, elements, output, elements)) return -1;
    for (size_t i = 0; i < elements; ++i) {
        if (!isfinite(gate[i]) || !isfinite(up[i])) return -3;
        const float value = (gate[i] / (1.0f + expf(-gate[i]))) * up[i];
        if (!isfinite(value)) return -3;
    }
    for (size_t i = 0; i < elements; ++i)
        output[i] = (gate[i] / (1.0f + expf(-gate[i]))) * up[i];
    return 0;
}

int rope_qk_bounded_checked_f32(float *q, size_t q_elements,
                                float *k, size_t k_elements,
                                size_t heads, size_t kv_heads,
                                size_t tokens, size_t head_dim,
                                size_t pos_offset, float freq_base)
{
    if (!heads || !kv_heads || !tokens || !head_dim ||
        heads > INT_MAX || kv_heads > INT_MAX || tokens > INT_MAX ||
        head_dim > INT_MAX || pos_offset > INT_MAX ||
        heads > SIZE_MAX / tokens || kv_heads > SIZE_MAX / tokens ||
        heads * tokens > SIZE_MAX / head_dim ||
        kv_heads * tokens > SIZE_MAX / head_dim)
        return -2;
    const size_t nq = heads * tokens * head_dim;
    const size_t nk = kv_heads * tokens * head_dim;
    if (q_elements < nq || k_elements < nk) return -2;
    if (!isfinite(freq_base) || freq_base <= 0 ||
        !ck_checked_region(q, nq) || !ck_checked_region(k, nk) ||
        ck_checked_overlap(q, nq, k, nk)) return -1;
    for (size_t i = 0; i < nq; ++i)
        if (!isfinite(q[i]) || fabsf(q[i]) > FLT_MAX / 4) return -3;
    for (size_t i = 0; i < nk; ++i)
        if (!isfinite(k[i]) || fabsf(k[i]) > FLT_MAX / 4) return -3;
    rope_forward_qk_split_direct_f32(q, k, NULL, 0, (int)heads,
        (int)kv_heads, (int)tokens, (int)head_dim, (int)head_dim,
        (int)pos_offset, (int)head_dim, freq_base);
    return 0;
}

int attention_causal_bounded_checked_f32(
    const float *q, size_t q_elements,
    const float *k, size_t k_elements,
    const float *v, size_t v_elements,
    float *work, size_t work_elements,
    float *output, size_t output_elements,
    size_t heads, size_t kv_heads, size_t tokens, size_t head_dim)
{
    if (!heads || !kv_heads || heads < kv_heads || heads % kv_heads ||
        !tokens || !head_dim || heads > INT_MAX || kv_heads > INT_MAX ||
        tokens > INT_MAX || head_dim > INT_MAX ||
        heads > SIZE_MAX / tokens || kv_heads > SIZE_MAX / tokens ||
        heads * tokens > SIZE_MAX / head_dim ||
        kv_heads * tokens > SIZE_MAX / head_dim)
        return -2;
    const size_t nq = heads * tokens * head_dim;
    const size_t nk = kv_heads * tokens * head_dim;
    if (q_elements < nq || k_elements < nk || v_elements < nk ||
        work_elements < nq || output_elements < nq) return -2;
    if (!ck_checked_region(q, nq) || !ck_checked_region(k, nk) ||
        !ck_checked_region(v, nk) || !ck_checked_region(work, nq) ||
        !ck_checked_region(output, nq) ||
        ck_checked_overlap(q, nq, work, nq) ||
        ck_checked_overlap(k, nk, work, nq) ||
        ck_checked_overlap(v, nk, work, nq) ||
        ck_checked_overlap(work, nq, output, nq) ||
        ck_checked_overlap(q, nq, output, nq) ||
        ck_checked_overlap(k, nk, output, nq) ||
        ck_checked_overlap(v, nk, output, nq)) return -1;
    if (!ck_checked_finite(q, nq) || !ck_checked_finite(k, nk) ||
        !ck_checked_finite(v, nk)) return -3;
    attention_forward_causal_head_major_gqa_flash_strided(
        q, k, v, work, (int)heads, (int)kv_heads, (int)tokens,
        (int)head_dim, (int)head_dim, (int)tokens);
    if (!ck_checked_finite(work, nq)) return -3;
    memcpy(output, work, nq * sizeof(float));
    return 0;
}
