/* Checked FP32 word/type/position embedding and per-token layer norm.
 * All buffers are caller owned. No model-specific tensor names are used here. */
#include "ckernel_embedding_checked.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

static int span_fits(size_t rows, size_t stride, size_t columns, size_t elements)
{
    return rows > 0 && columns > 0 && stride >= columns &&
           rows - 1 <= (SIZE_MAX - columns) / stride &&
           (rows - 1) * stride + columns <= elements;
}

int embedding_three_table_layer_norm_f32(
    const int32_t *word_ids, const int32_t *type_ids, size_t id_elements,
    const float *word, size_t word_elements, size_t vocab_size,
    const float *position, size_t position_elements, size_t position_count,
    const float *token_type, size_t type_elements, size_t type_count,
    const float *gamma, const float *beta, size_t norm_elements,
    float *output, size_t output_elements, size_t tokens, size_t channels,
    size_t weight_stride, size_t output_stride, float epsilon)
{
    if (!word_ids || !type_ids || !word || !position || !token_type ||
        !gamma || !beta || !output || !tokens || !channels ||
        id_elements < tokens || norm_elements < channels ||
        !isfinite(epsilon) || epsilon <= 0.0f ||
        !span_fits(vocab_size, weight_stride, channels, word_elements) ||
        !span_fits(position_count, weight_stride, channels, position_elements) ||
        !span_fits(type_count, weight_stride, channels, type_elements) ||
        !span_fits(tokens, output_stride, channels, output_elements) ||
        tokens > position_count)
        return -1;

    /* Validate every input before the first output write. */
    for (size_t c = 0; c < channels; ++c)
        if (!isfinite(gamma[c]) || !isfinite(beta[c])) return -1;
    for (size_t t = 0; t < tokens; ++t) {
        if (word_ids[t] < 0 || (size_t)word_ids[t] >= vocab_size ||
            type_ids[t] < 0 || (size_t)type_ids[t] >= type_count)
            return -1;
        const size_t wi = (size_t)word_ids[t] * weight_stride;
        const size_t ti = (size_t)type_ids[t] * weight_stride;
        const size_t pi = t * weight_stride;
        for (size_t c = 0; c < channels; ++c)
            if (!isfinite(word[wi + c]) || !isfinite(token_type[ti + c]) ||
                !isfinite(position[pi + c])) return -1;
    }

    for (size_t t = 0; t < tokens; ++t) {
        const float *w = word + (size_t)word_ids[t] * weight_stride;
        const float *ty = token_type + (size_t)type_ids[t] * weight_stride;
        const float *p = position + t * weight_stride;
        double sum = 0.0, square_sum = 0.0;
        for (size_t c = 0; c < channels; ++c) {
            const float a = w[c] + ty[c];
            const float x = a + p[c];
            sum += (double)x;
            square_sum += (double)x * (double)x;
        }
        const double mean = sum / (double)channels;
        const double variance = fmax(0.0, square_sum / (double)channels - mean * mean);
        const float inv_std = (float)(1.0 / sqrt(variance + (double)epsilon));
        float *dst = output + t * output_stride;
        for (size_t c = 0; c < channels; ++c) {
            const float a = w[c] + ty[c];
            const float x = a + p[c];
            dst[c] = ((x - (float)mean) * inv_std) * gamma[c] + beta[c];
        }
    }
    return 0;
}
