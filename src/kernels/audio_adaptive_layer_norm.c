/* FP32 style-conditioned LayerNorm with caller-owned projection scratch.
 * The style projection uses output-feature-major [2*C, S] weights.
 */
#include "ckernel_audio.h"

#include <math.h>
#include <stdint.h>

static int checked_product(size_t left, size_t right, size_t *result)
{
    if (right != 0u && left > SIZE_MAX / right) return 0;
    *result = left * right;
    return 1;
}

static int checked_span(size_t rows, size_t stride, size_t width,
                        size_t *result)
{
    size_t prefix;
    if (rows == 0u || !checked_product(rows - 1u, stride, &prefix) ||
        prefix > SIZE_MAX - width) return 0;
    *result = prefix + width;
    return 1;
}

static int checked_float_extent(size_t elements)
{
    return elements <= (size_t)PTRDIFF_MAX / sizeof(float);
}

static int row_stats(const float *row, size_t channels, float epsilon,
                     double *mean, double *inverse_std)
{
    double sum = 0.0;
    for (size_t channel = 0u; channel < channels; ++channel)
        sum += (double)row[channel];
    *mean = sum / (double)channels;
    double squared = 0.0;
    for (size_t channel = 0u; channel < channels; ++channel) {
        const double delta = (double)row[channel] - *mean;
        squared += delta * delta;
    }
    const double variance = squared / (double)channels;
    *inverse_std = 1.0 / sqrt(variance + (double)epsilon);
    return isfinite(*mean) && isfinite(*inverse_std);
}

int audio_adaptive_layer_norm_f32(
    const float *input, size_t input_elements,
    const float *style, size_t style_elements,
    const float *projection_weight, size_t projection_weight_elements,
    const float *projection_bias, size_t projection_bias_elements,
    float *output, size_t output_elements,
    float *projection_scratch, size_t projection_scratch_bytes,
    int tokens, int channels, int style_dim,
    size_t input_stride, size_t output_stride, float epsilon)
{
    if (input == NULL || style == NULL || projection_weight == NULL ||
        projection_bias == NULL || output == NULL ||
        projection_scratch == NULL) return -1;
    if (tokens <= 0 || channels <= 0 || style_dim <= 0 ||
        !isfinite(epsilon) || epsilon <= 0.0f) return -2;

    const size_t rows = (size_t)tokens;
    const size_t width = (size_t)channels;
    const size_t style_width = (size_t)style_dim;
    size_t input_required, output_required, projection_width;
    size_t weight_required, scratch_required;
    if (input_stride < width || output_stride < width ||
        !checked_span(rows, input_stride, width, &input_required) ||
        !checked_span(rows, output_stride, width, &output_required) ||
        !checked_product(width, 2u, &projection_width) ||
        !checked_product(projection_width, style_width, &weight_required) ||
        !checked_product(projection_width, sizeof(float), &scratch_required) ||
        !checked_float_extent(input_required) ||
        !checked_float_extent(output_required) ||
        !checked_float_extent(style_width) ||
        !checked_float_extent(weight_required) ||
        !checked_float_extent(projection_width))
        return -2;
    if (input_elements < input_required || output_elements < output_required ||
        style_elements < style_width ||
        projection_weight_elements < weight_required ||
        projection_bias_elements < projection_width ||
        projection_scratch_bytes < scratch_required) return -3;

    /* Validate all external values before changing output or scratch. Padding
     * is deliberately not read; only the declared logical columns are valid.
     */
    for (size_t index = 0u; index < style_width; ++index)
        if (!isfinite(style[index])) return -4;
    for (size_t index = 0u; index < weight_required; ++index)
        if (!isfinite(projection_weight[index])) return -4;
    for (size_t index = 0u; index < projection_width; ++index)
        if (!isfinite(projection_bias[index])) return -4;
    for (size_t row = 0u; row < rows; ++row)
        for (size_t channel = 0u; channel < width; ++channel)
            if (!isfinite(input[row * input_stride + channel])) return -4;

    for (size_t channel = 0u; channel < projection_width; ++channel) {
        float value = projection_bias[channel];
        const float *weight = projection_weight + channel * style_width;
        for (size_t index = 0u; index < style_width; ++index)
            value += weight[index] * style[index];
        if (!isfinite(value)) return -4;
        projection_scratch[channel] = value;
    }

    /* Validate every produced value before the first output write. The second
     * pass repeats cheap normalization arithmetic without a hidden T*C buffer.
     */
    for (int pass = 0; pass < 2; ++pass) {
        for (size_t row = 0u; row < rows; ++row) {
            const float *input_row = input + row * input_stride;
            float *output_row = output + row * output_stride;
            double mean, inverse_std;
            if (!row_stats(input_row, width, epsilon, &mean, &inverse_std))
                return -4;
            for (size_t channel = 0u; channel < width; ++channel) {
                const float normalized = (float)(
                    ((double)input_row[channel] - mean) * inverse_std);
                const float gamma = projection_scratch[channel];
                const float beta = projection_scratch[width + channel];
                const float value = (1.0f + gamma) * normalized + beta;
                if (!isfinite(value)) return -4;
                if (pass != 0) output_row[channel] = value;
            }
        }
    }
    return 0;
}
