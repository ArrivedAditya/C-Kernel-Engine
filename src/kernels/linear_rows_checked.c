#include "ckernel_linear_checked.h"

#include <float.h>
#include <math.h>
#include <stdint.h>

static int span_fits(size_t rows, size_t stride, size_t width, size_t elements)
{
    return rows && width && stride >= width &&
           rows - 1 <= (SIZE_MAX - width) / stride &&
           (rows - 1) * stride + width <= elements;
}

static double dot_bias(const float *input, const float *weight,
                       size_t width, float bias)
{
    double sum = (double)bias;
    for (size_t c = 0; c < width; ++c) {
        volatile double product = (double)input[c] * (double)weight[c];
        sum += product;
    }
    return sum;
}

int linear_rows_checked_f32(
    const float *input, size_t input_elements, size_t input_stride,
    const float *weight, size_t weight_elements, size_t weight_stride,
    const float *bias, size_t bias_elements,
    float *output, size_t output_elements, size_t output_stride,
    size_t rows, size_t input_channels, size_t output_channels)
{
    if (!input || !weight || !bias || !output ||
        !span_fits(rows, input_stride, input_channels, input_elements) ||
        !span_fits(output_channels, weight_stride, input_channels, weight_elements) ||
        !span_fits(rows, output_stride, output_channels, output_elements) ||
        bias_elements < output_channels)
        return -1;

    /* Validate every accessed value and every result before the first write. */
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < input_channels; ++c)
            if (!isfinite(input[r * input_stride + c])) return -1;
    for (size_t o = 0; o < output_channels; ++o) {
        if (!isfinite(bias[o])) return -1;
        for (size_t c = 0; c < input_channels; ++c)
            if (!isfinite(weight[o * weight_stride + c])) return -1;
    }
    for (size_t r = 0; r < rows; ++r)
        for (size_t o = 0; o < output_channels; ++o) {
            const double result = dot_bias(input + r * input_stride,
                                           weight + o * weight_stride,
                                           input_channels, bias[o]);
            if (!isfinite(result) || fabs(result) > FLT_MAX) return -1;
        }
    for (size_t r = 0; r < rows; ++r)
        for (size_t o = 0; o < output_channels; ++o)
            output[r * output_stride + o] = (float)dot_bias(
                input + r * input_stride, weight + o * weight_stride,
                input_channels, bias[o]);
    return 0;
}
