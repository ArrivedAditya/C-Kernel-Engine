#ifndef CKERNEL_STRIDED_UNARY_CHECKED_H
#define CKERNEL_STRIDED_UNARY_CHECKED_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Providers validate all used input before writing any output.
 * Input and output must not overlap; padding is left untouched. */
int transpose_strided_f32_checked(
    const float *input, size_t input_elements, size_t input_stride,
    float *output, size_t output_elements, size_t output_stride,
    size_t rows, size_t columns);

int leaky_relu_strided_f32_checked(
    const float *input, size_t input_elements, size_t input_stride,
    float *output, size_t output_elements, size_t output_stride,
    size_t rows, size_t columns, float negative_slope);

int tanh_strided_f32_checked(
    const float *input, size_t input_elements, size_t input_stride,
    float *output, size_t output_elements, size_t output_stride,
    size_t rows, size_t columns);

#ifdef __cplusplus
}
#endif

#endif
