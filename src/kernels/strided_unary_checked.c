#include "ckernel_strided_unary_checked.h"
#include "checked_float_buffer.h"

int transpose_strided_f32_checked(
    const float *input, size_t input_elements, size_t input_stride,
    float *output, size_t output_elements, size_t output_stride,
    size_t rows, size_t columns) {
    size_t ni, no;
    if (!ck_checked_span(rows, columns, input_stride, &ni) ||
        !ck_checked_span(columns, rows, output_stride, &no) ||
        input_elements < ni || output_elements < no)
        return -2;
    if (!ck_checked_region(input, ni) || !ck_checked_region(output, no) ||
        ck_checked_overlap(input, ni, output, no)) return -1;
    for (size_t row = 0; row < rows; ++row)
        if (!ck_checked_finite(input + row * input_stride, columns)) return -3;
    for (size_t row = 0; row < rows; ++row)
        for (size_t column = 0; column < columns; ++column)
            output[column * output_stride + row] = input[row * input_stride + column];
    return 0;
}

int leaky_relu_strided_f32_checked(
    const float *input, size_t input_elements, size_t input_stride,
    float *output, size_t output_elements, size_t output_stride,
    size_t rows, size_t columns, float negative_slope) {
    size_t ni, no;
    if (!ck_checked_span(rows, columns, input_stride, &ni) ||
        !ck_checked_span(rows, columns, output_stride, &no) ||
        input_elements < ni || output_elements < no)
        return -2;
    if (!isfinite(negative_slope) || negative_slope < 0 ||
        !ck_checked_region(input, ni) || !ck_checked_region(output, no) ||
        ck_checked_overlap(input, ni, output, no)) return -1;
    for (size_t row = 0; row < rows; ++row)
        for (size_t column = 0; column < columns; ++column) {
            float value = input[row * input_stride + column];
            if (!isfinite(value) || (value < 0 && !isfinite(value * negative_slope)))
                return -3;
        }
    for (size_t row = 0; row < rows; ++row)
        for (size_t column = 0; column < columns; ++column) {
            float value = input[row * input_stride + column];
            output[row * output_stride + column] =
                value >= 0 ? value : value * negative_slope;
        }
    return 0;
}

int tanh_strided_f32_checked(
    const float *input, size_t input_elements, size_t input_stride,
    float *output, size_t output_elements, size_t output_stride,
    size_t rows, size_t columns) {
    size_t ni, no;
    if (!ck_checked_span(rows, columns, input_stride, &ni) ||
        !ck_checked_span(rows, columns, output_stride, &no) ||
        input_elements < ni || output_elements < no)
        return -2;
    if (!ck_checked_region(input, ni) || !ck_checked_region(output, no) ||
        ck_checked_overlap(input, ni, output, no)) return -1;
    for (size_t row = 0; row < rows; ++row)
        if (!ck_checked_finite(input + row * input_stride, columns)) return -3;
    for (size_t row = 0; row < rows; ++row)
        for (size_t column = 0; column < columns; ++column)
            output[row * output_stride + column] =
                tanhf(input[row * input_stride + column]);
    return 0;
}
