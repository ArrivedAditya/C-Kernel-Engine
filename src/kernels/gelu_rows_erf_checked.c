#include "ckernel_engine.h"
#include "checked_float_buffer.h"

#include <string.h>

/* Checked row adapter for the existing FP64-erf/FP32-storage GELU arithmetic. */
int gelu_rows_erf_checked_f32(const float *input, size_t input_elements,
    size_t input_stride, float *output, size_t output_elements,
    size_t output_stride, float *scratch, size_t scratch_elements,
    size_t rows, size_t channels) {
    size_t input_span, output_span, staged;
    if (!ck_checked_span(rows, channels, input_stride, &input_span) ||
        !ck_checked_span(rows, channels, output_stride, &output_span) ||
        rows > SIZE_MAX / channels || rows * channels > SIZE_MAX / sizeof(float))
        return -2;
    staged = rows * channels;
    if (input_elements < input_span || output_elements < output_span ||
        scratch_elements < staged)
        return -3;
    if (!ck_checked_region(input, input_span) ||
        !ck_checked_region(output, output_span) ||
        !ck_checked_region(scratch, staged) ||
        ck_checked_overlap(input, input_span, output, output_span) ||
        ck_checked_overlap(input, input_span, scratch, staged) ||
        ck_checked_overlap(output, output_span, scratch, staged))
        return -1;
    for (size_t r = 0; r < rows; ++r)
        if (!ck_checked_finite(input + r * input_stride, channels)) return -4;
    for (size_t r = 0; r < rows; ++r)
        memcpy(scratch + r * channels, input + r * input_stride,
               channels * sizeof(float));
    gelu_erf_fp64_f32_inplace(scratch, staged);
    if (!ck_checked_finite(scratch, staged)) return -4;
    for (size_t r = 0; r < rows; ++r)
        memcpy(output + r * output_stride, scratch + r * channels,
               channels * sizeof(float));
    return 0;
}
