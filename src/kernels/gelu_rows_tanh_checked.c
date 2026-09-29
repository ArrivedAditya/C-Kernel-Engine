#include "ckernel_normalization_checked.h"
#include "ckernel_engine.h"
#include "checked_float_buffer.h"
#include <string.h>
int gelu_rows_tanh_checked_f32(const float *input, size_t input_elements, size_t input_stride,
    float *output, size_t output_elements, size_t output_stride,
    float *scratch, size_t scratch_elements, size_t rows, size_t channels) {
    size_t ni, no, ns;
    if (!ck_checked_span(rows, channels, input_stride, &ni) ||
        !ck_checked_span(rows, channels, output_stride, &no) ||
        !ck_checked_span(rows, channels, channels, &ns) ||
        input_elements < ni || output_elements < no || scratch_elements < ns) return -2;
    if (!ck_checked_region(input, ni) || !ck_checked_region(output, no) ||
        !ck_checked_region(scratch, ns) || ck_checked_overlap(input, ni, output, no) ||
        ck_checked_overlap(input, ni, scratch, ns) ||
        ck_checked_overlap(output, no, scratch, ns)) return -1;
    for (size_t r = 0; r < rows; ++r)
        if (!ck_checked_finite(input + r * input_stride, channels)) return -3;
    for (size_t r = 0; r < rows; ++r) {
        memcpy(scratch + r * channels, input + r * input_stride, channels * sizeof(float));
        /* Historical gelu_exact_inplace implements tanh GELU, matching gelu_new.
         * This preserves its arithmetic; an erf provider is a different contract. */
        gelu_exact_inplace(scratch + r * channels, channels);
    }
    if (!ck_checked_finite(scratch, ns)) return -4;
    for (size_t r = 0; r < rows; ++r)
        memcpy(output + r * output_stride, scratch + r * channels, channels * sizeof(float));
    return 0;
}
