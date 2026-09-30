#include "ckernel_feature_concat_checked.h"
#include "ckernel_engine.h"
#include "checked_float_buffer.h"
#include <limits.h>

int feature_concat_broadcast_rows_f32(
    const float *input, size_t input_elements, size_t input_stride,
    const float *feature, size_t feature_elements,
    float *output, size_t output_elements, size_t output_stride,
    size_t rows, size_t input_channels, size_t feature_channels,
    size_t output_channels) {
    size_t ni, no, width;
    if (!rows || !input_channels || !feature_channels ||
        input_channels > (size_t)INT_MAX ||
        feature_channels > (size_t)INT_MAX - input_channels) return -2;
    width = input_channels + feature_channels;
    if (output_channels != width) return -2;
    if (!ck_checked_span(rows, input_channels, input_stride, &ni) ||
        !ck_checked_span(rows, width, output_stride, &no) ||
        input_elements < ni || feature_elements < feature_channels ||
        output_elements < no) return -2;
    if (!ck_checked_region(input, ni) ||
        !ck_checked_region(feature, feature_channels) ||
        !ck_checked_region(output, no) ||
        ck_checked_overlap(output, no, input, ni) ||
        ck_checked_overlap(output, no, feature, feature_channels)) return -1;
    if (!ck_checked_finite(feature, feature_channels)) return -3;
    for (size_t r = 0; r < rows; ++r)
        if (!ck_checked_finite(input + r * input_stride, input_channels)) return -3;
    for (size_t r = 0; r < rows; ++r)
        /* The existing provider owns the exact copy semantics. A one-row
         * call broadcasts the same feature vector without staging memory. */
        feature_concat(input + r * input_stride, feature,
                       output + r * output_stride,
                       1, (int)input_channels, (int)feature_channels, 1);
    return 0;
}
