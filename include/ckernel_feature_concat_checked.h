#ifndef CKERNEL_FEATURE_CONCAT_CHECKED_H
#define CKERNEL_FEATURE_CONCAT_CHECKED_H

#include <stddef.h>

/* Concatenate a single feature vector to each logical row of a strided FP32
 * matrix. Output padding is untouched. All buffers are caller-owned.
 * Returns 0 on success; -1 for pointer/alias faults, -2 for geometry or
 * capacity faults, and -3 for nonfinite logical inputs. Rejection leaves the
 * entire output unchanged. No allocation or persistent state. */
int feature_concat_broadcast_rows_f32(
    const float *input, size_t input_elements, size_t input_stride,
    const float *feature, size_t feature_elements,
    float *output, size_t output_elements, size_t output_stride,
    size_t rows, size_t input_channels, size_t feature_channels,
    size_t output_channels);

#endif
