#ifndef CKERNEL_AUDIO_CONCAT_CHECKED_H
#define CKERNEL_AUDIO_CONCAT_CHECKED_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    CK_AUDIO_CONCAT_OK = 0,
    CK_AUDIO_CONCAT_INVALID = -1,
    CK_AUDIO_CONCAT_CAPACITY = -2,
    CK_AUDIO_CONCAT_OVERFLOW = -3,
    CK_AUDIO_CONCAT_NONFINITE = -4
};

/* Concatenate two channel-major FP32 tensors along the channel axis.
 * Each channel has `frames` valid samples at its declared physical stride.
 * All geometry, capacity, alias and finite-input checks complete before any
 * output write. Padding outside the valid frame extent is left untouched.
 * No allocation, persistent state or scratch is used. */
int audio_concat_channels_checked_f32(
    const float *left, size_t left_elements, size_t left_stride,
    const float *right, size_t right_elements, size_t right_stride,
    float *output, size_t output_elements, size_t output_stride,
    size_t left_channels, size_t right_channels, size_t output_channels,
    size_t frames);

#ifdef __cplusplus
}
#endif

#endif
