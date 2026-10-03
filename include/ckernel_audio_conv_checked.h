#ifndef CKERNEL_AUDIO_CONV_CHECKED_H
#define CKERNEL_AUDIO_CONV_CHECKED_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    CK_AUDIO_CONV_CHECKED_OK = 0,
    CK_AUDIO_CONV_CHECKED_INVALID = -1,
    CK_AUDIO_CONV_CHECKED_CAPACITY = -2,
    CK_AUDIO_CONV_CHECKED_OVERFLOW = -3,
    CK_AUDIO_CONV_CHECKED_NONFINITE = -4
};

/* Caller-owned scratch stages contiguous output.
 * No output is written until geometry, capacities, finite inputs and the
 * scalar Conv1D arithmetic have succeeded. Buffers must not alias. */
int audio_conv1d_checked_workspace(size_t input_channels, size_t output_channels,
    size_t input_frames, size_t output_frames, size_t *scratch_elements);

int audio_conv1d_checked_channel_major_f32(
    const float *input, size_t input_elements, size_t input_stride,
    const float *weight, size_t weight_elements,
    const float *bias, size_t bias_elements,
    float *output, size_t output_elements, size_t output_stride,
    float *scratch, size_t scratch_elements,
    size_t input_channels, size_t output_channels, size_t input_frames,
    size_t kernel_size, size_t stride, size_t padding, size_t output_frames);

/* Same channel/kernel FMA order with explicit tap dilation. The effective
 * kernel width is 1 + (kernel_size - 1) * dilation. Scratch and output
 * preservation follow the checked Conv1D contract above. */
int audio_conv1d_dilated_checked_channel_major_f32(
    const float *input, size_t input_elements, size_t input_stride,
    const float *weight, size_t weight_elements,
    const float *bias, size_t bias_elements,
    float *output, size_t output_elements, size_t output_stride,
    float *scratch, size_t scratch_elements,
    size_t input_channels, size_t output_channels, size_t input_frames,
    size_t kernel_size, size_t stride, size_t padding, size_t dilation,
    size_t output_frames);

#ifdef __cplusplus
}
#endif

#endif
