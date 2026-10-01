#ifndef CKERNEL_AUDIO_PROSODY_UPSAMPLE_CHECKED_H
#define CKERNEL_AUDIO_PROSODY_UPSAMPLE_CHECKED_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum CKAudioUpsampleCheckedStatus {
    CK_AUDIO_UPSAMPLE_OK = 0,
    CK_AUDIO_UPSAMPLE_INVALID = -1,
    CK_AUDIO_UPSAMPLE_CAPACITY = -2,
    CK_AUDIO_UPSAMPLE_NONFINITE = -3,
    CK_AUDIO_UPSAMPLE_OVERFLOW = -4
};

int audio_upsample_nearest_channel_major_f32_checked(
    const float *input, size_t input_elements, size_t input_stride,
    float *output, size_t output_elements, size_t output_stride,
    size_t channels, size_t input_frames, size_t factor,
    size_t output_frames);

int audio_conv_transpose1d_depthwise_channel_major_f32_checked(
    const float *input, size_t input_elements, size_t input_stride,
    const float *weight, size_t weight_elements,
    const float *bias, size_t bias_elements,
    float *output, size_t output_elements, size_t output_stride,
    float *scratch, size_t scratch_elements,
    size_t channels, size_t input_frames, size_t kernel_size,
    size_t stride, size_t padding, size_t output_padding,
    size_t output_frames);

#ifdef __cplusplus
}
#endif

#endif
