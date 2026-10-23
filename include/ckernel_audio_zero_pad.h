#ifndef CKERNEL_AUDIO_ZERO_PAD_H
#define CKERNEL_AUDIO_ZERO_PAD_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum CKAudioZeroPadStatus {
    CK_AUDIO_ZERO_PAD_OK = 0,
    CK_AUDIO_ZERO_PAD_INVALID = -1,
    CK_AUDIO_ZERO_PAD_CAPACITY = -2,
    CK_AUDIO_ZERO_PAD_NONFINITE = -3,
    CK_AUDIO_ZERO_PAD_OVERFLOW = -4
};

/* Prepend zeros to every channel-major FP32 row. Valid output frames equal
 * input_frames + left_padding; physical strides may be larger. Rejected calls
 * leave output unchanged, including its unused padding. */
int audio_zero_pad1d_left_channel_major_f32_checked(
    const float *input, size_t input_elements, size_t input_stride,
    float *output, size_t output_elements, size_t output_stride,
    size_t channels, size_t input_frames, size_t left_padding,
    size_t output_frames);

#ifdef __cplusplus
}
#endif

#endif
