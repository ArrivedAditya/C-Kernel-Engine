#ifndef CKERNEL_AUDIO_REFLECT_PAD_H
#define CKERNEL_AUDIO_REFLECT_PAD_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum CKAudioReflectPadStatus {
    CK_AUDIO_REFLECT_PAD_OK = 0,
    CK_AUDIO_REFLECT_PAD_INVALID = -1,
    CK_AUDIO_REFLECT_PAD_CAPACITY = -2,
    CK_AUDIO_REFLECT_PAD_NONFINITE = -3,
    CK_AUDIO_REFLECT_PAD_OVERFLOW = -4
};

/* Reflect-pad the left edge of each channel-major FP32 row without repeating
 * the edge sample. For one pad element, [a,b,c] becomes [b,a,b,c].
 * Valid output frames must equal input_frames + left_padding. Physical row
 * strides and buffer capacities are independent of those valid extents.
 * Output is unchanged for every rejected call; padding remains untouched.
 */
int audio_reflect_pad1d_left_channel_major_f32_checked(
    const float *input, size_t input_elements, size_t input_stride,
    float *output, size_t output_elements, size_t output_stride,
    size_t channels, size_t input_frames, size_t left_padding,
    size_t output_frames);

#ifdef __cplusplus
}
#endif

#endif
