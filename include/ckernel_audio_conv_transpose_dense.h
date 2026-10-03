#ifndef CKERNEL_AUDIO_CONV_TRANSPOSE_DENSE_H
#define CKERNEL_AUDIO_CONV_TRANSPOSE_DENSE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum CKAudioConvTransposeDenseStatus {
    CK_AUDIO_DECONV_DENSE_OK = 0,
    CK_AUDIO_DECONV_DENSE_INVALID = -1,
    CK_AUDIO_DECONV_DENSE_CAPACITY = -2,
    CK_AUDIO_DECONV_DENSE_NONFINITE = -3,
    CK_AUDIO_DECONV_DENSE_OVERFLOW = -4
};

/* Group-one ConvTranspose1D: input [I,T], weight [I,O,K], bias [O].
 * Input/output rows may be padded; weight is contiguous input/output/tap.
 * Accumulation is FP32 FMA in input-channel, input-frame, tap order.
 * All writes stage in caller-owned [O,U] scratch; output is unchanged on error.
 */
int audio_conv_transpose1d_dense_f32_workspace(
    size_t output_channels, size_t output_frames, size_t *scratch_elements);

int audio_conv_transpose1d_dense_channel_major_f32_checked(
    const float *input, size_t input_elements, size_t input_stride,
    const float *weight, size_t weight_elements,
    const float *bias, size_t bias_elements,
    float *output, size_t output_elements, size_t output_stride,
    float *scratch, size_t scratch_elements,
    size_t input_channels, size_t output_channels, size_t input_frames,
    size_t kernel_size, size_t stride, size_t padding,
    size_t output_padding, size_t output_frames);

#ifdef __cplusplus
}
#endif

#endif
