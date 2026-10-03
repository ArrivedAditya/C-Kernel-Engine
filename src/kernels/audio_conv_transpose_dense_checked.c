#include "ckernel_audio_conv_transpose_dense.h"
#include "checked_float_buffer.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

int audio_conv_transpose1d_dense_f32_workspace(
    size_t output_channels, size_t output_frames, size_t *scratch_elements) {
    if (!scratch_elements || !output_channels || !output_frames ||
        output_channels > SIZE_MAX / output_frames ||
        output_channels * output_frames > SIZE_MAX / sizeof(float))
        return CK_AUDIO_DECONV_DENSE_OVERFLOW;
    *scratch_elements = output_channels * output_frames;
    return CK_AUDIO_DECONV_DENSE_OK;
}

int audio_conv_transpose1d_dense_channel_major_f32_checked(
    const float *input, size_t input_elements, size_t input_stride,
    const float *weight, size_t weight_elements,
    const float *bias, size_t bias_elements,
    float *output, size_t output_elements, size_t output_stride,
    float *scratch, size_t scratch_elements,
    size_t input_channels, size_t output_channels, size_t input_frames,
    size_t kernel_size, size_t stride, size_t padding,
    size_t output_padding, size_t output_frames) {
    size_t geometry, input_span, output_span, weight_need, scratch_need;
    if (!input_channels || !output_channels || !input_frames ||
        !kernel_size || !stride || !output_frames || output_padding >= stride)
        return CK_AUDIO_DECONV_DENSE_INVALID;
    if (input_frames - 1 > SIZE_MAX / stride || padding > SIZE_MAX / 2)
        return CK_AUDIO_DECONV_DENSE_OVERFLOW;
    geometry = (input_frames - 1) * stride;
    if (geometry > SIZE_MAX - kernel_size ||
        geometry + kernel_size > SIZE_MAX - output_padding)
        return CK_AUDIO_DECONV_DENSE_OVERFLOW;
    geometry += kernel_size + output_padding;
    if (2 * padding >= geometry || geometry - 2 * padding != output_frames)
        return CK_AUDIO_DECONV_DENSE_INVALID;
    if (!ck_checked_span(input_channels, input_frames, input_stride,
                         &input_span) ||
        !ck_checked_span(output_channels, output_frames, output_stride,
                         &output_span) ||
        input_channels > SIZE_MAX / output_channels ||
        input_channels * output_channels > SIZE_MAX / kernel_size ||
        audio_conv_transpose1d_dense_f32_workspace(
            output_channels, output_frames, &scratch_need))
        return CK_AUDIO_DECONV_DENSE_OVERFLOW;
    weight_need = input_channels * output_channels * kernel_size;
    if (weight_need > SIZE_MAX / sizeof(float))
        return CK_AUDIO_DECONV_DENSE_OVERFLOW;
    if (input_elements < input_span || weight_elements < weight_need ||
        bias_elements < output_channels || output_elements < output_span ||
        scratch_elements < scratch_need)
        return CK_AUDIO_DECONV_DENSE_CAPACITY;
    if (!ck_checked_region(input, input_span) ||
        !ck_checked_region(weight, weight_need) ||
        !ck_checked_region(bias, output_channels) ||
        !ck_checked_region(output, output_span) ||
        !ck_checked_region(scratch, scratch_need) ||
        ck_checked_overlap(output, output_span, input, input_span) ||
        ck_checked_overlap(output, output_span, weight, weight_need) ||
        ck_checked_overlap(output, output_span, bias, output_channels) ||
        ck_checked_overlap(output, output_span, scratch, scratch_need) ||
        ck_checked_overlap(scratch, scratch_need, input, input_span) ||
        ck_checked_overlap(scratch, scratch_need, weight, weight_need) ||
        ck_checked_overlap(scratch, scratch_need, bias, output_channels))
        return CK_AUDIO_DECONV_DENSE_INVALID;
    if (!ck_checked_finite(weight, weight_need) ||
        !ck_checked_finite(bias, output_channels))
        return CK_AUDIO_DECONV_DENSE_NONFINITE;
    for (size_t ic = 0; ic < input_channels; ++ic)
        if (!ck_checked_finite(input + ic * input_stride, input_frames))
            return CK_AUDIO_DECONV_DENSE_NONFINITE;

    for (size_t oc = 0; oc < output_channels; ++oc)
        for (size_t frame = 0; frame < output_frames; ++frame)
            scratch[oc * output_frames + frame] = bias[oc];
    for (size_t ic = 0; ic < input_channels; ++ic)
        for (size_t frame = 0; frame < input_frames; ++frame)
            for (size_t tap = 0; tap < kernel_size; ++tap) {
                size_t target = frame * stride + tap;
                if (target < padding || target - padding >= output_frames)
                    continue;
                size_t out_frame = target - padding;
                float value = input[ic * input_stride + frame];
                for (size_t oc = 0; oc < output_channels; ++oc) {
                    size_t index = oc * output_frames + out_frame;
                    scratch[index] = fmaf(value,
                        weight[(ic * output_channels + oc) * kernel_size + tap],
                        scratch[index]);
                }
            }
    if (!ck_checked_finite(scratch, scratch_need))
        return CK_AUDIO_DECONV_DENSE_NONFINITE;
    for (size_t oc = 0; oc < output_channels; ++oc)
        memcpy(output + oc * output_stride,
               scratch + oc * output_frames,
               output_frames * sizeof(float));
    return CK_AUDIO_DECONV_DENSE_OK;
}
