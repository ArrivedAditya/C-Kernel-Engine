#include "ckernel_audio_prosody_upsample_checked.h"
#include "checked_float_buffer.h"

#include <stdint.h>
#include <string.h>

int audio_upsample_nearest_channel_major_f32_checked(
    const float *input, size_t input_elements, size_t input_stride,
    float *output, size_t output_elements, size_t output_stride,
    size_t channels, size_t input_frames, size_t factor,
    size_t output_frames) {
    size_t input_span, output_span;
    if (!channels || !input_frames || !factor)
        return CK_AUDIO_UPSAMPLE_INVALID;
    if (input_frames > SIZE_MAX / factor)
        return CK_AUDIO_UPSAMPLE_OVERFLOW;
    if (input_frames * factor != output_frames ||
        !ck_checked_span(channels, input_frames, input_stride, &input_span) ||
        !ck_checked_span(channels, output_frames, output_stride, &output_span))
        return CK_AUDIO_UPSAMPLE_INVALID;
    if (input_elements < input_span || output_elements < output_span)
        return CK_AUDIO_UPSAMPLE_CAPACITY;
    if (!ck_checked_region(input, input_span) ||
        !ck_checked_region(output, output_span) ||
        ck_checked_overlap(input, input_span, output, output_span))
        return CK_AUDIO_UPSAMPLE_INVALID;
    for (size_t channel = 0; channel < channels; ++channel)
        if (!ck_checked_finite(input + channel * input_stride, input_frames))
            return CK_AUDIO_UPSAMPLE_NONFINITE;
    for (size_t channel = 0; channel < channels; ++channel)
        for (size_t frame = 0; frame < output_frames; ++frame)
            output[channel * output_stride + frame] =
                input[channel * input_stride + frame / factor];
    return CK_AUDIO_UPSAMPLE_OK;
}

int audio_conv_transpose1d_depthwise_channel_major_f32_checked(
    const float *input, size_t input_elements, size_t input_stride,
    const float *weight, size_t weight_elements,
    const float *bias, size_t bias_elements,
    float *output, size_t output_elements, size_t output_stride,
    float *scratch, size_t scratch_elements,
    size_t channels, size_t input_frames, size_t kernel_size,
    size_t stride, size_t padding, size_t output_padding,
    size_t output_frames) {
    size_t input_span, output_span, weight_need, scratch_need, geometry;
    if (!channels || !input_frames || !kernel_size || !stride ||
        output_padding >= stride || !output_frames)
        return CK_AUDIO_UPSAMPLE_INVALID;
    if (input_frames - 1 > SIZE_MAX / stride)
        return CK_AUDIO_UPSAMPLE_OVERFLOW;
    geometry = (input_frames - 1) * stride;
    if (geometry > SIZE_MAX - kernel_size ||
        geometry + kernel_size > SIZE_MAX - output_padding ||
        padding > SIZE_MAX / 2)
        return CK_AUDIO_UPSAMPLE_OVERFLOW;
    geometry += kernel_size + output_padding;
    if (2 * padding >= geometry || geometry - 2 * padding != output_frames)
        return CK_AUDIO_UPSAMPLE_INVALID;
    if (!ck_checked_span(channels, input_frames, input_stride, &input_span) ||
        !ck_checked_span(channels, output_frames, output_stride, &output_span) ||
        channels > SIZE_MAX / kernel_size ||
        channels > SIZE_MAX / output_frames)
        return CK_AUDIO_UPSAMPLE_OVERFLOW;
    weight_need = channels * kernel_size;
    scratch_need = channels * output_frames;
    if (weight_need > SIZE_MAX / sizeof(float) ||
        scratch_need > SIZE_MAX / sizeof(float))
        return CK_AUDIO_UPSAMPLE_OVERFLOW;
    if (input_elements < input_span || weight_elements < weight_need ||
        bias_elements < channels || output_elements < output_span ||
        scratch_elements < scratch_need)
        return CK_AUDIO_UPSAMPLE_CAPACITY;
    if (!ck_checked_region(input, input_span) ||
        !ck_checked_region(weight, weight_need) ||
        !ck_checked_region(bias, channels) ||
        !ck_checked_region(output, output_span) ||
        !ck_checked_region(scratch, scratch_need) ||
        ck_checked_overlap(output, output_span, input, input_span) ||
        ck_checked_overlap(output, output_span, weight, weight_need) ||
        ck_checked_overlap(output, output_span, bias, channels) ||
        ck_checked_overlap(output, output_span, scratch, scratch_need) ||
        ck_checked_overlap(scratch, scratch_need, input, input_span) ||
        ck_checked_overlap(scratch, scratch_need, weight, weight_need) ||
        ck_checked_overlap(scratch, scratch_need, bias, channels))
        return CK_AUDIO_UPSAMPLE_INVALID;
    if (!ck_checked_finite(weight, weight_need) ||
        !ck_checked_finite(bias, channels))
        return CK_AUDIO_UPSAMPLE_NONFINITE;
    for (size_t channel = 0; channel < channels; ++channel)
        if (!ck_checked_finite(input + channel * input_stride, input_frames))
            return CK_AUDIO_UPSAMPLE_NONFINITE;
    for (size_t channel = 0; channel < channels; ++channel) {
        float *row = scratch + channel * output_frames;
        for (size_t frame = 0; frame < output_frames; ++frame)
            row[frame] = bias[channel];
        for (size_t frame = 0; frame < input_frames; ++frame)
            for (size_t tap = 0; tap < kernel_size; ++tap) {
                size_t target = frame * stride + tap;
                if (target >= padding && target - padding < output_frames)
                    row[target - padding] = fmaf(
                        input[channel * input_stride + frame],
                        weight[channel * kernel_size + tap],
                        row[target - padding]);
            }
        if (!ck_checked_finite(row, output_frames))
            return CK_AUDIO_UPSAMPLE_NONFINITE;
    }
    for (size_t channel = 0; channel < channels; ++channel)
        memcpy(output + channel * output_stride,
               scratch + channel * output_frames,
               output_frames * sizeof(float));
    return CK_AUDIO_UPSAMPLE_OK;
}
