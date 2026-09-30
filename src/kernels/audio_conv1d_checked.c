#include "ckernel_audio_conv_checked.h"
#include "checked_float_buffer.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

int audio_conv1d_checked_workspace(size_t input_channels, size_t output_channels,
    size_t input_frames, size_t output_frames, size_t *scratch_elements) {
    size_t staged;
    if (!scratch_elements || !input_channels || !output_channels ||
        !input_frames || !output_frames ||
        input_channels > SIZE_MAX / input_frames ||
        output_channels > SIZE_MAX / output_frames)
        return CK_AUDIO_CONV_CHECKED_OVERFLOW;
    staged = output_channels * output_frames;
    if (staged > SIZE_MAX / sizeof(float))
        return CK_AUDIO_CONV_CHECKED_OVERFLOW;
    *scratch_elements = staged;
    return CK_AUDIO_CONV_CHECKED_OK;
}

int audio_conv1d_checked_channel_major_f32(
    const float *input, size_t input_elements, size_t input_stride,
    const float *weight, size_t weight_elements,
    const float *bias, size_t bias_elements,
    float *output, size_t output_elements, size_t output_stride,
    float *scratch, size_t scratch_elements,
    size_t input_channels, size_t output_channels, size_t input_frames,
    size_t kernel_size, size_t stride, size_t padding, size_t output_frames) {
    size_t input_span, output_span, weight_count, scratch_need;
    if (!input_channels || !output_channels || !input_frames ||
        !kernel_size || !stride || !output_frames ||
        input_channels > INT_MAX || output_channels > INT_MAX ||
        input_frames > INT_MAX || kernel_size > INT_MAX ||
        stride > INT_MAX || padding > INT_MAX || output_frames > INT_MAX)
        return CK_AUDIO_CONV_CHECKED_INVALID;
    const int64_t padded = (int64_t)input_frames + 2 * (int64_t)padding - kernel_size;
    if (padded < 0 || padded / stride + 1 != output_frames)
        return CK_AUDIO_CONV_CHECKED_INVALID;
    if (!ck_checked_span((size_t)input_channels, (size_t)input_frames,
                         input_stride, &input_span) ||
        !ck_checked_span((size_t)output_channels, (size_t)output_frames,
                         output_stride, &output_span) ||
        audio_conv1d_checked_workspace((size_t)input_channels,
            (size_t)output_channels, (size_t)input_frames,
            (size_t)output_frames, &scratch_need))
        return CK_AUDIO_CONV_CHECKED_OVERFLOW;
    if ((size_t)output_channels > SIZE_MAX / (size_t)input_channels ||
        (size_t)output_channels * (size_t)input_channels > SIZE_MAX / (size_t)kernel_size)
        return CK_AUDIO_CONV_CHECKED_OVERFLOW;
    weight_count = (size_t)output_channels * (size_t)input_channels * (size_t)kernel_size;
    if (weight_count > SIZE_MAX / sizeof(float)) return CK_AUDIO_CONV_CHECKED_OVERFLOW;
    if (input_elements < input_span || weight_elements < weight_count ||
        bias_elements < (size_t)output_channels || output_elements < output_span ||
        scratch_elements < scratch_need)
        return CK_AUDIO_CONV_CHECKED_CAPACITY;
    if (!ck_checked_region(input, input_span) ||
        !ck_checked_region(weight, weight_count) ||
        !ck_checked_region(bias, (size_t)output_channels) ||
        !ck_checked_region(output, output_span) ||
        !ck_checked_region(scratch, scratch_need))
        return CK_AUDIO_CONV_CHECKED_INVALID;
    if (ck_checked_overlap(output, output_span, input, input_span) ||
        ck_checked_overlap(output, output_span, weight, weight_count) ||
        ck_checked_overlap(output, output_span, bias, (size_t)output_channels) ||
        ck_checked_overlap(output, output_span, scratch, scratch_need) ||
        ck_checked_overlap(scratch, scratch_need, input, input_span) ||
        ck_checked_overlap(scratch, scratch_need, weight, weight_count) ||
        ck_checked_overlap(scratch, scratch_need, bias, (size_t)output_channels))
        return CK_AUDIO_CONV_CHECKED_INVALID;
    if (!ck_checked_finite(weight, weight_count) ||
        !ck_checked_finite(bias, (size_t)output_channels))
        return CK_AUDIO_CONV_CHECKED_NONFINITE;
    for (size_t channel = 0; channel < input_channels; ++channel)
        if (!ck_checked_finite(input + (size_t)channel * input_stride,
                               (size_t)input_frames))
            return CK_AUDIO_CONV_CHECKED_NONFINITE;
    /* Scalar channel/kernel order matches the existing Conv1D reference. No
     * global thread-pool initialization or allocation occurs in this call. */
    float *staged = scratch;
    for (size_t oc = 0; oc < output_channels; ++oc) {
        for (size_t frame = 0; frame < output_frames; ++frame) {
            float sum = bias[oc];
            for (size_t ic = 0; ic < input_channels; ++ic) {
                const float *row = weight +
                    ((size_t)oc * (size_t)input_channels + (size_t)ic) *
                    (size_t)kernel_size;
                for (size_t tap = 0; tap < kernel_size; ++tap) {
                    int64_t source = (int64_t)frame * stride + tap - padding;
                    if (source >= 0 && (size_t)source < input_frames)
                        sum = fmaf(input[(size_t)ic * input_stride +
                            (size_t)source], row[tap], sum);
                }
            }
            staged[(size_t)oc * (size_t)output_frames + (size_t)frame] = sum;
        }
    }
    if (!ck_checked_finite(staged, (size_t)output_channels * (size_t)output_frames))
        return CK_AUDIO_CONV_CHECKED_NONFINITE;
    for (size_t channel = 0; channel < output_channels; ++channel)
        memcpy(output + (size_t)channel * output_stride,
               staged + (size_t)channel * (size_t)output_frames,
               (size_t)output_frames * sizeof(float));
    return CK_AUDIO_CONV_CHECKED_OK;
}
