#include "ckernel_audio_concat_checked.h"
#include "checked_float_buffer.h"

#include <stdint.h>
#include <string.h>

int audio_concat_channels_checked_f32(
    const float *left, size_t left_elements, size_t left_stride,
    const float *right, size_t right_elements, size_t right_stride,
    float *output, size_t output_elements, size_t output_stride,
    size_t left_channels, size_t right_channels, size_t output_channels,
    size_t frames) {
    size_t left_span, right_span, output_span, channels;
    if (!left_channels || !right_channels || !frames)
        return CK_AUDIO_CONCAT_INVALID;
    if (left_channels > SIZE_MAX - right_channels)
        return CK_AUDIO_CONCAT_OVERFLOW;
    channels = left_channels + right_channels;
    if (output_channels != channels) return CK_AUDIO_CONCAT_INVALID;
    if (!ck_checked_span(left_channels, frames, left_stride, &left_span) ||
        !ck_checked_span(right_channels, frames, right_stride, &right_span) ||
        !ck_checked_span(channels, frames, output_stride, &output_span))
        return CK_AUDIO_CONCAT_OVERFLOW;
    if (left_elements < left_span || right_elements < right_span ||
        output_elements < output_span)
        return CK_AUDIO_CONCAT_CAPACITY;
    if (!ck_checked_region(left, left_span) ||
        !ck_checked_region(right, right_span) ||
        !ck_checked_region(output, output_span) ||
        ck_checked_overlap(output, output_span, left, left_span) ||
        ck_checked_overlap(output, output_span, right, right_span))
        return CK_AUDIO_CONCAT_INVALID;
    for (size_t row = 0; row < left_channels; ++row)
        if (!ck_checked_finite(left + row * left_stride, frames))
            return CK_AUDIO_CONCAT_NONFINITE;
    for (size_t row = 0; row < right_channels; ++row)
        if (!ck_checked_finite(right + row * right_stride, frames))
            return CK_AUDIO_CONCAT_NONFINITE;
    for (size_t row = 0; row < left_channels; ++row)
        memcpy(output + row * output_stride, left + row * left_stride,
               frames * sizeof(float));
    for (size_t row = 0; row < right_channels; ++row)
        memcpy(output + (left_channels + row) * output_stride,
               right + row * right_stride, frames * sizeof(float));
    return CK_AUDIO_CONCAT_OK;
}
