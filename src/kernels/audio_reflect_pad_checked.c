#include "ckernel_audio_reflect_pad.h"
#include "checked_float_buffer.h"

#include <stdint.h>
#include <string.h>

int audio_reflect_pad1d_left_channel_major_f32_checked(
    const float *input, size_t input_elements, size_t input_stride,
    float *output, size_t output_elements, size_t output_stride,
    size_t channels, size_t input_frames, size_t left_padding,
    size_t output_frames) {
    size_t input_span, output_span;
    if (!channels || !input_frames || (left_padding &&
        (input_frames < 2 || left_padding >= input_frames)))
        return CK_AUDIO_REFLECT_PAD_INVALID;
    if (input_frames > SIZE_MAX - left_padding)
        return CK_AUDIO_REFLECT_PAD_OVERFLOW;
    if (output_frames != input_frames + left_padding)
        return CK_AUDIO_REFLECT_PAD_INVALID;
    if (!ck_checked_span(channels, input_frames, input_stride, &input_span) ||
        !ck_checked_span(channels, output_frames, output_stride, &output_span))
        return CK_AUDIO_REFLECT_PAD_OVERFLOW;
    if (input_elements < input_span || output_elements < output_span)
        return CK_AUDIO_REFLECT_PAD_CAPACITY;
    if (!ck_checked_region(input, input_span) ||
        !ck_checked_region(output, output_span) ||
        ck_checked_overlap(input, input_span, output, output_span))
        return CK_AUDIO_REFLECT_PAD_INVALID;
    for (size_t channel = 0; channel < channels; ++channel)
        if (!ck_checked_finite(input + channel * input_stride, input_frames))
            return CK_AUDIO_REFLECT_PAD_NONFINITE;
    for (size_t channel = 0; channel < channels; ++channel) {
        const float *src = input + channel * input_stride;
        float *dst = output + channel * output_stride;
        for (size_t pad = 0; pad < left_padding; ++pad)
            dst[pad] = src[left_padding - pad];
        memcpy(dst + left_padding, src, input_frames * sizeof(float));
    }
    return CK_AUDIO_REFLECT_PAD_OK;
}
