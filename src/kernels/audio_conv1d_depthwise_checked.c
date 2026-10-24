#include "checked_float_buffer.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

/* Channel-major depthwise valid convolution. Causal padding is a separate
 * declared graph operation, so the kernel's input has its physical extent. */
int audio_conv1d_depthwise_channel_major_f32_checked(
    const float *input, size_t input_elements, size_t input_stride,
    const float *weight, size_t weight_elements,
    const float *bias, size_t bias_elements,
    float *output, size_t output_elements, size_t output_stride,
    float *scratch, size_t scratch_elements,
    size_t channels, size_t input_frames, size_t kernel_size,
    size_t stride, size_t output_frames) {
    size_t input_span, output_span, weight_count, staged;
    if (!channels || !input_frames || !kernel_size || !stride || !output_frames ||
        input_frames < kernel_size ||
        (input_frames - kernel_size) / stride + 1 != output_frames ||
        channels > SIZE_MAX / kernel_size ||
        channels > SIZE_MAX / output_frames)
        return -2;
    weight_count = channels * kernel_size;
    staged = channels * output_frames;
    if (!ck_checked_span(channels, input_frames, input_stride, &input_span) ||
        !ck_checked_span(channels, output_frames, output_stride, &output_span) ||
        weight_count > SIZE_MAX / sizeof(float) ||
        staged > SIZE_MAX / sizeof(float))
        return -2;
    if (input_elements < input_span || weight_elements < weight_count ||
        bias_elements < channels || output_elements < output_span ||
        scratch_elements < staged)
        return -3;
    if (!ck_checked_region(input, input_span) ||
        !ck_checked_region(weight, weight_count) ||
        !ck_checked_region(bias, channels) ||
        !ck_checked_region(output, output_span) ||
        !ck_checked_region(scratch, staged) ||
        ck_checked_overlap(output, output_span, input, input_span) ||
        ck_checked_overlap(output, output_span, weight, weight_count) ||
        ck_checked_overlap(output, output_span, bias, channels) ||
        ck_checked_overlap(output, output_span, scratch, staged) ||
        ck_checked_overlap(scratch, staged, input, input_span) ||
        ck_checked_overlap(scratch, staged, weight, weight_count) ||
        ck_checked_overlap(scratch, staged, bias, channels))
        return -1;
    if (!ck_checked_finite(weight, weight_count) ||
        !ck_checked_finite(bias, channels))
        return -4;
    for (size_t c = 0; c < channels; ++c)
        if (!ck_checked_finite(input + c * input_stride, input_frames))
            return -4;
    for (size_t c = 0; c < channels; ++c) {
        for (size_t t = 0; t < output_frames; ++t) {
            float sum = bias[c];
            for (size_t k = 0; k < kernel_size; ++k)
                sum = fmaf(input[c * input_stride + t * stride + k],
                           weight[c * kernel_size + k], sum);
            scratch[c * output_frames + t] = sum;
        }
    }
    if (!ck_checked_finite(scratch, staged)) return -4;
    for (size_t c = 0; c < channels; ++c)
        memcpy(output + c * output_stride, scratch + c * output_frames,
               output_frames * sizeof(float));
    return 0;
}
