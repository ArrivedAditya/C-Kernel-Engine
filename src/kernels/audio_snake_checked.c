#include "ckernel_audio_snake_checked.h"
#include "checked_float_buffer.h"

#include <math.h>

static int snake_value(float x, float alpha, float *value) {
    const float argument = alpha * x;
    if (!isfinite(argument)) return 0;
    const float sine = sinf(argument);
    const float inverse = 1.0f / alpha;
    const float square = sine * sine;
    /* Preserve the FP32 multiplication before addition even when the model
     * library is built with FMA enabled. */
    volatile float scaled = inverse * square;
    const float result = x + scaled;
    if (!isfinite(sine) || !isfinite(inverse) || !isfinite(result)) return 0;
    *value = result;
    return 1;
}

int audio_snake_strided_f32_checked(
    const float *input, size_t input_elements, size_t input_stride,
    const float *alpha, size_t alpha_elements,
    float *output, size_t output_elements, size_t output_stride,
    size_t channels, size_t frames) {
    size_t input_span, output_span;
    if (!channels || !frames || input_stride < frames || output_stride < frames)
        return -1;
    if (!ck_checked_span(channels, frames, input_stride, &input_span) ||
        !ck_checked_span(channels, frames, output_stride, &output_span))
        return -2;
    if (input_elements < input_span || output_elements < output_span ||
        alpha_elements < channels) return -2;
    if (!ck_checked_region(input, input_span) ||
        !ck_checked_region(alpha, channels) ||
        !ck_checked_region(output, output_span) ||
        ck_checked_overlap(input, input_span, output, output_span) ||
        ck_checked_overlap(alpha, channels, input, input_span) ||
        ck_checked_overlap(alpha, channels, output, output_span))
        return -1;

    /* A preflight pass preserves all output, including valid cells, when a
     * late element overflows. No heap or variable-length stack storage. */
    for (size_t channel = 0; channel < channels; ++channel) {
        const float a = alpha[channel];
        if (!isfinite(a) || a == 0.0f) return -3;
        for (size_t frame = 0; frame < frames; ++frame) {
            const float x = input[channel * input_stride + frame];
            float result;
            if (!isfinite(x) || !snake_value(x, a, &result)) return -3;
        }
    }
    for (size_t channel = 0; channel < channels; ++channel)
        for (size_t frame = 0; frame < frames; ++frame) {
            float result = 0.0f;
            /* Preflight established that this evaluation is finite. */
            (void)snake_value(input[channel * input_stride + frame],
                              alpha[channel], &result);
            output[channel * output_stride + frame] = result;
        }
    return 0;
}
