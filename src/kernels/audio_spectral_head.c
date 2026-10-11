#include "ckernel_tts.h"

#include <math.h>
#include <stdint.h>

static int product(size_t a, size_t b, size_t *out) {
    if (a && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static int overlaps(const void *a, size_t a_bytes,
                    const void *b, size_t b_bytes) {
    const uintptr_t left = (uintptr_t)a;
    const uintptr_t right = (uintptr_t)b;
    if (a_bytes > UINTPTR_MAX - left || b_bytes > UINTPTR_MAX - right)
        return 1;
    return left < right + b_bytes && right < left + a_bytes;
}

int audio_spectral_head_mag_phase_f32(
    const float *input, size_t input_elements, size_t input_stride,
    float *magnitude, size_t magnitude_elements,
    float *phase, size_t phase_elements,
    size_t bins, size_t frames) {
    if (!input || !magnitude || !phase || !bins || !frames ||
        input_stride < frames) return CK_AUDIO_EXTENT_INVALID;

    size_t channels, input_required, output_required;
    if (!product(bins, 2, &channels) ||
        !product(channels, input_stride, &input_required) ||
        !product(bins, frames, &output_required) ||
        input_required > SIZE_MAX / sizeof(float) ||
        output_required > SIZE_MAX / sizeof(float))
        return CK_AUDIO_EXTENT_OVERFLOW;
    if (input_elements < input_required ||
        magnitude_elements < output_required ||
        phase_elements < output_required) return CK_AUDIO_EXTENT_LIMIT;

    const size_t input_bytes = input_required * sizeof(float);
    const size_t output_bytes = output_required * sizeof(float);
    if (overlaps(input, input_bytes, magnitude, output_bytes) ||
        overlaps(input, input_bytes, phase, output_bytes) ||
        overlaps(magnitude, output_bytes, phase, output_bytes))
        return CK_AUDIO_EXTENT_INVALID;

    for (size_t bin = 0; bin < bins; ++bin) {
        for (size_t frame = 0; frame < frames; ++frame) {
            const float log_magnitude = input[bin * input_stride + frame];
            const float raw_phase = input[(bins + bin) * input_stride + frame];
            if (!isfinite(log_magnitude) || !isfinite(raw_phase) ||
                !isfinite(expf(log_magnitude)))
                return CK_AUDIO_EXTENT_INVALID;
        }
    }
    for (size_t frame = 0; frame < frames; ++frame) {
        for (size_t bin = 0; bin < bins; ++bin) {
            const size_t output_index = frame * bins + bin;
            magnitude[output_index] = expf(input[bin * input_stride + frame]);
            phase[output_index] = sinf(input[(bins + bin) * input_stride + frame]);
        }
    }
    return CK_AUDIO_EXTENT_OK;
}
