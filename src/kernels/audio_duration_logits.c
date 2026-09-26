#include "ckernel_tts.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>

static float sigmoid_f32(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    const float e = expf(x);
    return e / (1.0f + e);
}

static int token_frames(const float *logits, size_t bins, float speed,
                        size_t max_duration_per_token, int32_t *frames) {
    float sum = 0.0f;
    for (size_t bin = 0; bin < bins; ++bin) {
        const float value = logits[bin];
        if (!isfinite(value)) return CK_AUDIO_EXTENT_INVALID;
        sum += sigmoid_f32(value);
    }
    const float scaled = sum / speed;
    if (!isfinite(scaled) || scaled >= 2147483648.0f)
        return CK_AUDIO_EXTENT_LIMIT;

    /* Nonnegative scaled values only. Implement nearest-even independently of
     * the process floating-point rounding mode. */
    const float lower = floorf(scaled);
    const float fraction = scaled - lower;
    float rounded = lower;
    if (fraction > 0.5f ||
        (fraction == 0.5f && fmodf(lower, 2.0f) != 0.0f))
        rounded += 1.0f;
    if (rounded < 1.0f) rounded = 1.0f;
    if (rounded > (float)max_duration_per_token)
        return CK_AUDIO_EXTENT_LIMIT;
    *frames = (int32_t)rounded;
    return CK_AUDIO_EXTENT_OK;
}

int audio_duration_logits_to_frames_f32(
    const float *logits, size_t input_elements, size_t tokens, size_t bins,
    size_t input_stride, float speed, int32_t *durations,
    size_t duration_capacity, size_t max_duration_per_token,
    size_t max_expanded_frames, int32_t *expanded_frames) {
    if (!logits || !durations || !expanded_frames || !tokens || !bins ||
        input_stride < bins || !isfinite(speed) || speed <= 0.0f ||
        !max_duration_per_token || max_duration_per_token > INT32_MAX ||
        !max_expanded_frames || max_expanded_frames > INT32_MAX)
        return CK_AUDIO_EXTENT_INVALID;
    if (tokens - 1 > (SIZE_MAX - bins) / input_stride ||
        tokens > SIZE_MAX / sizeof(int32_t))
        return CK_AUDIO_EXTENT_OVERFLOW;
    const size_t required_input = (tokens - 1) * input_stride + bins;
    if (required_input > SIZE_MAX / sizeof(float))
        return CK_AUDIO_EXTENT_OVERFLOW;
    if (input_elements < required_input || duration_capacity < tokens)
        return CK_AUDIO_EXTENT_LIMIT;

    size_t total = 0;
    for (size_t token = 0; token < tokens; ++token) {
        int32_t frames = 0;
        const int rc = token_frames(logits + token * input_stride, bins, speed,
                                    max_duration_per_token, &frames);
        if (rc != CK_AUDIO_EXTENT_OK) return rc;
        if ((size_t)frames > max_expanded_frames - total)
            return CK_AUDIO_EXTENT_LIMIT;
        total += (size_t)frames;
    }

    for (size_t token = 0; token < tokens; ++token) {
        int32_t frames = 0;
        /* The first pass validated all inputs and capacities. */
        (void)token_frames(logits + token * input_stride, bins, speed,
                           max_duration_per_token, &frames);
        durations[token] = frames;
    }
    *expanded_frames = (int32_t)total;
    return CK_AUDIO_EXTENT_OK;
}
