#include "ckernel_tts.h"
#include "checked_float_buffer.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#define CK_SOURCE_PI_F 3.14159265358979323846f

/* A bounded harmonic oscillator with caller-supplied Gaussian excitation.
 * Frame F0 is nearest-expanded by `upsample`; the phase trajectory follows
 * align_corners=false linear down/cumsum/up interpolation. The first-sample
 * phase perturbation in Kokoro's non-pulse reference is discarded by the
 * downsampling midpoint, so no initial-phase input is needed here. */
int audio_harmonic_source_checked_f32(
    const float *f0, size_t f0_capacity,
    const float *gaussian, size_t gaussian_capacity,
    float *output, size_t output_capacity,
    float *phase_scratch, size_t scratch_capacity,
    size_t frames, size_t upsample, size_t harmonics,
    float sample_rate, float voiced_threshold,
    float sine_amp, float noise_std) {
    if (!frames || upsample < 4 || !harmonics || !isfinite(sample_rate) ||
        !isfinite(voiced_threshold) || !isfinite(sine_amp) ||
        !isfinite(noise_std) || sample_rate <= 0.0f || sine_amp < 0.0f ||
        noise_std < 0.0f || frames > SIZE_MAX / upsample)
        return CK_AUDIO_EXTENT_INVALID;
    const size_t samples = frames * upsample;
    if (samples > SIZE_MAX / harmonics ||
        frames > SIZE_MAX / harmonics)
        return CK_AUDIO_EXTENT_OVERFLOW;
    if (samples > (1u << 24)) return CK_AUDIO_EXTENT_LIMIT;
    const size_t output_need = samples * harmonics;
    const size_t phase_need = frames * harmonics;
    if (phase_need > SIZE_MAX - output_need)
        return CK_AUDIO_EXTENT_OVERFLOW;
    const size_t scratch_need = phase_need + output_need;
    if (f0_capacity < frames || gaussian_capacity < output_need ||
        output_capacity < output_need || scratch_capacity < scratch_need)
        return CK_AUDIO_EXTENT_LIMIT;
    if (!ck_checked_region(f0, frames) ||
        !ck_checked_region(gaussian, output_need) ||
        !ck_checked_region(output, output_need) ||
        !ck_checked_region(phase_scratch, scratch_need) ||
        ck_checked_overlap(output, output_need, f0, frames) ||
        ck_checked_overlap(output, output_need, gaussian, output_need) ||
        ck_checked_overlap(output, output_need, phase_scratch, scratch_need) ||
        ck_checked_overlap(phase_scratch, scratch_need, f0, frames) ||
        ck_checked_overlap(phase_scratch, scratch_need, gaussian, output_need) ||
        !ck_checked_finite(f0, frames) ||
        !ck_checked_finite(gaussian, output_need))
        return CK_AUDIO_EXTENT_INVALID;
    for (size_t harmonic = 0; harmonic < harmonics; ++harmonic) {
        /* The pinned CPU oracle widens FP32 cumsum's accumulator before
         * storing each FP32 prefix result. */
        double running = 0.0;
        for (size_t frame = 0; frame < frames; ++frame) {
            const float hz = f0[frame] * (float)(harmonic + 1);
            const float radians = hz / sample_rate;
            const float fraction = radians - floorf(radians);
            running += fraction;
            const float phase = (float)running * (2.0f * CK_SOURCE_PI_F);
            phase_scratch[frame * harmonics + harmonic] =
                phase * (float)upsample;
            if (!isfinite(phase_scratch[frame * harmonics + harmonic]))
                return CK_AUDIO_EXTENT_INVALID;
        }
    }
    float *staged = phase_scratch + phase_need;
    /* Stage the entire output so a late arithmetic failure preserves the
     * caller's output buffer. */
    for (size_t sample = 0; sample < samples; ++sample) {
        const size_t frame = sample / upsample;
        const float uv = f0[frame] > voiced_threshold ? 1.0f : 0.0f;
        const float amplitude = uv * noise_std +
            (1.0f - uv) * (sine_amp / 3.0f);
        /* PyTorch's align_corners=false interpolation multiplies by a
         * pre-rounded FP32 reciprocal; division changes long-phase samples. */
        const float inverse_scale = 1.0f / (float)upsample;
        float position = inverse_scale * ((float)sample + 0.5f) - 0.5f;
        if (position < 0.0f) position = 0.0f;
        if (position > (float)(frames - 1)) position = (float)(frames - 1);
        const size_t left = (size_t)position;
        const size_t right = left + 1 < frames ? left + 1 : left;
        const float right_weight = position - (float)left;
        for (size_t harmonic = 0; harmonic < harmonics; ++harmonic) {
            const float a = phase_scratch[left * harmonics + harmonic];
            const float b = phase_scratch[right * harmonics + harmonic];
            const float phase = a + (b - a) * right_weight;
            const float value = sinf(phase) * sine_amp * uv +
                amplitude * gaussian[sample * harmonics + harmonic];
            if (!isfinite(value)) return CK_AUDIO_EXTENT_INVALID;
            staged[sample * harmonics + harmonic] = value;
        }
    }
    memcpy(output, staged, output_need * sizeof(float));
    return CK_AUDIO_EXTENT_OK;
}
