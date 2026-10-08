#include "ckernel_tts.h"
#include "checked_float_buffer.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

int audio_stft_mag_phase_plan_f32(size_t samples, size_t n_fft, size_t hop,
    size_t *frames, size_t *spectral_elements) {
    if (!frames || !spectral_elements || n_fft < 2 || (n_fft & 1) ||
        !hop || samples <= n_fft / 2 || samples > INT64_MAX / 2 ||
        n_fft > INT64_MAX / 2 || samples > SIZE_MAX / sizeof(float) ||
        n_fft > SIZE_MAX / sizeof(float)) return CK_AUDIO_EXTENT_INVALID;
    const size_t count = samples / hop + 1;
    const size_t bins = n_fft / 2 + 1;
    if (count > SIZE_MAX / (2 * bins) ||
        count * (2 * bins) > SIZE_MAX / sizeof(float))
        return CK_AUDIO_EXTENT_OVERFLOW;
    *frames = count;
    *spectral_elements = count * 2 * bins;
    return CK_AUDIO_EXTENT_OK;
}

/* Centered reflect-pad STFT, periodic-Hann window, unnormalized forward DFT.
 * Output rows are magnitude bins followed by phase bins, matching a channel
 * concatenation of torch.abs(torch.stft(...)) and torch.angle(...).
 * Tables and staging scratch are supplied by the caller. */
static int stft_trace_region(const double *trace, size_t elements) {
    return trace && (uintptr_t)trace % _Alignof(double) == 0 &&
        elements <= SIZE_MAX / sizeof(double) &&
        (uintptr_t)trace <= UINTPTR_MAX - elements * sizeof(double);
}

static int stft_trace_overlaps(const double *trace, size_t elements,
    const float *other, size_t other_elements) {
    return (uintptr_t)trace < (uintptr_t)other + other_elements * sizeof(float) &&
        (uintptr_t)other < (uintptr_t)trace + elements * sizeof(double);
}

static int stft_mag_phase_impl(const float *samples, size_t samples_capacity,
    const float *window, size_t window_capacity,
    const float *cos_table, const float *sin_table, size_t table_capacity,
    float *output, size_t output_capacity, size_t output_stride,
    float *scratch, size_t scratch_capacity,
    size_t samples_count, size_t n_fft, size_t hop, size_t frames,
    double *complex_trace, size_t complex_trace_capacity) {
    size_t required_frames, spectral_elements, span, bins, table_need;
    const int planned = audio_stft_mag_phase_plan_f32(samples_count, n_fft,
        hop, &required_frames, &spectral_elements);
    if (planned != CK_AUDIO_EXTENT_OK) return planned;
    if (frames != required_frames || output_stride < frames ||
        n_fft > SIZE_MAX / (n_fft / 2 + 1)) return CK_AUDIO_EXTENT_INVALID;
    bins = n_fft / 2 + 1;
    table_need = bins * n_fft;
    if (!ck_checked_span(2 * bins, frames, output_stride, &span))
        return CK_AUDIO_EXTENT_OVERFLOW;
    if (samples_capacity < samples_count || window_capacity < n_fft ||
        table_capacity < table_need || output_capacity < span ||
        scratch_capacity < spectral_elements) return CK_AUDIO_EXTENT_LIMIT;
    if (!ck_checked_region(samples, samples_count) ||
        !ck_checked_region(window, n_fft) ||
        !ck_checked_region(cos_table, table_need) ||
        !ck_checked_region(sin_table, table_need) ||
        !ck_checked_region(output, span) ||
        !ck_checked_region(scratch, spectral_elements) ||
        ck_checked_overlap(output, span, samples, samples_count) ||
        ck_checked_overlap(output, span, window, n_fft) ||
        ck_checked_overlap(output, span, cos_table, table_need) ||
        ck_checked_overlap(output, span, sin_table, table_need) ||
        ck_checked_overlap(output, span, scratch, spectral_elements) ||
        ck_checked_overlap(scratch, spectral_elements, samples, samples_count) ||
        ck_checked_overlap(scratch, spectral_elements, window, n_fft) ||
        ck_checked_overlap(scratch, spectral_elements, cos_table, table_need) ||
        ck_checked_overlap(scratch, spectral_elements, sin_table, table_need))
        return CK_AUDIO_EXTENT_INVALID;
    if (!ck_checked_finite(samples, samples_count) ||
        !ck_checked_finite(window, n_fft) ||
        !ck_checked_finite(cos_table, table_need) ||
        !ck_checked_finite(sin_table, table_need))
        return CK_AUDIO_EXTENT_INVALID;
    if (complex_trace) {
        if (complex_trace_capacity < spectral_elements)
            return CK_AUDIO_EXTENT_LIMIT;
        if (!stft_trace_region(complex_trace, spectral_elements) ||
            stft_trace_overlaps(complex_trace, spectral_elements, samples, samples_count) ||
            stft_trace_overlaps(complex_trace, spectral_elements, window, n_fft) ||
            stft_trace_overlaps(complex_trace, spectral_elements, cos_table, table_need) ||
            stft_trace_overlaps(complex_trace, spectral_elements, sin_table, table_need) ||
            stft_trace_overlaps(complex_trace, spectral_elements, output, span) ||
            stft_trace_overlaps(complex_trace, spectral_elements, scratch, spectral_elements))
            return CK_AUDIO_EXTENT_INVALID;
    } else if (complex_trace_capacity) return CK_AUDIO_EXTENT_INVALID;
    const size_t center = n_fft / 2;
    for (size_t frame = 0; frame < frames; ++frame) {
        for (size_t bin = 0; bin < bins; ++bin) {
            double real = 0.0, imag = 0.0;
            for (size_t tap = 0; tap < n_fft; ++tap) {
                /* samples_count > center guarantees one reflection suffices. */
                const int64_t unreflected = (int64_t)(frame * hop + tap) -
                    (int64_t)center;
                size_t source;
                if (unreflected < 0) source = (size_t)(-unreflected);
                else if ((uint64_t)unreflected >= samples_count)
                    source = 2 * (samples_count - 1) - (size_t)unreflected;
                else source = (size_t)unreflected;
                const double value = (double)samples[source] * window[tap];
                real += value * cos_table[bin * n_fft + tap];
                imag += value * sin_table[bin * n_fft + tap];
            }
            /* The Nyquist bin of a real-input even FFT has exactly zero
             * imaginary part. Trig-table roundoff otherwise flips +pi to
             * -pi, which is not interchangeable for downstream convolutions. */
            if (bin == bins - 1) imag = 0.0;
            const float magnitude = (float)hypot(real, imag);
            const float phase = (float)atan2(imag, real);
            if (!isfinite(magnitude) || !isfinite(phase))
                return CK_AUDIO_EXTENT_INVALID;
            if (complex_trace) {
                const size_t index = 2 * (bin * frames + frame);
                complex_trace[index] = real;
                complex_trace[index + 1] = imag;
            }
            scratch[bin * frames + frame] = magnitude;
            scratch[(bins + bin) * frames + frame] = phase;
        }
    }
    for (size_t row = 0; row < 2 * bins; ++row)
        memcpy(output + row * output_stride, scratch + row * frames,
            frames * sizeof(float));
    return CK_AUDIO_EXTENT_OK;
}

int audio_stft_mag_phase_checked_f32(const float *samples, size_t samples_capacity,
    const float *window, size_t window_capacity,
    const float *cos_table, const float *sin_table, size_t table_capacity,
    float *output, size_t output_capacity, size_t output_stride,
    float *scratch, size_t scratch_capacity,
    size_t samples_count, size_t n_fft, size_t hop, size_t frames) {
    return stft_mag_phase_impl(samples, samples_capacity, window,
        window_capacity, cos_table, sin_table, table_capacity, output,
        output_capacity, output_stride, scratch, scratch_capacity,
        samples_count, n_fft, hop, frames, NULL, 0);
}

/* Diagnostic-only: records the actual FP64 real/imag values passed to atan2.
 * Production calls the same implementation with no trace buffer. */
int audio_stft_mag_phase_trace_checked_f32(const float *samples,
    size_t samples_capacity, const float *window, size_t window_capacity,
    const float *cos_table, const float *sin_table, size_t table_capacity,
    float *output, size_t output_capacity, size_t output_stride,
    float *scratch, size_t scratch_capacity,
    size_t samples_count, size_t n_fft, size_t hop, size_t frames,
    double *complex_trace, size_t complex_trace_capacity) {
    if (!complex_trace) return CK_AUDIO_EXTENT_INVALID;
    return stft_mag_phase_impl(samples, samples_capacity, window,
        window_capacity, cos_table, sin_table, table_capacity, output,
        output_capacity, output_stride, scratch, scratch_capacity,
        samples_count, n_fft, hop, frames, complex_trace,
        complex_trace_capacity);
}
