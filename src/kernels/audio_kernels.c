/**
 * @file audio_kernels.c
 * @brief Numerically explicit audio frontend reference kernels.
 */

#include "ckernel_audio.h"

#include <math.h>
#include <stddef.h>

#define CK_AUDIO_PI_F 3.14159265358979323846f

static int reflect_index(int index, int length)
{
    while (index < 0 || index >= length) {
        if (index < 0) {
            index = -index;
        } else {
            index = 2 * length - index - 2;
        }
    }
    return index;
}

int audio_whisper_stft_power_reference_f32(
    const float *samples,
    int n_samples,
    float *power,
    int n_frames)
{
    if (samples == NULL || power == NULL) {
        return -1;
    }
    if (n_samples <= CK_AUDIO_WHISPER_N_FFT / 2 || n_frames <= 0) {
        return -2;
    }
    if (n_frames != n_samples / CK_AUDIO_WHISPER_HOP_LENGTH) {
        return -3;
    }

    const int center = CK_AUDIO_WHISPER_N_FFT / 2;
    for (int frame = 0; frame < n_frames; ++frame) {
        for (int bin = 0; bin < CK_AUDIO_WHISPER_POWER_BINS; ++bin) {
            float real = 0.0f;
            float imag = 0.0f;
            for (int sample = 0; sample < CK_AUDIO_WHISPER_N_FFT; ++sample) {
                const int source = reflect_index(
                    frame * CK_AUDIO_WHISPER_HOP_LENGTH + sample - center,
                    n_samples);
                const float window = 0.5f - 0.5f * cosf(
                    2.0f * CK_AUDIO_PI_F * (float)sample /
                    (float)CK_AUDIO_WHISPER_N_FFT);
                const float value = samples[source] * window;
                const float angle = -2.0f * CK_AUDIO_PI_F *
                    (float)(bin * sample) / (float)CK_AUDIO_WHISPER_N_FFT;
                real = fmaf(value, cosf(angle), real);
                imag = fmaf(value, sinf(angle), imag);
            }
            power[(size_t)frame * CK_AUDIO_WHISPER_POWER_BINS + bin] =
                fmaf(real, real, imag * imag);
        }
    }
    return 0;
}

int audio_whisper_log_mel_from_power_reference_f32(
    const float *power,
    const float *mel_filters,
    int n_mels,
    int n_frames,
    float *log_mel)
{
    if (power == NULL || mel_filters == NULL || log_mel == NULL) {
        return -1;
    }
    if (n_mels <= 0 || n_frames <= 0) {
        return -2;
    }

    float maximum = -INFINITY;
    for (int mel = 0; mel < n_mels; ++mel) {
        const float *filter = mel_filters + (size_t)mel * CK_AUDIO_WHISPER_POWER_BINS;
        float *output = log_mel + (size_t)mel * n_frames;
        for (int frame = 0; frame < n_frames; ++frame) {
            const float *spectrum = power + (size_t)frame * CK_AUDIO_WHISPER_POWER_BINS;
            float sum = 0.0f;
            for (int bin = 0; bin < CK_AUDIO_WHISPER_POWER_BINS; ++bin) {
                sum = fmaf(filter[bin], spectrum[bin], sum);
            }
            const float value = log10f(fmaxf(sum, 1.0e-10f));
            output[frame] = value;
            maximum = fmaxf(maximum, value);
        }
    }

    const float floor = maximum - 8.0f;
    for (int mel = 0; mel < n_mels; ++mel) {
        float *output = log_mel + (size_t)mel * n_frames;
        for (int frame = 0; frame < n_frames; ++frame) {
            output[frame] = (fmaxf(output[frame], floor) + 4.0f) / 4.0f;
        }
    }
    return 0;
}

int audio_whisper_log_mel_reference_f32(
    const float *samples,
    int n_samples,
    const float *mel_filters,
    int n_mels,
    float *power_scratch,
    float *log_mel,
    int n_frames)
{
    const int stft_status = audio_whisper_stft_power_reference_f32(
        samples, n_samples, power_scratch, n_frames);
    if (stft_status != 0) {
        return stft_status;
    }
    return audio_whisper_log_mel_from_power_reference_f32(
        power_scratch, mel_filters, n_mels, n_frames, log_mel);
}
