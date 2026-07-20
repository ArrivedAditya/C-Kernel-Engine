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

int audio_pcm_s16_to_mono_f32(
    const int16_t *interleaved,
    int n_frames,
    int n_channels,
    float *mono)
{
    if (interleaved == NULL || mono == NULL) {
        return -1;
    }
    if (n_frames <= 0 || n_channels <= 0) {
        return -2;
    }
    const float scale = 1.0f / 32768.0f;
    for (int frame = 0; frame < n_frames; ++frame) {
        float sum = 0.0f;
        for (int channel = 0; channel < n_channels; ++channel) {
            sum += (float)interleaved[(size_t)frame * n_channels + channel];
        }
        mono[frame] = (sum / (float)n_channels) * scale;
    }
    return 0;
}

int audio_resampled_frame_count(
    int input_frames,
    int input_rate,
    int output_rate)
{
    if (input_frames <= 0 || input_rate <= 0 || output_rate <= 0) {
        return -1;
    }
    return 1 + (int)(((long long)(input_frames - 1) * output_rate) / input_rate);
}

int audio_resample_linear_f32(
    const float *input,
    int input_frames,
    int input_rate,
    float *output,
    int output_frames,
    int output_rate)
{
    if (input == NULL || output == NULL) {
        return -1;
    }
    const int expected = audio_resampled_frame_count(input_frames, input_rate, output_rate);
    if (expected <= 0 || output_frames != expected) {
        return -2;
    }
    for (int frame = 0; frame < output_frames; ++frame) {
        const long long numerator = (long long)frame * input_rate;
        const int left = (int)(numerator / output_rate);
        const int right = left + 1 < input_frames ? left + 1 : left;
        const float fraction = (float)(numerator % output_rate) / (float)output_rate;
        output[frame] = fmaf(input[right] - input[left], fraction, input[left]);
    }
    return 0;
}

int audio_stft_precompute_tables_f32(
    int n_fft,
    float *window,
    float *cos_table,
    float *sin_table)
{
    if (window == NULL || cos_table == NULL || sin_table == NULL) {
        return -1;
    }
    if (n_fft <= 0 || (n_fft & 1) != 0) {
        return -2;
    }
    const int bins = n_fft / 2 + 1;
    for (int sample = 0; sample < n_fft; ++sample) {
        window[sample] = 0.5f - 0.5f * cosf(
            2.0f * CK_AUDIO_PI_F * (float)sample / (float)n_fft);
    }
    for (int bin = 0; bin < bins; ++bin) {
        for (int sample = 0; sample < n_fft; ++sample) {
            const float angle = -2.0f * CK_AUDIO_PI_F *
                (float)(bin * sample) / (float)n_fft;
            const size_t index = (size_t)bin * n_fft + sample;
            cos_table[index] = cosf(angle);
            sin_table[index] = sinf(angle);
        }
    }
    return 0;
}

int audio_stft_power_precomputed_f32(
    const float *samples,
    int n_samples,
    const float *window,
    const float *cos_table,
    const float *sin_table,
    int n_fft,
    int hop_length,
    float *power,
    int n_frames)
{
    if (samples == NULL || window == NULL || cos_table == NULL ||
        sin_table == NULL || power == NULL) {
        return -1;
    }
    if (n_fft <= 0 || hop_length <= 0 || n_samples <= n_fft / 2 ||
        (n_fft & 1) != 0 || n_frames <= 0) {
        return -2;
    }
    if (n_frames != n_samples / hop_length) {
        return -3;
    }
    const int bins = n_fft / 2 + 1;
    const int center = n_fft / 2;
    for (int frame = 0; frame < n_frames; ++frame) {
        for (int bin = 0; bin < bins; ++bin) {
            const float *cos_row = cos_table + (size_t)bin * n_fft;
            const float *sin_row = sin_table + (size_t)bin * n_fft;
            float real = 0.0f;
            float imag = 0.0f;
            for (int sample = 0; sample < n_fft; ++sample) {
                const int source = reflect_index(
                    frame * hop_length + sample - center, n_samples);
                const float value = samples[source] * window[sample];
                real = fmaf(value, cos_row[sample], real);
                imag = fmaf(value, sin_row[sample], imag);
            }
            power[(size_t)frame * bins + bin] =
                fmaf(real, real, imag * imag);
        }
    }
    return 0;
}

int audio_conv1d_channel_major_f32(
    const float *input,
    const float *weight,
    const float *bias,
    float *output,
    int input_channels,
    int output_channels,
    int input_frames,
    int kernel_size,
    int stride,
    int padding,
    int output_frames)
{
    if (input == NULL || weight == NULL || output == NULL) {
        return -1;
    }
    if (input_channels <= 0 || output_channels <= 0 || input_frames <= 0 ||
        kernel_size <= 0 || stride <= 0 || padding < 0 || output_frames <= 0) {
        return -2;
    }
    const int expected = (input_frames + 2 * padding - kernel_size) / stride + 1;
    if (output_frames != expected) {
        return -3;
    }
    for (int out_channel = 0; out_channel < output_channels; ++out_channel) {
        const float *weight_channel = weight +
            (size_t)out_channel * input_channels * kernel_size;
        float *output_channel = output + (size_t)out_channel * output_frames;
        for (int out_frame = 0; out_frame < output_frames; ++out_frame) {
            float sum = bias != NULL ? bias[out_channel] : 0.0f;
            for (int in_channel = 0; in_channel < input_channels; ++in_channel) {
                const float *input_channel = input + (size_t)in_channel * input_frames;
                const float *weight_row = weight_channel + (size_t)in_channel * kernel_size;
                for (int kernel = 0; kernel < kernel_size; ++kernel) {
                    const int in_frame = out_frame * stride + kernel - padding;
                    if (in_frame >= 0 && in_frame < input_frames) {
                        sum = fmaf(input_channel[in_frame], weight_row[kernel], sum);
                    }
                }
            }
            output_channel[out_frame] = sum;
        }
    }
    return 0;
}

int audio_transpose_channel_to_token_f32(
    const float *input,
    float *output,
    int channels,
    int frames)
{
    if (input == NULL || output == NULL) {
        return -1;
    }
    if (channels <= 0 || frames <= 0) {
        return -2;
    }
    for (int frame = 0; frame < frames; ++frame) {
        for (int channel = 0; channel < channels; ++channel) {
            output[(size_t)frame * channels + channel] =
                input[(size_t)channel * frames + frame];
        }
    }
    return 0;
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
