/**
 * @file audio_kernels.c
 * @brief Numerically explicit audio frontend reference kernels.
 */

#include "ckernel_audio.h"

#include <math.h>
#include <limits.h>
#include <stddef.h>
#include <string.h>

#define CK_AUDIO_PI_F 3.14159265358979323846f
#define CK_AUDIO_PI_D 3.14159265358979323846264338327950288

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

static uint16_t read_u16_le(const uint8_t *p)
{
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t read_u32_le(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
        ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

int audio_wav_parse_memory(
    const uint8_t *bytes,
    size_t byte_count,
    CKAudioWavInfo *info)
{
    if (bytes == NULL || info == NULL) {
        return -1;
    }
    if (byte_count < 12 || memcmp(bytes, "RIFF", 4) != 0 ||
        memcmp(bytes + 8, "WAVE", 4) != 0) {
        return -2;
    }
    const size_t riff_end = (size_t)read_u32_le(bytes + 4) + 8u;
    if (riff_end < 12 || riff_end > byte_count) {
        return -3;
    }
    memset(info, 0, sizeof(*info));
    int found_format = 0;
    int found_data = 0;
    size_t offset = 12;
    while (offset + 8 <= riff_end) {
        const uint8_t *chunk = bytes + offset;
        const uint32_t chunk_bytes = read_u32_le(chunk + 4);
        const size_t payload = offset + 8;
        if ((size_t)chunk_bytes > riff_end - payload) {
            return -3;
        }
        if (!found_format && memcmp(chunk, "fmt ", 4) == 0) {
            if (chunk_bytes < 16) {
                return -4;
            }
            info->format_tag = (int)read_u16_le(bytes + payload);
            info->channels = (int)read_u16_le(bytes + payload + 2);
            info->sample_rate = (int)read_u32_le(bytes + payload + 4);
            info->bits_per_sample = (int)read_u16_le(bytes + payload + 14);
            found_format = 1;
        } else if (!found_data && memcmp(chunk, "data", 4) == 0) {
            info->data_offset = payload;
            info->data_bytes = chunk_bytes;
            found_data = 1;
        }
        const size_t padded = (size_t)chunk_bytes + ((size_t)chunk_bytes & 1u);
        if (padded > SIZE_MAX - payload) {
            return -3;
        }
        offset = payload + padded;
    }
    if (!found_format || !found_data || info->format_tag != 1 ||
        info->channels <= 0 || info->sample_rate <= 0 ||
        info->bits_per_sample != 16) {
        return -5;
    }
    const size_t bytes_per_frame = (size_t)info->channels * 2u;
    if (bytes_per_frame == 0 || info->data_bytes % bytes_per_frame != 0 ||
        info->data_bytes / bytes_per_frame > (size_t)INT_MAX) {
        return -6;
    }
    info->frames = (int)(info->data_bytes / bytes_per_frame);
    return info->frames > 0 ? 0 : -6;
}

int audio_wav_decode_pcm16_mono_f32(
    const uint8_t *bytes,
    size_t byte_count,
    const CKAudioWavInfo *info,
    float *mono,
    int mono_capacity)
{
    if (bytes == NULL || info == NULL || mono == NULL) {
        return -1;
    }
    if (info->format_tag != 1 || info->bits_per_sample != 16 ||
        info->channels <= 0 || info->frames <= 0 || mono_capacity < info->frames ||
        info->data_offset > byte_count || info->data_bytes > byte_count - info->data_offset) {
        return -2;
    }
    const uint8_t *pcm = bytes + info->data_offset;
    const float scale = 1.0f / 32768.0f;
    for (int frame = 0; frame < info->frames; ++frame) {
        float sum = 0.0f;
        for (int channel = 0; channel < info->channels; ++channel) {
            const size_t index = ((size_t)frame * info->channels + channel) * 2u;
            sum += (float)(int16_t)read_u16_le(pcm + index);
        }
        mono[frame] = (sum / (float)info->channels) * scale;
    }
    return info->frames;
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

int audio_resample_windowed_sinc_f32(
    const float *input,
    int input_frames,
    int input_rate,
    float *output,
    int output_frames,
    int output_rate,
    int radius)
{
    if (input == NULL || output == NULL) {
        return -1;
    }
    const int expected = audio_resampled_frame_count(input_frames, input_rate, output_rate);
    if (expected <= 0 || output_frames != expected || radius < 2 || radius > 128) {
        return -2;
    }
    const double ratio = (double)output_rate / (double)input_rate;
    const double cutoff = ratio < 1.0 ? ratio : 1.0;
    for (int frame = 0; frame < output_frames; ++frame) {
        const double source = (double)frame * (double)input_rate / (double)output_rate;
        const int center = (int)floor(source);
        double weighted = 0.0;
        double weight_sum = 0.0;
        for (int tap = center - radius + 1; tap <= center + radius; ++tap) {
            if (tap < 0 || tap >= input_frames) {
                continue;
            }
            const double distance = source - (double)tap;
            const double scaled = cutoff * distance;
            const double sinc = fabs(scaled) < 1.0e-12 ? 1.0 :
                sin(CK_AUDIO_PI_D * scaled) / (CK_AUDIO_PI_D * scaled);
            const double window_x = distance / (double)radius;
            if (fabs(window_x) >= 1.0) {
                continue;
            }
            const double window = 0.5 * (1.0 + cos(CK_AUDIO_PI_D * window_x));
            const double weight = cutoff * sinc * window;
            weighted += (double)input[tap] * weight;
            weight_sum += weight;
        }
        output[frame] = weight_sum != 0.0 ? (float)(weighted / weight_sum) : 0.0f;
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

int audio_stft_power_fft400_f32(
    const float *samples,
    int n_samples,
    const float *window,
    const float *cos_table,
    const float *sin_table,
    int hop_length,
    float *power,
    int n_frames,
    float *fft_scratch)
{
    const int n_fft = CK_AUDIO_WHISPER_N_FFT;
    const int radix = 20;
    const int bins = CK_AUDIO_WHISPER_POWER_BINS;
    if (samples == NULL || window == NULL || cos_table == NULL ||
        sin_table == NULL || power == NULL || fft_scratch == NULL) {
        return -1;
    }
    if (hop_length <= 0 || n_samples <= n_fft / 2 || n_frames <= 0 ||
        n_frames != n_samples / hop_length) {
        return -2;
    }
    float *stage_real = fft_scratch;
    float *stage_imag = fft_scratch + n_fft;
    const int center = n_fft / 2;
    for (int frame = 0; frame < n_frames; ++frame) {
        for (int p = 0; p < radix; ++p) {
            for (int k = 0; k < radix; ++k) {
                float real = 0.0f;
                float imag = 0.0f;
                for (int q = 0; q < radix; ++q) {
                    const int sample = p + radix * q;
                    const int source = reflect_index(
                        frame * hop_length + sample - center, n_samples);
                    const float value = samples[source] * window[sample];
                    const size_t twiddle = (size_t)k * n_fft + radix * q;
                    real = fmaf(value, cos_table[twiddle], real);
                    imag = fmaf(value, sin_table[twiddle], imag);
                }
                stage_real[p * radix + k] = real;
                stage_imag[p * radix + k] = imag;
            }
        }
        for (int frequency = 0; frequency < bins; ++frequency) {
            const int k = frequency % radix;
            float real = 0.0f;
            float imag = 0.0f;
            for (int p = 0; p < radix; ++p) {
                const float a = stage_real[p * radix + k];
                const float b = stage_imag[p * radix + k];
                const size_t twiddle = (size_t)frequency * n_fft + p;
                const float c = cos_table[twiddle];
                const float s = sin_table[twiddle];
                real = fmaf(a, c, fmaf(-b, s, real));
                imag = fmaf(a, s, fmaf(b, c, imag));
            }
            power[(size_t)frame * bins + frequency] =
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
