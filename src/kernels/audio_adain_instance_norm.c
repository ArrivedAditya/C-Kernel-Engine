/* FP32 channelwise InstanceNorm1d followed by style-conditioned affine.
 *
 * PyTorch reference: norm = InstanceNorm1d(affine=True)(x), then
 * (1 + style_gamma) * norm + style_beta. The statistics use the valid frames
 * of each channel, biased variance, and ascending FP64 reductions. The style
 * projection is a separate operation in the circuit.
 */
#include "ckernel_audio.h"

#include <math.h>
#include <stdint.h>

static int checked_span(size_t channels, size_t stride, size_t frames,
                        size_t *result)
{
    if (channels == 0u || frames == 0u || stride < frames ||
        (channels - 1u) > (SIZE_MAX - frames) / stride) return 0;
    *result = (channels - 1u) * stride + frames;
    return *result <= (size_t)PTRDIFF_MAX / sizeof(float);
}

static int channel_stats(const float *row, size_t frames, float epsilon,
                         double *mean, double *inverse_std)
{
    double total = 0.0;
    for (size_t frame = 0u; frame < frames; ++frame)
        total += (double)row[frame];
    *mean = total / (double)frames;
    double squared = 0.0;
    for (size_t frame = 0u; frame < frames; ++frame) {
        const double delta = (double)row[frame] - *mean;
        squared += delta * delta;
    }
    *inverse_std = 1.0 / sqrt(squared / (double)frames + (double)epsilon);
    return isfinite(*mean) && isfinite(*inverse_std);
}

int audio_adain_instance_norm_f32(
    const float *input, size_t input_elements, size_t input_stride,
    const float *norm_weight, size_t norm_weight_elements,
    const float *norm_bias, size_t norm_bias_elements,
    const float *style_affine, size_t style_affine_elements,
    float *output, size_t output_elements, size_t output_stride,
    size_t channels, size_t frames, float epsilon)
{
    if (input == NULL || norm_weight == NULL || norm_bias == NULL ||
        style_affine == NULL || output == NULL ||
        !isfinite(epsilon) || epsilon <= 0.0f ||
        channels == 0u || channels > SIZE_MAX / 2u) return -1;
    size_t input_required, output_required;
    if (!checked_span(channels, input_stride, frames, &input_required) ||
        !checked_span(channels, output_stride, frames, &output_required))
        return -1;
    if (input_elements < input_required || output_elements < output_required ||
        norm_weight_elements < channels || norm_bias_elements < channels ||
        style_affine_elements < 2u * channels) return -2;

    for (size_t channel = 0u; channel < channels; ++channel) {
        if (!isfinite(norm_weight[channel]) || !isfinite(norm_bias[channel]) ||
            !isfinite(style_affine[channel]) ||
            !isfinite(style_affine[channels + channel])) return -3;
        const float *row = input + channel * input_stride;
        for (size_t frame = 0u; frame < frames; ++frame)
            if (!isfinite(row[frame])) return -3;
    }

    /* Dry-run every result before the first output write. The repeated scalar
     * pass trades throughput for a small, allocation-free reference kernel. */
    for (int pass = 0; pass < 2; ++pass) {
        for (size_t channel = 0u; channel < channels; ++channel) {
            const float *row = input + channel * input_stride;
            float *output_row = output + channel * output_stride;
            double mean, inverse_std;
            if (!channel_stats(row, frames, epsilon, &mean, &inverse_std))
                return -3;
            const float weight = norm_weight[channel];
            const float bias = norm_bias[channel];
            const float gamma = style_affine[channel];
            const float beta = style_affine[channels + channel];
            for (size_t frame = 0u; frame < frames; ++frame) {
                const float normalized = (float)(
                    ((double)row[frame] - mean) * inverse_std);
                const float affine = normalized * weight + bias;
                const float value = (1.0f + gamma) * affine + beta;
                if (!isfinite(value)) return -3;
                if (pass != 0) output_row[frame] = value;
            }
        }
    }
    return 0;
}
