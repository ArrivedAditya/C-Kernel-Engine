#include "ckernel_audio_text_embedding.h"
#include "ckernel_tts.h"

#include <math.h>
#include <stdint.h>

int audio_text_embedding_channel_major_f32(
    const int32_t *ids, size_t id_elements,
    const float *table, size_t table_elements,
    size_t vocabulary, size_t channels, size_t table_stride,
    float *output, size_t output_elements,
    size_t tokens, size_t output_stride) {
    if (!ids || !table || !output || !vocabulary || !channels ||
        !tokens || table_stride < channels || output_stride < tokens)
        return CK_AUDIO_EXTENT_INVALID;
    if (vocabulary - 1 > (SIZE_MAX - channels) / table_stride ||
        channels - 1 > (SIZE_MAX - tokens) / output_stride)
        return CK_AUDIO_EXTENT_OVERFLOW;
    size_t table_required = (vocabulary - 1) * table_stride + channels;
    size_t output_required = (channels - 1) * output_stride + tokens;
    if (table_required > SIZE_MAX / sizeof(float) ||
        output_required > SIZE_MAX / sizeof(float))
        return CK_AUDIO_EXTENT_OVERFLOW;
    if (id_elements < tokens || table_elements < table_required ||
        output_elements < output_required)
        return CK_AUDIO_EXTENT_LIMIT;
    for (size_t token = 0; token < tokens; ++token) {
        int32_t id = ids[token];
        if (id < 0 || (size_t)id >= vocabulary)
            return CK_AUDIO_EXTENT_INVALID;
        const float *row = table + (size_t)id * table_stride;
        for (size_t channel = 0; channel < channels; ++channel)
            if (!isfinite(row[channel])) return CK_AUDIO_EXTENT_INVALID;
    }
    for (size_t channel = 0; channel < channels; ++channel) {
        float *out = output + channel * output_stride;
        for (size_t token = 0; token < tokens; ++token)
            out[token] = table[(size_t)ids[token] * table_stride + channel];
    }
    return CK_AUDIO_EXTENT_OK;
}
