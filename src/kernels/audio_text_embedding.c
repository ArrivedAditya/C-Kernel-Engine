#include "ckernel_audio_text_embedding.h"
#include "ckernel_tts.h"
#include "checked_float_buffer.h"

#include <math.h>
#include <limits.h>
#include <stdint.h>

int audio_argmax_last_row_checked_f32(
    const float *values, size_t values_elements,
    size_t rows, size_t channels, size_t row_stride,
    int32_t *selected, size_t selected_elements)
{
    size_t span;
    if (!ck_checked_span(rows, channels, row_stride, &span) ||
        channels > INT32_MAX || values_elements < span || selected_elements < 1)
        return -2;
    if (!ck_checked_region(values, span) || !selected ||
        (uintptr_t)selected % _Alignof(int32_t) != 0 ||
        (uintptr_t)selected > UINTPTR_MAX - sizeof(int32_t) ||
        ck_checked_overlap(values, span, (const float *)selected, 1))
        return -1;
    const float *row = values + (rows - 1) * row_stride;
    if (!isfinite(row[0])) return -3;
    size_t best = 0;
    float maximum = row[0];
    for (size_t index = 1; index < channels; ++index) {
        const float value = row[index];
        if (!isfinite(value)) return -3;
        if (value > maximum) {
            maximum = value;
            best = index;
        }
    }
    *selected = (int32_t)best;
    return 0;
}

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

int audio_text_embedding_token_major_f32(
    const int32_t *ids, size_t id_elements,
    const float *table, size_t table_elements,
    size_t vocabulary, size_t channels, size_t table_stride,
    float *output, size_t output_elements,
    size_t tokens, size_t output_stride) {
    size_t table_span, output_span;
    if (!vocabulary || !tokens || vocabulary > INT32_MAX ||
        !ck_checked_span(vocabulary, channels, table_stride, &table_span) ||
        !ck_checked_span(tokens, channels, output_stride, &output_span))
        return CK_AUDIO_EXTENT_INVALID;
    if (id_elements < tokens || table_elements < table_span ||
        output_elements < output_span) return CK_AUDIO_EXTENT_LIMIT;
    if (!ids || (uintptr_t)ids % _Alignof(int32_t) != 0 ||
        tokens > SIZE_MAX / sizeof(int32_t) ||
        (uintptr_t)ids > UINTPTR_MAX - tokens * sizeof(int32_t) ||
        !ck_checked_region(table, table_span) ||
        !ck_checked_region(output, output_span) ||
        ck_checked_overlap(table, table_span, output, output_span) ||
        ((uintptr_t)ids < (uintptr_t)output + output_span * sizeof(float) &&
         (uintptr_t)output < (uintptr_t)ids + tokens * sizeof(int32_t)))
        return CK_AUDIO_EXTENT_INVALID;
    for (size_t token = 0; token < tokens; ++token) {
        const int32_t id = ids[token];
        if (id < 0 || (size_t)id >= vocabulary) return CK_AUDIO_EXTENT_INVALID;
        const float *row = table + (size_t)id * table_stride;
        if (!ck_checked_finite(row, channels)) return CK_AUDIO_EXTENT_INVALID;
    }
    for (size_t token = 0; token < tokens; ++token) {
        const float *row = table + (size_t)ids[token] * table_stride;
        float *dest = output + token * output_stride;
        for (size_t channel = 0; channel < channels; ++channel)
            dest[channel] = row[channel];
    }
    return CK_AUDIO_EXTENT_OK;
}
