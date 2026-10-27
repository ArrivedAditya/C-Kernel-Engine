#include "ckernel_repetition_penalty_checked.h"

#include <math.h>
#include <stdint.h>

static int overlaps(uintptr_t a, size_t a_bytes, uintptr_t b, size_t b_bytes)
{
    return a < b + b_bytes && b < a + a_bytes;
}

int ck_repetition_penalty_last_row_checked_f32(
    const float *values, size_t values_elements,
    size_t rows, size_t channels, size_t row_stride,
    const int32_t *history, size_t history_elements, size_t history_count,
    size_t penalty_numerator, size_t penalty_denominator,
    float *output, size_t output_elements)
{
    if (!rows || !channels || channels > INT32_MAX || row_stride < channels ||
        !penalty_numerator || !penalty_denominator ||
        penalty_numerator > UINT32_MAX || penalty_denominator > UINT32_MAX ||
        history_count > history_elements || output_elements < channels ||
        rows - 1 > (SIZE_MAX - channels) / row_stride)
        return -2;
    const size_t values_span = (rows - 1) * row_stride + channels;
    if (values_span > SIZE_MAX / sizeof(float) ||
        history_count > SIZE_MAX / sizeof(int32_t) ||
        channels > SIZE_MAX / sizeof(float) || values_elements < values_span)
        return -2;
    const float penalty = (float)penalty_numerator / (float)penalty_denominator;
    if (!isfinite(penalty) || penalty <= 0.0f)
        return -2;
    const size_t values_bytes = values_span * sizeof(float);
    const size_t history_bytes = history_count * sizeof(int32_t);
    const size_t output_bytes = channels * sizeof(float);
    const uintptr_t v = (uintptr_t)values, h = (uintptr_t)history;
    const uintptr_t o = (uintptr_t)output;
    if (!values || !output || (history_count && !history) ||
        v % _Alignof(float) || o % _Alignof(float) ||
        (history_count && h % _Alignof(int32_t)) ||
        v > UINTPTR_MAX - values_bytes || o > UINTPTR_MAX - output_bytes ||
        (history_count && h > UINTPTR_MAX - history_bytes) ||
        overlaps(v, values_bytes, o, output_bytes) ||
        (history_count && overlaps(h, history_bytes, o, output_bytes)))
        return -1;
    for (size_t i = 0; i < history_count; ++i)
        if (history[i] < 0 || (size_t)history[i] >= channels)
            return -2;
    const float *last = values + (rows - 1) * row_stride;
    /* Check the whole output before writing any element. */
    for (size_t j = 0; j < channels; ++j) {
        if (!isfinite(last[j])) return -3;
        int seen = 0;
        for (size_t i = 0; i < history_count; ++i)
            if ((size_t)history[i] == j) { seen = 1; break; }
        const float adjusted = seen ?
            (last[j] < 0.0f ? last[j] * penalty : last[j] / penalty) : last[j];
        if (!isfinite(adjusted)) return -3;
    }
    for (size_t j = 0; j < channels; ++j) {
        int seen = 0;
        for (size_t i = 0; i < history_count; ++i)
            if ((size_t)history[i] == j) { seen = 1; break; }
        output[j] = seen ?
            (last[j] < 0.0f ? last[j] * penalty : last[j] / penalty) : last[j];
    }
    return 0;
}
