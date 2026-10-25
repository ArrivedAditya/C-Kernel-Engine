#include "ckernel_indexed_sum_rows_checked.h"
#include "checked_float_buffer.h"

#include <math.h>
#include <stdint.h>

static int checked_i32_region(const int32_t *ptr, size_t count)
{
    return ptr && (uintptr_t)ptr % _Alignof(int32_t) == 0 &&
           count <= SIZE_MAX / sizeof(int32_t) &&
           (uintptr_t)ptr <= UINTPTR_MAX - count * sizeof(int32_t);
}

static int overlaps_bytes(const void *a, size_t a_bytes,
                          const void *b, size_t b_bytes)
{
    const uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x < y + b_bytes && y < x + a_bytes;
}

int indexed_sum_rows_checked_f32(
    const float *left, size_t left_elements, size_t left_stride, size_t left_rows,
    const float *right, size_t right_elements, size_t right_stride, size_t right_rows,
    const int32_t *left_indices, size_t left_index_elements,
    const int32_t *right_indices, size_t right_index_elements,
    float *output, size_t output_elements, size_t output_stride,
    size_t rows, size_t channels)
{
    size_t left_span, right_span, output_span;
    if (!ck_checked_span(left_rows, channels, left_stride, &left_span) ||
        !ck_checked_span(right_rows, channels, right_stride, &right_span) ||
        !ck_checked_span(rows, channels, output_stride, &output_span) ||
        left_elements < left_span || right_elements < right_span ||
        output_elements < output_span || left_index_elements < rows ||
        right_index_elements < rows || left_rows > INT32_MAX ||
        right_rows > INT32_MAX) return -2;
    if (!ck_checked_region(left, left_span) ||
        !ck_checked_region(right, right_span) ||
        !ck_checked_region(output, output_span) ||
        !checked_i32_region(left_indices, rows) ||
        !checked_i32_region(right_indices, rows) ||
        overlaps_bytes(left, left_span * sizeof(float), output,
                       output_span * sizeof(float)) ||
        overlaps_bytes(right, right_span * sizeof(float), output,
                       output_span * sizeof(float)) ||
        overlaps_bytes(left_indices, rows * sizeof(int32_t), output,
                       output_span * sizeof(float)) ||
        overlaps_bytes(right_indices, rows * sizeof(int32_t), output,
                       output_span * sizeof(float))) return -1;

    /* Preflight every row so rejection cannot expose a partial output. */
    for (size_t row = 0; row < rows; ++row) {
        const int32_t li = left_indices[row], ri = right_indices[row];
        if (li < -1 || ri < -1 || (li == -1 && ri == -1) ||
            (li >= 0 && (size_t)li >= left_rows) ||
            (ri >= 0 && (size_t)ri >= right_rows)) return -2;
        for (size_t channel = 0; channel < channels; ++channel) {
            const float l = li < 0 ? 0.0f : left[(size_t)li * left_stride + channel];
            const float r = ri < 0 ? 0.0f : right[(size_t)ri * right_stride + channel];
            if (!isfinite(l) || !isfinite(r) ||
                (li >= 0 && ri >= 0 && !isfinite(l + r))) return -3;
        }
    }
    for (size_t row = 0; row < rows; ++row) {
        const int32_t li = left_indices[row], ri = right_indices[row];
        for (size_t channel = 0; channel < channels; ++channel) {
            const float l = li < 0 ? 0.0f : left[(size_t)li * left_stride + channel];
            const float r = ri < 0 ? 0.0f : right[(size_t)ri * right_stride + channel];
            output[row * output_stride + channel] =
                li < 0 ? r : (ri < 0 ? l : l + r);
        }
    }
    return 0;
}
