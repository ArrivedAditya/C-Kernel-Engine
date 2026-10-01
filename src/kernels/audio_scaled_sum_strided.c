#include "checked_float_buffer.h"

/* Output[c, t] = (left[c, t] + right[c, t]) * scale.
 * The three physical row strides may differ. Only the valid columns are read
 * and written; capacities describe reserved storage, not valid extents.
 */
int audio_scaled_sum_strided_f32_checked(
    const float *left, size_t left_elements, size_t left_stride,
    const float *right, size_t right_elements, size_t right_stride,
    float *output, size_t output_elements, size_t output_stride,
    size_t rows, size_t columns, float scale)
{
    size_t nl, nr, no;
    if (!ck_checked_span(rows, columns, left_stride, &nl) ||
        !ck_checked_span(rows, columns, right_stride, &nr) ||
        !ck_checked_span(rows, columns, output_stride, &no) ||
        left_elements < nl || right_elements < nr || output_elements < no)
        return -2;
    if (!isfinite(scale) ||
        !ck_checked_region(left, nl) || !ck_checked_region(right, nr) ||
        !ck_checked_region(output, no) ||
        ck_checked_overlap(left, nl, output, no) ||
        ck_checked_overlap(right, nr, output, no))
        return -1;
    /* Preflight every element so failure never publishes a partial output. */
    for (size_t row = 0; row < rows; ++row) {
        for (size_t column = 0; column < columns; ++column) {
            const float a = left[row * left_stride + column];
            const float b = right[row * right_stride + column];
            const float sum = a + b;
            if (!isfinite(a) || !isfinite(b) || !isfinite(sum) ||
                !isfinite(sum * scale))
                return -3;
        }
    }
    for (size_t row = 0; row < rows; ++row)
        for (size_t column = 0; column < columns; ++column)
            output[row * output_stride + column] =
                (left[row * left_stride + column] +
                 right[row * right_stride + column]) * scale;
    return 0;
}
