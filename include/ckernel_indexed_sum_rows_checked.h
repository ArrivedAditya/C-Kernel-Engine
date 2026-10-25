#ifndef CKERNEL_INDEXED_SUM_ROWS_CHECKED_H
#define CKERNEL_INDEXED_SUM_ROWS_CHECKED_H

#include <stddef.h>
#include <stdint.h>

/* Each output row sums the selected left and right rows. An index of -1
 * contributes zero; every output row must select at least one input row. */
int indexed_sum_rows_checked_f32(
    const float *left, size_t left_elements, size_t left_stride, size_t left_rows,
    const float *right, size_t right_elements, size_t right_stride, size_t right_rows,
    const int32_t *left_indices, size_t left_index_elements,
    const int32_t *right_indices, size_t right_index_elements,
    float *output, size_t output_elements, size_t output_stride,
    size_t rows, size_t channels);

#endif
