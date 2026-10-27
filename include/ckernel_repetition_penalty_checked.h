#ifndef CKERNEL_REPETITION_PENALTY_CHECKED_H
#define CKERNEL_REPETITION_PENALTY_CHECKED_H

#include <stddef.h>
#include <stdint.h>

/* Apply the Transformers sign-dependent repetition rule to the final row.
 * Repeated IDs in history are penalized once. Rejected calls preserve output. */
int ck_repetition_penalty_last_row_checked_f32(
    const float *values, size_t values_elements,
    size_t rows, size_t channels, size_t row_stride,
    const int32_t *history, size_t history_elements, size_t history_count,
    size_t penalty_numerator, size_t penalty_denominator,
    float *output, size_t output_elements);

#endif
