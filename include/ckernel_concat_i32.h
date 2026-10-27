#ifndef CKERNEL_CONCAT_I32_H
#define CKERNEL_CONCAT_I32_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Concatenate two nonempty i32 spans into caller-owned output. The inputs may
 * overlap each other, but output must be disjoint. Invalid calls leave output
 * unchanged. Return 0 on success, -1 for invalid/overlapping storage, and -2
 * for invalid counts or insufficient capacity. */
int ck_concat_i32_checked(const int32_t *left, size_t left_elements,
                          size_t left_count, const int32_t *right,
                          size_t right_elements, size_t right_count,
                          int32_t *output, size_t output_elements);

#ifdef __cplusplus
}
#endif

#endif
