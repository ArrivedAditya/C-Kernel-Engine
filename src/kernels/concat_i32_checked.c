#include "ckernel_concat_i32.h"

#include <stdint.h>
#include <string.h>

static int valid_region(const int32_t *pointer, size_t elements) {
    return pointer && (uintptr_t)pointer % _Alignof(int32_t) == 0 &&
           elements <= SIZE_MAX / sizeof(int32_t) &&
           (uintptr_t)pointer <= UINTPTR_MAX - elements * sizeof(int32_t);
}

static int overlaps(const int32_t *left, size_t left_count,
                    const int32_t *right, size_t right_count) {
    const uintptr_t a = (uintptr_t)left;
    const uintptr_t b = (uintptr_t)right;
    return a < b + right_count * sizeof(int32_t) &&
           b < a + left_count * sizeof(int32_t);
}

int ck_concat_i32_checked(const int32_t *left, size_t left_elements,
                          size_t left_count, const int32_t *right,
                          size_t right_elements, size_t right_count,
                          int32_t *output, size_t output_elements) {
    if (!left_count || !right_count || left_count > SIZE_MAX - right_count ||
        left_count > left_elements || right_count > right_elements ||
        left_count + right_count > output_elements)
        return -2;
    if (!valid_region(left, left_count) || !valid_region(right, right_count) ||
        !valid_region(output, left_count + right_count) ||
        overlaps(left, left_count, output, left_count + right_count) ||
        overlaps(right, right_count, output, left_count + right_count))
        return -1;
    memcpy(output, left, left_count * sizeof(int32_t));
    memcpy(output + left_count, right, right_count * sizeof(int32_t));
    return 0;
}
