#include "ckernel_equal_i32_scalar.h"

static int valid_scalar(const int32_t *pointer, size_t elements) {
    return pointer && elements >= 1 &&
           (uintptr_t)pointer % _Alignof(int32_t) == 0 &&
           (uintptr_t)pointer <= UINTPTR_MAX - sizeof(int32_t);
}

int ck_equal_i32_scalar_checked(const int32_t *requested,
                                size_t requested_elements,
                                const int32_t *expected,
                                size_t expected_elements,
                                int32_t *checked,
                                size_t checked_elements) {
    if (!valid_scalar(requested, requested_elements) ||
        !valid_scalar(expected, expected_elements) ||
        !valid_scalar(checked, checked_elements))
        return -1;
    const uintptr_t out = (uintptr_t)checked;
    const uintptr_t in_a = (uintptr_t)requested;
    const uintptr_t in_b = (uintptr_t)expected;
    if ((out < in_a + sizeof(int32_t) && in_a < out + sizeof(int32_t)) ||
        (out < in_b + sizeof(int32_t) && in_b < out + sizeof(int32_t)))
        return -1;
    const int32_t value = *requested;
    if (value != *expected)
        return -2;
    *checked = value;
    return 0;
}
