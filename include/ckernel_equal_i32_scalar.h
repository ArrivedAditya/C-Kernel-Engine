#ifndef CKERNEL_EQUAL_I32_SCALAR_H
#define CKERNEL_EQUAL_I32_SCALAR_H

#include <stddef.h>
#include <stdint.h>

int ck_equal_i32_scalar_checked(const int32_t *requested,
                                size_t requested_elements,
                                const int32_t *expected,
                                size_t expected_elements,
                                int32_t *checked,
                                size_t checked_elements);

#endif
