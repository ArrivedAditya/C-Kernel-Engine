#ifndef CK_CHECKED_FLOAT_BUFFER_H
#define CK_CHECKED_FLOAT_BUFFER_H
#include <stdint.h>
#include <stddef.h>
#include <math.h>
static inline int ck_checked_span(size_t rows, size_t width, size_t stride, size_t *span) {
    if (!rows || !width || stride < width || rows - 1 > (SIZE_MAX - width) / stride) return 0;
    *span = (rows - 1) * stride + width;
    return *span <= SIZE_MAX / sizeof(float);
}
static inline int ck_checked_region(const float *ptr, size_t n) {
    return ptr && (uintptr_t)ptr % _Alignof(float) == 0 && n <= SIZE_MAX / sizeof(float)
        && (uintptr_t)ptr <= UINTPTR_MAX - n * sizeof(float);
}
static inline int ck_checked_overlap(const float *a, size_t na, const float *b, size_t nb) {
    return (uintptr_t)a < (uintptr_t)b + nb * sizeof(float)
        && (uintptr_t)b < (uintptr_t)a + na * sizeof(float);
}
static inline int ck_checked_finite(const float *ptr, size_t n) {
    for (size_t i = 0; i < n; ++i) if (!isfinite(ptr[i])) return 0;
    return 1;
}
#endif
