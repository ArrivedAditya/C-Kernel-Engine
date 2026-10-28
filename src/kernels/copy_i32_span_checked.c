#include "ckernel_copy_i32_span.h"

#include <string.h>

static int valid_region(const int32_t *pointer, size_t elements) {
    return pointer && (uintptr_t)pointer % _Alignof(int32_t) == 0 &&
           elements <= SIZE_MAX / sizeof(int32_t) &&
           (uintptr_t)pointer <= UINTPTR_MAX - elements * sizeof(int32_t);
}

int ck_copy_i32_span_checked(const int32_t *source, size_t source_elements,
                             size_t source_offset, int32_t *destination,
                             size_t destination_elements,
                             size_t destination_offset, size_t count) {
    if (!count || source_offset > source_elements ||
        count > source_elements - source_offset ||
        destination_offset > destination_elements ||
        count > destination_elements - destination_offset)
        return -2;
    if (!valid_region(source, source_elements) ||
        !valid_region(destination, destination_elements))
        return -1;
    const uintptr_t src = (uintptr_t)(source + source_offset);
    const uintptr_t dst = (uintptr_t)(destination + destination_offset);
    const size_t bytes = count * sizeof(int32_t);
    if (src < dst + bytes && dst < src + bytes)
        return -1;
    memcpy(destination + destination_offset, source + source_offset, bytes);
    return 0;
}
