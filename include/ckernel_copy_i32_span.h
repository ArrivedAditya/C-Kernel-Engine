#ifndef CKERNEL_COPY_I32_SPAN_H
#define CKERNEL_COPY_I32_SPAN_H

#include <stddef.h>
#include <stdint.h>

int ck_copy_i32_span_checked(const int32_t *source, size_t source_elements,
                             size_t source_offset, int32_t *destination,
                             size_t destination_elements,
                             size_t destination_offset, size_t count);

#endif
