#ifndef CKERNEL_SELECT_NEXT_ID_CHECKED_H
#define CKERNEL_SELECT_NEXT_ID_CHECKED_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    CK_NEXT_ID_CONTINUE = 0,
    CK_NEXT_ID_STOP = 1,
    CK_NEXT_ID_BUDGET_EXHAUSTED = 2
};

/* Select the first maximum in the final finite row. The stop ID is excluded
 * while emitted_frames < min_frames. A declared ID range is suppressed except
 * for the stop ID once the minimum is met. A positive status is a normal control
 * outcome and stops dependent generated calls. Invalid calls return a negative
 * status and leave selected unchanged; budget exhaustion also leaves it
 * unchanged. EOS writes selected before returning CK_NEXT_ID_STOP. */
int ck_select_next_id_checked_f32(
    const float *values, size_t values_elements,
    size_t rows, size_t channels, size_t row_stride,
    size_t emitted_frames, size_t min_frames, size_t max_frames,
    size_t stop_id, size_t suppressed_begin, size_t suppressed_end,
    int32_t *selected, size_t selected_elements);

#ifdef __cplusplus
}
#endif

#endif
