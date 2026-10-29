#include "ckernel_select_next_id_checked.h"

#include <math.h>
#include <stdint.h>

int ck_select_next_id_checked_f32(
    const float *values, size_t values_elements,
    size_t rows, size_t channels, size_t row_stride,
    size_t emitted_frames, size_t min_frames, size_t max_frames,
    size_t stop_id, size_t suppressed_begin, size_t suppressed_end,
    int32_t *selected, size_t selected_elements)
{
    if (!rows || !channels || channels > INT32_MAX || row_stride < channels ||
        !max_frames || min_frames > max_frames || emitted_frames > max_frames ||
        stop_id >= channels || suppressed_begin > suppressed_end ||
        suppressed_end > channels)
        return -2;
    if (rows - 1 > (SIZE_MAX - channels) / row_stride)
        return -2;
    const size_t span = (rows - 1) * row_stride + channels;
    if (span > SIZE_MAX / sizeof(float) || values_elements < span ||
        selected_elements < 1)
        return -2;
    if (!values || !selected ||
        (uintptr_t)values % _Alignof(float) != 0 ||
        (uintptr_t)selected % _Alignof(int32_t) != 0 ||
        (uintptr_t)values > UINTPTR_MAX - span * sizeof(float) ||
        (uintptr_t)selected > UINTPTR_MAX - sizeof(int32_t))
        return -1;
    const uintptr_t begin = (uintptr_t)values;
    const uintptr_t end = begin + span * sizeof(float);
    const uintptr_t output = (uintptr_t)selected;
    if (begin < output + sizeof(int32_t) && output < end)
        return -1;
    if (emitted_frames == max_frames)
        return CK_NEXT_ID_BUDGET_EXHAUSTED;

    const float *row = values + (rows - 1) * row_stride;
    size_t best = SIZE_MAX;
    float maximum = 0.0f;
    for (size_t index = 0; index < channels; ++index) {
        const float value = row[index];
        if (!isfinite(value)) return -3;
        if ((emitted_frames < min_frames && index == stop_id) ||
            (index >= suppressed_begin && index < suppressed_end &&
             index != stop_id))
            continue;
        if (best == SIZE_MAX || value > maximum) {
            best = index;
            maximum = value;
        }
    }
    if (best == SIZE_MAX) return -2;
    *selected = (int32_t)best;
    return best == stop_id ? CK_NEXT_ID_STOP : CK_NEXT_ID_CONTINUE;
}
