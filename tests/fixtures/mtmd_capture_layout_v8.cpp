#include <cassert>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "mtmd_capture_layout_v8.h"

int main() {
    struct Node { Node *src[4]{}; } parent, child;
    parent.src[0] = &child;
    assert(ck_mtmd_capture_tensor(&parent, -1) == &parent);
    assert(ck_mtmd_capture_tensor(&parent, 0) == &child);
    assert(ck_mtmd_capture_tensor(&parent, 1) == nullptr);
    assert(ck_mtmd_capture_tensor(&parent, -2) == nullptr);
    assert(ck_mtmd_capture_tensor(&parent, 4) == nullptr);
    char storage[16]{};
    assert(ck_mtmd_capture_storage_overlaps(storage, 8, storage + 7, 4));
    assert(ck_mtmd_capture_storage_overlaps(storage + 7, 4, storage, 8));
    assert(!ck_mtmd_capture_storage_overlaps(storage, 8, storage + 8, 4));
    assert(!ck_mtmd_capture_storage_overlaps(nullptr, 8, storage, 4));
    const float source[] = {0, 1, 2, 3, 4, 5};
    const int64_t transposed_ne[] = {2, 3, 1, 1};
    const size_t transposed_nb[] = {12, 4, 24, 24};
    std::vector<uint8_t> dense;
    assert(ck_mtmd_gather_strided_elements(
        reinterpret_cast<const uint8_t *>(source), sizeof(source),
        transposed_ne, transposed_nb, sizeof(float), dense));
    const float expected[] = {0, 3, 1, 4, 2, 5};
    assert(dense.size() == sizeof(expected));
    assert(std::memcmp(dense.data(), expected, sizeof(expected)) == 0);

    const uint16_t half_source[] = {100, 101, 102, 103, 104, 105};
    const int64_t padded_ne[] = {2, 2, 1, 1};
    const size_t padded_nb[] = {2, 8, 12, 12};
    assert(ck_mtmd_gather_strided_elements(
        reinterpret_cast<const uint8_t *>(half_source), sizeof(half_source),
        padded_ne, padded_nb, sizeof(uint16_t), dense));
    const uint16_t half_expected[] = {100, 101, 104, 105};
    assert(dense.size() == sizeof(half_expected));
    assert(std::memcmp(dense.data(), half_expected, sizeof(half_expected)) == 0);

    // Physical [head, token, channel] storage exposed as logical [channel, head, token].
    const uint16_t head_major[] = {
        100, 101, 102, 110, 111, 112,
        200, 201, 202, 210, 211, 212,
    };
    const int64_t view_ne[] = {3, 2, 2, 1};
    const size_t view_nb[] = {2, 12, 6, 24};
    assert(ck_mtmd_gather_strided_elements(
        reinterpret_cast<const uint8_t *>(head_major), sizeof(head_major),
        view_ne, view_nb, sizeof(uint16_t), dense));
    const uint16_t view_expected[] = {
        100, 101, 102, 200, 201, 202,
        110, 111, 112, 210, 211, 212,
    };
    assert(dense.size() == sizeof(view_expected));
    assert(std::memcmp(dense.data(), view_expected, sizeof(view_expected)) == 0);

    assert(!ck_mtmd_gather_strided_elements(
        reinterpret_cast<const uint8_t *>(source), sizeof(source) - 1,
        transposed_ne, transposed_nb, sizeof(float), dense));
    const int64_t empty_ne[] = {0, 3, 1, 1};
    assert(!ck_mtmd_gather_strided_elements(
        reinterpret_cast<const uint8_t *>(source), sizeof(source),
        empty_ne, transposed_nb, sizeof(float), dense));
    const size_t overflow_nb[] = {std::numeric_limits<size_t>::max(), 4, 24, 24};
    assert(!ck_mtmd_gather_strided_elements(
        reinterpret_cast<const uint8_t *>(source), sizeof(source),
        transposed_ne, overflow_nb, sizeof(float), dense));
    assert(!ck_mtmd_gather_strided_elements(
        reinterpret_cast<const uint8_t *>(source), sizeof(source),
        transposed_ne, transposed_nb, 3, dense));
    return 0;
}
