#ifndef CK_MTMD_CAPTURE_LAYOUT_V8_H
#define CK_MTMD_CAPTURE_LAYOUT_V8_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

template <typename Tensor>
inline Tensor *ck_mtmd_capture_tensor(Tensor *node, int source_index) {
    if (!node || source_index < -1 || source_index >= 4) return nullptr;
    if (source_index < 0) return node;
    return node->src[source_index];
}

inline bool ck_mtmd_capture_storage_overlaps(
    const void *left, size_t left_bytes, const void *right, size_t right_bytes) {
    if (!left || !right || !left_bytes || !right_bytes) return false;
    const uintptr_t l = reinterpret_cast<uintptr_t>(left);
    const uintptr_t r = reinterpret_cast<uintptr_t>(right);
    return l <= r ? r - l < left_bytes : l - r < right_bytes;
}

inline bool ck_mtmd_gather_strided_elements(
    const uint8_t *storage, size_t storage_bytes, const int64_t ne[4],
    const size_t nb[4], size_t element_bytes, std::vector<uint8_t> &dense) {
    if (!storage || !ne || !nb || (element_bytes != 2 && element_bytes != 4) ||
        storage_bytes < element_bytes) {
        return false;
    }
    size_t count = 1;
    size_t last_offset = 0;
    for (int axis = 0; axis < 4; ++axis) {
        if (ne[axis] <= 0 || static_cast<uint64_t>(ne[axis]) >
                std::numeric_limits<size_t>::max() / count) {
            return false;
        }
        const size_t extent = static_cast<size_t>(ne[axis]);
        count *= extent;
        if (extent > 1 && nb[axis] >
                (std::numeric_limits<size_t>::max() - last_offset) / (extent - 1)) {
            return false;
        }
        last_offset += (extent - 1) * nb[axis];
    }
    if (count > 64u * 1024u * 1024u ||
        count > std::numeric_limits<size_t>::max() / element_bytes ||
        last_offset > storage_bytes - element_bytes) {
        return false;
    }
    dense.resize(count * element_bytes);
    for (size_t flat = 0; flat < count; ++flat) {
        size_t remaining = flat;
        size_t offset = 0;
        for (int axis = 0; axis < 4; ++axis) {
            const size_t extent = static_cast<size_t>(ne[axis]);
            offset += (remaining % extent) * nb[axis];
            remaining /= extent;
        }
        std::memcpy(dense.data() + flat * element_bytes,
                    storage + offset, element_bytes);
    }
    return true;
}

#endif
