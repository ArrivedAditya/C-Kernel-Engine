#include "ckernel_tts.h"
#include "checked_float_buffer.h"

#include <math.h>
#include <stdint.h>

/* Counter-based excitation. Each element depends only on its index, seed and
 * stream, so requests cannot inherit RNG state from previous calls. The
 * Box-Muller transform uses FP64 libm and stores FP32 samples. It does not
 * promise the same random stream as PyTorch. */
static uint64_t mix64(uint64_t value) {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

int audio_noise_counter_gaussian_f32(float *output, size_t output_capacity,
    size_t count, uint64_t seed, uint64_t stream) {
    if (!count) return CK_AUDIO_EXTENT_INVALID;
    if (count > (1u << 24) || output_capacity < count)
        return CK_AUDIO_EXTENT_LIMIT;
    if (!ck_checked_region(output, count)) return CK_AUDIO_EXTENT_INVALID;
    const uint64_t domain = mix64(seed ^ mix64(stream));
    for (size_t index = 0; index < count; ++index) {
        const uint64_t counter = (uint64_t)index * UINT64_C(2);
        const uint64_t first = mix64(domain + counter);
        const uint64_t second = mix64(domain + counter + UINT64_C(1));
        const double uniform_positive =
            ((double)(first >> 11) + 0.5) * 0x1.0p-53;
        const double uniform_angle = (double)(second >> 11) * 0x1.0p-53;
        const double radius = sqrt(-2.0 * log(uniform_positive));
        output[index] = (float)(radius * cos(
            6.283185307179586476925286766559 * uniform_angle));
    }
    return CK_AUDIO_EXTENT_OK;
}
