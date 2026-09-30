#ifndef CKERNEL_AUDIO_DURATION_EXPAND_TOKEN_MAJOR_H
#define CKERNEL_AUDIO_DURATION_EXPAND_TOKEN_MAJOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Expand token-major [tokens, channels] features into channel-major
 * [channels, valid_frames]. Strides describe physical storage; capacities are
 * element counts. All geometry and durations are checked before any write.
 * Buffers are caller owned and must not overlap. Output padding is untouched.
 * Returns CK_AUDIO_EXTENT_* from ckernel_tts.h. */
int audio_duration_expand_token_major_f32(
    const float *features,
    size_t input_elements,
    size_t channels,
    size_t phoneme_count,
    size_t input_stride,
    const int32_t *durations,
    size_t expanded_frames,
    float *output,
    size_t output_elements,
    size_t output_stride);

#ifdef __cplusplus
}
#endif

#endif
