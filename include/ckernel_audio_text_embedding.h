#ifndef CKERNEL_AUDIO_TEXT_EMBEDDING_H
#define CKERNEL_AUDIO_TEXT_EMBEDDING_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Exact FP32 table lookup from int32 IDs to channel-major [C,T] output.
 * Every extent and selected ID is validated before output writes. The caller
 * owns nonoverlapping input, table and output buffers; padding is untouched.
 * Returns CK_AUDIO_EXTENT_* from ckernel_tts.h. */
int audio_text_embedding_channel_major_f32(
    const int32_t *ids, size_t id_elements,
    const float *table, size_t table_elements,
    size_t vocabulary, size_t channels, size_t table_stride,
    float *output, size_t output_elements,
    size_t tokens, size_t output_stride);

#ifdef __cplusplus
}
#endif

#endif
