#ifndef CKERNEL_ATTENTION_FULL_TOKEN_MAJOR_H
#define CKERNEL_ATTENTION_FULL_TOKEN_MAJOR_H

#include <stddef.h>

/* Full, unmasked self-attention over token-major [tokens, heads * head_dim]
 * FP32 Q/K/V. Scores and softmax use FP64 scalar accumulation. Output uses the
 * same token-major layout. All buffers are caller owned and must not alias.
 * scratch_bytes reserves at least tokens floats. Returns zero on success; invalid
 * geometry, capacity, or nonfinite input returns -1 before output is written.
 */
int attention_full_token_major_f32_checked(
    const float *query, size_t query_elements,
    const float *key, size_t key_elements,
    const float *value, size_t value_elements,
    float *output, size_t output_elements,
    float *scratch, size_t scratch_bytes,
    size_t tokens, size_t heads, size_t head_dim);

#endif
