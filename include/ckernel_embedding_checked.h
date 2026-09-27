#ifndef CKE_EMBEDDING_CHECKED_H
#define CKE_EMBEDDING_CHECKED_H

#include <stddef.h>
#include <stdint.h>

/* Token-major FP32 word + type + position lookup, followed by per-token
 * population-variance LayerNorm. Positions are 0..tokens-1. All tables use
 * weight_stride; output uses output_stride. Padding is never written.
 * Returns -1 before any output write for invalid geometry, IDs or nonfinite
 * inputs. All pointers are nonaliasing and caller owned. Inference only. */
int embedding_three_table_layer_norm_f32(
    const int32_t *word_ids, const int32_t *type_ids, size_t id_elements,
    const float *word, size_t word_elements, size_t vocab_size,
    const float *position, size_t position_elements, size_t position_count,
    const float *token_type, size_t type_elements, size_t type_count,
    const float *gamma, const float *beta, size_t norm_elements,
    float *output, size_t output_elements, size_t tokens, size_t channels,
    size_t weight_stride, size_t output_stride, float epsilon);

#endif
