/**
 * @file quantize_row_q8_k_avx2.c
 * @brief AVX2 entrypoint for exact Q8_K row quantization
 *
 * The previous AVX2 entrypoint delegated to the SSE implementation. On this
 * CPU the scalar reference compiles into a faster loop than both the SSE body
 * and a hand-written AVX2 pack path, while preserving byte-exact Q8_K output.
 */

#include "ckernel_quant.h"

void quantize_row_q8_k_ref(const float *x, void *vy, int k);

void quantize_row_q8_k_avx2(const float *x, void *vy, int k) {
    quantize_row_q8_k_ref(x, vy, k);
}
