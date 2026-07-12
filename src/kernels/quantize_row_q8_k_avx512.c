/**
 * @file quantize_row_q8_k_avx512.c
 * @brief AVX-512 entrypoint for exact Q8_K row quantization
 */

void quantize_row_q8_k_ref(const float *x, void *vy, int k);

void quantize_row_q8_k_avx512(const float *x, void *vy, int k) {
    /*
     * Q8_K bytes are part of the numerical ABI consumed by Q4_K/Q6_K dots.
     * Keep AVX-512 on the reference contract until a vector implementation is
     * byte-exact for every block, including multi-block activation rows.
     */
    quantize_row_q8_k_ref(x, vy, k);
}
