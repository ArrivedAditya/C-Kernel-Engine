/**
 * @file relu_kernels_bf16.c
 * @brief ReLU activation kernels for BF16 tensors
 *
 * CK-ENGINE KERNEL RULES:
 * =======================
 * 1. NO malloc/free - memory via bump allocator, pointers passed in
 * 2. NO OpenMP - parallelization at orchestrator/codegen layer
 * 3. API must define: inputs, outputs, workspace, and memory layouts
 * 4. Pure computation - deterministic, no side effects
 *
 * After changes: make test && make llamacpp-parity-full
 *
 * ReLU: y = max(0, x)
 */

#include <stddef.h>
#include <stdint.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include "bf16_utils.h"
#include "ckernel_engine.h"

void relu_forward_bf16(const uint16_t *input, uint16_t *output, size_t n)
{
    if (!input || !output) {
        return;
    }

    size_t i = 0;
#if defined(__AVX2__)
    const __m256i zero = _mm256_setzero_si256();
    for (; i + 16 <= n; i += 16) {
        const __m256i x = _mm256_loadu_si256((const __m256i *)(input + i));
        const __m256i mask = _mm256_cmpgt_epi16(x, zero);
        const __m256i y = _mm256_and_si256(x, mask);
        _mm256_storeu_si256((__m256i *)(output + i), y);
    }
#endif
    for (; i < n; ++i) {
        output[i] = (input[i] & 0x8000u) ? 0u : input[i];
    }
}

void relu_forward_inplace_bf16(uint16_t *data, size_t n)
{
    if (!data) {
        return;
    }

    size_t i = 0;
#if defined(__AVX2__)
    const __m256i zero = _mm256_setzero_si256();
    for (; i + 16 <= n; i += 16) {
        const __m256i x = _mm256_loadu_si256((const __m256i *)(data + i));
        const __m256i mask = _mm256_cmpgt_epi16(x, zero);
        const __m256i y = _mm256_and_si256(x, mask);
        _mm256_storeu_si256((__m256i *)(data + i), y);
    }
#endif
    for (; i < n; ++i) {
        data[i] = (data[i] & 0x8000u) ? 0u : data[i];
    }
}

void relu_backward_bf16(const uint16_t *input,
                        const uint16_t *d_output,
                        uint16_t *d_input,
                        size_t n)
{
    if (!input || !d_output || !d_input) {
        return;
    }

    size_t i = 0;
#if defined(__AVX2__)
    const __m256i zero = _mm256_setzero_si256();
    for (; i + 16 <= n; i += 16) {
        const __m256i x = _mm256_loadu_si256((const __m256i *)(input + i));
        const __m256i dy = _mm256_loadu_si256((const __m256i *)(d_output + i));
        const __m256i mask = _mm256_cmpgt_epi16(x, zero);
        const __m256i dx = _mm256_and_si256(dy, mask);
        _mm256_storeu_si256((__m256i *)(d_input + i), dx);
    }
#endif
    for (; i < n; ++i) {
        d_input[i] = ((input[i] & 0x8000u) == 0u && (input[i] & 0x7fffu) != 0u) ? d_output[i] : 0u;
    }
}
