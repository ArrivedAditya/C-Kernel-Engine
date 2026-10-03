#ifndef CKERNEL_AUDIO_SNAKE_CHECKED_H
#define CKERNEL_AUDIO_SNAKE_CHECKED_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Channel-major Snake: y[c,t] = x[c,t] + sin(alpha[c] * x[c,t])^2 / alpha[c].
 * All valid input and outputs are checked before output writes. Input, alpha,
 * and output must be disjoint; physical row padding remains untouched.
 * Returns 0 on success, -1 for invalid pointers/geometry, -2 for insufficient
 * capacity or overflow, and -3 for nonfinite arithmetic or zero alpha. */
int audio_snake_strided_f32_checked(
    const float *input, size_t input_elements, size_t input_stride,
    const float *alpha, size_t alpha_elements,
    float *output, size_t output_elements, size_t output_stride,
    size_t channels, size_t frames);

#ifdef __cplusplus
}
#endif

#endif
