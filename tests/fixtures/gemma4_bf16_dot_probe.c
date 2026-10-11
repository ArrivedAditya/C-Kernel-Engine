#include "src/kernels/gemma4_per_layer_embed.c"

int ck_test_gemma4_compiled_avx2(void)
{
#if defined(__AVX2__)
    return 1;
#else
    return 0;
#endif
}

float ck_test_gemma4_bf16_dot(const uint16_t *x, const uint16_t *y, int n)
{
    return ck_gemma4_bf16_dot(x, y, n);
}
