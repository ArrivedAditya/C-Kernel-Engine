/* Exact physical-layout conversions used when no direct-layout provider wins. */

#include <stddef.h>
#include <string.h>

void ck_layout_token_to_head_f32(const float *src, float *dst,
                                 int tokens, int heads, int head_dim)
{
    if (!src || !dst || tokens <= 0 || heads <= 0 || head_dim <= 0) return;
    for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < heads; ++h) {
            memcpy(dst + ((size_t)h * (size_t)tokens + (size_t)t) * (size_t)head_dim,
                   src + ((size_t)t * (size_t)heads + (size_t)h) * (size_t)head_dim,
                   (size_t)head_dim * sizeof(float));
        }
    }
}

void ck_layout_head_to_token_f32(const float *src, float *dst,
                                 int heads, int tokens, int head_dim)
{
    if (!src || !dst || heads <= 0 || tokens <= 0 || head_dim <= 0) return;
    for (int h = 0; h < heads; ++h) {
        for (int t = 0; t < tokens; ++t) {
            memcpy(dst + ((size_t)t * (size_t)heads + (size_t)h) * (size_t)head_dim,
                   src + ((size_t)h * (size_t)tokens + (size_t)t) * (size_t)head_dim,
                   (size_t)head_dim * sizeof(float));
        }
    }
}
