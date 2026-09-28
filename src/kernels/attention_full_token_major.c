#include "ckernel_attention_full_token_major.h"

#include <float.h>
#include <math.h>
#include <stdint.h>

static int checked_geometry(size_t tokens, size_t heads, size_t head_dim,
                            size_t *width, size_t *elements)
{
    if (!tokens || !heads || !head_dim || heads > SIZE_MAX / head_dim)
        return 0;
    *width = heads * head_dim;
    if (tokens > SIZE_MAX / *width)
        return 0;
    *elements = tokens * *width;
    return 1;
}

static double score(const float *query, const float *key, size_t head_dim,
                    double scale)
{
    double sum = 0.0;
    for (size_t d = 0; d < head_dim; ++d) {
        volatile double product = (double)query[d] * (double)key[d];
        sum += product;
    }
    return sum * scale;
}

int attention_full_token_major_f32_checked(
    const float *query, size_t query_elements,
    const float *key, size_t key_elements,
    const float *value, size_t value_elements,
    float *output, size_t output_elements,
    float *scratch, size_t scratch_bytes,
    size_t tokens, size_t heads, size_t head_dim)
{
    size_t width, elements;
    if (!query || !key || !value || !output || !scratch ||
        !checked_geometry(tokens, heads, head_dim, &width, &elements) ||
        query_elements < elements || key_elements < elements ||
        value_elements < elements || output_elements < elements ||
        tokens > scratch_bytes / sizeof(float))
        return -1;

    /* A rejected input never publishes a partially updated output. */
    for (size_t i = 0; i < elements; ++i)
        if (!isfinite(query[i]) || !isfinite(key[i]) || !isfinite(value[i]))
            return -1;

    const double scale = 1.0 / sqrt((double)head_dim);
    for (size_t t = 0; t < tokens; ++t)
        for (size_t h = 0; h < heads; ++h)
            for (size_t k = 0; k < tokens; ++k) {
                const double s = score(query + t * width + h * head_dim,
                                       key + k * width + h * head_dim,
                                       head_dim, scale);
                if (!isfinite(s) || fabs(s) > FLT_MAX)
                    return -1;
            }

    for (size_t t = 0; t < tokens; ++t) {
        for (size_t h = 0; h < heads; ++h) {
            const float *q = query + t * width + h * head_dim;
            double maximum = -INFINITY;
            for (size_t k = 0; k < tokens; ++k) {
                const float *kv = key + k * width + h * head_dim;
                const double s = score(q, kv, head_dim, scale);
                scratch[k] = (float)s;
                /* Softmax consumes rounded FP32 scores. Its maximum must be
                 * drawn from those same values: using the unrounded dot can
                 * make even a one-token softmax overflow or underflow. At least
                 * one exponent is then exactly exp(0), and all are in [0, 1]. */
                if ((double)scratch[k] > maximum) maximum = scratch[k];
            }
            double denominator = 0.0;
            for (size_t k = 0; k < tokens; ++k) {
                scratch[k] = (float)exp((double)scratch[k] - maximum);
                denominator += scratch[k];
            }
            for (size_t d = 0; d < head_dim; ++d) {
                double weighted = 0.0;
                for (size_t k = 0; k < tokens; ++k) {
                    volatile double product = (double)scratch[k] *
                        (double)value[k * width + h * head_dim + d];
                    weighted += product;
                }
                output[t * width + h * head_dim + d] = (float)(weighted / denominator);
            }
        }
    }
    return 0;
}
