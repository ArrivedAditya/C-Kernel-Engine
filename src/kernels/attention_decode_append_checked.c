/* Checked FP32 KV seeding and single-query grouped attention.
 * All storage is caller owned. Decode stages the result before publishing
 * either the output or the new KV row. */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static int ck_mul_size(size_t a, size_t b, size_t *result) {
    if (a && b > SIZE_MAX / a) return 0;
    *result = a * b;
    return 1;
}

static int ck_region(const void *pointer, size_t elements,
                     uintptr_t *begin, uintptr_t *end) {
    size_t bytes;
    if (!pointer || !ck_mul_size(elements, sizeof(float), &bytes)) return 0;
    *begin = (uintptr_t)pointer;
    if (bytes > UINTPTR_MAX - *begin) return 0;
    *end = *begin + bytes;
    return 1;
}

static int ck_disjoint(const void *a, size_t a_elements,
                       const void *b, size_t b_elements) {
    uintptr_t ab, ae, bb, be;
    return ck_region(a, a_elements, &ab, &ae) &&
           ck_region(b, b_elements, &bb, &be) &&
           (ae <= bb || be <= ab);
}

int ck_kv_cache_seed_checked_f32(
    const float *k, size_t k_elements, const float *v, size_t v_elements,
    float *k_cache, size_t k_cache_elements,
    float *v_cache, size_t v_cache_elements,
    size_t kv_heads, size_t tokens, size_t cache_capacity, size_t head_dim) {
    size_t token_head, input_need, cache_head, cache_need;
    if (!k || !v || !k_cache || !v_cache || !kv_heads || !tokens ||
        !head_dim || tokens > cache_capacity ||
        !ck_mul_size(tokens, head_dim, &token_head) ||
        !ck_mul_size(kv_heads, token_head, &input_need) ||
        !ck_mul_size(cache_capacity, head_dim, &cache_head) ||
        !ck_mul_size(kv_heads, cache_head, &cache_need) ||
        input_need > k_elements || input_need > v_elements ||
        cache_need > k_cache_elements || cache_need > v_cache_elements ||
        !ck_disjoint(k, input_need, k_cache, cache_need) ||
        !ck_disjoint(k, input_need, v_cache, cache_need) ||
        !ck_disjoint(v, input_need, k_cache, cache_need) ||
        !ck_disjoint(v, input_need, v_cache, cache_need) ||
        !ck_disjoint(k_cache, cache_need, v_cache, cache_need)) return -1;
    for (size_t i = 0; i < input_need; ++i)
        if (!isfinite(k[i]) || !isfinite(v[i])) return -2;
    for (size_t head = 0; head < kv_heads; ++head) {
        memcpy(k_cache + head * cache_head, k + head * token_head,
               token_head * sizeof(float));
        memcpy(v_cache + head * cache_head, v + head * token_head,
               token_head * sizeof(float));
    }
    return 0;
}

int ck_attention_decode_append_checked_f32(
    const float *q, size_t q_elements,
    const float *k_new, size_t k_new_elements,
    const float *v_new, size_t v_new_elements,
    float *k_cache, size_t k_cache_elements,
    float *v_cache, size_t v_cache_elements,
    float *scores, size_t score_elements,
    float *staged_output, size_t staged_output_elements,
    float *output, size_t output_elements,
    size_t heads, size_t kv_heads, size_t cache_capacity,
    size_t head_dim, size_t position) {
    size_t query_need, new_need, cache_head, cache_need, score_need;
    if (!q || !k_new || !v_new || !k_cache || !v_cache || !scores ||
        !staged_output || !output || !heads || !kv_heads ||
        heads % kv_heads || !head_dim || !cache_capacity ||
        position >= cache_capacity ||
        !ck_mul_size(heads, head_dim, &query_need) ||
        !ck_mul_size(kv_heads, head_dim, &new_need) ||
        !ck_mul_size(cache_capacity, head_dim, &cache_head) ||
        !ck_mul_size(kv_heads, cache_head, &cache_need) ||
        !ck_mul_size(heads, cache_capacity, &score_need) ||
        query_need > q_elements || new_need > k_new_elements ||
        new_need > v_new_elements || cache_need > k_cache_elements ||
        cache_need > v_cache_elements || score_need > score_elements ||
        query_need > staged_output_elements || query_need > output_elements)
        return -1;

    const void *reads[] = {q, k_new, v_new};
    const size_t read_sizes[] = {query_need, new_need, new_need};
    const void *writes[] = {k_cache, v_cache, scores, staged_output, output};
    const size_t write_sizes[] = {cache_need, cache_need, score_need,
                                  query_need, query_need};
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 5; ++j)
            if (!ck_disjoint(reads[i], read_sizes[i], writes[j], write_sizes[j]))
                return -1;
    for (size_t i = 0; i < 5; ++i)
        for (size_t j = i + 1; j < 5; ++j)
            if (!ck_disjoint(writes[i], write_sizes[i], writes[j], write_sizes[j]))
                return -1;

    for (size_t i = 0; i < query_need; ++i)
        if (!isfinite(q[i])) return -2;
    for (size_t i = 0; i < new_need; ++i)
        if (!isfinite(k_new[i]) || !isfinite(v_new[i])) return -2;
    for (size_t head = 0; head < kv_heads; ++head)
        for (size_t token = 0; token < position; ++token)
            for (size_t d = 0; d < head_dim; ++d) {
                size_t index = head * cache_head + token * head_dim + d;
                if (!isfinite(k_cache[index]) || !isfinite(v_cache[index]))
                    return -2;
            }

    const double scale = 1.0 / sqrt((double)head_dim);
    const size_t group_size = heads / kv_heads;
    for (size_t head = 0; head < heads; ++head) {
        size_t kv_head = head / group_size;
        float maximum = -INFINITY;
        for (size_t token = 0; token <= position; ++token) {
            const float *key = token == position ?
                k_new + kv_head * head_dim :
                k_cache + kv_head * cache_head + token * head_dim;
            double dot = 0.0;
            for (size_t d = 0; d < head_dim; ++d)
                dot += (double)q[head * head_dim + d] * (double)key[d];
            float score = (float)(dot * scale);
            if (!isfinite(score)) return -2;
            scores[head * cache_capacity + token] = score;
            if (score > maximum) maximum = score;
        }
        double denominator = 0.0;
        for (size_t token = 0; token <= position; ++token)
            denominator += exp((double)scores[head * cache_capacity + token] -
                               (double)maximum);
        if (!(denominator > 0.0) || !isfinite(denominator)) return -2;
        for (size_t d = 0; d < head_dim; ++d) {
            double numerator = 0.0;
            for (size_t token = 0; token <= position; ++token) {
                const float *value = token == position ?
                    v_new + kv_head * head_dim :
                    v_cache + kv_head * cache_head + token * head_dim;
                double probability = exp(
                    (double)scores[head * cache_capacity + token] -
                    (double)maximum) / denominator;
                numerator += probability * (double)value[d];
            }
            float result = (float)numerator;
            if (!isfinite(result)) return -2;
            staged_output[head * head_dim + d] = result;
        }
    }
    for (size_t head = 0; head < kv_heads; ++head) {
        memcpy(k_cache + head * cache_head + position * head_dim,
               k_new + head * head_dim, head_dim * sizeof(float));
        memcpy(v_cache + head * cache_head + position * head_dim,
               v_new + head * head_dim, head_dim * sizeof(float));
    }
    memcpy(output, staged_output, query_need * sizeof(float));
    return 0;
}
