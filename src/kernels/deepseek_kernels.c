/**
 * @file deepseek_kernels.c
 * @brief Scalar reference kernels for DeepSeek-style research ops.
 *
 * These kernels intentionally prioritize explicit math contracts over speed.
 * They are used to pin PyTorch parity before adding SIMD/threaded variants.
 */

#include <math.h>
#include <float.h>
#include <stddef.h>
#include <stdint.h>

static inline size_t ds_mhc_idx(int t, int s, int d, int n_streams, int dim)
{
    return ((size_t)t * (size_t)n_streams + (size_t)s) * (size_t)dim + (size_t)d;
}

static inline size_t ds_mix_idx(int t, int out_s, int in_s, int n_streams)
{
    return ((size_t)t * (size_t)n_streams + (size_t)out_s) * (size_t)n_streams + (size_t)in_s;
}

void deepseek_mhc_mix_f32(const float *streams,
                          const float *mix,
                          float *out,
                          int tokens,
                          int n_streams,
                          int dim)
{
    if (!streams || !mix || !out || tokens <= 0 || n_streams <= 0 || dim <= 0) return;

    for (int t = 0; t < tokens; ++t) {
        for (int os = 0; os < n_streams; ++os) {
            for (int d = 0; d < dim; ++d) {
                float acc = 0.0f;
                for (int is = 0; is < n_streams; ++is) {
                    acc += mix[ds_mix_idx(t, os, is, n_streams)] *
                           streams[ds_mhc_idx(t, is, d, n_streams, dim)];
                }
                out[ds_mhc_idx(t, os, d, n_streams, dim)] = acc;
            }
        }
    }
}

void deepseek_mhc_mix_backward_f32(const float *d_out,
                                   const float *streams,
                                   const float *mix,
                                   float *d_streams,
                                   float *d_mix,
                                   int tokens,
                                   int n_streams,
                                   int dim)
{
    if (!d_out || !streams || !mix || !d_streams || !d_mix ||
        tokens <= 0 || n_streams <= 0 || dim <= 0) return;

    const size_t stream_count = (size_t)tokens * (size_t)n_streams * (size_t)dim;
    const size_t mix_count = (size_t)tokens * (size_t)n_streams * (size_t)n_streams;
    for (size_t i = 0; i < stream_count; ++i) d_streams[i] = 0.0f;
    for (size_t i = 0; i < mix_count; ++i) d_mix[i] = 0.0f;

    for (int t = 0; t < tokens; ++t) {
        for (int os = 0; os < n_streams; ++os) {
            for (int is = 0; is < n_streams; ++is) {
                float d_mix_acc = 0.0f;
                const float m = mix[ds_mix_idx(t, os, is, n_streams)];
                for (int d = 0; d < dim; ++d) {
                    const float go = d_out[ds_mhc_idx(t, os, d, n_streams, dim)];
                    d_streams[ds_mhc_idx(t, is, d, n_streams, dim)] += m * go;
                    d_mix_acc += go * streams[ds_mhc_idx(t, is, d, n_streams, dim)];
                }
                d_mix[ds_mix_idx(t, os, is, n_streams)] = d_mix_acc;
            }
        }
    }
}

void deepseek_dsa_topk_softmax_f32(const float *scores,
                                   int *indices,
                                   float *weights,
                                   int tokens,
                                   int heads,
                                   int key_count,
                                   int top_k)
{
    if (!scores || !indices || !weights || tokens <= 0 || heads <= 0 ||
        key_count <= 0 || top_k <= 0) return;

    if (top_k > key_count) top_k = key_count;

    for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < heads; ++h) {
            const float *row = scores + ((size_t)t * (size_t)heads + (size_t)h) * (size_t)key_count;
            int *idx = indices + ((size_t)t * (size_t)heads + (size_t)h) * (size_t)top_k;
            float *w = weights + ((size_t)t * (size_t)heads + (size_t)h) * (size_t)top_k;

            for (int k = 0; k < top_k; ++k) {
                idx[k] = -1;
                w[k] = -FLT_MAX;
            }

            for (int j = 0; j < key_count; ++j) {
                const float v = row[j];
                int pos = -1;
                for (int k = 0; k < top_k; ++k) {
                    if (idx[k] < 0 || v > w[k] || (v == w[k] && j < idx[k])) {
                        pos = k;
                        break;
                    }
                }
                if (pos >= 0) {
                    for (int k = top_k - 1; k > pos; --k) {
                        idx[k] = idx[k - 1];
                        w[k] = w[k - 1];
                    }
                    idx[pos] = j;
                    w[pos] = v;
                }
            }

            float max_v = w[0];
            for (int k = 1; k < top_k; ++k) if (w[k] > max_v) max_v = w[k];
            float sum = 0.0f;
            for (int k = 0; k < top_k; ++k) {
                w[k] = expf(w[k] - max_v);
                sum += w[k];
            }
            if (sum > 0.0f) {
                const float inv = 1.0f / sum;
                for (int k = 0; k < top_k; ++k) w[k] *= inv;
            }
        }
    }
}


extern void topk_softmax_backward_f32(const int *indices,
                                      const float *weights,
                                      const float *d_weights,
                                      float *d_scores,
                                      int num_tokens,
                                      int n_experts_or_keys,
                                      int k);

void deepseek_dsa_topk_softmax_backward_f32(const int *indices,
                                            const float *weights,
                                            const float *d_weights,
                                            float *d_scores,
                                            int tokens,
                                            int heads,
                                            int key_count,
                                            int top_k)
{
    if (!indices || !weights || !d_weights || !d_scores ||
        tokens <= 0 || heads <= 0 || key_count <= 0 || top_k <= 0) return;

    topk_softmax_backward_f32(indices,
                              weights,
                              d_weights,
                              d_scores,
                              tokens * heads,
                              key_count,
                              top_k);
}

static inline size_t ds_qkv_idx(int token, int head, int d, int heads, int dim)
{
    return ((size_t)token * (size_t)heads + (size_t)head) * (size_t)dim + (size_t)d;
}

static void ds_softmax(float *x, int n)
{
    if (n <= 0) return;
    float max_v = x[0];
    for (int i = 1; i < n; ++i) if (x[i] > max_v) max_v = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        x[i] = expf(x[i] - max_v);
        sum += x[i];
    }
    if (sum > 0.0f) {
        const float inv = 1.0f / sum;
        for (int i = 0; i < n; ++i) x[i] *= inv;
    }
}

void deepseek_csa_attention_f32(const float *q,
                                const float *k,
                                const float *v,
                                const int *indices,
                                float *out,
                                float *attn,
                                int query_tokens,
                                int key_tokens,
                                int heads,
                                int dim,
                                int top_k,
                                float scale)
{
    if (!q || !k || !v || !indices || !out || query_tokens <= 0 || key_tokens <= 0 ||
        heads <= 0 || dim <= 0 || top_k <= 0) return;

    for (int tq = 0; tq < query_tokens; ++tq) {
        for (int h = 0; h < heads; ++h) {
            float local_scores[top_k];
            int valid = 0;
            for (int j = 0; j < top_k; ++j) {
                const int tk = indices[((size_t)tq * (size_t)heads + (size_t)h) * (size_t)top_k + (size_t)j];
                if (tk < 0 || tk >= key_tokens) {
                    local_scores[j] = -FLT_MAX;
                    continue;
                }
                float dot = 0.0f;
                for (int d = 0; d < dim; ++d) {
                    dot += q[ds_qkv_idx(tq, h, d, heads, dim)] * k[ds_qkv_idx(tk, h, d, heads, dim)];
                }
                local_scores[j] = dot * scale;
                valid++;
            }

            float *out_row = out + ds_qkv_idx(tq, h, 0, heads, dim);
            for (int d = 0; d < dim; ++d) out_row[d] = 0.0f;
            if (valid == 0) {
                if (attn) {
                    for (int j = 0; j < top_k; ++j) {
                        attn[((size_t)tq * (size_t)heads + (size_t)h) * (size_t)top_k + (size_t)j] = 0.0f;
                    }
                }
                continue;
            }

            ds_softmax(local_scores, top_k);
            for (int j = 0; j < top_k; ++j) {
                const int tk = indices[((size_t)tq * (size_t)heads + (size_t)h) * (size_t)top_k + (size_t)j];
                const float a = (tk >= 0 && tk < key_tokens) ? local_scores[j] : 0.0f;
                if (attn) attn[((size_t)tq * (size_t)heads + (size_t)h) * (size_t)top_k + (size_t)j] = a;
                if (a == 0.0f) continue;
                for (int d = 0; d < dim; ++d) {
                    out_row[d] += a * v[ds_qkv_idx(tk, h, d, heads, dim)];
                }
            }
        }
    }
}

void deepseek_csa_attention_backward_f32(const float *d_out,
                                         const float *q,
                                         const float *k,
                                         const float *v,
                                         const int *indices,
                                         const float *attn,
                                         float *d_q,
                                         float *d_k,
                                         float *d_v,
                                         int query_tokens,
                                         int key_tokens,
                                         int heads,
                                         int dim,
                                         int top_k,
                                         float scale)
{
    if (!d_out || !q || !k || !v || !indices || !attn || !d_q || !d_k || !d_v ||
        query_tokens <= 0 || key_tokens <= 0 || heads <= 0 || dim <= 0 || top_k <= 0) return;

    const size_t q_count = (size_t)query_tokens * (size_t)heads * (size_t)dim;
    const size_t kv_count = (size_t)key_tokens * (size_t)heads * (size_t)dim;
    for (size_t i = 0; i < q_count; ++i) d_q[i] = 0.0f;
    for (size_t i = 0; i < kv_count; ++i) {
        d_k[i] = 0.0f;
        d_v[i] = 0.0f;
    }

    for (int tq = 0; tq < query_tokens; ++tq) {
        for (int h = 0; h < heads; ++h) {
            float d_attn[top_k];
            float attn_dot = 0.0f;
            for (int j = 0; j < top_k; ++j) {
                const int tk = indices[((size_t)tq * (size_t)heads + (size_t)h) * (size_t)top_k + (size_t)j];
                float da = 0.0f;
                if (tk >= 0 && tk < key_tokens) {
                    const float a = attn[((size_t)tq * (size_t)heads + (size_t)h) * (size_t)top_k + (size_t)j];
                    for (int d = 0; d < dim; ++d) {
                        const float go = d_out[ds_qkv_idx(tq, h, d, heads, dim)];
                        da += go * v[ds_qkv_idx(tk, h, d, heads, dim)];
                        d_v[ds_qkv_idx(tk, h, d, heads, dim)] += a * go;
                    }
                    attn_dot += a * da;
                }
                d_attn[j] = da;
            }

            for (int j = 0; j < top_k; ++j) {
                const int tk = indices[((size_t)tq * (size_t)heads + (size_t)h) * (size_t)top_k + (size_t)j];
                if (tk < 0 || tk >= key_tokens) continue;
                const float a = attn[((size_t)tq * (size_t)heads + (size_t)h) * (size_t)top_k + (size_t)j];
                const float d_score = a * (d_attn[j] - attn_dot);
                for (int d = 0; d < dim; ++d) {
                    const float qv = q[ds_qkv_idx(tq, h, d, heads, dim)];
                    const float kv = k[ds_qkv_idx(tk, h, d, heads, dim)];
                    d_q[ds_qkv_idx(tq, h, d, heads, dim)] += scale * d_score * kv;
                    d_k[ds_qkv_idx(tk, h, d, heads, dim)] += scale * d_score * qv;
                }
            }
        }
    }
}

void deepseek_hybrid_attention_f32(const float *q,
                                   const float *k,
                                   const float *v,
                                   const int *indices,
                                   float *out,
                                   float *attn,
                                   int query_tokens,
                                   int key_tokens,
                                   int heads,
                                   int dim,
                                   int top_k,
                                   float scale,
                                   int mode)
{
    if (mode != 0) {
        deepseek_csa_attention_f32(q, k, v, indices, out, attn,
                                   query_tokens, key_tokens, heads, dim, top_k, scale);
        return;
    }

    int dense_indices[query_tokens * heads * key_tokens];
    for (int tq = 0; tq < query_tokens; ++tq) {
        for (int h = 0; h < heads; ++h) {
            for (int tk = 0; tk < key_tokens; ++tk) {
                dense_indices[((size_t)tq * (size_t)heads + (size_t)h) * (size_t)key_tokens + (size_t)tk] = tk;
            }
        }
    }
    deepseek_csa_attention_f32(q, k, v, dense_indices, out, attn,
                               query_tokens, key_tokens, heads, dim, key_tokens, scale);
}
