#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include "ck_threadpool.h"
#include "ckernel_quant.h"
#include "bf16_utils.h"

void gemm_nt_f32_llama_production_output_range(
    const float *A, const float *B, const float *bias, float *C,
    int M, int N, int K, int output_begin, int output_end);
void rmsnorm_forward_llama_production(
    const float *input, const float *gamma, float *output, float *rstd_cache,
    int tokens, int d_model, int aligned_embed_dim, float eps);

typedef struct {
    ck_half d;
    ck_half dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K / 8];
    uint8_t qs[QK_K / 2];
} ck_gemma4_block_q5_K;

#define CK_GEMMA4_MAX_EMBED_DIM 16384

static inline float ck_bf16_to_f32(uint16_t v)
{
    uint32_t bits = ((uint32_t)v) << 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static float ck_gemma4_bf16_dot(const uint16_t *x, const uint16_t *y, int n)
{
    int i = 0;
    double sum = 0.0;
#if defined(__AVX2__)
    // Match ggml-cpu's BF16 AVX2 reduction for this selected numerical contract.
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();
    __m256 c4 = _mm256_setzero_ps();
#define CK_GEMMA4_LOAD_BF16(p) _mm256_castsi256_ps(_mm256_slli_epi32( \
    _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(p))), 16))
    for (; i + 32 <= n; i += 32) {
        c1 = _mm256_add_ps(_mm256_mul_ps(CK_GEMMA4_LOAD_BF16(x + i),
                                         CK_GEMMA4_LOAD_BF16(y + i)), c1);
        c2 = _mm256_add_ps(_mm256_mul_ps(CK_GEMMA4_LOAD_BF16(x + i + 8),
                                         CK_GEMMA4_LOAD_BF16(y + i + 8)), c2);
        c3 = _mm256_add_ps(_mm256_mul_ps(CK_GEMMA4_LOAD_BF16(x + i + 16),
                                         CK_GEMMA4_LOAD_BF16(y + i + 16)), c3);
        c4 = _mm256_add_ps(_mm256_mul_ps(CK_GEMMA4_LOAD_BF16(x + i + 24),
                                         CK_GEMMA4_LOAD_BF16(y + i + 24)), c4);
    }
#undef CK_GEMMA4_LOAD_BF16
    c1 = _mm256_add_ps(_mm256_add_ps(c1, c3), _mm256_add_ps(c2, c4));
    __m128 g = _mm_add_ps(_mm256_extractf128_ps(c1, 1), _mm256_castps256_ps128(c1));
    g = _mm_add_ps(g, _mm_movehl_ps(g, g));
    g = _mm_add_ss(g, _mm_movehdup_ps(g));
    sum = (double)_mm_cvtss_f32(g);
#endif
    for (; i < n; ++i) {
        sum += (double)(ck_bf16_to_f32(x[i]) * ck_bf16_to_f32(y[i]));
    }
    return (float)sum;
}

static inline float ck_gemma4_gelu(float x)
{
    if (x <= -10.0f) return 0.0f;
    if (x >= 10.0f) return x;
    x = CK_FP16_TO_FP32(CK_FP32_TO_FP16(x));
    const float c0 = 0.7978845608028654f;
    const float c1 = 0.044715f;
    const float y = 0.5f * x * (1.0f + tanhf(c0 * x * (1.0f + c1 * x * x)));
    return CK_FP16_TO_FP32(CK_FP32_TO_FP16(y));
}

static inline void ck_gemma4_unpack_q5_k_scales(const uint8_t *scales, uint8_t *sc, uint8_t *m)
{
    sc[0] = scales[0] & 0x3F;
    sc[1] = scales[1] & 0x3F;
    sc[2] = scales[2] & 0x3F;
    sc[3] = scales[3] & 0x3F;

    m[0] = scales[4] & 0x3F;
    m[1] = scales[5] & 0x3F;
    m[2] = scales[6] & 0x3F;
    m[3] = scales[7] & 0x3F;

    sc[4] = (scales[8]  & 0x0F) | ((scales[0] >> 6) << 4);
    sc[5] = (scales[9]  & 0x0F) | ((scales[1] >> 6) << 4);
    sc[6] = (scales[10] & 0x0F) | ((scales[2] >> 6) << 4);
    sc[7] = (scales[11] & 0x0F) | ((scales[3] >> 6) << 4);

    m[4] = (scales[8]  >> 4) | ((scales[4] >> 6) << 4);
    m[5] = (scales[9]  >> 4) | ((scales[5] >> 6) << 4);
    m[6] = (scales[10] >> 4) | ((scales[6] >> 6) << 4);
    m[7] = (scales[11] >> 4) | ((scales[7] >> 6) << 4);
}

static inline uint8_t ck_gemma4_q5_k_value(const ck_gemma4_block_q5_K *block, int subblock, int i)
{
    const uint8_t *ql = block->qs + (subblock / 2) * 32;
    const uint8_t low = (subblock & 1) ? (uint8_t)(ql[i] >> 4) : (uint8_t)(ql[i] & 0x0F);
    const uint8_t high = (block->qh[i] & (uint8_t)(1u << subblock)) ? 16u : 0u;
    return (uint8_t)(low | high);
}

static void ck_gemma4_dequant_q5_k_block(const ck_gemma4_block_q5_K *block, float *out)
{
    uint8_t sc[8];
    uint8_t m[8];
    ck_gemma4_unpack_q5_k_scales(block->scales, sc, m);
    const float d = CK_FP16_TO_FP32(block->d);
    const float dmin = CK_FP16_TO_FP32(block->dmin);
    for (int s = 0; s < 8; ++s) {
        const float scale = d * (float)sc[s];
        const float minv = dmin * (float)m[s];
        for (int i = 0; i < 32; ++i) {
            out[s * 32 + i] = scale * (float)ck_gemma4_q5_k_value(block, s, i) - minv;
        }
    }
}

static void ck_gemma4_rmsnorm_tmp(const float *x, const float *gamma, float *out, int n, float eps)
{
    double ss = 0.0;
    for (int i = 0; i < n; ++i) {
        ss += (double)x[i] * (double)x[i];
    }
    const float scale = 1.0f / sqrtf((float)(ss / (double)n) + eps);
    for (int i = 0; i < n; ++i) {
        out[i] = x[i] * scale * gamma[i];
    }
}

typedef struct {
    float *per_layer_input;
    const float *hidden;
    const int32_t *token_ids;
    const void *per_layer_token_emb;
    const uint16_t *per_layer_model_proj;
    const float *per_layer_proj_norm;
    int num_layers;
    int embed_dim;
    int per_layer_dim;
    int vocab_size;
    float eps;
} ck_gemma4_prepare_args_t;

static void ck_gemma4_prepare_q5_range(int begin, int end, void *opaque)
{
    const ck_gemma4_prepare_args_t *args =
        (const ck_gemma4_prepare_args_t *)opaque;
    const ck_gemma4_block_q5_K *token_blocks =
        (const ck_gemma4_block_q5_K *)args->per_layer_token_emb;
    const size_t token_blocks_per_row = (size_t)args->num_layers;
    const float token_scale = sqrtf((float)args->per_layer_dim);
    const float model_scale = 1.0f / sqrtf((float)args->embed_dim);
    const float mix_scale = 0.7071067811865475f;
    float token_vec[QK_K];
    float proj_vec[QK_K];
    float proj_normed[QK_K];

    for (int t = begin; t < end; ++t) {
        const int token = args->token_ids[t];
        if (token < 0 || token >= args->vocab_size) continue;
        const float *h = args->hidden + (size_t)t * (size_t)args->embed_dim;
        uint16_t rounded_hidden[args->embed_dim];
        for (int j = 0; j < args->embed_dim; ++j) {
            rounded_hidden[j] = float_to_bf16(h[j]);
        }
        for (int layer = 0; layer < args->num_layers; ++layer) {
            float *dst = args->per_layer_input +
                ((size_t)t * (size_t)args->num_layers + (size_t)layer) *
                    (size_t)args->per_layer_dim;
            const ck_gemma4_block_q5_K *tok_block = token_blocks +
                (size_t)token * token_blocks_per_row + (size_t)layer;
            ck_gemma4_dequant_q5_k_block(tok_block, token_vec);
            for (int i = 0; i < args->per_layer_dim; ++i) {
                token_vec[i] *= token_scale;
            }

            const uint16_t *model_proj_base = args->per_layer_model_proj +
                (size_t)layer * (size_t)args->per_layer_dim *
                    (size_t)args->embed_dim;
            for (int i = 0; i < args->per_layer_dim; ++i) {
                const uint16_t *row = model_proj_base +
                    (size_t)i * (size_t)args->embed_dim;
                const float acc = ck_gemma4_bf16_dot(row, rounded_hidden, args->embed_dim);
                proj_vec[i] = acc * model_scale;
            }
            ck_gemma4_rmsnorm_tmp(
                proj_vec, args->per_layer_proj_norm, proj_normed,
                args->per_layer_dim, args->eps);
            for (int i = 0; i < args->per_layer_dim; ++i) {
                dst[i] = (token_vec[i] + proj_normed[i]) * mix_scale;
            }
        }
    }
}

static void ck_gemma4_prepare_bf16_range(int begin, int end, void *opaque)
{
    const ck_gemma4_prepare_args_t *args =
        (const ck_gemma4_prepare_args_t *)opaque;
    const uint16_t *token_embeddings =
        (const uint16_t *)args->per_layer_token_emb;
    const float token_scale = sqrtf((float)args->per_layer_dim);
    const float model_scale = 1.0f / sqrtf((float)args->embed_dim);
    const float mix_scale = 0.7071067811865475f;
    float token_vec[QK_K];
    float proj_vec[QK_K];
    float proj_normed[QK_K];

    for (int t = begin; t < end; ++t) {
        const int token = args->token_ids[t];
        if (token < 0 || token >= args->vocab_size) continue;
        const float *h = args->hidden + (size_t)t * (size_t)args->embed_dim;
        uint16_t rounded_hidden[args->embed_dim];
        for (int j = 0; j < args->embed_dim; ++j) {
            rounded_hidden[j] = float_to_bf16(h[j]);
        }
        for (int layer = 0; layer < args->num_layers; ++layer) {
            float *dst = args->per_layer_input +
                ((size_t)t * (size_t)args->num_layers + (size_t)layer) *
                    (size_t)args->per_layer_dim;
            const uint16_t *tok_row = token_embeddings +
                ((size_t)token * (size_t)args->num_layers + (size_t)layer) *
                    (size_t)args->per_layer_dim;
            for (int i = 0; i < args->per_layer_dim; ++i) {
                token_vec[i] = ck_bf16_to_f32(tok_row[i]) * token_scale;
            }

            const uint16_t *model_proj_base = args->per_layer_model_proj +
                (size_t)layer * (size_t)args->per_layer_dim *
                    (size_t)args->embed_dim;
            for (int i = 0; i < args->per_layer_dim; ++i) {
                const uint16_t *row = model_proj_base +
                    (size_t)i * (size_t)args->embed_dim;
                const float acc = ck_gemma4_bf16_dot(row, rounded_hidden, args->embed_dim);
                proj_vec[i] = acc * model_scale;
            }
            ck_gemma4_rmsnorm_tmp(
                proj_vec, args->per_layer_proj_norm, proj_normed,
                args->per_layer_dim, args->eps);
            for (int i = 0; i < args->per_layer_dim; ++i) {
                dst[i] = (token_vec[i] + proj_normed[i]) * mix_scale;
            }
        }
    }
}

static void ck_gemma4_prepare_parallel(
    int tokens, ck_range_fn_t fn, ck_gemma4_prepare_args_t *args)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    int active = pool ? ck_threadpool_n_threads(pool) : 1;
    const char *disabled = getenv("CK_DISABLE_GEMMA4_PREPARE_PARALLEL");
    if (disabled && disabled[0] && strcmp(disabled, "0") != 0) active = 1;
    if (active > tokens) active = tokens;
    ck_threadpool_parallel_for_n(pool, active, 0, tokens, 1, fn, args);
}

void gemma4_per_layer_prepare_forward(float *per_layer_input,
                                      const float *hidden,
                                      const int32_t *token_ids,
                                      const void *per_layer_token_emb,
                                      const uint16_t *per_layer_model_proj,
                                      const float *per_layer_proj_norm,
                                      int tokens,
                                      int num_layers,
                                      int embed_dim,
                                      int per_layer_dim,
                                      int vocab_size,
                                      float eps)
{
    if (!per_layer_input || !hidden || !token_ids || !per_layer_token_emb ||
        !per_layer_model_proj || !per_layer_proj_norm || tokens <= 0 ||
        num_layers <= 0 || embed_dim <= 0 || embed_dim > CK_GEMMA4_MAX_EMBED_DIM ||
        per_layer_dim != QK_K || vocab_size <= 0) {
        return;
    }

    ck_gemma4_prepare_args_t args = {
        .per_layer_input = per_layer_input,
        .hidden = hidden,
        .token_ids = token_ids,
        .per_layer_token_emb = per_layer_token_emb,
        .per_layer_model_proj = per_layer_model_proj,
        .per_layer_proj_norm = per_layer_proj_norm,
        .num_layers = num_layers,
        .embed_dim = embed_dim,
        .per_layer_dim = per_layer_dim,
        .vocab_size = vocab_size,
        .eps = eps,
    };
    ck_gemma4_prepare_parallel(tokens, ck_gemma4_prepare_q5_range, &args);
}


void gemma4_per_layer_prepare_bf16_forward(float *per_layer_input,
                                           const float *hidden,
                                           const int32_t *token_ids,
                                           const uint16_t *per_layer_token_emb,
                                           const uint16_t *per_layer_model_proj,
                                           const float *per_layer_proj_norm,
                                           int tokens,
                                           int num_layers,
                                           int embed_dim,
                                           int per_layer_dim,
                                           int vocab_size,
                                           float eps)
{
    if (!per_layer_input || !hidden || !token_ids || !per_layer_token_emb ||
        !per_layer_model_proj || !per_layer_proj_norm || tokens <= 0 ||
        num_layers <= 0 || embed_dim <= 0 || embed_dim > CK_GEMMA4_MAX_EMBED_DIM ||
        per_layer_dim <= 0 || vocab_size <= 0) {
        return;
    }

    if (per_layer_dim > QK_K) {
        return;
    }
    ck_gemma4_prepare_args_t args = {
        .per_layer_input = per_layer_input,
        .hidden = hidden,
        .token_ids = token_ids,
        .per_layer_token_emb = per_layer_token_emb,
        .per_layer_model_proj = per_layer_model_proj,
        .per_layer_proj_norm = per_layer_proj_norm,
        .num_layers = num_layers,
        .embed_dim = embed_dim,
        .per_layer_dim = per_layer_dim,
        .vocab_size = vocab_size,
        .eps = eps,
    };
    ck_gemma4_prepare_parallel(tokens, ck_gemma4_prepare_bf16_range, &args);
}

typedef struct {
    float *hidden;
    const float *per_layer_input;
    const float *inp_gate;
    const float *proj;
    const float *post_norm;
    const float *out_scale;
    int layer;
    int num_layers;
    int embed_dim;
    int per_layer_dim;
    int tokens;
    float eps;
    const int *segment_lengths;
    int num_segments;
    int singleton_tail;
} ck_gemma4_embed_args_t;

static int ck_gemma4_embed_row_group(const ck_gemma4_embed_args_t *args, int token, int tokens)
{
    if (!args->segment_lengths) return tokens > 1 ? 2 : 1;
    int start = 0;
    for (int segment = 0; segment < args->num_segments; ++segment) {
        const int length = args->segment_lengths[segment];
        if (token < start + length) {
            if (length == 1 || (args->singleton_tail && segment == args->num_segments - 1)) {
                return 1;
            }
            return 2;
        }
        start += length;
    }
    return 1;
}

static void ck_gemma4_embed_range(int begin, int end, void *opaque)
{
    const ck_gemma4_embed_args_t *args =
        (const ck_gemma4_embed_args_t *)opaque;
    float gate_vec[QK_K];
    float branch[4096];
    float branch_normed[4096];
    for (int t = begin; t < end; ++t) {
        float *h = args->hidden + (size_t)t * (size_t)args->embed_dim;
        const float *inp_vec = args->per_layer_input +
            ((size_t)t * (size_t)args->num_layers + (size_t)args->layer) *
                (size_t)args->per_layer_dim;

        const int row_group = ck_gemma4_embed_row_group(args, t, args->tokens);
        gemm_nt_f32_llama_production_output_range(
            h, args->inp_gate, NULL, gate_vec, row_group,
            args->per_layer_dim, args->embed_dim, 0, args->per_layer_dim);
        for (int i = 0; i < args->per_layer_dim; ++i) {
            gate_vec[i] = ck_gemma4_gelu(gate_vec[i]) * inp_vec[i];
        }

        gemm_nt_f32_llama_production_output_range(
            gate_vec, args->proj, NULL, branch, row_group,
            args->embed_dim, args->per_layer_dim, 0, args->embed_dim);
        rmsnorm_forward_llama_production(
            branch, args->post_norm, branch_normed, NULL,
            1, args->embed_dim, args->embed_dim, args->eps);
        const float layer_scale = args->out_scale ? args->out_scale[0] : 1.0f;
        for (int j = 0; j < args->embed_dim; ++j) {
            h[j] = (h[j] + branch_normed[j]) * layer_scale;
        }
    }
}

void gemma4_per_layer_embed_forward(float *hidden,
                                    const float *per_layer_input,
                                    const float *inp_gate,
                                    const float *proj,
                                    const float *post_norm,
                                    const float *out_scale,
                                    int tokens,
                                    int layer,
                                    int num_layers,
                                    int embed_dim,
                                    int per_layer_dim,
                                    float eps,
                                    const int *segment_lengths,
                                    int num_segments,
                                    int singleton_tail)
{
    if (!hidden || !per_layer_input || !inp_gate || !proj || !post_norm ||
        tokens <= 0 || layer < 0 || layer >= num_layers || embed_dim <= 0 ||
        per_layer_dim != QK_K || embed_dim > 4096 ||
        (segment_lengths == NULL) != (num_segments == 0) ||
        (singleton_tail != 0 && singleton_tail != 1)) {
        return;
    }
    if (segment_lengths) {
        int remaining = tokens;
        for (int i = 0; i < num_segments; ++i) {
            if (segment_lengths[i] < 0 || segment_lengths[i] > remaining) return;
            remaining -= segment_lengths[i];
        }
        if (remaining != 0) return;
    }

    ck_gemma4_embed_args_t args = {
        .hidden = hidden,
        .per_layer_input = per_layer_input,
        .inp_gate = inp_gate,
        .proj = proj,
        .post_norm = post_norm,
        .out_scale = out_scale,
        .layer = layer,
        .num_layers = num_layers,
        .embed_dim = embed_dim,
        .per_layer_dim = per_layer_dim,
        .tokens = tokens,
        .eps = eps,
        .segment_lengths = segment_lengths,
        .num_segments = num_segments,
        .singleton_tail = singleton_tail,
    };
    ck_threadpool_t *pool = ck_threadpool_global();
    int active = pool ? ck_threadpool_n_threads(pool) : 1;
    const char *disabled = getenv("CK_DISABLE_GEMMA4_EMBED_PARALLEL");
    if (disabled && disabled[0] && strcmp(disabled, "0") != 0) active = 1;
    if (active > tokens) active = tokens;
    ck_threadpool_parallel_for_n(
        pool, active, 0, tokens, 1, ck_gemma4_embed_range, &args);
}

void assistant_layer_scale_forward(float *hidden,
                                   const float *scale,
                                   int tokens,
                                   int embed_dim)
{
    if (!hidden || !scale || tokens <= 0 || embed_dim <= 0) {
        return;
    }

    const float s = scale[0];
    const size_t n = (size_t)tokens * (size_t)embed_dim;
    for (size_t i = 0; i < n; ++i) {
        hidden[i] *= s;
    }
}

void gemma4_final_logit_softcap_forward(float *logits,
                                        int tokens,
                                        int vocab_size,
                                        float cap)
{
    if (!logits || tokens <= 0 || vocab_size <= 0 || cap <= 0.0f) {
        return;
    }
    const float inv_cap = 1.0f / cap;
    const size_t total = (size_t)tokens * (size_t)vocab_size;
    for (size_t i = 0; i < total; ++i) {
        logits[i] = tanhf(logits[i] * inv_cap) * cap;
    }
}
