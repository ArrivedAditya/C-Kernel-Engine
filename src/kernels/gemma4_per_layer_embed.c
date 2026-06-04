#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#include "ckernel_quant.h"

typedef struct {
    ck_half d;
    ck_half dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K / 8];
    uint8_t qs[QK_K / 2];
} ck_gemma4_block_q5_K;

static inline float ck_bf16_to_f32(uint16_t v)
{
    uint32_t bits = ((uint32_t)v) << 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static inline float ck_gemma4_gelu(float x)
{
    const float c0 = 0.7978845608028654f;
    const float c1 = 0.044715f;
    return 0.5f * x * (1.0f + tanhf(c0 * x * (1.0f + c1 * x * x)));
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
        num_layers <= 0 || embed_dim <= 0 || per_layer_dim != QK_K || vocab_size <= 0) {
        return;
    }

    const ck_gemma4_block_q5_K *token_blocks = (const ck_gemma4_block_q5_K *)per_layer_token_emb;
    const size_t token_blocks_per_row = (size_t)num_layers;
    const float token_scale = sqrtf((float)per_layer_dim);
    const float model_scale = 1.0f / sqrtf((float)embed_dim);
    const float mix_scale = 0.7071067811865475f;

    float token_vec[QK_K];
    float proj_vec[QK_K];
    float proj_normed[QK_K];

    for (int t = 0; t < tokens; ++t) {
        const int token = token_ids[t];
        if (token < 0 || token >= vocab_size) {
            continue;
        }
        const float *h = hidden + (size_t)t * (size_t)embed_dim;
        for (int layer = 0; layer < num_layers; ++layer) {
            float *dst = per_layer_input + ((size_t)t * (size_t)num_layers + (size_t)layer) * (size_t)per_layer_dim;
            const ck_gemma4_block_q5_K *tok_block = token_blocks + (size_t)token * token_blocks_per_row + (size_t)layer;
            ck_gemma4_dequant_q5_k_block(tok_block, token_vec);
            for (int i = 0; i < per_layer_dim; ++i) {
                token_vec[i] *= token_scale;
            }

            const uint16_t *model_proj_base = per_layer_model_proj + (size_t)layer * (size_t)per_layer_dim * (size_t)embed_dim;
            for (int i = 0; i < per_layer_dim; ++i) {
                const uint16_t *row = model_proj_base + (size_t)i * (size_t)embed_dim;
                float acc = 0.0f;
                for (int j = 0; j < embed_dim; ++j) {
                    acc += ck_bf16_to_f32(row[j]) * h[j];
                }
                proj_vec[i] = acc * model_scale;
            }
            ck_gemma4_rmsnorm_tmp(proj_vec, per_layer_proj_norm, proj_normed, per_layer_dim, eps);
            for (int i = 0; i < per_layer_dim; ++i) {
                dst[i] = (token_vec[i] + proj_normed[i]) * mix_scale;
            }
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
                                    float eps)
{
    if (!hidden || !per_layer_input || !inp_gate || !proj || !post_norm ||
        tokens <= 0 || layer < 0 || layer >= num_layers || embed_dim <= 0 || per_layer_dim != QK_K) {
        return;
    }

    float gate_vec[QK_K];
    float branch[4096];
    float branch_normed[4096];
    if (embed_dim > (int)(sizeof(branch) / sizeof(branch[0]))) {
        return;
    }

    for (int t = 0; t < tokens; ++t) {
        float *h = hidden + (size_t)t * (size_t)embed_dim;
        const float *inp_vec = per_layer_input + ((size_t)t * (size_t)num_layers + (size_t)layer) * (size_t)per_layer_dim;

        for (int i = 0; i < per_layer_dim; ++i) {
            const float *row = inp_gate + (size_t)i * (size_t)embed_dim;
            float acc = 0.0f;
            for (int j = 0; j < embed_dim; ++j) {
                acc += row[j] * h[j];
            }
            gate_vec[i] = ck_gemma4_gelu(acc) * inp_vec[i];
        }

        for (int j = 0; j < embed_dim; ++j) {
            const float *row = proj + (size_t)j * (size_t)per_layer_dim;
            float acc = 0.0f;
            for (int i = 0; i < per_layer_dim; ++i) {
                acc += row[i] * gate_vec[i];
            }
            branch[j] = acc;
        }
        ck_gemma4_rmsnorm_tmp(branch, post_norm, branch_normed, embed_dim, eps);
        const float layer_scale = out_scale ? out_scale[0] : 1.0f;
        for (int j = 0; j < embed_dim; ++j) {
            h[j] = (h[j] + branch_normed[j]) * layer_scale;
        }
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
