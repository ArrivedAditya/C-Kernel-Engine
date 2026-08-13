#include "ckernel_engine.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void recurrent_conv_state_update_forward(const float *state_in,
                                         const float *q,
                                         const float *k,
                                         const float *v,
                                         float *conv_x,
                                         float *state_out,
                                         int history_len,
                                         int num_seqs,
                                         int num_tokens,
                                         int q_dim,
                                         int k_dim,
                                         int v_dim) {
    const int channels = q_dim + k_dim + v_dim;
    const int total_len = history_len + num_tokens;
    for (int seq = 0; seq < num_seqs; ++seq) {
        const float *state_seq = state_in + (size_t) seq * (size_t) channels * (size_t) history_len;
        float *conv_seq = conv_x + (size_t) seq * (size_t) channels * (size_t) total_len;
        float *state_out_seq = state_out + (size_t) seq * (size_t) channels * (size_t) history_len;
        for (int ch = 0; ch < channels; ++ch) {
            memcpy(
                conv_seq + (size_t) ch * (size_t) total_len,
                state_seq + (size_t) ch * (size_t) history_len,
                (size_t) history_len * sizeof(float));
        }

        for (int tok = 0; tok < num_tokens; ++tok) {
            const int row = seq * num_tokens + tok;
            const float *q_row = q + (size_t) row * (size_t) q_dim;
            const float *k_row = k + (size_t) row * (size_t) k_dim;
            const float *v_row = v + (size_t) row * (size_t) v_dim;
            for (int col = 0; col < q_dim; ++col) {
                conv_seq[(size_t) col * (size_t) total_len + (size_t) (history_len + tok)] = q_row[col];
            }
            for (int col = 0; col < k_dim; ++col) {
                conv_seq[(size_t) (q_dim + col) * (size_t) total_len + (size_t) (history_len + tok)] = k_row[col];
            }
            for (int col = 0; col < v_dim; ++col) {
                conv_seq[(size_t) (q_dim + k_dim + col) * (size_t) total_len + (size_t) (history_len + tok)] = v_row[col];
            }
        }

        for (int ch = 0; ch < channels; ++ch) {
            memcpy(
                state_out_seq + (size_t) ch * (size_t) history_len,
                conv_seq + (size_t) ch * (size_t) total_len + (size_t) num_tokens,
                (size_t) history_len * sizeof(float));
        }
    }
}

static int recurrent_conv_backward_extents(int history_len,
                                           int num_seqs,
                                           int num_tokens,
                                           int q_dim,
                                           int k_dim,
                                           int v_dim,
                                           int *channels_out,
                                           int *total_len_out,
                                           size_t *elements_out) {
    if (history_len < 0 || num_seqs <= 0 || num_tokens < 0 || q_dim < 0 ||
        k_dim < 0 || v_dim < 0 || q_dim > INT_MAX - k_dim ||
        q_dim + k_dim > INT_MAX - v_dim || history_len > INT_MAX - num_tokens) {
        return 0;
    }
    const int channels = q_dim + k_dim + v_dim;
    const int total_len = history_len + num_tokens;
    if (channels == 0 || total_len == 0) {
        return 0;
    }
    size_t elements = (size_t)num_seqs;
    if ((size_t)total_len > SIZE_MAX / elements) {
        return 0;
    }
    elements *= (size_t)total_len;
    if ((size_t)channels > SIZE_MAX / elements) {
        return 0;
    }
    *channels_out = channels;
    *total_len_out = total_len;
    *elements_out = elements * (size_t)channels;
    return 1;
}

void recurrent_conv_state_update_backward_workspace(const float *d_conv_x,
                                                     const float *d_state_out,
                                                     float *d_state_in,
                                                     float *d_q,
                                                     float *d_k,
                                                     float *d_v,
                                                     float *d_conv_total,
                                                     int history_len,
                                                     int num_seqs,
                                                     int num_tokens,
                                                     int q_dim,
                                                     int k_dim,
                                                     int v_dim) {
    int channels = 0;
    int total_len = 0;
    size_t elements = 0;
    if (!d_conv_x || !d_state_out || !d_state_in || !d_q || !d_k || !d_v ||
        !d_conv_total || !recurrent_conv_backward_extents(
            history_len, num_seqs, num_tokens, q_dim, k_dim, v_dim,
            &channels, &total_len, &elements)) {
        return;
    }
    if (elements > SIZE_MAX / sizeof(float)) {
        return;
    }

    memcpy(d_conv_total, d_conv_x, elements * sizeof(float));

    for (int seq = 0; seq < num_seqs; ++seq) {
        const float *d_state_out_seq = d_state_out + (size_t) seq * (size_t) channels * (size_t) history_len;
        float *d_conv_seq = d_conv_total + (size_t) seq * (size_t) channels * (size_t) total_len;
        for (int ch = 0; ch < channels; ++ch) {
            float *dst = d_conv_seq + (size_t) ch * (size_t) total_len + (size_t) num_tokens;
            const float *src = d_state_out_seq + (size_t) ch * (size_t) history_len;
            for (int idx = 0; idx < history_len; ++idx) {
                dst[idx] += src[idx];
            }
        }
    }

    for (int seq = 0; seq < num_seqs; ++seq) {
        const float *d_conv_seq = d_conv_total + (size_t) seq * (size_t) channels * (size_t) total_len;
        float *d_state_in_seq = d_state_in + (size_t) seq * (size_t) channels * (size_t) history_len;

        for (int ch = 0; ch < channels; ++ch) {
            memcpy(
                d_state_in_seq + (size_t) ch * (size_t) history_len,
                d_conv_seq + (size_t) ch * (size_t) total_len,
                (size_t) history_len * sizeof(float));
        }

        for (int tok = 0; tok < num_tokens; ++tok) {
            const int row = seq * num_tokens + tok;
            float *d_q_row = d_q + (size_t) row * (size_t) q_dim;
            float *d_k_row = d_k + (size_t) row * (size_t) k_dim;
            float *d_v_row = d_v + (size_t) row * (size_t) v_dim;
            for (int col = 0; col < q_dim; ++col) {
                d_q_row[col] = d_conv_seq[(size_t) col * (size_t) total_len + (size_t) (history_len + tok)];
            }
            for (int col = 0; col < k_dim; ++col) {
                d_k_row[col] = d_conv_seq[(size_t) (q_dim + col) * (size_t) total_len + (size_t) (history_len + tok)];
            }
            for (int col = 0; col < v_dim; ++col) {
                d_v_row[col] = d_conv_seq[(size_t) (q_dim + k_dim + col) * (size_t) total_len + (size_t) (history_len + tok)];
            }
        }
    }
}

void recurrent_conv_state_update_backward(const float *d_conv_x,
                                          const float *d_state_out,
                                          float *d_state_in,
                                          float *d_q,
                                          float *d_k,
                                          float *d_v,
                                          int history_len,
                                          int num_seqs,
                                          int num_tokens,
                                          int q_dim,
                                          int k_dim,
                                          int v_dim) {
    int channels = 0;
    int total_len = 0;
    size_t elements = 0;
    if (!recurrent_conv_backward_extents(
            history_len, num_seqs, num_tokens, q_dim, k_dim, v_dim,
            &channels, &total_len, &elements) ||
        elements > SIZE_MAX / sizeof(float)) {
        return;
    }
    (void)channels;
    (void)total_len;
    float *workspace = (float *)malloc(elements * sizeof(float));
    if (!workspace) {
        return;
    }
    recurrent_conv_state_update_backward_workspace(
        d_conv_x, d_state_out, d_state_in, d_q, d_k, d_v, workspace,
        history_len, num_seqs, num_tokens, q_dim, k_dim, v_dim);
    free(workspace);
}
