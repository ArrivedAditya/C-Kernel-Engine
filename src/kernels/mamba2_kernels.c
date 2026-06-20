#include "ckernel_engine.h"

#include <math.h>
#include <string.h>

static inline float mamba2_sigmoid_f32(float x) {
    if (x >= 0.0f) {
        const float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    {
        const float z = expf(x);
        return z / (1.0f + z);
    }
}

static inline float mamba2_silu_f32(float x) {
    return x * mamba2_sigmoid_f32(x);
}

static inline float mamba2_softplus_f32(float x) {
    if (x > 20.0f) {
        return x;
    }
    if (x < -20.0f) {
        return expf(x);
    }
    return log1pf(expf(x));
}

void mamba2_in_proj_split_f32(const float *projected,
                              float *gate,
                              float *hidden_bc,
                              float *dt,
                              int rows,
                              int d_mlp,
                              int intermediate_dim,
                              int conv_dim,
                              int num_heads) {
    if (!projected || !gate || !hidden_bc || !dt ||
        rows <= 0 || d_mlp < 0 || intermediate_dim <= 0 || conv_dim <= 0 || num_heads <= 0) {
        return;
    }

    const int projection_dim = 2 * d_mlp + intermediate_dim + conv_dim + num_heads;
    const int gate_offset = 2 * d_mlp;
    const int hidden_bc_offset = gate_offset + intermediate_dim;
    const int dt_offset = hidden_bc_offset + conv_dim;

    for (int row = 0; row < rows; ++row) {
        const float *src = projected + (size_t)row * (size_t)projection_dim;
        memcpy(gate + (size_t)row * (size_t)intermediate_dim,
               src + gate_offset,
               (size_t)intermediate_dim * sizeof(float));
        memcpy(hidden_bc + (size_t)row * (size_t)conv_dim,
               src + hidden_bc_offset,
               (size_t)conv_dim * sizeof(float));
        memcpy(dt + (size_t)row * (size_t)num_heads,
               src + dt_offset,
               (size_t)num_heads * sizeof(float));
    }
}

void mamba2_conv1d_decode_f32(const float *state_in,
                              const float *x,
                              const float *weight,
                              const float *bias,
                              float *conv_out,
                              float *state_out,
                              int rows,
                              int conv_dim,
                              int kernel_size) {
    if (!state_in || !x || !weight || !conv_out || !state_out ||
        rows <= 0 || conv_dim <= 0 || kernel_size <= 0) {
        return;
    }

    for (int row = 0; row < rows; ++row) {
        for (int ch = 0; ch < conv_dim; ++ch) {
            const size_t base = ((size_t)row * (size_t)conv_dim + (size_t)ch) * (size_t)kernel_size;
            for (int k = 0; k < kernel_size - 1; ++k) {
                state_out[base + (size_t)k] = state_in[base + (size_t)k + 1u];
            }
            state_out[base + (size_t)kernel_size - 1u] =
                x[(size_t)row * (size_t)conv_dim + (size_t)ch];

            float acc = bias ? bias[ch] : 0.0f;
            for (int k = 0; k < kernel_size; ++k) {
                acc += state_out[base + (size_t)k] *
                       weight[(size_t)ch * (size_t)kernel_size + (size_t)k];
            }
            conv_out[(size_t)row * (size_t)conv_dim + (size_t)ch] = mamba2_silu_f32(acc);
        }
    }
}

void mamba2_dt_softplus_f32(const float *dt,
                            const float *dt_bias,
                            float *dt_out,
                            int rows,
                            int num_heads,
                            float dt_min,
                            float dt_max) {
    if (!dt || !dt_out || rows <= 0 || num_heads <= 0) {
        return;
    }

    for (int row = 0; row < rows; ++row) {
        for (int h = 0; h < num_heads; ++h) {
            float v = dt[(size_t)row * (size_t)num_heads + (size_t)h];
            if (dt_bias) {
                v += dt_bias[h];
            }
            v = mamba2_softplus_f32(v);
            if (dt_min < dt_max) {
                if (v < dt_min) {
                    v = dt_min;
                } else if (v > dt_max) {
                    v = dt_max;
                }
            }
            dt_out[(size_t)row * (size_t)num_heads + (size_t)h] = v;
        }
    }
}

void mamba2_selective_state_update_decode_f32(const float *state_in,
                                              const float *x,
                                              const float *dt,
                                              const float *a,
                                              const float *b,
                                              const float *c,
                                              const float *d,
                                              float *state_out,
                                              float *y,
                                              int rows,
                                              int num_heads,
                                              int head_dim,
                                              int state_dim,
                                              int num_groups) {
    if (!state_in || !x || !dt || !a || !b || !c || !d || !state_out || !y ||
        rows <= 0 || num_heads <= 0 || head_dim <= 0 || state_dim <= 0 || num_groups <= 0) {
        return;
    }

    for (int row = 0; row < rows; ++row) {
        for (int h = 0; h < num_heads; ++h) {
            const int group = (int)(((long long)h * (long long)num_groups) / (long long)num_heads);
            const float dt_h = dt[(size_t)row * (size_t)num_heads + (size_t)h];
            const float d_a = expf(dt_h * a[h]);
            const float d_h = d[h];
            const float *b_row = b + ((size_t)row * (size_t)num_groups + (size_t)group) * (size_t)state_dim;
            const float *c_row = c + ((size_t)row * (size_t)num_groups + (size_t)group) * (size_t)state_dim;

            for (int hd = 0; hd < head_dim; ++hd) {
                const size_t x_idx = ((size_t)row * (size_t)num_heads + (size_t)h) * (size_t)head_dim + (size_t)hd;
                const float x_val = x[x_idx];
                const size_t state_base =
                    (((size_t)row * (size_t)num_heads + (size_t)h) * (size_t)head_dim + (size_t)hd) *
                    (size_t)state_dim;
                float acc = 0.0f;
                for (int s = 0; s < state_dim; ++s) {
                    const size_t si = state_base + (size_t)s;
                    const float new_state = state_in[si] * d_a + dt_h * b_row[s] * x_val;
                    state_out[si] = new_state;
                    acc += new_state * c_row[s];
                }
                y[x_idx] = acc + d_h * x_val;
            }
        }
    }
}


void mamba2_selective_scan_f32(const float *state_init,
                               const float *x,
                               const float *dt,
                               const float *a,
                               const float *b,
                               const float *c,
                               const float *d,
                               float *state_out,
                               float *y,
                               int batch,
                               int seq_len,
                               int num_heads,
                               int head_dim,
                               int state_dim,
                               int num_groups) {
    if (!state_init || !x || !dt || !a || !b || !c || !d || !state_out || !y ||
        batch <= 0 || seq_len <= 0 || num_heads <= 0 || head_dim <= 0 || state_dim <= 0 || num_groups <= 0) {
        return;
    }

    const size_t state_per_batch = (size_t)num_heads * (size_t)head_dim * (size_t)state_dim;
    memcpy(state_out, state_init, (size_t)batch * state_per_batch * sizeof(float));

    for (int bs = 0; bs < batch; ++bs) {
        float *state_batch = state_out + (size_t)bs * state_per_batch;
        for (int t = 0; t < seq_len; ++t) {
            for (int h = 0; h < num_heads; ++h) {
                const int group = (int)(((long long)h * (long long)num_groups) / (long long)num_heads);
                const float dt_h = dt[((size_t)bs * (size_t)seq_len + (size_t)t) * (size_t)num_heads + (size_t)h];
                const float d_a = expf(dt_h * a[h]);
                const float d_h = d[h];
                const float *b_row = b + (((size_t)bs * (size_t)seq_len + (size_t)t) * (size_t)num_groups + (size_t)group) * (size_t)state_dim;
                const float *c_row = c + (((size_t)bs * (size_t)seq_len + (size_t)t) * (size_t)num_groups + (size_t)group) * (size_t)state_dim;

                for (int hd = 0; hd < head_dim; ++hd) {
                    const size_t x_idx = (((size_t)bs * (size_t)seq_len + (size_t)t) * (size_t)num_heads + (size_t)h) * (size_t)head_dim + (size_t)hd;
                    const float x_val = x[x_idx];
                    const size_t state_base = ((size_t)h * (size_t)head_dim + (size_t)hd) * (size_t)state_dim;
                    float acc = 0.0f;
                    for (int st = 0; st < state_dim; ++st) {
                        const size_t si = state_base + (size_t)st;
                        const float new_state = state_batch[si] * d_a + dt_h * b_row[st] * x_val;
                        state_batch[si] = new_state;
                        acc += new_state * c_row[st];
                    }
                    y[x_idx] = acc + d_h * x_val;
                }
            }
        }
    }
}

void mamba2_rmsnorm_gate_f32(const float *x,
                             const float *gate,
                             const float *weight,
                             float *out,
                             int rows,
                             int inner_dim,
                             int group_size,
                             float eps) {
    if (!x || !gate || !weight || !out || rows <= 0 || inner_dim <= 0 || group_size <= 0) {
        return;
    }

    for (int row = 0; row < rows; ++row) {
        const float *x_row = x + (size_t)row * (size_t)inner_dim;
        const float *gate_row = gate + (size_t)row * (size_t)inner_dim;
        float *out_row = out + (size_t)row * (size_t)inner_dim;

        for (int start = 0; start < inner_dim; start += group_size) {
            int end = start + group_size;
            if (end > inner_dim) {
                end = inner_dim;
            }
            const int count = end - start;
            float ms = 0.0f;
            for (int col = start; col < end; ++col) {
                const float gated = x_row[col] * mamba2_silu_f32(gate_row[col]);
                ms += gated * gated;
            }
            const float inv_rms = 1.0f / sqrtf(ms / (float)count + eps);
            for (int col = start; col < end; ++col) {
                const float gated = x_row[col] * mamba2_silu_f32(gate_row[col]);
                out_row[col] = gated * inv_rms * weight[col];
            }
        }
    }
}
