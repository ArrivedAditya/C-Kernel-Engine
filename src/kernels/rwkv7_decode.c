/* RWKV-7 T=1 FP32 decode reference. See include/ckernel_rwkv7_decode.h. */
#include "ckernel_rwkv7_decode.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

/* Compare byte ranges without undefined relational comparisons of pointers. */
static int overlaps(const void *a, size_t an, const void *b, size_t bn) {
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    if (an > UINTPTR_MAX - x || bn > UINTPTR_MAX - y) return 1;
    return x < y + bn && y < x + an;
}

int ck_rwkv7_token_shift_lerp(const float *x,
                                  const float *shift_in,
                                  const float *mu,
                                  float *xm,
                                  float *shift_out,
                                  int dim) {
    if (!x || !shift_in || !mu || !xm || !shift_out || dim <= 0) {
        return -1;
    }
    size_t bytes = (size_t)dim * sizeof(float);
    if (overlaps(xm, bytes, shift_out, bytes)) return -1;
    const float *inputs[] = {x, shift_in, mu};
    for (int j = 0; j < 3; ++j) {
        if ((xm != inputs[j] && overlaps(xm, bytes, inputs[j], bytes)) ||
            (shift_out != inputs[j] && overlaps(shift_out, bytes, inputs[j], bytes))) return -1;
    }
    for (int i = 0; i < dim; ++i) {
        const float previous = shift_in[i];
        const float current = x[i];
        const float m = mu[i];
        xm[i] = current + m * (previous - current);
        shift_out[i] = current;
    }
    return 0;
}

int ck_rwkv7_norm_kk(const float *kk_in,
                         float *kk_out,
                         int num_heads,
                         int head_dim) {
    if (!kk_in || !kk_out || num_heads <= 0 || head_dim <= 0 ||
        num_heads > 256 || head_dim > 512) {
        return -1;
    }
    for (int h = 0; h < num_heads; ++h) {
        const float *src = kk_in + (size_t)h * (size_t)head_dim;
        float *dst = kk_out + (size_t)h * (size_t)head_dim;
        float norm = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            norm += src[i] * src[i];
        }
        norm = sqrtf(norm);
        /* Divisor max(norm,1e-12); an all-zero row stays zero. */
        if (norm < 1e-12f) {
            norm = 1e-12f;
        }
        const float inv = 1.0f / norm;
        for (int i = 0; i < head_dim; ++i) {
            dst[i] = src[i] * inv;
        }
    }
    return 0;
}

int ck_rwkv7_scale_k(const float *k_in,
                         const float *a,
                         const float *k_a,
                         float *k_out,
                         int dim) {
    if (!k_in || !a || !k_a || !k_out || dim <= 0) {
        return -1;
    }
    for (int i = 0; i < dim; ++i) {
        k_out[i] = k_in[i] * (1.0f + (a[i] - 1.0f) * k_a[i]);
    }
    return 0;
}

static inline float ck_rwkv7_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

int ck_rwkv7_lora_gates(const float *xw,
                            const float *xa,
                            const float *xg,
                            const float *w1,
                            const float *w2,
                            const float *w0,
                            const float *a1,
                            const float *a2,
                            const float *a0,
                            const float *g1,
                            const float *g2,
                            int dim,
                            int rank_w,
                            int rank_a,
                            int rank_g,
                            float *w_log,
                            float *a_out,
                            float *g_out) {
    if (!xw || !xa || !xg || !w1 || !w2 || !w0 || !a1 || !a2 || !a0 || !g1 ||
        !g2 || !w_log || !a_out || !g_out || dim <= 0 || rank_w <= 0 ||
        rank_a <= 0 || rank_g <= 0 || rank_w > 512 || rank_a > 512 ||
        rank_g > 512) {
        return -1;
    }
    /* w: tanh between factors, sigmoid after (+w0), scaled by -e^-0.5. */
    {
        float tmp[512];
        for (int r = 0; r < rank_w; ++r) {
            float acc = 0.0f;
            for (int i = 0; i < dim; ++i) {
                acc += xw[i] * w1[(size_t)i * (size_t)rank_w + (size_t)r];
            }
            tmp[r] = tanhf(acc);
        }
        for (int i = 0; i < dim; ++i) {
            float acc = w0[i];
            for (int r = 0; r < rank_w; ++r) {
                acc += tmp[r] * w2[(size_t)r * (size_t)dim + (size_t)i];
            }
            w_log[i] = -0.6065306597126334f * ck_rwkv7_sigmoid(acc);
        }
    }
    /* a: no activation between, sigmoid after (+a0). */
    {
        float tmp[512];
        for (int r = 0; r < rank_a; ++r) {
            float acc = 0.0f;
            for (int i = 0; i < dim; ++i) {
                acc += xa[i] * a1[(size_t)i * (size_t)rank_a + (size_t)r];
            }
            tmp[r] = acc;
        }
        for (int i = 0; i < dim; ++i) {
            float acc = a0[i];
            for (int r = 0; r < rank_a; ++r) {
                acc += tmp[r] * a2[(size_t)r * (size_t)dim + (size_t)i];
            }
            a_out[i] = ck_rwkv7_sigmoid(acc);
        }
    }
    /* g: sigmoid between factors, nothing after, no bias. */
    {
        float tmp[512];
        for (int r = 0; r < rank_g; ++r) {
            float acc = 0.0f;
            for (int i = 0; i < dim; ++i) {
                acc += xg[i] * g1[(size_t)i * (size_t)rank_g + (size_t)r];
            }
            tmp[r] = ck_rwkv7_sigmoid(acc);
        }
        for (int i = 0; i < dim; ++i) {
            float acc = 0.0f;
            for (int r = 0; r < rank_g; ++r) {
                acc += tmp[r] * g2[(size_t)r * (size_t)dim + (size_t)i];
            }
            g_out[i] = acc;
        }
    }
    return 0;
}

int ck_rwkv7_vgate_vmix(const float *v_in,
                            const float *v_first,
                            const float *xv,
                            const float *v1,
                            const float *v2,
                            const float *v0,
                            int dim,
                            int rank_v,
                            float *v_out,
                            float *v_first_out,
                            float *v_gate) {
    if (!v_in || !v_out || !v_first_out || dim <= 0) {
        return -1;
    }
    if (xv == NULL) {
        /* Layer 0: produce v_first, no gate read. */
        for (int i = 0; i < dim; ++i) {
            v_out[i] = v_in[i];
            v_first_out[i] = v_in[i];
        }
        return 0;
    }
    if (!v_first || !v1 || !v2 || !v0 || rank_v <= 0 || rank_v > 512) {
        return -1;
    }
    float tmp[512];
    for (int r = 0; r < rank_v; ++r) {
        float acc = 0.0f;
        for (int i = 0; i < dim; ++i) {
            acc += xv[i] * v1[(size_t)i * (size_t)rank_v + (size_t)r];
        }
        tmp[r] = acc;
    }
    for (int i = 0; i < dim; ++i) {
        float acc = v0[i];
        for (int r = 0; r < rank_v; ++r) {
            acc += tmp[r] * v2[(size_t)r * (size_t)dim + (size_t)i];
        }
        const float gate = ck_rwkv7_sigmoid(acc);
        if (v_gate) {
            v_gate[i] = gate;
        }
        v_out[i] = v_in[i] + (v_first[i] - v_in[i]) * gate;
        v_first_out[i] = v_first[i];
    }
    return 0;
}

int ck_rwkv7_state_step(const float *w_log,
                        const float *k,
                        const float *v,
                        const float *kk,
                        const float *a,
                        const float *state_in,
                        float *state_out,
                        int head_dim) {
    if (!w_log || !k || !v || !kk || !a || !state_in || !state_out ||
        head_dim <= 0 || head_dim > 512) {
        return -1;
    }
    float decay[512];
    float bvec[512];
    for (int j = 0; j < head_dim; ++j) {
        decay[j] = expf(w_log[j]);
        bvec[j] = kk[j] * a[j];
    }
    for (int i = 0; i < head_dim; ++i) {
        float t = 0.0f;
        for (int l = 0; l < head_dim; ++l) {
            t += state_in[(size_t)i * (size_t)head_dim + (size_t)l] * kk[l];
        }
        for (int j = 0; j < head_dim; ++j) {
            state_out[(size_t)i * (size_t)head_dim + (size_t)j] =
                state_in[(size_t)i * (size_t)head_dim + (size_t)j] * decay[j] -
                t * bvec[j] + v[i] * k[j];
        }
    }
    return 0;
}

int ck_rwkv7_state_readout(const float *state,
                           const float *r,
                           float *y,
                           int head_dim) {
    if (!state || !r || !y || head_dim <= 0 || head_dim > 512) {
        return -1;
    }
    for (int i = 0; i < head_dim; ++i) {
        float acc = 0.0f;
        for (int l = 0; l < head_dim; ++l) {
            acc += state[(size_t)i * (size_t)head_dim + (size_t)l] * r[l];
        }
        y[i] = acc;
    }
    return 0;
}

int ck_rwkv7_wkv_decode(const float *r,
                            const float *w_log,
                            const float *k,
                            const float *v,
                            const float *kk,
                            const float *a,
                            const float *g,
                            const float *rk,
                            const float *ln_w,
                            const float *ln_b,
                            float norm_eps,
                            const float *state_in,
                            float *state_out,
                            float *y_out,
                            int num_heads,
                            int head_dim) {
    if (!r || !w_log || !k || !v || !kk || !a || !g || !rk || !ln_w || !ln_b ||
        !state_in || !state_out || !y_out || num_heads <= 0 || head_dim <= 0 ||
        !(norm_eps > 0.0f) || !isfinite(norm_eps) ||
        !isfinite(norm_eps * (float)head_dim)) {
        return -1;
    }
    if (num_heads > 256 || head_dim > 512) {
        return -1;
    }

    /* Per-head generalised delta-rule step + raw readout (pre-norm). */
    for (int h = 0; h < num_heads; ++h) {
        const float *rh = r + (size_t)h * (size_t)head_dim;
        const float *kh = k + (size_t)h * (size_t)head_dim;
        const float *vh = v + (size_t)h * (size_t)head_dim;
        const float *wh = w_log + (size_t)h * (size_t)head_dim;
        const float *khh = kk + (size_t)h * (size_t)head_dim;
        const float *ah = a + (size_t)h * (size_t)head_dim;
        const float *sin = state_in + (size_t)h * (size_t)head_dim * (size_t)head_dim;
        float *sout = state_out + (size_t)h * (size_t)head_dim * (size_t)head_dim;
        float *yh = y_out + (size_t)h * (size_t)head_dim;

        if (ck_rwkv7_state_step(wh, kh, vh, khh, ah, sin, sout, head_dim) != 0 ||
            ck_rwkv7_state_readout(sout, rh, yh, head_dim) != 0) {
            return -1;
        }
    }

    /* GroupNorm over C with H groups, eps = norm_eps*head_dim. */
    const float eps = norm_eps * (float)head_dim;
    for (int h = 0; h < num_heads; ++h) {
        float *yh = y_out + (size_t)h * (size_t)head_dim;
        const float *lwh = ln_w + (size_t)h * (size_t)head_dim;
        const float *lbh = ln_b + (size_t)h * (size_t)head_dim;
        float mean = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            mean += yh[i];
        }
        mean /= (float)head_dim;
        float var = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            const float d = yh[i] - mean;
            var += d * d;
        }
        var /= (float)head_dim;
        const float inv = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < head_dim; ++i) {
            yh[i] = (yh[i] - mean) * inv * lwh[i] + lbh[i];
        }
    }

    /* Per-head bonus s_h = sum(r_h*k_h*rk_h); y += s_h*v; then y *= g. */
    const int dim = num_heads * head_dim;
    for (int h = 0; h < num_heads; ++h) {
        const float *rh = r + (size_t)h * (size_t)head_dim;
        const float *kh = k + (size_t)h * (size_t)head_dim;
        const float *vh = v + (size_t)h * (size_t)head_dim;
        const float *rkh = rk + (size_t)h * (size_t)head_dim;
        float *yh = y_out + (size_t)h * (size_t)head_dim;
        float s = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            s += rh[i] * kh[i] * rkh[i];
        }
        for (int i = 0; i < head_dim; ++i) {
            yh[i] += s * vh[i];
        }
    }
    for (int i = 0; i < dim; ++i) {
        y_out[i] *= g[i];
    }
    return 0;
}

size_t ck_rwkv7_channelmix_scratch_elems(int dim, int hidden_dim) {
    if (dim <= 0 || hidden_dim <= 0 || hidden_dim > 8192) {
        return 0;
    }
    return (size_t)dim + (size_t)hidden_dim;
}

int ck_rwkv7_channelmix_decode(const float *x,
                               const float *shift_in,
                               const float *mu,
                               const float *Wk,
                               const float *Wv,
                               float *shift_out,
                               float *out,
                               int dim,
                               int hidden_dim,
                               float *scratch,
                               size_t scratch_elems) {
    if (!x || !shift_in || !mu || !Wk || !Wv || !shift_out || !out || dim <= 0 ||
        hidden_dim <= 0 || hidden_dim > 8192) {
        return -1;
    }
    if (shift_out == shift_in) {
        return -1;
    }
    if (!scratch ||
        scratch_elems < (size_t)dim + (size_t)hidden_dim ||
        scratch == (const float *)x || scratch == (const float *)shift_in ||
        scratch == (const float *)mu || scratch == (const float *)Wk ||
        scratch == (const float *)Wv || scratch == (const float *)shift_out ||
        scratch == (const float *)out) {
        return -1;
    }
    const size_t bytes = (size_t)dim * sizeof(float);
    if ((size_t)hidden_dim > SIZE_MAX / bytes) return -1;
    const size_t weight_bytes = bytes * (size_t)hidden_dim;
    const size_t scratch_bytes = ((size_t)dim + (size_t)hidden_dim) * sizeof(float);
    if (overlaps(shift_out, bytes, out, bytes)) return -1;
    const float *inputs[] = {x, shift_in, mu, Wk, Wv};
    const size_t sizes[] = {bytes, bytes, bytes, weight_bytes, weight_bytes};
    for (int j = 0; j < 5; ++j) {
        if (overlaps(scratch, scratch_bytes, inputs[j], sizes[j])) return -1;
        if (!(j == 0 && shift_out == x) &&
            overlaps(shift_out, bytes, inputs[j], sizes[j])) return -1;
        if (!(j == 0 && out == x) &&
            overlaps(out, bytes, inputs[j], sizes[j])) return -1;
    }
    if (overlaps(scratch, scratch_bytes, out, bytes) ||
        overlaps(scratch, scratch_bytes, shift_out, bytes)) return -1;
    /* Hoisted token-shift mix; per-index read-before-write keeps
     * shift_out == x aliasing safe. */
    float *xk = scratch;
    float *hidden = scratch + dim;
    for (int i = 0; i < dim; ++i) {
        xk[i] = x[i] + mu[i] * (shift_in[i] - x[i]);
        shift_out[i] = x[i];
    }
    /* hidden[j] = relu(Wk[j,:] @ xk)^2, Wk [K,C] row-major. */
    for (int j = 0; j < hidden_dim; ++j) {
        const float *row = Wk + (size_t)j * (size_t)dim;
        float acc = 0.0f;
        for (int i = 0; i < dim; ++i) {
            acc += row[i] * xk[i];
        }
        const float rel = acc > 0.0f ? acc : 0.0f;
        hidden[j] = rel * rel;
    }
    /* out = Wv @ hidden, Wv [C,K] row-major; runs after the last xk read,
     * so out == x aliasing is safe. */
    for (int i = 0; i < dim; ++i) {
        const float *row = Wv + (size_t)i * (size_t)hidden_dim;
        float acc = 0.0f;
        for (int j = 0; j < hidden_dim; ++j) {
            acc += row[j] * hidden[j];
        }
        out[i] = acc;
    }
    return 0;
}
