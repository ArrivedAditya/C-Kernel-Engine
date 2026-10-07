#ifndef CKERNEL_RWKV7_DECODE_H
#define CKERNEL_RWKV7_DECODE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * RWKV-7 single-token (T=1, B=1) FP32 decode reference.
 *
 * Token shift: xm = x + mu*(shift-x) with shift_out = x.
 *
 * WKV generalised delta rule, multiplicative decay, per head with state
 * [N,N] in (value,key) axes:
 *   decay[j]     = exp(w_log[j])
 *   b[j]         = kk[j]*a[j]
 *   S_new[i,j]   = S[i,j]*decay[j] - (S[i,:].kk)*b[j] + v[i]*k[j]
 *   y[i]         = sum_l S_new[i,l]*r[l]
 * GroupNorm eps is norm_eps*head_dim. Bonus is per head:
 * s_h = sum(r_h*k_h*rk_h), y_h += s_h*v_h, with rk [H,N].
 * Output is y_out = (groupnorm(y)+bonus)*g elementwise; the final CxC
 * output projection stays with the caller's GEMV.
 * v-mixing towards v_first and the LoRA chains are separate units below;
 * the WKV takes final w_log/k/v/kk/a/g (kk L2-normalised, k scaled by
 * 1+(a-1)*k_a).
 *
 * Layouts (all FP32, row-major, flat C = H*N):
 *  - vectors: [C]; rk: [H,N] (row per head)
 *  - state:   [H,N,N], head h at offset h*N*N, row i at +i*N+j
 *  - LoRA down factors [C,R], up factors [R,C]; biases [C] (g has none)
 *  - channelmix Wk [K,C] (expand), Wv [C,K] (project)
 *
 * Limits (violations are hard errors): head_dim <= 512, num_heads <= 256,
 * LoRA ranks <= 512, hidden_dim <= 8192, norm_eps > 0.
 *
 * All functions return 0 on success, -1 on bad args, except
 * ck_rwkv7_channelmix_scratch_elems, which returns the required scratch
 * element count (0 on bad args). No malloc/free.
 * Unless an alias is explicitly documented, outputs must be disjoint from
 * inputs and each other; callers own storage sizes. State-step/state-decode
 * permit state_out == state_in, and norm_kk permits kk_out == kk_in.
 */

int ck_rwkv7_token_shift_lerp(const float *x,
                                  const float *shift_in,
                                  const float *mu,
                                  float *xm,
                                  float *shift_out,
                                  int dim);
/* Each output may exactly alias an input. The two outputs must be disjoint;
 * shifted/partial output-input overlaps are rejected before any write. */

/* kk_raw -> per-head L2-normalised kk (divisor max(norm,1e-12)). */
int ck_rwkv7_norm_kk(const float *kk_in,
                         float *kk_out,
                         int num_heads,
                         int head_dim);

/* k_out = k_in * (1 + (a-1) * k_a), elementwise over [C]. */
int ck_rwkv7_scale_k(const float *k_in,
                         const float *a,
                         const float *k_a,
                         float *k_out,
                         int dim);

/* Unfused FP32 LoRA chains (decode, B=1,T=1).
 * Computes w_log = -e^-0.5*sigmoid(tanh(xw@w1)@w2+w0),
 * a = sigmoid((xa@a1)@a2+a0), g = sigmoid(xg@g1)@g2.
 * Fusion is a later perf change behind its own parity gate. */
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
                            float *g_out);

/* Value-residual gate and mix. Layer 0: pass xv/v1/v2/v0 as NULL to produce
 * v_first (v_out = v_first_out = v_in copy). Layer >0: v_gate =
 * sigmoid((xv@v1)@v2+v0), v_out = v_in+(v_first-v_in)*v_gate,
 * v_first_out = v_first copy. v_gate may be NULL if unneeded. */
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
                            float *v_gate);

/* Single-head state transition (the unit a SIMD/chunked variant replaces):
 * S_new[i,j] = S[i,j]*exp(w_log[j]) - (S[i,:].kk)*(kk[j]*a[j]) + v[i]*k[j],
 * state axes (value,key), all vectors [N], states [N,N] row-major. */
int ck_rwkv7_state_step(const float *w_log,
                        const float *k,
                        const float *v,
                        const float *kk,
                        const float *a,
                        const float *state_in,
                        float *state_out,
                        int head_dim);

/* Single-head readout y = S_new.r, state [N,N] row-major, r/y [N]. */
int ck_rwkv7_state_readout(const float *state,
                           const float *r,
                           float *y,
                           int head_dim);

/* Full single-token time-mix step (composition): per head, ck_rwkv7_state_step
 * then ck_rwkv7_state_readout; then GroupNorm over C (eps = norm_eps*head_dim),
 * per-head bonus s_h = sum(r_h*k_h*rk_h) added as s_h*v_h, and y *= g.
 * The final CxC output projection stays with the caller's GEMV. */
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
                            int head_dim);

/* Scratch elements needed by ck_rwkv7_channelmix_decode (xk + hidden);
 * 0 on bad args. */
size_t ck_rwkv7_channelmix_scratch_elems(int dim, int hidden_dim);

/* Squared-ReLU feedforward decode step with caller-provided scratch.
 * xk = x + mu*(shift_in-x); hidden = relu(Wk@xk)^2, Wk [K,C] row-major;
 * out = Wv@hidden, Wv [C,K] row-major; shift_out = x.
 * `scratch` must hold ck_rwkv7_channelmix_scratch_elems(dim,hidden_dim)
 * floats. Either shift_out or out may exactly alias x, but the two outputs
 * must be disjoint. Other output-input overlap and all scratch overlap are
 * rejected before any write. Read-only inputs may overlap each other. */
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
                               size_t scratch_elems);

#ifdef __cplusplus
}
#endif

#endif
