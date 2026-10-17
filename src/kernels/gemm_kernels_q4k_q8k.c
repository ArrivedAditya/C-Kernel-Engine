/**
 * @file gemm_kernels_q4k_q8k.c
 * @brief Q4_K (weights) x Q8_K (activations) kernels for inference
 *
 * CK-ENGINE KERNEL RULES:
 * =======================
 * 1. NO malloc/free - memory via bump allocator, pointers passed in
 * 2. NO OpenMP - parallelization at orchestrator/codegen layer
 * 3. API must define: inputs, outputs, workspace, and memory layouts
 * 4. Pure computation - deterministic, no side effects
 *
 * After changes: make test && make llamacpp-parity-full
 *
 * Implements decode-style matvec/matmul where weights are Q4_K and the
 * activations are quantized on-the-fly to Q8_K. This is inference-only;
 * no backward pass is provided here.
 */

#include <assert.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "ck_threadpool.h"
#include "ckernel_quant.h"

void gemv_q4_k_q8_k_avx2(float *y,
                         const void *W,
                         const void *x_q8,
                         int M, int K);

void gemv_q4_k_q8_k_vnni(float *y,
                         const void *W,
                         const void *x_q8,
                         int M, int K);

void gemv_q4_k_q8_k_avx(float *y,
                        const void *W,
                        const void *x_q8,
                        int M, int K);

void gemv_q4_k_q8_k_sse(float *y,
                        const void *W,
                        const void *x_q8,
                        int M, int K);

static inline int ck_nearest_int(float fval) {
    /* Bit-level round-to-nearest from llama.cpp (fast + deterministic). */
    float val = fval + 12582912.f;
    int i;
    memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
__attribute__((optimize("fp-contract=off")))
#endif
void quantize_row_q8_k_ref(const float *x, void *vy, int k) {
    if (!x || !vy || k <= 0) {
        return;
    }
    assert(k % QK_K == 0);
    const int nb = k / QK_K;
    block_q8_K *y = (block_q8_K *)vy;

    for (int i = 0; i < nb; ++i) {
        float max = 0.0f;
        float amax = 0.0f;
        for (int j = 0; j < QK_K; ++j) {
            float ax = fabsf(x[j]);
            if (ax > amax) {
                amax = ax;
                max = x[j];
            }
        }
        if (!amax) {
            y[i].d = 0.0f;
            memset(y[i].qs, 0, sizeof(y[i].qs));
            memset(y[i].bsums, 0, sizeof(y[i].bsums));
            x += QK_K;
            continue;
        }

        const float iscale = -127.0f / max;
        for (int j = 0; j < QK_K; ++j) {
            /* llama.cpp rounds the multiply before adding nearest_int's magic
             * constant. Contracting both operations changes Q8_K tie cases. */
            float scaled = iscale * x[j];
            int v = ck_nearest_int(scaled);
            if (v > 127) {
                v = 127;
            }
            if (v < -128) {
                v = -128;
            }
            y[i].qs[j] = (int8_t)v;
        }

        for (int j = 0; j < QK_K / 16; ++j) {
            int sum = 0;
            const int8_t *qs = &y[i].qs[j * 16];
            for (int ii = 0; ii < 16; ++ii) {
                sum += qs[ii];
            }
            y[i].bsums[j] = (int16_t)sum;
        }

        y[i].d = 1.0f / iscale;
        x += QK_K;
    }
}

#if defined(__ARM_NEON) || defined(__aarch64__)
/* NEON Q8_K row quantizer. Byte-exact against quantize_row_q8_k_ref:
 * - amax via IEEE max reduction (exact), then the first element whose |x|
 *   equals amax, matching the scalar strict-> tie-break.
 * - Plain single-precision multiply (fp-contract off), then the same
 *   magic-constant round-to-nearest-even as ck_nearest_int, vectorized.
 * - bsums recomputed from the stored int8 payload, integer-exact. */
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
__attribute__((optimize("fp-contract=off")))
#endif
void quantize_row_q8_k_neon(const float *x, void *vy, int k) {
    if (!x || !vy || k <= 0) {
        return;
    }
    assert(k % QK_K == 0);
    const int nb = k / QK_K;
    block_q8_K *y = (block_q8_K *)vy;

    for (int i = 0; i < nb; ++i) {
        float32x4_t vmax = vdupq_n_f32(0.0f);
        for (int j = 0; j < QK_K; j += 16) {
            const float32x4_t a0 = vabsq_f32(vld1q_f32(x + j + 0));
            const float32x4_t a1 = vabsq_f32(vld1q_f32(x + j + 4));
            const float32x4_t a2 = vabsq_f32(vld1q_f32(x + j + 8));
            const float32x4_t a3 = vabsq_f32(vld1q_f32(x + j + 12));
            vmax = vmaxq_f32(vmax, vmaxq_f32(vmaxq_f32(a0, a1),
                                             vmaxq_f32(a2, a3)));
        }
        float lanes[4];
        vst1q_f32(lanes, vmax);
        const float amax = fmaxf(fmaxf(lanes[0], lanes[1]),
                                 fmaxf(lanes[2], lanes[3]));
        if (!amax) {
            y[i].d = 0.0f;
            memset(y[i].qs, 0, sizeof(y[i].qs));
            memset(y[i].bsums, 0, sizeof(y[i].bsums));
            x += QK_K;
            continue;
        }

        /* Scalar ref keeps the FIRST element reaching amax (strict >). */
        float max = 0.0f;
        for (int j = 0; j < QK_K; ++j) {
            if (fabsf(x[j]) == amax) {
                max = x[j];
                break;
            }
        }

        const float iscale = -127.0f / max;
        const float32x4_t viscale = vdupq_n_f32(iscale);
        const float32x4_t vmagic = vdupq_n_f32(12582912.0f);
        const int32x4_t mmant = vdupq_n_s32(0x007fffff);
        const int32x4_t mbias = vdupq_n_s32(0x00400000);
        const int32x4_t vlo = vdupq_n_s32(-128);
        const int32x4_t vhi = vdupq_n_s32(127);

        for (int j = 0; j < QK_K; j += 16) {
            /* ck_nearest_int(iscale * x[j]): one rounded multiply, then the
             * magic-constant trick (round-to-nearest-even), vectorized. */
            int32x4_t v0 = vreinterpretq_s32_f32(vaddq_f32(
                vmulq_f32(viscale, vld1q_f32(x + j + 0)), vmagic));
            int32x4_t v1 = vreinterpretq_s32_f32(vaddq_f32(
                vmulq_f32(viscale, vld1q_f32(x + j + 4)), vmagic));
            int32x4_t v2 = vreinterpretq_s32_f32(vaddq_f32(
                vmulq_f32(viscale, vld1q_f32(x + j + 8)), vmagic));
            int32x4_t v3 = vreinterpretq_s32_f32(vaddq_f32(
                vmulq_f32(viscale, vld1q_f32(x + j + 12)), vmagic));
            v0 = vsubq_s32(vandq_s32(v0, mmant), mbias);
            v1 = vsubq_s32(vandq_s32(v1, mmant), mbias);
            v2 = vsubq_s32(vandq_s32(v2, mmant), mbias);
            v3 = vsubq_s32(vandq_s32(v3, mmant), mbias);
            v0 = vminq_s32(vmaxq_s32(v0, vlo), vhi);
            v1 = vminq_s32(vmaxq_s32(v1, vlo), vhi);
            v2 = vminq_s32(vmaxq_s32(v2, vlo), vhi);
            v3 = vminq_s32(vmaxq_s32(v3, vlo), vhi);
            const int16x8_t p0 = vcombine_s16(vmovn_s32(v0), vmovn_s32(v1));
            const int16x8_t p1 = vcombine_s16(vmovn_s32(v2), vmovn_s32(v3));
            vst1q_s8(&y[i].qs[j], vcombine_s8(vmovn_s16(p0), vmovn_s16(p1)));
        }

        for (int j = 0; j < QK_K / 16; ++j) {
            const int8x16_t q = vld1q_s8(&y[i].qs[j * 16]);
            const int32x4_t s0 = vpaddlq_s16(vmovl_s8(vget_low_s8(q)));
            const int32x4_t s1 = vpaddlq_s16(vmovl_s8(vget_high_s8(q)));
            const int32x4_t s = vaddq_s32(s0, s1);
            int32_t sl[4];
            vst1q_s32(sl, s);
            y[i].bsums[j] = (int16_t)(sl[0] + sl[1] + sl[2] + sl[3]);
        }

        y[i].d = 1.0f / iscale;
        x += QK_K;
    }
}
#endif /* __ARM_NEON || __aarch64__ */

void quantize_row_q8_k_sse(const float *x, void *vy, int k);
void quantize_row_q8_k_avx(const float *x, void *vy, int k);
void quantize_row_q8_k_avx2(const float *x, void *vy, int k);
void quantize_row_q8_k_avx512(const float *x, void *vy, int k);

#if defined(__ARM_NEON) || defined(__aarch64__)
void quantize_row_q8_k_neon(const float *x, void *vy, int k);
#endif

void quantize_row_q8_k(const float *x, void *vy, int k) {
    const char *ref_env = getenv("CK_DEBUG_Q8K_REF");
    if (ref_env && atoi(ref_env) != 0) {
        quantize_row_q8_k_ref(x, vy, k);
        return;
    }
#if defined(__AVX512F__) && defined(__AVX512BW__)
    quantize_row_q8_k_avx512(x, vy, k);
#elif defined(__AVX2__)
    quantize_row_q8_k_avx2(x, vy, k);
#elif defined(__AVX__)
    quantize_row_q8_k_avx(x, vy, k);
#elif defined(__SSE4_1__)
    quantize_row_q8_k_sse(x, vy, k);
#elif defined(__ARM_NEON) || defined(__aarch64__)
    quantize_row_q8_k_neon(x, vy, k);
#else
    quantize_row_q8_k_ref(x, vy, k);
#endif
}

const char *ck_quantize_row_q8_k_provider_name(void) {
    const char *ref_env = getenv("CK_DEBUG_Q8K_REF");
    if (ref_env && atoi(ref_env) != 0) {
        return "q8_k_quantize_ref";
    }
#if defined(__AVX512F__) && defined(__AVX512BW__)
    return "q8_k_quantize_avx512";
#elif defined(__AVX2__)
    return "q8_k_quantize_avx2";
#elif defined(__AVX__)
    return "q8_k_quantize_avx";
#elif defined(__SSE4_1__)
    return "q8_k_quantize_sse";
#elif defined(__ARM_NEON) || defined(__aarch64__)
    return "q8_k_quantize_neon";
#else
    return "q8_k_quantize_ref";
#endif
}

void quantize_batch_q8_k_4row_nearest_even(const float *x, void *vy,
                                            int num_rows, int k) {
    if (!x || !vy || num_rows <= 0 || k <= 0) {
        return;
    }
    assert(k % QK_K == 0);

    block_q8_K *y = (block_q8_K *)vy;
    const int blocks_per_row = k / QK_K;

    /* Q8_K bytes are a numerical ABI. The previous four-row AVX2 path used
     * a different max/scale evaluation order and changed real Qwen3-VL
     * visual-prefix scales by one FP32 ULP. Keep this public grouped ABI on
     * the canonical row provider until an optimized implementation is
     * byte-exact against llama.cpp on both synthetic and production inputs. */
    for (int row = 0; row < num_rows; ++row) {
        quantize_row_q8_k(
                x + (size_t)row * (size_t)k,
                y + (size_t)row * (size_t)blocks_per_row,
                k);
    }
}

static float dot_q4_k_q8_k_ref(const block_q4_K *w,
                               const block_q8_K *x,
                               int k)
{
    const int nb = k / QK_K;
    float sumf = 0.0f;

    for (int i = 0; i < nb; ++i) {
        uint8_t sc[8], m_val[8];
        unpack_q4_k_scales(w[i].scales, sc, m_val);

        const float d = CK_FP16_TO_FP32(w[i].d) * x[i].d;
        const float dmin = CK_FP16_TO_FP32(w[i].dmin) * x[i].d;

        int sumi = 0;
        for (int j = 0; j < QK_K / 16; ++j) {
            sumi += (int)x[i].bsums[j] * (int)m_val[j / 2];
        }

        int32_t scaled_sum = 0;
        for (int group = 0; group < 4; ++group) {
            const uint8_t *qs = &w[i].qs[group * 32];
            const int8_t *q8_lo = &x[i].qs[group * 64];
            const int8_t *q8_hi = q8_lo + 32;
            int32_t lo = 0;
            int32_t hi = 0;
            for (int l = 0; l < 32; ++l) {
                lo += (int32_t)(qs[l] & 0x0F) * (int32_t)q8_lo[l];
                hi += (int32_t)(qs[l] >> 4) * (int32_t)q8_hi[l];
            }
            scaled_sum += (int32_t)sc[2 * group] * lo;
            scaled_sum += (int32_t)sc[2 * group + 1] * hi;
        }
        sumf += d * (float)scaled_sum - dmin * (float)sumi;
    }
    return sumf;
}

#if defined(__ARM_FEATURE_DOTPROD)
static int32_t reduce_s32x4(int32x4_t v) {
    int32_t lanes[4];
    vst1q_s32(lanes, v);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3];
}

/* Q4_K x Q8_K dot using the AArch64 dot-product extension (SDOT). Nibbles
 * are in [0,15], so they are valid non-negative int8 values for vdotq_s32.
 * Integer arithmetic matches the scalar reference exactly; the FP32
 * per-super-block epilogue keeps the reference expression.
 *
 * Gated on __ARM_FEATURE_DOTPROD rather than plain NEON: on cores without
 * SDOT the widening (vmovl/vmlal) variant is no faster than the compiler's
 * own autovectorization of the scalar reference, so those hosts keep the
 * reference path (baseline armv8-a builds are unaffected). */
static float dot_q4_k_q8_k_neon(const block_q4_K *w,
                                const block_q8_K *x,
                                int k)
{
    const int nb = k / QK_K;
    float sumf = 0.0f;
    const uint8x16_t m4 = vdupq_n_u8(0x0F);

    for (int i = 0; i < nb; ++i) {
        uint8_t sc[8], m_val[8];
        unpack_q4_k_scales(w[i].scales, sc, m_val);

        const float d = CK_FP16_TO_FP32(w[i].d) * x[i].d;
        const float dmin = CK_FP16_TO_FP32(w[i].dmin) * x[i].d;

        /* sumi = sum_j bsums[j] * m[j/2] */
        int16_t mv[16];
        for (int j = 0; j < 16; ++j) {
            mv[j] = (int16_t)m_val[j / 2];
        }
        const int16x8_t b0 = vld1q_s16(x[i].bsums);
        const int16x8_t b1 = vld1q_s16(x[i].bsums + 8);
        const int16x8_t m0 = vld1q_s16(mv);
        const int16x8_t m1 = vld1q_s16(mv + 8);
        int32x4_t sumi_v = vmull_s16(vget_low_s16(b0), vget_low_s16(m0));
        sumi_v = vmlal_s16(sumi_v, vget_high_s16(b0), vget_high_s16(m0));
        sumi_v = vmlal_s16(sumi_v, vget_low_s16(b1), vget_low_s16(m1));
        sumi_v = vmlal_s16(sumi_v, vget_high_s16(b1), vget_high_s16(m1));
        const int32_t sumi = reduce_s32x4(sumi_v);

        /* Scale each group's partial dots in vector form so only one
         * horizontal reduction is needed per super-block. */
        int32x4_t scaled_v = vdupq_n_s32(0);
        for (int group = 0; group < 4; ++group) {
            const uint8_t *qs = &w[i].qs[group * 32];
            const int8_t *q8_lo = &x[i].qs[group * 64];
            const int8_t *q8_hi = q8_lo + 32;

            const uint8x16_t q0 = vld1q_u8(qs);
            const uint8x16_t q1 = vld1q_u8(qs + 16);

            const int8x16_t lo0 = vreinterpretq_s8_u8(vandq_u8(q0, m4));
            const int8x16_t lo1 = vreinterpretq_s8_u8(vandq_u8(q1, m4));
            const int8x16_t hi0 = vreinterpretq_s8_u8(vshrq_n_u8(q0, 4));
            const int8x16_t hi1 = vreinterpretq_s8_u8(vshrq_n_u8(q1, 4));

            int32x4_t lo = vdotq_s32(vdupq_n_s32(0), lo0, vld1q_s8(q8_lo));
            lo = vdotq_s32(lo, lo1, vld1q_s8(q8_lo + 16));
            int32x4_t hi = vdotq_s32(vdupq_n_s32(0), hi0, vld1q_s8(q8_hi));
            hi = vdotq_s32(hi, hi1, vld1q_s8(q8_hi + 16));

            scaled_v = vmlaq_s32(scaled_v, lo,
                                 vdupq_n_s32((int32_t)sc[2 * group]));
            scaled_v = vmlaq_s32(scaled_v, hi,
                                 vdupq_n_s32((int32_t)sc[2 * group + 1]));
        }
        const int32_t scaled_sum = reduce_s32x4(scaled_v);
        sumf += d * (float)scaled_sum - dmin * (float)sumi;
    }
    return sumf;
}

static void gemv_q4_k_q8_k_neon(float *y,
                                const void *W,
                                const void *x_q8,
                                int M, int K)
{
    const block_q4_K *blocks = (const block_q4_K *)W;
    const block_q8_K *x = (const block_q8_K *)x_q8;
    const int blocks_per_row = K / QK_K;

    for (int row = 0; row < M; ++row) {
        const block_q4_K *w_row = blocks + (size_t)row * (size_t)blocks_per_row;
        y[row] = dot_q4_k_q8_k_neon(w_row, x, K);
    }
}
#endif /* __ARM_FEATURE_DOTPROD */

void gemv_q4_k_q8_k_ref(float *y,
                        const void *W,
                        const void *x_q8,
                        int M, int K)
{
    if (!y || !W || !x_q8 || M <= 0 || K <= 0) {
        return;
    }

    const block_q4_K *blocks = (const block_q4_K *)W;
    const block_q8_K *x = (const block_q8_K *)x_q8;
    const int blocks_per_row = K / QK_K;

    for (int row = 0; row < M; ++row) {
        const block_q4_K *w_row = blocks + (size_t)row * (size_t)blocks_per_row;
        y[row] = dot_q4_k_q8_k_ref(w_row, x, K);
    }
}

static int ck_q4k_q8k_force_ref(void)
{
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("CK_DEBUG_Q4K_Q8_REF");
        cached = (env && env[0] && env[0] != '0') ? 1 : 0;
    }
    return cached;
}

/* ============================================================================
 * PARALLEL VERSIONS (for parallel orchestration)
 *
 * These receive ith (thread index) and nth (total threads) from orchestration.
 * OpenMP lives in orchestration layer, NOT here.
 *
 * Naming: *_parallel = receives ith/nth, processes only its portion
 *         *_ref/_avx  = single-threaded, processes all rows
 * ============================================================================ */

void gemv_q4_k_q8_k_parallel(float *y,
                             const void *W,
                             const void *x_q8,
                             int M, int K,
                             int ith, int nth)
{
    if (!y || !W || !x_q8 || M <= 0 || K <= 0) {
        return;
    }
    if (ith < 0 || nth <= 0 || ith >= nth) {
        return;
    }

    /* Compute row range for this thread */
    const int dr = (M + nth - 1) / nth;
    const int r0 = dr * ith;
    const int r1 = (r0 + dr < M) ? (r0 + dr) : M;

    if (r0 >= M) {
        return;  /* This thread has no work */
    }

    const block_q4_K *blocks = (const block_q4_K *)W;
    const block_q8_K *x = (const block_q8_K *)x_q8;
    const int blocks_per_row = K / QK_K;

    /* Only process rows [r0, r1) */
    for (int row = r0; row < r1; ++row) {
        const block_q4_K *w_row = blocks + (size_t)row * (size_t)blocks_per_row;
        y[row] = dot_q4_k_q8_k_ref(w_row, x, K);
    }
}

void gemv_q4_k_q8_k(float *y,
                    const void *W,
                    const void *x_q8,
                    int M, int K)
{
    if (ck_q4k_q8k_force_ref()) {
        gemv_q4_k_q8_k_ref(y, W, x_q8, M, K);
        return;
    }
#if defined(__AVX512VNNI__) && defined(__AVX512VL__) && !defined(CK_NO_AVX512_VNNI)
    /* VNNI: Best for decode (single token) - INT8 dot product acceleration */
    gemv_q4_k_q8_k_vnni(y, W, x_q8, M, K);
#elif defined(__AVX2__)
    gemv_q4_k_q8_k_avx2(y, W, x_q8, M, K);
#elif defined(__AVX__)
    /* AVX version uses maddubs_epi16 (more efficient than SSE) */
    gemv_q4_k_q8_k_avx(y, W, x_q8, M, K);
#elif defined(__SSE4_1__)
    gemv_q4_k_q8_k_sse(y, W, x_q8, M, K);
#elif defined(__ARM_FEATURE_DOTPROD)
    /* SDOT NEON is integer-exact against the scalar reference (identical
     * block arithmetic and FP32 epilogue); CK_DEBUG_Q4K_Q8_REF=1 still
     * forces the reference above. Baseline armv8-a builds (no dot-product
     * extension) keep the reference path, which the compiler already
     * autovectorizes well. */
    gemv_q4_k_q8_k_neon(y, W, x_q8, M, K);
#else
    gemv_q4_k_q8_k_ref(y, W, x_q8, M, K);
#endif
}

const char *ck_q4_k_q8_k_provider_name(void)
{
    if (ck_q4k_q8k_force_ref()) {
        return "q4_k_q8_k_ref";
    }
#if defined(__AVX512VNNI__) && defined(__AVX512VL__) && !defined(CK_NO_AVX512_VNNI)
    return "q4_k_q8_k_vnni";
#elif defined(__AVX2__)
    return "q4_k_q8_k_avx2";
#elif defined(__AVX__)
    return "q4_k_q8_k_avx";
#elif defined(__SSE4_1__)
    return "q4_k_q8_k_sse";
#elif defined(__ARM_FEATURE_DOTPROD)
    return "q4_k_q8_k_neon";
#else
    return "q4_k_q8_k_ref";
#endif
}

void gemm_q4_k_q8_k_ref(float *Y,
                        const void *W,
                        const void *X_q8,
                        int M, int N, int K)
{
    if (!Y || !W || !X_q8 || M <= 0 || N <= 0 || K <= 0) {
        return;
    }

    const block_q8_K *X = (const block_q8_K *)X_q8;
    const int blocks_per_vec = K / QK_K;

    for (int n = 0; n < N; ++n) {
        const block_q8_K *x_row = X + (size_t)n * (size_t)blocks_per_vec;
        gemv_q4_k_q8_k_ref(&Y[n * M], W, x_row, M, K);
    }
}

typedef struct {
    float *Y;
    const void *W;
    const block_q8_K *X;
    int M_out;
    int N_batch;
    int K;
    int blocks_per_vec;
    int blocks_per_row;
} gemm_q4_k_q8_k_work_t;

static void gemm_q4_k_q8_k_thread_fn(int ith, int nth, void *args)
{
    gemm_q4_k_q8_k_work_t *a = (gemm_q4_k_q8_k_work_t *)args;
    if (!a || ith < 0 || nth <= 0 || ith >= nth) {
        return;
    }

    const int dr = (a->M_out + nth - 1) / nth;
    const int r0 = dr * ith;
    const int r1 = (r0 + dr < a->M_out) ? (r0 + dr) : a->M_out;
    if (r0 >= a->M_out) {
        return;
    }

    const block_q4_K *blocks = (const block_q4_K *)a->W;
    const block_q4_K *w_start = blocks + (size_t)r0 * (size_t)a->blocks_per_row;
    const int rows = r1 - r0;

    for (int n = 0; n < a->N_batch; ++n) {
        const block_q8_K *x_row = a->X + (size_t)n * (size_t)a->blocks_per_vec;
        gemv_q4_k_q8_k(a->Y + (size_t)n * (size_t)a->M_out + (size_t)r0,
                       w_start,
                       x_row,
                       rows,
                       a->K);
    }
}

void gemm_q4_k_q8_k(float *Y,
                    const void *W,
                    const void *X_q8,
                    int M, int N, int K)
{
    if (!Y || !W || !X_q8 || M <= 0 || N <= 0 || K <= 0) {
        return;
    }

    const block_q8_K *X = (const block_q8_K *)X_q8;
    const int blocks_per_vec = K / QK_K;
    const int blocks_per_row = K / QK_K;
    const size_t work_items = (size_t)M * (size_t)N;

    if (work_items >= 4096u && M >= 512 && N > 1) {
        ck_threadpool_t *pool = ck_threadpool_global();
        const int pool_threads = pool ? ck_threadpool_n_threads(pool) : 1;
        int active_threads = pool_threads;
        if (active_threads > M) {
            active_threads = M;
        }
        if (active_threads > 1) {
            gemm_q4_k_q8_k_work_t work = {
                .Y = Y,
                .W = W,
                .X = X,
                .M_out = M,
                .N_batch = N,
                .K = K,
                .blocks_per_vec = blocks_per_vec,
                .blocks_per_row = blocks_per_row,
            };
            ck_threadpool_dispatch_n(pool, active_threads, gemm_q4_k_q8_k_thread_fn, &work);
            return;
        }
    }

    for (int n = 0; n < N; ++n) {
        const block_q8_K *x_row = X + (size_t)n * (size_t)blocks_per_vec;
        gemv_q4_k_q8_k(&Y[n * M], W, x_row, M, K);
    }
}

void gemm_nt_q4_k_q8_k(const void *A_q8,
                       const void *B,
                       const float *bias,
                       float *C,
                       int M, int N, int K)
{
    if (!A_q8 || !B || !C) {
        return;
    }
    if (M <= 0 || N <= 0 || K <= 0) {
        return;
    }

    gemm_q4_k_q8_k(C, B, A_q8, /*M_out=*/N, /*N_batch=*/M, K);

    if (!bias) {
        return;
    }

    for (int i = 0; i < M; ++i) {
        float *row = C + (size_t)i * (size_t)N;
        for (int j = 0; j < N; ++j) {
            row[j] += bias[j];
        }
    }
}
