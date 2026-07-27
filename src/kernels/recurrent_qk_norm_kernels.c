#include "bf16_utils.h"
#include "ckernel_engine.h"

#include <math.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

static void recurrent_l2_norm_rows_forward_one(float *x,
                                               int rows,
                                               int dim,
                                               int head_dim,
                                               float eps) {
    if (!x || rows <= 0 || dim <= 0 || head_dim <= 0) {
        return;
    }
    const int num_heads = dim / head_dim;
    if (num_heads <= 0 || num_heads * head_dim != dim) {
        return;
    }

    for (int row = 0; row < rows; ++row) {
        float *row_ptr = x + (size_t) row * (size_t) dim;
        for (int head = 0; head < num_heads; ++head) {
            float *head_ptr = row_ptr + (size_t) head * (size_t) head_dim;
            double sum_sq = 0.0;
            for (int col = 0; col < head_dim; ++col) {
                sum_sq += (double) (head_ptr[col] * head_ptr[col]);
            }
            const float norm = sqrtf((float) sum_sq);
            const float inv_norm = 1.0f / fmaxf(norm, eps);
            for (int col = 0; col < head_dim; ++col) {
                head_ptr[col] *= inv_norm;
            }
        }
    }
}

static void recurrent_l2_norm_rows_backward_one(const float *d_out,
                                                const float *x,
                                                float *d_x,
                                                int rows,
                                                int dim,
                                                int head_dim,
                                                float eps) {
    if (!d_out || !x || !d_x || rows <= 0 || dim <= 0 || head_dim <= 0) {
        return;
    }
    const int num_heads = dim / head_dim;
    if (num_heads <= 0 || num_heads * head_dim != dim) {
        return;
    }

    for (int row = 0; row < rows; ++row) {
        const float *d_row = d_out + (size_t) row * (size_t) dim;
        const float *x_row = x + (size_t) row * (size_t) dim;
        float *dx_row = d_x + (size_t) row * (size_t) dim;
        for (int head = 0; head < num_heads; ++head) {
            const float *d_head = d_row + (size_t) head * (size_t) head_dim;
            const float *x_head = x_row + (size_t) head * (size_t) head_dim;
            float *dx_head = dx_row + (size_t) head * (size_t) head_dim;

            double sum_sq = 0.0;
            double dot = 0.0;
            for (int col = 0; col < head_dim; ++col) {
                sum_sq += (double) (x_head[col] * x_head[col]);
                dot += (double) (d_head[col] * x_head[col]);
            }
            const float norm = sqrtf((float) sum_sq);
            const float inv_norm = 1.0f / fmaxf(norm, eps);
            if (norm <= eps) {
                for (int col = 0; col < head_dim; ++col) {
                    dx_head[col] = d_head[col] / eps;
                }
                continue;
            }
            const float proj_scale = inv_norm * inv_norm * inv_norm * (float) dot;
            for (int col = 0; col < head_dim; ++col) {
                dx_head[col] = inv_norm * d_head[col] - proj_scale * x_head[col];
            }
        }
    }
}

void recurrent_qk_l2_norm_forward(float *q,
                                  float *k,
                                  int rows,
                                  int q_dim,
                                  int k_dim,
                                  int head_dim,
                                  float eps) {
    recurrent_l2_norm_rows_forward_one(q, rows, q_dim, head_dim, eps);
    recurrent_l2_norm_rows_forward_one(k, rows, k_dim, head_dim, eps);
}

static float recurrent_pytorch_bf16_square_sum(const float *x, int dim)
{
#if defined(__AVX2__)
    if (dim >= 32) {
        __m256 streams[4] = {
            _mm256_setzero_ps(), _mm256_setzero_ps(),
            _mm256_setzero_ps(), _mm256_setzero_ps()
        };
        int d = 0;
        for (; d + 32 <= dim; d += 32) {
            for (int stream = 0; stream < 4; ++stream) {
                _Alignas(32) float lanes[8];
                const __m256 values = _mm256_loadu_ps(x + d + stream * 8);
                _mm256_store_ps(lanes, _mm256_mul_ps(values, values));
                for (int lane = 0; lane < 8; ++lane) {
                    lanes[lane] =
                        bf16_to_float(float_to_bf16(lanes[lane]));
                }
                streams[stream] =
                    _mm256_add_ps(streams[stream], _mm256_load_ps(lanes));
            }
        }
        __m256 reduced = _mm256_add_ps(streams[0], streams[1]);
        reduced = _mm256_add_ps(reduced, streams[2]);
        reduced = _mm256_add_ps(reduced, streams[3]);
        _Alignas(32) float lanes[8];
        _mm256_store_ps(lanes, reduced);
        volatile float sum = 0.0f;
        for (int lane = 0; lane < 8; ++lane) {
            sum = sum + lanes[lane];
        }
        for (; d < dim; ++d) {
            const float value = bf16_to_float(float_to_bf16(x[d]));
            const float square =
                bf16_to_float(float_to_bf16(value * value));
            sum = sum + square;
        }
        return sum;
    }
#endif
    volatile float sum = 0.0f;
    for (int d = 0; d < dim; ++d) {
        const float value = bf16_to_float(float_to_bf16(x[d]));
        const float square = bf16_to_float(float_to_bf16(value * value));
        sum = sum + square;
    }
    return sum;
}

static void recurrent_pytorch_bf16_l2_rows(float *x,
                                            int rows,
                                            int dim,
                                            int expanded_heads,
                                            int head_dim,
                                            float eps)
{
    if (!x || rows <= 0 || dim <= 0 || head_dim <= 0 ||
        dim % head_dim != 0) {
        return;
    }
    const int num_heads = dim / head_dim;
    if (expanded_heads <= 0 || expanded_heads % num_heads != 0) {
        return;
    }
    const int heads_per_group = expanded_heads / num_heads;
    const int total_heads = rows * expanded_heads;
    const int vector_limit = (total_heads / 32) * 32;
    for (int row = 0; row < rows; ++row) {
        float *row_ptr = x + (size_t)row * (size_t)dim;
        for (int head = 0; head < num_heads; ++head) {
            float *head_ptr =
                row_ptr + (size_t)head * (size_t)head_dim;
            float sum = recurrent_pytorch_bf16_square_sum(
                head_ptr, head_dim);
            sum = bf16_to_float(float_to_bf16(sum));
            const float denominator =
                bf16_to_float(float_to_bf16(sum + eps));
            /*
             * Qwen3Next repeat_interleave() expands compact Q/K groups to
             * value heads before l2norm.  TensorIterator's rsqrt tail is
             * therefore determined by the expanded tensor, not this compact
             * storage.  Select the arithmetic used by the first repeated
             * lane; all lanes are vector lanes for production prefill.
             */
            const int flat_head =
                (row * expanded_heads) + head * heads_per_group;
            float inverse;
            if (flat_head < vector_limit) {
                /* ATen Vec<BF16> evaluates 32 scalar outputs per vector
                 * iteration using FP32 rsqrt followed by BF16 storage. */
                inverse = bf16_to_float(
                    float_to_bf16(1.0f / sqrtf(denominator)));
            } else {
                /* TensorIterator's BF16 scalar tail materializes sqrt to
                 * BF16 before applying the BF16 reciprocal. */
                const float root =
                    bf16_to_float(float_to_bf16(sqrtf(denominator)));
                inverse =
                    bf16_to_float(float_to_bf16(1.0f / root));
            }
            for (int col = 0; col < head_dim; ++col) {
                const float value =
                    bf16_to_float(float_to_bf16(head_ptr[col]));
                head_ptr[col] = bf16_to_float(
                    float_to_bf16(value * inverse));
            }
        }
    }
}

void recurrent_qk_l2_norm_pytorch_bf16_storage(float *q,
                                                float *k,
                                                int rows,
                                                int q_dim,
                                                int k_dim,
                                                int expanded_heads,
                                                int head_dim,
                                                float eps)
{
    recurrent_pytorch_bf16_l2_rows(
        q, rows, q_dim, expanded_heads, head_dim, eps);
    recurrent_pytorch_bf16_l2_rows(
        k, rows, k_dim, expanded_heads, head_dim, eps);
}

void recurrent_qk_l2_norm_backward(const float *d_q_out,
                                   const float *d_k_out,
                                   const float *q,
                                   const float *k,
                                   float *d_q,
                                   float *d_k,
                                   int rows,
                                   int q_dim,
                                   int k_dim,
                                   int head_dim,
                                   float eps) {
    recurrent_l2_norm_rows_backward_one(d_q_out, q, d_q, rows, q_dim, head_dim, eps);
    recurrent_l2_norm_rows_backward_one(d_k_out, k, d_k, rows, k_dim, head_dim, eps);
}
