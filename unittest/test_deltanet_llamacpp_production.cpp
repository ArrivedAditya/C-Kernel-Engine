// Grouped Gated DeltaNet parity against llama.cpp's production fused CPU op
// for both decode and multi-token prefill.

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
void gated_deltanet_llama_avx2_forward(
        const float *, const float *, const float *, const float *, const float *,
        const float *, float *, float *, int, int, int, float);
void gated_deltanet_llama_avx2_parallel_forward(
        const float *, const float *, const float *, const float *, const float *,
        const float *, float *, float *, int, int, int, float);
void gated_deltanet_llama_avx2_prefill_forward(
        const float *, const float *, const float *, const float *, const float *,
        const float *, float *, float *, int, int, int, int, float);
void gated_deltanet_llama_chunk64_prefill_forward(
        const float *, const float *, const float *, const float *, const float *,
        const float *, float *, float *, int, int, int, int, float);
void recurrent_norm_gate_llama_avx2_forward(
        const float *, const float *, const float *, float *, int, int, int, float);
}

namespace {

static float fixture(size_t i, float scale, float phase) {
    const float x = static_cast<float>(i);
    return std::sin(x * 0.017f + phase) * scale
         + std::cos(x * 0.0031f - phase) * scale * 0.37f;
}

static bool close(const char * label, const float * ck, const float * oracle,
                  size_t count, float atol) {
    if (std::memcmp(ck, oracle, count * sizeof(float)) == 0) {
        std::printf("  %-24s bit_exact (%zu values) [PASS]\n", label, count);
        return true;
    }
    size_t first = count;
    size_t different = 0;
    float max_abs = 0.0f;
    size_t worst = 0;
    for (size_t i = 0; i < count; ++i) {
        if (std::memcmp(ck + i, oracle + i, sizeof(float)) != 0) {
            if (first == count) first = i;
            ++different;
        }
        const float diff = std::fabs(ck[i] - oracle[i]);
        if (diff > max_abs) {
            max_abs = diff;
            worst = i;
        }
    }
    const bool ok = max_abs <= atol;
    std::printf("  %-24s different=%zu/%zu first=%zu worst=%zu max_abs=%.9g [%s]\n",
            label, different, count, first, worst, max_abs, ok ? "PASS" : "FAIL");
    return ok;
    return false;
}

static bool llama_fused_graph(
        const std::vector<float> & q,
        const std::vector<float> & k,
        const std::vector<float> & v,
        const std::vector<float> & g,
        const std::vector<float> & beta_raw,
        const std::vector<float> & state_ck,
        std::vector<float> & output,
        std::vector<float> & state_ck_out,
        int rows, int heads, int groups, int dim) {
    const size_t arena = 64u * 1024u * 1024u;
    ggml_init_params params = {arena, nullptr, false};
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor * tq = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, groups, rows, 1);
    ggml_tensor * tk = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, groups, rows, 1);
    ggml_tensor * tv = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, heads, rows, 1);
    ggml_tensor * tg = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, heads, rows, 1);
    ggml_tensor * tb = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, heads, rows, 1);
    ggml_tensor * ts = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, dim, heads, 1);
    std::memcpy(ggml_get_data(tq), q.data(), q.size() * sizeof(float));
    std::memcpy(ggml_get_data(tk), k.data(), k.size() * sizeof(float));
    std::memcpy(ggml_get_data(tv), v.data(), v.size() * sizeof(float));
    std::memcpy(ggml_get_data(tg), g.data(), g.size() * sizeof(float));

    std::vector<float> beta(beta_raw.size());
    float (*volatile llama_expf)(float) = expf;
    for (size_t i = 0; i < beta.size(); ++i) {
        beta[i] = 1.0f / (1.0f + llama_expf(-beta_raw[i]));
    }
    std::memcpy(ggml_get_data(tb), beta.data(), beta.size() * sizeof(float));

    float * llama_state = ggml_get_data_f32(ts);
    for (int h = 0; h < heads; ++h) {
        for (int key = 0; key < dim; ++key) {
            for (int value = 0; value < dim; ++value) {
                llama_state[(static_cast<size_t>(h) * dim + value) * dim + key] =
                    state_ck[(static_cast<size_t>(h) * dim + key) * dim + value];
            }
        }
    }

    ggml_tensor * result = ggml_gated_delta_net(ctx, tq, tk, tv, tg, tb, ts, 1);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    const int threads = std::max(1, std::atoi(
            std::getenv("CK_NUM_THREADS") ? std::getenv("CK_NUM_THREADS") : "1"));
    const bool ok = ggml_graph_compute_with_ctx(ctx, graph, threads) == GGML_STATUS_SUCCESS;
    if (ok) {
        const float * packed = ggml_get_data_f32(result);
        const size_t output_count = static_cast<size_t>(rows) * heads * dim;
        std::memcpy(output.data(), packed, output_count * sizeof(float));
        const float * final_state = packed + output_count;
        for (int h = 0; h < heads; ++h) {
            for (int key = 0; key < dim; ++key) {
                for (int value = 0; value < dim; ++value) {
                    state_ck_out[(static_cast<size_t>(h) * dim + key) * dim + value] =
                        final_state[(static_cast<size_t>(h) * dim + value) * dim + key];
                }
            }
        }
    }
    ggml_free(ctx);
    return ok;
}

static ggml_tensor * chunk_slice_2d(ggml_context * ctx, ggml_tensor * tensor, int64_t chunk) {
    return ggml_view_4d(ctx, tensor, tensor->ne[0], tensor->ne[1], 1, tensor->ne[3],
            tensor->nb[1], tensor->nb[2], tensor->nb[3], tensor->nb[2] * chunk);
}

static bool llama_chunk_graph(
        const std::vector<float> & q_compact,
        const std::vector<float> & k_compact,
        const std::vector<float> & v_input,
        const std::vector<float> & g_input,
        const std::vector<float> & beta_raw,
        const std::vector<float> & state_ck,
        std::vector<float> & output,
        std::vector<float> & state_ck_out,
        int rows, int heads, int groups, int dim) {
    const size_t arena = 256u * 1024u * 1024u;
    ggml_init_params params = {arena, nullptr, false};
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor * tq_compact = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, groups, rows, 1);
    ggml_tensor * tk_compact = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, groups, rows, 1);
    ggml_tensor * tv = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, heads, rows, 1);
    ggml_tensor * tg = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, heads, rows, 1);
    ggml_tensor * tb = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, heads, rows, 1);
    ggml_tensor * ts = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, dim, heads, 1);
    std::memcpy(ggml_get_data(tq_compact), q_compact.data(), q_compact.size() * sizeof(float));
    std::memcpy(ggml_get_data(tk_compact), k_compact.data(), k_compact.size() * sizeof(float));
    std::memcpy(ggml_get_data(tv), v_input.data(), v_input.size() * sizeof(float));
    std::memcpy(ggml_get_data(tg), g_input.data(), g_input.size() * sizeof(float));
    std::vector<float> beta(beta_raw.size());
    float (*volatile llama_expf)(float) = expf;
    for (size_t i = 0; i < beta.size(); ++i) {
        beta[i] = 1.0f / (1.0f + llama_expf(-beta_raw[i]));
    }
    std::memcpy(ggml_get_data(tb), beta.data(), beta.size() * sizeof(float));
    float * llama_state = ggml_get_data_f32(ts);
    for (int h = 0; h < heads; ++h) {
        for (int key = 0; key < dim; ++key) {
            for (int value = 0; value < dim; ++value) {
                llama_state[(static_cast<size_t>(h) * dim + value) * dim + key] =
                    state_ck[(static_cast<size_t>(h) * dim + key) * dim + value];
            }
        }
    }

    ggml_tensor * q = ggml_repeat_4d(ctx, tq_compact, dim, heads, rows, 1);
    ggml_tensor * k = ggml_repeat_4d(ctx, tk_compact, dim, heads, rows, 1);
    q = ggml_scale(ctx, q, 1.0f / sqrtf(static_cast<float>(dim)));
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    ggml_tensor * v = ggml_permute(ctx, tv, 0, 2, 1, 3);
    ggml_tensor * g = ggml_permute(ctx, tg, 0, 2, 1, 3);
    ggml_tensor * b = ggml_permute(ctx, tb, 0, 2, 1, 3);

    constexpr int CS = 64;
    const int pad = (CS - rows % CS) % CS;
    const int chunks = (rows + pad) / CS;
    q = ggml_pad(ctx, q, 0, pad, 0, 0);
    k = ggml_pad(ctx, k, 0, pad, 0, 0);
    v = ggml_pad(ctx, v, 0, pad, 0, 0);
    g = ggml_pad(ctx, g, 0, pad, 0, 0);
    b = ggml_pad(ctx, b, 0, pad, 0, 0);

    ggml_tensor * v_b = ggml_mul(ctx, v, b);
    ggml_tensor * k_b = ggml_mul(ctx, k, b);
    q = ggml_reshape_4d(ctx, q, dim, CS, chunks, heads);
    k = ggml_reshape_4d(ctx, k, dim, CS, chunks, heads);
    k_b = ggml_reshape_4d(ctx, k_b, dim, CS, chunks, heads);
    v = ggml_reshape_4d(ctx, v, dim, CS, chunks, heads);
    v_b = ggml_reshape_4d(ctx, v_b, dim, CS, chunks, heads);
    g = ggml_reshape_4d(ctx, g, 1, CS, chunks, heads);

    ggml_tensor * g_cs = ggml_cumsum(ctx, ggml_cont(ctx, ggml_transpose(ctx, g)));
    ggml_tensor * g_cs_j = ggml_reshape_4d(ctx, g_cs, 1, CS, chunks, heads);
    g_cs_j = ggml_repeat_4d(ctx, g_cs_j, CS, CS, chunks, heads);
    ggml_tensor * decay = ggml_sub(ctx, g_cs_j, g_cs);
    decay = ggml_exp(ctx, ggml_tri(ctx, decay, GGML_TRI_TYPE_LOWER_DIAG));

    ggml_tensor * kb = ggml_mul(ctx, ggml_mul_mat(ctx, k, k_b), decay);
    ggml_tensor * kq = ggml_mul(ctx, ggml_mul_mat(ctx, k, q), decay);
    kq = ggml_tri(ctx, kq, GGML_TRI_TYPE_LOWER_DIAG);
    ggml_tensor * attn = ggml_tri(ctx, kb, GGML_TRI_TYPE_LOWER);
    ggml_tensor * identity = ggml_diag(ctx, ggml_fill(
            ctx, ggml_view_1d(ctx, attn, CS, 0), 1.0f));
    ggml_tensor * lhs = ggml_add(ctx, attn, identity);
    attn = ggml_add(ctx, ggml_solve_tri(
            ctx, lhs, ggml_neg(ctx, attn), true, true, false), identity);

    v = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, v_b)), attn);
    ggml_tensor * g_exp = ggml_exp(ctx, g_cs);
    k_b = ggml_cont(ctx, ggml_transpose(ctx, k_b));
    ggml_tensor * k_cd = ggml_mul_mat(ctx, ggml_mul(ctx, k_b, g_exp), attn);
    ggml_tensor * g_exp_t = ggml_cont(ctx, ggml_transpose(ctx, g_exp));
    ggml_tensor * q_g_exp = ggml_mul(ctx, q, g_exp_t);

    ggml_tensor * g_last = ggml_view_4d(ctx, g_cs, 1, g_cs->ne[1], g_cs->ne[2], g_cs->ne[3],
            g_cs->nb[1], g_cs->nb[2], g_cs->nb[3],
            ggml_row_size(g_cs->type, g_cs->ne[0] - 1));
    g_last = ggml_cont(ctx, g_last);
    ggml_tensor * g_last_exp_t = ggml_transpose(ctx, ggml_exp(ctx, g_last));
    ggml_tensor * g_diff = ggml_neg(ctx, ggml_sub(ctx, g_cs, g_last));
    ggml_tensor * g_diff_exp_t = ggml_cont(ctx, ggml_transpose(ctx, ggml_exp(ctx, g_diff)));
    ggml_tensor * kg = ggml_mul(ctx, k, g_diff_exp_t);
    ggml_tensor * kg_t = ggml_cont(ctx, ggml_transpose(ctx, kg));

    ggml_tensor * s = ggml_reshape_4d(ctx, ts, dim, dim, 1, heads);
    ggml_tensor * v_t = ggml_cont(ctx, ggml_transpose(ctx, v));
    for (int chunk = 0; chunk < chunks; ++chunk) {
        ggml_tensor * ch_k_cd = chunk_slice_2d(ctx, k_cd, chunk);
        ggml_tensor * ch_v_t = chunk_slice_2d(ctx, v_t, chunk);
        ggml_tensor * ch_kq = chunk_slice_2d(ctx, kq, chunk);
        ggml_tensor * ch_q_g_exp = chunk_slice_2d(ctx, q_g_exp, chunk);
        ggml_tensor * ch_kg_t = chunk_slice_2d(ctx, kg_t, chunk);
        ggml_tensor * v_t_new = ggml_sub(ctx, ch_v_t, ggml_mul_mat(ctx, ch_k_cd, s));
        ggml_tensor * o_ch = ggml_add(ctx,
                ggml_mul_mat(ctx, s, ch_q_g_exp),
                ggml_mul_mat(ctx, v_t_new, ch_kq));
        v = ggml_set_inplace(ctx, v, o_ch, v->nb[1], v->nb[2], v->nb[3],
                chunk * v->nb[2]);
        ggml_tensor * kgv = ggml_mul_mat(ctx, ch_kg_t, v_t_new);
        s = ggml_add(ctx, ggml_mul(ctx, s,
                chunk_slice_2d(ctx, g_last_exp_t, chunk)), kgv);
    }

    ggml_tensor * o = ggml_view_4d(ctx, v, dim, rows, heads, 1,
            ggml_row_size(v->type, dim),
            ggml_row_size(v->type, dim * CS * chunks),
            ggml_row_size(v->type, dim * CS * chunks * heads), 0);
    o = ggml_permute(ctx, o, 0, 2, 1, 3);
    s = ggml_reshape_4d(ctx, s, dim, dim, heads, 1);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE * 8, false);
    ggml_build_forward_expand(graph, o);
    ggml_build_forward_expand(graph, s);
    const int threads = std::max(1, std::atoi(
            std::getenv("CK_NUM_THREADS") ? std::getenv("CK_NUM_THREADS") : "1"));
    const bool ok = ggml_graph_compute_with_ctx(ctx, graph, threads) == GGML_STATUS_SUCCESS;
    if (ok) {
        const char * o_data = static_cast<const char *>(o->data);
        for (int token = 0; token < rows; ++token) {
            for (int h = 0; h < heads; ++h) {
                for (int d = 0; d < dim; ++d) {
                    output[(static_cast<size_t>(token) * heads + h) * dim + d] =
                        *reinterpret_cast<const float *>(
                            o_data + d * o->nb[0] + h * o->nb[1] + token * o->nb[2]);
                }
            }
        }
        const float * final_state = ggml_get_data_f32(s);
        for (int h = 0; h < heads; ++h) {
            for (int key = 0; key < dim; ++key) {
                for (int value = 0; value < dim; ++value) {
                    state_ck_out[(static_cast<size_t>(h) * dim + key) * dim + value] =
                        final_state[(static_cast<size_t>(h) * dim + value) * dim + key];
                }
            }
        }
    }
    ggml_free(ctx);
    return ok;
}

static bool run_case(int rows, int heads = 16, int groups = 4,
                     bool parallel_decode = false) {
    constexpr int dim = 128;
    const size_t qk_vectors = static_cast<size_t>(rows) * groups * dim;
    const size_t vectors = static_cast<size_t>(rows) * heads * dim;
    const size_t gates = static_cast<size_t>(rows) * heads;
    const size_t states = static_cast<size_t>(heads) * dim * dim;
    std::vector<float> q(qk_vectors), k(qk_vectors), v(vectors), g(gates), beta(gates), state(states);
    for (size_t i = 0; i < qk_vectors; ++i) {
        q[i] = fixture(i, 0.09f, 0.13f);
        k[i] = fixture(i, 0.08f, 0.29f);
    }
    for (size_t i = 0; i < vectors; ++i) {
        v[i] = fixture(i, 0.21f, 0.47f);
    }
    for (size_t i = 0; i < gates; ++i) {
        g[i] = -0.03f - std::fabs(fixture(i, 0.08f, 0.61f));
        beta[i] = fixture(i, 0.7f, 0.83f);
    }
    for (size_t i = 0; i < states; ++i) state[i] = fixture(i, 0.04f, 1.07f);

    std::vector<float> ck_out(vectors), llama_out(vectors);
    std::vector<float> ck_state(states), llama_state(states);
    const bool oracle_ok =
        llama_fused_graph(q, k, v, g, beta, state, llama_out, llama_state,
                rows, heads, groups, dim);
    if (!oracle_ok) {
        std::printf("llama.cpp graph execution failed\n");
        return false;
    }
    if (rows == 1) {
        if (parallel_decode) {
            gated_deltanet_llama_avx2_parallel_forward(
                    q.data(), k.data(), v.data(), g.data(), beta.data(),
                    state.data(), ck_state.data(), ck_out.data(),
                    heads, groups, dim, 1e-6f);
        } else {
            gated_deltanet_llama_avx2_forward(
                    q.data(), k.data(), v.data(), g.data(), beta.data(),
                    state.data(), ck_state.data(), ck_out.data(),
                    heads, groups, dim, 1e-6f);
        }
    } else {
        gated_deltanet_llama_avx2_prefill_forward(q.data(), k.data(), v.data(), g.data(), beta.data(),
                state.data(), ck_state.data(), ck_out.data(), rows, heads, groups, dim, 1e-6f);
    }
    std::printf("rows=%d heads=%d groups=%d dim=%d provider=%s threads=%s\n",
            rows, heads, groups, dim, parallel_decode ? "parallel" : "serial",
            std::getenv("CK_NUM_THREADS") ? std::getenv("CK_NUM_THREADS") : "1");
    const bool output_ok =
        close("attention output", ck_out.data(), llama_out.data(), vectors, 2e-8f);
    const bool state_ok =
        close("recurrent state", ck_state.data(), llama_state.data(), states, 2e-8f);
    return output_ok && state_ok;
}

static bool run_chunk_case(int rows) {
    constexpr int heads = 16;
    constexpr int groups = heads / 4;
    constexpr int dim = 128;
    const size_t qk_vectors = static_cast<size_t>(rows) * groups * dim;
    const size_t vectors = static_cast<size_t>(rows) * heads * dim;
    const size_t gates = static_cast<size_t>(rows) * heads;
    const size_t states = static_cast<size_t>(heads) * dim * dim;
    std::vector<float> q(qk_vectors), k(qk_vectors), v(vectors), g(gates), beta(gates), state(states);
    for (size_t i = 0; i < qk_vectors; ++i) {
        q[i] = fixture(i, 0.09f, 0.13f);
        k[i] = fixture(i, 0.08f, 0.29f);
    }
    for (size_t i = 0; i < vectors; ++i) {
        v[i] = fixture(i, 0.21f, 0.47f);
    }
    for (size_t i = 0; i < gates; ++i) {
        g[i] = -0.03f - std::fabs(fixture(i, 0.08f, 0.61f));
        beta[i] = fixture(i, 0.7f, 0.83f);
    }
    for (size_t i = 0; i < states; ++i) state[i] = fixture(i, 0.04f, 1.07f);

    std::vector<float> ck_out(vectors), llama_out(vectors);
    std::vector<float> ck_state(states), llama_state(states);
    if (!llama_chunk_graph(
            q, k, v, g, beta, state, llama_out, llama_state,
            rows, heads, groups, dim)) {
        std::printf("llama.cpp chunk graph execution failed\n");
        return false;
    }
    gated_deltanet_llama_chunk64_prefill_forward(
        q.data(), k.data(), v.data(), g.data(), beta.data(),
        state.data(), ck_state.data(), ck_out.data(),
        rows, heads, groups, dim, 1e-6f);
    std::printf("chunk rows=%d heads=%d dim=%d threads=%s\n", rows, heads, dim,
            std::getenv("CK_NUM_THREADS") ? std::getenv("CK_NUM_THREADS") : "1");
    const bool output_ok =
        close("chunk attention output", ck_out.data(), llama_out.data(), vectors, 2e-8f);
    const bool state_ok =
        close("chunk recurrent state", ck_state.data(), llama_state.data(), states, 2e-8f);
    return output_ok && state_ok;
}

static bool run_norm_gate_case(int rows) {
    constexpr int heads = 16;
    constexpr int dim = 128;
    const size_t count = static_cast<size_t>(rows) * heads * dim;
    std::vector<float> x(count), gate(count), weight(dim), ck(count), oracle(count);
    for (size_t i = 0; i < count; ++i) {
        x[i] = fixture(i, 0.19f, 0.37f);
        gate[i] = fixture(i, 1.7f, 0.71f);
    }
    for (int i = 0; i < dim; ++i) weight[i] = 0.8f + fixture(i, 0.12f, 1.19f);

    ggml_init_params params = {32u * 1024u * 1024u, nullptr, false};
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;
    ggml_tensor * tx = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, heads, rows);
    ggml_tensor * tg = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, heads, rows);
    ggml_tensor * tw = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
    std::memcpy(ggml_get_data(tx), x.data(), count * sizeof(float));
    std::memcpy(ggml_get_data(tg), gate.data(), count * sizeof(float));
    std::memcpy(ggml_get_data(tw), weight.data(), weight.size() * sizeof(float));
    ggml_tensor * normalized = ggml_rms_norm(ctx, tx, 1e-6f);
    normalized = ggml_mul(ctx, normalized, tw);
    ggml_tensor * silu = ggml_silu(ctx, tg);
    ggml_tensor * result = ggml_mul(ctx, normalized, silu);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    const int threads = std::max(1, std::atoi(
            std::getenv("CK_NUM_THREADS") ? std::getenv("CK_NUM_THREADS") : "1"));
    const bool ok = ggml_graph_compute_with_ctx(ctx, graph, threads) == GGML_STATUS_SUCCESS;
    if (ok) std::memcpy(oracle.data(), ggml_get_data_f32(result), count * sizeof(float));
    ggml_free(ctx);
    if (!ok) return false;

    recurrent_norm_gate_llama_avx2_forward(
        x.data(), gate.data(), weight.data(), ck.data(), rows, heads, dim, 1e-6f);
    std::printf("norm_gate rows=%d heads=%d dim=%d threads=%s\n", rows, heads, dim,
            std::getenv("CK_NUM_THREADS") ? std::getenv("CK_NUM_THREADS") : "1");
    return close("gated normalization", ck.data(), oracle.data(), count, 1e-6f);
}

} // namespace

int main() {
    const bool decode = run_case(1);
    const bool qwen36_parallel_decode = run_case(1, 48, 16, true);
    const bool prefill = run_case(18);
    const bool prefill_cross_chunk = run_case(65);
    const bool chunk_prefill = run_chunk_case(18);
    const bool chunk_cross_chunk = run_chunk_case(65);
    const bool norm_decode = run_norm_gate_case(1);
    const bool norm_prefill = run_norm_gate_case(18);
    return decode && qwen36_parallel_decode && prefill && prefill_cross_chunk &&
        chunk_prefill && chunk_cross_chunk &&
        norm_decode && norm_prefill ? 0 : 1;
}
