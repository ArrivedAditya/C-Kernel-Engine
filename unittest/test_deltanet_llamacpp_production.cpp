// Authoritative Gated DeltaNet parity against llama.cpp's production CPU graph.

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
        const float *, float *, float *, int, int, float);
void gated_deltanet_llama_avx2_prefill_forward(
        const float *, const float *, const float *, const float *, const float *,
        const float *, float *, float *, int, int, int, float);
void recurrent_norm_gate_llama_avx2_forward(
        const float *, const float *, const float *, float *, int, int, int, float);
}

namespace {

static float fixture(size_t i, float scale, float phase) {
    const float x = static_cast<float>(i);
    return std::sin(x * 0.017f + phase) * scale
         + std::cos(x * 0.0031f - phase) * scale * 0.37f;
}

static bool exact(const char * label, const float * ck, const float * oracle, size_t count) {
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
    std::printf("  %-24s different=%zu/%zu first=%zu worst=%zu max_abs=%.9g [FAIL]\n",
            label, different, count, first, worst, max_abs);
    return false;
}

static bool llama_graph(
        const std::vector<float> & q,
        const std::vector<float> & k,
        const std::vector<float> & v,
        const std::vector<float> & g,
        const std::vector<float> & beta_raw,
        const std::vector<float> & state_ck,
        std::vector<float> & output,
        std::vector<float> & state_ck_out,
        int rows, int heads, int dim) {
    const size_t arena = 64u * 1024u * 1024u;
    ggml_init_params params = {arena, nullptr, false};
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor * tq = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, heads, rows, 1);
    ggml_tensor * tk = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, heads, rows, 1);
    ggml_tensor * tv = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, heads, rows, 1);
    ggml_tensor * tg = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, heads, rows, 1);
    ggml_tensor * tb = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, heads, rows, 1);
    ggml_tensor * ts = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, dim, heads, 1);
    std::memcpy(ggml_get_data(tq), q.data(), q.size() * sizeof(float));
    std::memcpy(ggml_get_data(tk), k.data(), k.size() * sizeof(float));
    std::memcpy(ggml_get_data(tv), v.data(), v.size() * sizeof(float));
    std::memcpy(ggml_get_data(tg), g.data(), g.size() * sizeof(float));

    std::vector<float> beta(beta_raw.size());
    for (size_t i = 0; i < beta.size(); ++i) {
        beta[i] = 1.0f / (1.0f + expf(-beta_raw[i]));
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

static bool run_case(int rows) {
    constexpr int heads = 16;
    constexpr int dim = 128;
    const size_t vectors = static_cast<size_t>(rows) * heads * dim;
    const size_t gates = static_cast<size_t>(rows) * heads;
    const size_t states = static_cast<size_t>(heads) * dim * dim;
    std::vector<float> q(vectors), k(vectors), v(vectors), g(gates), beta(gates), state(states);
    for (size_t i = 0; i < vectors; ++i) {
        q[i] = fixture(i, 0.09f, 0.13f);
        k[i] = fixture(i, 0.08f, 0.29f);
        v[i] = fixture(i, 0.21f, 0.47f);
    }
    for (size_t i = 0; i < gates; ++i) {
        g[i] = -0.03f - std::fabs(fixture(i, 0.08f, 0.61f));
        beta[i] = fixture(i, 0.7f, 0.83f);
    }
    for (size_t i = 0; i < states; ++i) state[i] = fixture(i, 0.04f, 1.07f);

    std::vector<float> ck_out(vectors), llama_out(vectors);
    std::vector<float> ck_state(states), llama_state(states);
    if (!llama_graph(q, k, v, g, beta, state, llama_out, llama_state, rows, heads, dim)) {
        std::printf("llama.cpp graph execution failed\n");
        return false;
    }
    if (rows == 1) {
        gated_deltanet_llama_avx2_forward(q.data(), k.data(), v.data(), g.data(), beta.data(),
                state.data(), ck_state.data(), ck_out.data(), heads, dim, 1e-6f);
    } else {
        gated_deltanet_llama_avx2_prefill_forward(q.data(), k.data(), v.data(), g.data(), beta.data(),
                state.data(), ck_state.data(), ck_out.data(), rows, heads, dim, 1e-6f);
    }
    std::printf("rows=%d heads=%d dim=%d threads=%s\n", rows, heads, dim,
            std::getenv("CK_NUM_THREADS") ? std::getenv("CK_NUM_THREADS") : "1");
    return exact("attention output", ck_out.data(), llama_out.data(), vectors)
        && exact("recurrent state", ck_state.data(), llama_state.data(), states);
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
    return exact("gated normalization", ck.data(), oracle.data(), count);
}

} // namespace

int main() {
    const bool decode = run_case(1);
    const bool prefill = run_case(18);
    const bool norm_decode = run_norm_gate_case(1);
    const bool norm_prefill = run_norm_gate_case(18);
    return decode && prefill && norm_decode && norm_prefill ? 0 : 1;
}
