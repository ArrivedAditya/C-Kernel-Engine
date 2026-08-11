// Authoritative Q5_1 x Q8_1 parity against llama.cpp's production CPU graph.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" void gemm_nt_q5_1_q8_1_parallel_dispatch(
        const float *, const void *, const float *, float *, int, int, int);

namespace {

struct case_spec { const char * name; int m; int n; int k; };

static float fixture(int row, int col, float scale, float phase) {
    const float r = static_cast<float>(row);
    const float c = static_cast<float>(col);
    return std::sin(c * 0.017f + r * 0.071f + phase) * scale
         + std::cos(c * 0.0031f - r * 0.013f + phase * 0.5f) * scale * 0.37f;
}

static bool llama_graph(
        const std::vector<unsigned char> & weights,
        const std::vector<float> & activations,
        std::vector<float> & output,
        int m, int n, int k) {
    const size_t arena = 64u * 1024u * 1024u + weights.size()
            + activations.size() * sizeof(float) + output.size() * sizeof(float);
    ggml_init_params params = {arena, nullptr, false};
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q5_1, k, n);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
    std::memcpy(ggml_get_data(w), weights.data(), weights.size());
    std::memcpy(ggml_get_data(x), activations.data(), activations.size() * sizeof(float));
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    const int threads = std::max(1, std::atoi(
            std::getenv("CK_NUM_THREADS") ? std::getenv("CK_NUM_THREADS") : "1"));
    const bool ok = ggml_graph_compute_with_ctx(ctx, graph, threads) == GGML_STATUS_SUCCESS;
    if (ok) {
        std::memcpy(output.data(), ggml_get_data_f32(y), output.size() * sizeof(float));
    }
    ggml_free(ctx);
    return ok;
}

static bool run_case(const case_spec & spec) {
    std::vector<float> activations(static_cast<size_t>(spec.m) * spec.k);
    std::vector<float> weights_f32(static_cast<size_t>(spec.n) * spec.k);
    for (int row = 0; row < spec.m; ++row) {
        for (int col = 0; col < spec.k; ++col) {
            activations[static_cast<size_t>(row) * spec.k + col] =
                    fixture(row, col, 0.31f, 0.19f);
        }
    }
    for (int row = 0; row < spec.n; ++row) {
        for (int col = 0; col < spec.k; ++col) {
            weights_f32[static_cast<size_t>(row) * spec.k + col] =
                    fixture(row, col, 0.13f, 0.47f);
        }
    }

    const size_t q5_row = static_cast<size_t>(spec.k / QK5_1) * sizeof(block_q5_1);
    std::vector<unsigned char> weights(static_cast<size_t>(spec.n) * q5_row);
    for (int row = 0; row < spec.n; ++row) {
        quantize_row_q5_1_ref(
                weights_f32.data() + static_cast<size_t>(row) * spec.k,
                reinterpret_cast<block_q5_1 *>(
                    weights.data() + static_cast<size_t>(row) * q5_row),
                spec.k);
    }

    std::vector<float> ck(static_cast<size_t>(spec.m) * spec.n);
    std::vector<float> llama(ck.size());
    gemm_nt_q5_1_q8_1_parallel_dispatch(
            activations.data(), weights.data(), nullptr, ck.data(),
            spec.m, spec.n, spec.k);
    if (!llama_graph(weights, activations, llama, spec.m, spec.n, spec.k)) {
        return false;
    }
    size_t first = ck.size();
    float max_abs = 0.0f;
    for (size_t i = 0; i < ck.size(); ++i) {
        if (first == ck.size() && std::memcmp(&ck[i], &llama[i], sizeof(float)) != 0) {
            first = i;
        }
        max_abs = std::max(max_abs, std::fabs(ck[i] - llama[i]));
    }
    const bool pass = max_abs <= 1.0e-4f;
    std::printf("%-20s M=%d N=%d K=%d first=%zu max_abs=%.9g [%s]\n",
            spec.name, spec.m, spec.n, spec.k, first, max_abs,
            pass ? "PASS" : "FAIL");
    return pass;
}

} // namespace

int main() {
    ggml_cpu_init();
    const case_spec cases[] = {
        {"single_row", 1, 64, 640},
        {"m4_provider", 4, 128, 640},
        {"row_tail", 7, 96, 1024},
    };
    int failed = 0;
    for (const auto & spec : cases) failed += !run_case(spec);
    std::printf("Q5_1 x Q8_1 llama.cpp production parity: %s\n",
            failed ? "FAIL" : "PASS");
    return failed ? 1 : 0;
}
