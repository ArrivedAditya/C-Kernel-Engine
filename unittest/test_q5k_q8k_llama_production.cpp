// Authoritative Q5_K x Q8_K parity against llama.cpp's production CPU graph.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpu/repack.h"
#include "ggml-quants.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
void quantize_row_q8_k(const float *, void *, int);
void gemv_q5_k(float *, const void *, const float *, int, int);
void gemm_nt_q5_k(const float *, const void *, const float *, float *, int, int, int);
void ggml_vec_dot_q5_K_q8_K(
        int, float *, size_t, const void *, size_t, const void *, size_t, int);
}

namespace {

struct case_spec { const char * name; int m; int n; int k; };

static float fixture(int row, int col, float scale, float phase) {
    const float r = static_cast<float>(row);
    const float c = static_cast<float>(col);
    float value = std::sin(c * 0.017f + r * 0.071f + phase) * scale;
    value += std::cos(c * 0.0031f - r * 0.013f + phase * 0.5f) * scale * 0.37f;
    return value;
}

static bool compare_bytes(const char * label, const void * ck, const void * llama, size_t size) {
    if (std::memcmp(ck, llama, size) == 0) {
        std::printf("  %-34s byte_exact (%zu bytes) [PASS]\n", label, size);
        return true;
    }
    const auto * a = static_cast<const unsigned char *>(ck);
    const auto * b = static_cast<const unsigned char *>(llama);
    size_t first = size;
    size_t different = 0;
    for (size_t i = 0; i < size; ++i) {
        if (a[i] != b[i]) {
            if (first == size) first = i;
            ++different;
        }
    }
    std::printf("  %-34s different=%zu/%zu first=%zu ck=%u llama=%u [FAIL]\n",
            label, different, size, first, a[first], b[first]);
    return false;
}

static bool compare_f32(const char * label, const float * ck, const float * llama, size_t count) {
    size_t first = count;
    size_t worst = 0;
    size_t different = 0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        if (std::memcmp(ck + i, llama + i, sizeof(float)) != 0) {
            if (first == count) first = i;
            ++different;
        }
        const float diff = std::fabs(ck[i] - llama[i]);
        if (diff > max_abs) {
            max_abs = diff;
            worst = i;
        }
    }
    if (different == 0) {
        std::printf("  %-34s bit_exact (%zu values) [PASS]\n", label, count);
        return true;
    }
    std::printf("  %-34s different=%zu/%zu first=%zu worst=%zu max_abs=%.9g "
                "ck=%.9g llama=%.9g [FAIL]\n",
            label, different, count, first, worst, max_abs, ck[worst], llama[worst]);
    return false;
}

static bool llama_graph(
        const std::vector<unsigned char> & weights,
        const std::vector<float> & activations,
        std::vector<float> & output,
        int m, int n, int k, bool repack) {
    ggml_context * weight_ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * w = nullptr;
    if (repack) {
        ggml_init_params params = {ggml_tensor_overhead() * 2, nullptr, true};
        weight_ctx = ggml_init(params);
        if (!weight_ctx) return false;
        w = ggml_new_tensor_2d(weight_ctx, GGML_TYPE_Q5_K, k, n);
        ggml_backend_buffer_type_t buft = ggml_backend_cpu_repack_buffer_type();
        buffer = ggml_backend_buft_alloc_buffer(
                buft, ggml_backend_buft_get_alloc_size(buft, w));
        if (!buffer || ggml_backend_tensor_alloc(
                buffer, w, ggml_backend_buffer_get_base(buffer)) != GGML_STATUS_SUCCESS) {
            if (buffer) ggml_backend_buffer_free(buffer);
            ggml_free(weight_ctx);
            return false;
        }
        ggml_backend_tensor_set(w, weights.data(), 0, weights.size());
    }

    const size_t arena = 128u * 1024u * 1024u + weights.size()
            + activations.size() * sizeof(float) + output.size() * sizeof(float);
    ggml_init_params params = {arena, nullptr, false};
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;
    if (!repack) {
        w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q5_K, k, n);
        std::memcpy(ggml_get_data(w), weights.data(), weights.size());
    }
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
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
    if (buffer) ggml_backend_buffer_free(buffer);
    if (weight_ctx) ggml_free(weight_ctx);
    return ok;
}

static bool run_case(const case_spec & spec) {
    std::printf("\n%s M=%d N=%d K=%d threads=%s\n", spec.name, spec.m, spec.n, spec.k,
            std::getenv("CK_NUM_THREADS") ? std::getenv("CK_NUM_THREADS") : "1");
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

    const size_t q8_row = static_cast<size_t>(spec.k / QK_K) * sizeof(block_q8_K);
    const size_t q5_row = static_cast<size_t>(spec.k / QK_K) * sizeof(block_q5_K);
    std::vector<unsigned char> ck_q8(static_cast<size_t>(spec.m) * q8_row);
    std::vector<unsigned char> llama_q8(ck_q8.size());
    std::vector<unsigned char> weights(static_cast<size_t>(spec.n) * q5_row);
    for (int row = 0; row < spec.m; ++row) {
        const float * input = activations.data() + static_cast<size_t>(row) * spec.k;
        quantize_row_q8_k(input, ck_q8.data() + static_cast<size_t>(row) * q8_row, spec.k);
        quantize_row_q8_K_ref(input, reinterpret_cast<block_q8_K *>(
                llama_q8.data() + static_cast<size_t>(row) * q8_row), spec.k);
    }
    for (int row = 0; row < spec.n; ++row) {
        quantize_row_q5_K_ref(weights_f32.data() + static_cast<size_t>(row) * spec.k,
                reinterpret_cast<block_q5_K *>(weights.data() + static_cast<size_t>(row) * q5_row), spec.k);
    }

    bool pass = compare_bytes(
            "Q8_K activation quantizer", ck_q8.data(), llama_q8.data(), ck_q8.size());
    std::vector<float> leaf(static_cast<size_t>(spec.m) * spec.n);
    std::vector<float> canonical(leaf.size());
    std::vector<float> production(leaf.size());
    std::vector<float> ck(leaf.size());
    for (int row = 0; row < spec.m; ++row) {
        for (int col = 0; col < spec.n; ++col) {
            ggml_vec_dot_q5_K_q8_K(spec.k, &leaf[static_cast<size_t>(row) * spec.n + col], 0,
                    weights.data() + static_cast<size_t>(col) * q5_row, 0,
                    llama_q8.data() + static_cast<size_t>(row) * q8_row, 0, 1);
        }
    }
    if (spec.m == 1) {
        gemv_q5_k(ck.data(), weights.data(), activations.data(), spec.n, spec.k);
    } else {
        gemm_nt_q5_k(activations.data(), weights.data(), nullptr,
                ck.data(), spec.m, spec.n, spec.k);
    }
    if (!llama_graph(weights, activations, canonical, spec.m, spec.n, spec.k, false)) return false;
    const bool llama_q5_repack_selected = ggml_cpu_has_neon();
    if (llama_q5_repack_selected) {
        if (!llama_graph(weights, activations, production, spec.m, spec.n, spec.k, true)) return false;
    } else {
        production = canonical;
        std::printf("  %-34s x86 Q5 uses canonical graph [SKIP]\n", "llama Q5 repack graph");
    }
    pass &= compare_f32("llama leaf vs canonical graph", leaf.data(), canonical.data(), leaf.size());
    if (llama_q5_repack_selected) {
        pass &= compare_f32("llama canonical vs repack graph", canonical.data(), production.data(), leaf.size());
    }
    pass &= compare_f32("CK adapter vs llama production", ck.data(), production.data(), leaf.size());
    return pass;
}

} // namespace

int main(int argc, char ** argv) {
    ggml_cpu_init();
    const bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
    const case_spec quick_cases[] = {
        {"decode_leaf", 1, 64, 256},
        {"qwen35_recurrent_decode", 1, 6144, 1024},
        {"short_prefill", 7, 512, 1024},
    };
    const case_spec full_cases[] = {
        {"decode_leaf", 1, 64, 256},
        {"qwen35_recurrent_decode", 1, 6144, 1024},
        {"qwen35_recurrent_prefill", 33, 6144, 1024},
    };
    const case_spec * cases = quick ? quick_cases : full_cases;
    const size_t count = quick ? sizeof(quick_cases) / sizeof(quick_cases[0])
                               : sizeof(full_cases) / sizeof(full_cases[0]);
    int failed = 0;
    for (size_t i = 0; i < count; ++i) {
        if (!run_case(cases[i])) ++failed;
    }
    std::printf("\nQ5_K x Q8_K llama.cpp production parity: %s (%zu cases, %d failed)\n",
            failed ? "FAIL" : "PASS", count, failed);
    return failed ? 1 : 0;
}
