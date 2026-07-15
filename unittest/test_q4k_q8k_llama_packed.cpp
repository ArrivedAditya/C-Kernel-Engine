// Authoritative Q4_K x Q8_K parity against the production llama.cpp CPU graph.
//
// This deliberately tests the v8 public dispatcher, not only a leaf dot
// product. The dispatcher may repack output rows and reuse activation rows;
// those transformations and their accumulation order are part of the
// production numerical contract.

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
void quantize_row_q8_k(const float * x, void * y, int k);
void gemv_q4_k_q8_k(float * y, const void * w, const void * x_q8, int n, int k);
void gemm_nt_q4_k_q8_k_parallel_dispatch(
        const void * a_q8, const void * w, const float * bias, float * out,
        int m, int n, int k);
void ck_threadpool_global_destroy(void);
void ggml_vec_dot_q4_K_q8_K(
        int n, float * s, size_t bs,
        const void * vx, size_t bx,
        const void * vy, size_t by, int nrc);
}

namespace {

struct case_spec {
    const char * name;
    int m;
    int n;
    int k;
    bool production_dispatch;
    bool with_bias;
};

static float fixture_value(int row, int col, float scale, float phase) {
    const float r = static_cast<float>(row);
    const float c = static_cast<float>(col);
    float value = std::sin(c * 0.017f + r * 0.071f + phase) * scale;
    value += std::cos(c * 0.0031f - r * 0.013f + phase * 0.5f) * (scale * 0.37f);
    return value;
}

static void fill_activations(std::vector<float> & x, int m, int k) {
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < k; ++col) {
            x[static_cast<size_t>(row) * k + col] = fixture_value(row, col, 0.31f, 0.19f);
        }
        // Exercise signed-max selection and values close to quantization bins.
        for (int block = 0; block < k / QK_K; ++block) {
            const int col = block * QK_K + ((row * 17 + block * 29) % QK_K);
            x[static_cast<size_t>(row) * k + col] =
                    ((row + block) & 1 ? -1.0f : 1.0f) * (0.91f + 0.001f * block);
        }
    }
}

static void fill_weights(std::vector<float> & w, int n, int k) {
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < k; ++col) {
            w[static_cast<size_t>(row) * k + col] = fixture_value(row, col, 0.13f, 0.47f);
        }
    }
}

static bool compare_bytes(
        const char * label, const uint8_t * ck, const uint8_t * llama, size_t size) {
    size_t different = 0;
    size_t first = size;
    for (size_t i = 0; i < size; ++i) {
        if (ck[i] != llama[i]) {
            if (first == size) {
                first = i;
            }
            ++different;
        }
    }
    if (different == 0) {
        std::printf("  %-32s byte_exact (%zu bytes) [PASS]\n", label, size);
        return true;
    }
    std::printf("  %-32s different=%zu/%zu first_byte=%zu ck=%u llama=%u [FAIL]\n",
            label, different, size, first,
            static_cast<unsigned>(ck[first]), static_cast<unsigned>(llama[first]));
    return false;
}

static bool compare_f32(
        const char * label, const float * ck, const float * llama, int m, int n) {
    size_t different = 0;
    size_t first = static_cast<size_t>(m) * n;
    size_t worst = 0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < static_cast<size_t>(m) * n; ++i) {
        if (std::memcmp(&ck[i], &llama[i], sizeof(float)) != 0) {
            if (first == static_cast<size_t>(m) * n) {
                first = i;
            }
            ++different;
        }
        const float diff = std::fabs(ck[i] - llama[i]);
        if (diff > max_abs) {
            max_abs = diff;
            worst = i;
        }
    }
    if (different == 0) {
        std::printf("  %-32s bit_exact (%zu values) [PASS]\n",
                label, static_cast<size_t>(m) * n);
        return true;
    }
    std::printf(
            "  %-32s different=%zu/%zu first=(%zu,%zu) worst=(%zu,%zu) "
            "max_abs=%.9g ck=%.9g llama=%.9g [FAIL]\n",
            label, different, static_cast<size_t>(m) * n,
            first / n, first % n, worst / n, worst % n,
            max_abs, ck[worst], llama[worst]);
    return false;
}

static bool llama_mul_mat(
        const std::vector<uint8_t> & weights,
        const std::vector<float> & activations,
        std::vector<float> & output,
        int m, int n, int k) {
    const size_t arena_size = 64u * 1024u * 1024u
            + weights.size() + activations.size() * sizeof(float)
            + output.size() * sizeof(float);
    ggml_init_params params = {arena_size, nullptr, false};
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "ggml_init failed for %zu-byte arena\n", arena_size);
        return false;
    }

    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, k, n);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
    bool ok = w && x && ggml_nbytes(w) == weights.size();
    if (ok) {
        std::memcpy(ggml_get_data(w), weights.data(), weights.size());
        std::memcpy(ggml_get_data(x), activations.data(), activations.size() * sizeof(float));
        ggml_tensor * y = ggml_mul_mat(ctx, w, x);
        ggml_cgraph * graph = y ? ggml_new_graph(ctx) : nullptr;
        if (!graph) {
            ok = false;
        } else {
            ggml_build_forward_expand(graph, y);
            const int threads = std::max(1, std::atoi(std::getenv("CK_NUM_THREADS")
                    ? std::getenv("CK_NUM_THREADS") : "4"));
            ok = ggml_graph_compute_with_ctx(ctx, graph, threads) == GGML_STATUS_SUCCESS;
            if (ok) {
                std::memcpy(output.data(), ggml_get_data_f32(y), output.size() * sizeof(float));
            }
        }
    }
    ggml_free(ctx);
    return ok;
}

static bool run_case(const case_spec & spec) {
    std::printf("\n%s: M=%d N=%d K=%d provider=%s bias=%s\n",
            spec.name, spec.m, spec.n, spec.k,
            spec.production_dispatch ? "v8_dispatch" : "decode_leaf",
            spec.with_bias ? "yes" : "no");

    std::vector<float> activations(static_cast<size_t>(spec.m) * spec.k);
    std::vector<float> weights_f32(static_cast<size_t>(spec.n) * spec.k);
    fill_activations(activations, spec.m, spec.k);
    fill_weights(weights_f32, spec.n, spec.k);

    const size_t q8_row_bytes = static_cast<size_t>(spec.k / QK_K) * sizeof(block_q8_K);
    const size_t q4_row_bytes = static_cast<size_t>(spec.k / QK_K) * sizeof(block_q4_K);
    std::vector<uint8_t> ck_q8(static_cast<size_t>(spec.m) * q8_row_bytes);
    std::vector<uint8_t> llama_q8(ck_q8.size());
    std::vector<uint8_t> weights(static_cast<size_t>(spec.n) * q4_row_bytes);

    for (int row = 0; row < spec.m; ++row) {
        const float * src = activations.data() + static_cast<size_t>(row) * spec.k;
        quantize_row_q8_k(src, ck_q8.data() + static_cast<size_t>(row) * q8_row_bytes, spec.k);
        quantize_row_q8_K_ref(src,
                reinterpret_cast<block_q8_K *>(llama_q8.data() + static_cast<size_t>(row) * q8_row_bytes),
                spec.k);
    }
    for (int row = 0; row < spec.n; ++row) {
        quantize_row_q4_K_ref(weights_f32.data() + static_cast<size_t>(row) * spec.k,
                reinterpret_cast<block_q4_K *>(weights.data() + static_cast<size_t>(row) * q4_row_bytes),
                spec.k);
    }

    bool passed = compare_bytes("Q8_K activation quantizer", ck_q8.data(), llama_q8.data(), ck_q8.size());
    std::vector<float> ck_output(static_cast<size_t>(spec.m) * spec.n);
    std::vector<float> llama_output(ck_output.size());
    std::vector<float> llama_leaf_output(ck_output.size());
    std::vector<float> bias(spec.with_bias ? spec.n : 0);
    for (int col = 0; col < spec.n && spec.with_bias; ++col) {
        bias[col] = fixture_value(0, col, 0.017f, 0.83f);
    }
    for (int row = 0; row < spec.m; ++row) {
        const void * a_row = llama_q8.data() + static_cast<size_t>(row) * q8_row_bytes;
        for (int col = 0; col < spec.n; ++col) {
            const void * w_row = weights.data() + static_cast<size_t>(col) * q4_row_bytes;
            ggml_vec_dot_q4_K_q8_K(
                    spec.k, &llama_leaf_output[static_cast<size_t>(row) * spec.n + col],
                    0, w_row, 0, a_row, 0, 1);
        }
    }
    if (spec.production_dispatch) {
        gemm_nt_q4_k_q8_k_parallel_dispatch(
                ck_q8.data(), weights.data(), spec.with_bias ? bias.data() : nullptr, ck_output.data(),
                spec.m, spec.n, spec.k);
    } else {
        gemv_q4_k_q8_k(ck_output.data(), weights.data(), ck_q8.data(), spec.n, spec.k);
    }
    if (!llama_mul_mat(weights, activations, llama_output, spec.m, spec.n, spec.k)) {
        std::fprintf(stderr, "llama.cpp graph execution failed for %s\n", spec.name);
        return false;
    }
    if (spec.with_bias) {
        for (int row = 0; row < spec.m; ++row) {
            for (int col = 0; col < spec.n; ++col) {
                const size_t index = static_cast<size_t>(row) * spec.n + col;
                llama_output[index] += bias[col];
                llama_leaf_output[index] += bias[col];
            }
        }
    }
    if (spec.production_dispatch) {
        compare_f32("llama leaf vs graph control",
                llama_leaf_output.data(), llama_output.data(), spec.m, spec.n);
        passed &= compare_f32("production graph output",
                ck_output.data(), llama_output.data(), spec.m, spec.n);
    } else {
        passed &= compare_f32("decode leaf primitive",
                ck_output.data(), llama_leaf_output.data(), spec.m, spec.n);
        passed &= compare_f32("decode production graph",
                ck_output.data(), llama_output.data(), spec.m, spec.n);
    }
    return passed;
}

} // namespace

int main(int argc, char ** argv) {
    ggml_cpu_init();
    if (!std::getenv("CK_NUM_THREADS")) {
        setenv("CK_NUM_THREADS", "4", 1);
    }

    const bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
    const case_spec quick_cases[] = {
        {"decode_leaf", 1, 64, 256, false, false},
        {"packed_prefill_threshold", 16, 512, 1024, true, true},
        {"packed_prefill_multirow", 33, 1024, 1024, true, false},
    };
    const case_spec full_cases[] = {
        {"decode_leaf", 1, 64, 256, false, false},
        {"packed_prefill_threshold", 16, 512, 1024, true, true},
        {"packed_prefill_multirow", 33, 1024, 1024, true, false},
        {"qwen3vl_qkv_width", 16, 4096, 4096, true, false},
        {"qwen3vl_down_k_width", 16, 512, 11008, true, false},
    };
    const case_spec * cases = quick ? quick_cases : full_cases;
    const size_t case_count = quick
            ? sizeof(quick_cases) / sizeof(quick_cases[0])
            : sizeof(full_cases) / sizeof(full_cases[0]);

    int failed = 0;
    for (size_t i = 0; i < case_count; ++i) {
        if (!run_case(cases[i])) {
            ++failed;
        }
    }
    ck_threadpool_global_destroy();
    std::printf("\nQ4_K x Q8_K llama.cpp production parity: %s (%zu cases, %d failed)\n",
            failed == 0 ? "PASS" : "FAIL", case_count, failed);
    return failed == 0 ? 0 : 1;
}
