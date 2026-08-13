// Authoritative SSM convolution parity against the llama.cpp CPU graph.

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
void ssm_conv1d_forward_llama_production(
        const float * conv_x, const float * kernel, float * out,
        int kernel_size, int num_channels, int num_tokens, int num_seqs);
void ssm_conv1d_forward_llama_production_serial(
        const float * conv_x, const float * kernel, float * out,
        int kernel_size, int num_channels, int num_tokens, int num_seqs);
void ssm_conv1d_forward_llama_fma(
        const float * conv_x, const float * kernel, float * out,
        int kernel_size, int num_channels, int num_tokens, int num_seqs);
}

namespace {

struct case_spec {
    const char * name;
    int kernel_size;
    int channels;
    int tokens;
    int sequences;
    bool rounding_sensitive;
};

static float float_from_bits(uint32_t bits) {
    float value;
    static_assert(sizeof(value) == sizeof(bits), "float must be binary32");
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static float separated_mul_add(
        const float * input, const float * kernel, int count) {
    float sum = 0.0f;
    for (int i = 0; i < count; ++i) {
        volatile float product = input[i] * kernel[i];
        volatile float next = sum + product;
        sum = next;
    }
    return sum;
}

static float fused_mul_add(
        const float * input, const float * kernel, int count) {
    float sum = 0.0f;
    for (int i = 0; i < count; ++i) {
        sum = std::fma(input[i], kernel[i], sum);
    }
    return sum;
}

static float input_value(int sequence, int channel, int position) {
    float value = 0.31f * std::sin(
            0.017f * static_cast<float>(channel)
            + 0.071f * static_cast<float>(position)
            + 0.11f * static_cast<float>(sequence));
    if ((channel + position) % 127 == 0) {
        value += ((channel + position) & 1) ? -0.9375f : 0.9375f;
    }
    return value;
}

static float kernel_value(int channel, int tap) {
    return 0.23f * std::cos(
            0.013f * static_cast<float>(channel)
            - 0.19f * static_cast<float>(tap));
}

static bool llama_ssm_conv(
        const std::vector<float> & input,
        const std::vector<float> & kernel,
        std::vector<float> & output,
        const case_spec & spec) {
    const size_t arena_size = 16u * 1024u * 1024u
            + (input.size() + kernel.size() + output.size()) * sizeof(float);
    ggml_init_params params = {arena_size, nullptr, false};
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }

    const int sequence_width = spec.kernel_size - 1 + spec.tokens;
    ggml_tensor * x = ggml_new_tensor_3d(
            ctx, GGML_TYPE_F32, sequence_width, spec.channels, spec.sequences);
    ggml_tensor * w = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, spec.kernel_size, spec.channels);
    std::memcpy(ggml_get_data(x), input.data(), input.size() * sizeof(float));
    std::memcpy(ggml_get_data(w), kernel.data(), kernel.size() * sizeof(float));

    ggml_tensor * result = ggml_ssm_conv(ctx, x, w);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    const int threads = std::max(1, std::atoi(
            std::getenv("CK_NUM_THREADS") ? std::getenv("CK_NUM_THREADS") : "1"));
    const bool ok =
            ggml_graph_compute_with_ctx(ctx, graph, threads) == GGML_STATUS_SUCCESS;
    if (ok) {
        std::memcpy(
                output.data(), ggml_get_data_f32(result),
                output.size() * sizeof(float));
    }
    ggml_free(ctx);
    return ok;
}

static bool run_case(const case_spec & spec) {
    const int sequence_width = spec.kernel_size - 1 + spec.tokens;
    const size_t input_count =
            static_cast<size_t>(spec.sequences) * spec.channels * sequence_width;
    const size_t kernel_count =
            static_cast<size_t>(spec.channels) * spec.kernel_size;
    const size_t output_count =
            static_cast<size_t>(spec.sequences) * spec.tokens * spec.channels;
    std::vector<float> input(input_count);
    std::vector<float> kernel(kernel_count);
    std::vector<float> separated(output_count, 0.0f);
    std::vector<float> serial(output_count, 0.0f);
    std::vector<float> fused(output_count, 0.0f);
    std::vector<float> llama(output_count, 0.0f);

    if (spec.rounding_sensitive) {
        // This fixture differs by one ULP between a rounded multiply followed
        // by add and a contracted FMA chain.  Smooth trigonometric fixtures do
        // not reliably distinguish those two production contracts.
        const uint32_t input_bits[] = {
            0x3cac8de8u, 0x3d2fedc9u, 0x3d8458d7u, 0x3db01010u,
        };
        const uint32_t kernel_bits[] = {
            0x3e6b339fu, 0x3e6947ecu, 0x3e5ef6ceu, 0x3e4c9f57u,
        };
        for (int channel = 0; channel < spec.channels; ++channel) {
            for (int i = 0; i < spec.kernel_size; ++i) {
                input[static_cast<size_t>(channel) * sequence_width + i] =
                        float_from_bits(input_bits[i]);
                kernel[static_cast<size_t>(channel) * spec.kernel_size + i] =
                        float_from_bits(kernel_bits[i]);
            }
        }
        const float separated = separated_mul_add(
                input.data(), kernel.data(), spec.kernel_size);
        const float fused = fused_mul_add(
                input.data(), kernel.data(), spec.kernel_size);
        if (std::memcmp(&separated, &fused, sizeof(float)) == 0) {
            std::fprintf(
                    stderr, "%s: fixture does not distinguish arithmetic contracts\n",
                    spec.name);
            return false;
        }
    } else {
        for (int sequence = 0; sequence < spec.sequences; ++sequence) {
            for (int channel = 0; channel < spec.channels; ++channel) {
                for (int position = 0; position < sequence_width; ++position) {
                    const size_t index =
                            (static_cast<size_t>(sequence) * spec.channels + channel)
                            * sequence_width + position;
                    input[index] = input_value(sequence, channel, position);
                }
            }
        }
        for (int channel = 0; channel < spec.channels; ++channel) {
            for (int tap = 0; tap < spec.kernel_size; ++tap) {
                kernel[static_cast<size_t>(channel) * spec.kernel_size + tap] =
                        kernel_value(channel, tap);
            }
        }
    }

    ssm_conv1d_forward_llama_production(
            input.data(), kernel.data(), separated.data(),
            spec.kernel_size, spec.channels, spec.tokens, spec.sequences);
    ssm_conv1d_forward_llama_production_serial(
            input.data(), kernel.data(), serial.data(),
            spec.kernel_size, spec.channels, spec.tokens, spec.sequences);
    if (std::memcmp(
                separated.data(), serial.data(), output_count * sizeof(float)) != 0) {
        std::fprintf(
                stderr, "%s: parallel output differs from serial reference\n",
                spec.name);
        return false;
    }
    ssm_conv1d_forward_llama_fma(
            input.data(), kernel.data(), fused.data(),
            spec.kernel_size, spec.channels, spec.tokens, spec.sequences);
    if (!llama_ssm_conv(input, kernel, llama, spec)) {
        std::fprintf(stderr, "%s: llama.cpp graph execution failed\n", spec.name);
        return false;
    }

    size_t separated_different = 0;
    size_t fused_different = 0;
    for (size_t i = 0; i < output_count; ++i) {
        if (std::memcmp(&separated[i], &llama[i], sizeof(float)) != 0) {
            ++separated_different;
        }
        if (std::memcmp(&fused[i], &llama[i], sizeof(float)) != 0) {
            ++fused_different;
        }
    }
    const bool separated_exact = separated_different == 0;
    const bool fused_exact = fused_different == 0;
    if (spec.rounding_sensitive && separated_exact == fused_exact) {
        std::printf(
                "%-24s separated_diff=%zu fused_diff=%zu/%zu "
                "oracle=ambiguous [FAIL]\n",
                spec.name, separated_different, fused_different, output_count);
        return false;
    }
    if (!separated_exact && !fused_exact) {
        std::printf(
                "%-24s separated_diff=%zu fused_diff=%zu/%zu "
                "oracle=unclassified [FAIL]\n",
                spec.name, separated_different, fused_different, output_count);
        return false;
    }
    const char * oracle_arithmetic = fused_exact
            ? "fused_fma" : "separated_mul_add";
    std::printf(
            "%-24s bit_exact (%zu values) oracle=%s [PASS]\n",
            spec.name, output_count, oracle_arithmetic);
    return true;
}

} // namespace

int main() {
    const case_spec cases[] = {
        {"rounding_sensitive", 4, 48, 1, 1, true},
        {"decode_small", 4, 48, 1, 1, false},
        {"prefill_small", 4, 96, 13, 3, false},
        {"qwen36_decode", 4, 10240, 1, 1, false},
        {"qwen36_prefill", 4, 10240, 65, 1, false},
    };
    int passed = 0;
    for (const case_spec & spec : cases) {
        passed += run_case(spec) ? 1 : 0;
    }
    std::printf(
            "SSM conv llama production: %d/%zu passed\n",
            passed, sizeof(cases) / sizeof(cases[0]));
    return passed == static_cast<int>(sizeof(cases) / sizeof(cases[0])) ? 0 : 1;
}
