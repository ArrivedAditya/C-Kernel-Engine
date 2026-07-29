// Authoritative SSM convolution parity against the llama.cpp CPU graph.

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
void ssm_conv1d_forward_llama_production(
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
};

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
    std::vector<float> ck(output_count, 0.0f);
    std::vector<float> llama(output_count, 0.0f);

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

    ssm_conv1d_forward_llama_production(
            input.data(), kernel.data(), ck.data(),
            spec.kernel_size, spec.channels, spec.tokens, spec.sequences);
    if (!llama_ssm_conv(input, kernel, llama, spec)) {
        std::fprintf(stderr, "%s: llama.cpp graph execution failed\n", spec.name);
        return false;
    }

    size_t different = 0;
    size_t first = output_count;
    size_t worst = 0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < output_count; ++i) {
        if (std::memcmp(&ck[i], &llama[i], sizeof(float)) != 0) {
            if (first == output_count) {
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
    if (different != 0) {
        std::printf(
                "%-24s different=%zu/%zu first=%zu worst=%zu "
                "max_abs=%.9g ck=%.9g llama=%.9g [FAIL]\n",
                spec.name, different, output_count, first, worst,
                max_abs, ck[worst], llama[worst]);
        return false;
    }
    std::printf(
            "%-24s bit_exact (%zu values) [PASS]\n",
            spec.name, output_count);
    return true;
}

} // namespace

int main() {
    const case_spec cases[] = {
        {"decode_small", 4, 48, 1, 1},
        {"prefill_small", 4, 96, 13, 3},
        {"qwen36_decode", 4, 10240, 1, 1},
        {"qwen36_prefill", 4, 10240, 65, 1},
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
