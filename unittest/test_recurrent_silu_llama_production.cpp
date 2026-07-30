// Bit-exact recurrent SiLU parity against the linked llama.cpp CPU provider.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" {
void recurrent_silu_forward_ggml(
        const float * x, float * out, int rows, int dim);
void ggml_vec_silu_f32(const int n, float * y, const float * x);
}

namespace {

struct case_spec {
    const char * name;
    int rows;
    int dim;
};

static bool run_case(const case_spec & spec) {
    const size_t count = static_cast<size_t>(spec.rows) * spec.dim;
    std::vector<float> input(count);
    std::vector<float> ck(count, 0.0f);
    std::vector<float> llama(count, 0.0f);
    for (size_t i = 0; i < count; ++i) {
        float value = 2.75f * std::sin(0.013f * static_cast<float>(i));
        if (i % 127 == 0) {
            value += (i & 1) ? -7.5f : 7.5f;
        }
        input[i] = value;
    }

    recurrent_silu_forward_ggml(
            input.data(), ck.data(), spec.rows, spec.dim);
    for (int row = 0; row < spec.rows; ++row) {
        ggml_vec_silu_f32(
                spec.dim,
                llama.data() + static_cast<size_t>(row) * spec.dim,
                input.data() + static_cast<size_t>(row) * spec.dim);
    }

    size_t different = 0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        different += std::memcmp(&ck[i], &llama[i], sizeof(float)) != 0;
        max_abs = std::fmax(max_abs, std::fabs(ck[i] - llama[i]));
    }
    std::printf(
            "%-24s different=%zu/%zu max_abs=%.9g [%s]\n",
            spec.name, different, count, max_abs,
            different == 0 ? "PASS" : "FAIL");
    return different == 0;
}

} // namespace

int main() {
    const case_spec cases[] = {
        {"scalar_tail", 3, 131},
        {"qwen36_decode", 1, 10240},
        {"qwen36_prefill_13", 13, 10240},
        {"qwen36_prefill_65", 65, 10240},
    };
    int passed = 0;
    for (const case_spec & spec : cases) {
        passed += run_case(spec) ? 1 : 0;
    }
    std::printf(
            "Recurrent SiLU llama production: %d/%zu passed\n",
            passed, sizeof(cases) / sizeof(cases[0]));
    return passed == static_cast<int>(sizeof(cases) / sizeof(cases[0])) ? 0 : 1;
}
