#include "../../version/v8/tools/mtmd_clip_shim.cpp"
#include "ggml-cpu.h"

#include <cassert>
#include <fstream>

static bool assert_early_operand_capture(ggml_tensor *tensor, bool ask, void *opaque) {
    auto *state = static_cast<DumpState *>(opaque);
    if (ask && tensor->op == GGML_OP_MUL_MAT && state->conv_seen == state->conv_index &&
        unwrap_conv_view(tensor->src[0])) {
        assert(std::filesystem::exists(state->directory / "conv_1_conv_image.raw"));
        assert(std::filesystem::exists(state->directory / "conv_1_conv_im2col.raw"));
    }
    return ck_mtmd_dump_eval_callback(tensor, ask, opaque);
}

int main(int argc, char **argv) {
    assert(argc == 2);
    ggml_init_params params{32 * 1024 * 1024, nullptr, true};
    ggml_context *ctx = ggml_init(params);
    assert(ctx);
    ggml_tensor *images[2];
    ggml_tensor *weights[2];
    ggml_tensor *outputs[2];
    ggml_cgraph *graph = ggml_new_graph(ctx);
    for (int i = 0; i < 2; ++i) {
        images[i] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 5, 3, 2, 1);
        weights[i] = ggml_new_tensor_4d(ctx, i == 1 ? GGML_TYPE_F32 : GGML_TYPE_F16, 2, 2, 2, 3);
        outputs[i] = ggml_conv_2d(ctx, weights[i], images[i], 2, 1, 1, 0, 1, 1);
        ggml_build_forward_expand(graph, outputs[i]);
    }
    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend);
    std::vector<float> image_values[2];
    std::vector<ggml_fp16_t> weight_values_f16;
    std::vector<float> weight_values_f32;
    for (int i = 0; i < 2; ++i) {
        image_values[i].resize(5 * 3 * 2);
        for (size_t j = 0; j < image_values[i].size(); ++j) {
            image_values[i][j] = float(i * 100 + j) / 7.0f;
        }
    }
    weight_values_f16.resize(2 * 2 * 2 * 3);
    weight_values_f32.resize(weight_values_f16.size());
    for (size_t j = 0; j < weight_values_f16.size(); ++j) {
        weight_values_f16[j] = ggml_fp32_to_fp16(float(j) / 13.0f);
        weight_values_f32[j] = float(10 + j) / 13.0f;
    }

    DumpState state;
    state.directory = argv[1];
    state.file = std::fopen((state.directory / "dump.bin").c_str(), "wb");
    state.meta_file = std::fopen((state.directory / "meta.jsonl").c_str(), "wb");
    assert(state.file && state.meta_file);
    state.capture_conv_operands = true;
    state.conv_index = 1;
    ggml_backend_t backends[] = {backend};
    ggml_backend_buffer_type_t bufts[] = {ggml_backend_get_default_buffer_type(backend)};
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 1, 128, false, true);
    assert(sched);
    assert(ggml_backend_sched_alloc_graph(sched, graph));
    for (int i = 0; i < 2; ++i) {
        ggml_backend_tensor_set(images[i], image_values[i].data(), 0,
                                image_values[i].size() * sizeof(float));
        if (i == 0) {
            ggml_backend_tensor_set(weights[i], weight_values_f16.data(), 0,
                                    weight_values_f16.size() * sizeof(ggml_fp16_t));
        } else {
            ggml_backend_tensor_set(weights[i], weight_values_f32.data(), 0,
                                    weight_values_f32.size() * sizeof(float));
        }
    }
    std::vector<float> baseline;
    for (int pass = 0; pass < 2; ++pass) {
        if (pass == 1) {
            ggml_backend_sched_set_eval_callback(sched, assert_early_operand_capture, &state);
        }
        assert(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
        std::vector<float> result(ggml_nelements(outputs[1]));
        ggml_backend_tensor_get(outputs[1], result.data(), 0, result.size() * sizeof(float));
        if (pass == 0) {
            baseline = result;
        } else {
            assert(result == baseline);
        }
    }
    assert(state.conv_seen == 2);
    dump_state_close(state);
    ggml_backend_sched_free(sched);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 0;
}
