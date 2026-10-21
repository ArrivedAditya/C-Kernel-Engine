#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

struct CaptureStage {
    std::string name;
    std::string path;
    std::string error;
    bool seen = false;
    int64_t columns = 0;
    int64_t rows = 0;
};

static bool capture_stage(ggml_tensor * tensor, bool ask, void * opaque) {
    auto & capture = *static_cast<CaptureStage *>(opaque);
    if (capture.name != tensor->name) return false;
    if (ask) return true;
    if (capture.seen) {
        capture.error = "selected MTMD stage appeared more than once";
        return true;
    }
    const int64_t count = ggml_nelements(tensor);
    if (count <= 0 || count > 64 * 1024 * 1024 || !ggml_is_contiguous(tensor) ||
        (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16)) {
        capture.error = "selected MTMD stage has unsupported geometry, layout, or dtype";
        return true;
    }
    std::vector<uint8_t> storage(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, storage.data(), 0, storage.size());
    std::vector<float> values(static_cast<size_t>(count));
    if (tensor->type == GGML_TYPE_F32) {
        std::memcpy(values.data(), storage.data(), storage.size());
    } else {
        for (int64_t i = 0; i < count; ++i) {
            ggml_fp16_t half;
            std::memcpy(&half, storage.data() + static_cast<size_t>(i) * sizeof(half), sizeof(half));
            values[i] = ggml_fp16_to_fp32(half);
        }
    }
    std::ofstream output(capture.path, std::ios::binary);
    output.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float));
    if (!output) {
        capture.error = "failed to write MTMD stage capture";
        return true;
    }
    capture.columns = tensor->ne[0];
    capture.rows = count / tensor->ne[0];
    capture.seen = true;
    return true;
}

static std::string ppm_token(std::istream & stream) {
    std::string token;
    char ch;
    while (stream.get(ch)) {
        if (ch == '#') {
            stream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        } else if (!std::isspace(static_cast<unsigned char>(ch))) {
            token.push_back(ch);
            break;
        }
    }
    while (stream.get(ch) && !std::isspace(static_cast<unsigned char>(ch))) {
        token.push_back(ch);
    }
    if (token.empty()) throw std::runtime_error("incomplete PPM header");
    return token;
}

int main(int argc, char ** argv) {
    if (argc != 6 && argc != 9) {
        std::cerr << "usage: llama_mtmd_prefix_probe_v8 MODEL MMPROJ IMAGE OUTPUT_F32 OUTPUT_RGB8"
                     " [--capture-stage NAME OUTPUT_F32]\n";
        return 2;
    }
    try {
        CaptureStage capture;
        if (argc == 9) {
            if (std::strcmp(argv[6], "--capture-stage") != 0 || !*argv[7] || !*argv[8]) {
                throw std::runtime_error("invalid MTMD stage capture arguments");
            }
            capture.name = argv[7];
            capture.path = argv[8];
        }
        std::ifstream image(argv[3], std::ios::binary);
        if (!image) throw std::runtime_error("cannot open image");
        char magic[2] = {};
        image.read(magic, sizeof(magic));
        const bool is_p6 = image.gcount() == 2 && magic[0] == 'P' && magic[1] == '6';
        image.clear();
        image.seekg(0);
        int width = 0;
        int height = 0;
        std::vector<uint8_t> rgb;
        if (is_p6) {
            (void) ppm_token(image);
            width = std::stoi(ppm_token(image));
            height = std::stoi(ppm_token(image));
            const int maximum = std::stoi(ppm_token(image));
            if (width <= 0 || height <= 0 || width > 16384 || height > 16384 ||
                static_cast<int64_t>(width) * height > 16000000 || maximum != 255) {
                throw std::runtime_error("unsupported P6 geometry or channel range");
            }
            rgb.resize(static_cast<size_t>(width) * height * 3);
            image.read(reinterpret_cast<char *>(rgb.data()), rgb.size());
            if (static_cast<size_t>(image.gcount()) != rgb.size() || image.peek() != EOF) {
                throw std::runtime_error("incomplete or trailing P6 pixel data");
            }
        }

        llama_backend_init();
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], model_params), llama_model_free);
        if (!model) throw std::runtime_error("failed to load decoder model");
        mtmd_context_params params = mtmd_context_params_default();
        params.use_gpu = false;
        params.n_threads = 8;
        params.warmup = false;
        if (!capture.name.empty()) {
            params.cb_eval = capture_stage;
            params.cb_eval_user_data = &capture;
        }
        std::unique_ptr<mtmd_context, decltype(&mtmd_free)> mtmd(
            mtmd_init_from_file(argv[2], model.get(), params), mtmd_free);
        if (!mtmd || !mtmd_support_vision(mtmd.get())) {
            throw std::runtime_error("failed to load vision encoder");
        }
        std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)> bitmap(nullptr, mtmd_bitmap_free);
        if (is_p6) {
            bitmap.reset(mtmd_bitmap_init(width, height, rgb.data()));
        } else {
            const auto decoded = mtmd_helper_bitmap_init_from_file(
                mtmd.get(), argv[3], false, mtmd_helper_init_opt_default());
            if (decoded.video_ctx) mtmd_helper_video_free(decoded.video_ctx);
            bitmap.reset(decoded.bitmap);
            if (decoded.video_ctx || !bitmap || mtmd_bitmap_is_audio(bitmap.get())) {
                throw std::runtime_error("image decoder did not return one RGB8 image");
            }
            width = static_cast<int>(mtmd_bitmap_get_nx(bitmap.get()));
            height = static_cast<int>(mtmd_bitmap_get_ny(bitmap.get()));
            if (width <= 0 || height <= 0 || width > 16384 || height > 16384 ||
                static_cast<int64_t>(width) * height > 16000000 ||
                mtmd_bitmap_get_n_bytes(bitmap.get()) != static_cast<size_t>(width) * height * 3) {
                throw std::runtime_error("unsupported decoded image geometry");
            }
            const auto * data = mtmd_bitmap_get_data(bitmap.get());
            if (!data) throw std::runtime_error("image decoder returned no pixels");
            rgb.assign(data, data + static_cast<size_t>(width) * height * 3);
        }
        if (!bitmap) throw std::runtime_error("failed to prepare image bitmap");
        std::ofstream rgb_output(argv[5], std::ios::binary);
        rgb_output.write(reinterpret_cast<const char *>(rgb.data()), rgb.size());
        if (!rgb_output) throw std::runtime_error("failed to write decoded RGB8 pixels");
        std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)> chunks(
            mtmd_input_chunks_init(), mtmd_input_chunks_free);
        if (!bitmap || !chunks) throw std::runtime_error("failed to prepare image chunk");
        const mtmd_input_part part{nullptr, bitmap.get()};
        const mtmd_input_part * parts[] = {&part};
        if (mtmd_tokenize_from_parts(mtmd.get(), chunks.get(), parts, 1, false) != 0) {
            throw std::runtime_error("MTMD image preprocessing failed");
        }

        const int dim = llama_model_n_embd_inp(model.get());
        if (dim <= 0) throw std::runtime_error("invalid decoder input width");
        std::vector<float> prefix;
        size_t image_chunks = 0;
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks.get()); ++i) {
            const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks.get(), i);
            if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_IMAGE) continue;
            const size_t tokens = mtmd_input_chunk_get_n_tokens(chunk);
            if (tokens == 0 || tokens > 1000000 ||
                prefix.size() / static_cast<size_t>(dim) + tokens > 1000000) {
                throw std::runtime_error("invalid visual-token extent");
            }
            if (mtmd_encode_chunk(mtmd.get(), chunk) != 0) {
                throw std::runtime_error("MTMD image encoding failed");
            }
            const float * values = mtmd_get_output_embd(mtmd.get());
            if (!values) throw std::runtime_error("MTMD produced no embeddings");
            prefix.insert(prefix.end(), values, values + tokens * static_cast<size_t>(dim));
            ++image_chunks;
        }
        if (image_chunks == 0) throw std::runtime_error("MTMD produced no image chunks");
        if (!capture.name.empty()) {
            if (!capture.error.empty()) throw std::runtime_error(capture.error);
            if (!capture.seen) throw std::runtime_error("selected MTMD stage was not executed");
        }
        std::ofstream output(argv[4], std::ios::binary);
        output.write(reinterpret_cast<const char *>(prefix.data()),
                     prefix.size() * sizeof(float));
        if (!output) throw std::runtime_error("failed to write embeddings");
        std::cout << "tokens=" << prefix.size() / dim << " dim=" << dim
                  << " image_chunks=" << image_chunks;
        if (capture.seen) {
            std::cout << " capture_columns=" << capture.columns << " capture_rows=" << capture.rows;
        }
        std::cout << "\n";
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
