#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "llama.h"
#include "mtmd.h"

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
    if (argc != 6) {
        std::cerr << "usage: llama_mtmd_prefix_probe_v8 MODEL MMPROJ IMAGE_P6 OUTPUT_F32 OUTPUT_RGB8\n";
        return 2;
    }
    try {
        std::ifstream image(argv[3], std::ios::binary);
        if (!image || ppm_token(image) != "P6") throw std::runtime_error("expected P6 image");
        const int width = std::stoi(ppm_token(image));
        const int height = std::stoi(ppm_token(image));
        const int maximum = std::stoi(ppm_token(image));
        if (width <= 0 || height <= 0 || width > 16384 || height > 16384 ||
            static_cast<int64_t>(width) * height > 16000000 || maximum != 255) {
            throw std::runtime_error("unsupported P6 geometry or channel range");
        }
        std::vector<uint8_t> rgb(static_cast<size_t>(width) * height * 3);
        image.read(reinterpret_cast<char *>(rgb.data()), rgb.size());
        if (static_cast<size_t>(image.gcount()) != rgb.size() || image.peek() != EOF) {
            throw std::runtime_error("incomplete or trailing P6 pixel data");
        }
        std::ofstream rgb_output(argv[5], std::ios::binary);
        rgb_output.write(reinterpret_cast<const char *>(rgb.data()), rgb.size());
        if (!rgb_output) throw std::runtime_error("failed to write decoded RGB8 pixels");

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
        std::unique_ptr<mtmd_context, decltype(&mtmd_free)> mtmd(
            mtmd_init_from_file(argv[2], model.get(), params), mtmd_free);
        if (!mtmd || !mtmd_support_vision(mtmd.get())) {
            throw std::runtime_error("failed to load vision encoder");
        }
        std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)> bitmap(
            mtmd_bitmap_init(width, height, rgb.data()), mtmd_bitmap_free);
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
        std::ofstream output(argv[4], std::ios::binary);
        output.write(reinterpret_cast<const char *>(prefix.data()),
                     prefix.size() * sizeof(float));
        if (!output) throw std::runtime_error("failed to write embeddings");
        std::cout << "tokens=" << prefix.size() / dim << " dim=" << dim
                  << " image_chunks=" << image_chunks << "\n";
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
