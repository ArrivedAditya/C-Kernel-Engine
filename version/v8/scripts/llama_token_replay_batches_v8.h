#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace cke_oracle {

inline bool read_bounded_token_file(const std::string & path, size_t limit,
                                    std::string & text, std::string & error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "cannot open --tokens-file";
        return false;
    }
    text.clear();
    char block[8192];
    while (input) {
        input.read(block, sizeof(block));
        const auto count = input.gcount();
        if (count <= 0) break;
        if (static_cast<size_t>(count) > limit - text.size()) {
            error = "oversized --tokens-file";
            return false;
        }
        text.append(block, static_cast<size_t>(count));
    }
    if (input.bad()) {
        error = "invalid --tokens-file";
        return false;
    }
    return true;
}

struct TokenRow {
    int32_t token;
    int32_t position;
    int32_t sequence_id;
    bool logits;
};

// Build the same rows submitted to llama_decode, independent of libllama.
// A nonzero submit result stops later chunks without advancing their positions.
template <typename Submit>
int32_t submit_token_batches(const std::vector<int32_t> & tokens,
                             int32_t pos0, size_t max_rows,
                             bool logits_each_chunk, Submit submit) {
    if (max_rows == 0 || pos0 < 0 ||
        tokens.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max() - pos0))
        return -1;
    for (size_t start = 0; start < tokens.size(); start += max_rows) {
        const size_t count = std::min(max_rows, tokens.size() - start);
        std::vector<TokenRow> rows;
        rows.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            const size_t absolute = start + i;
            rows.push_back({tokens[absolute], pos0 + static_cast<int32_t>(absolute), 0,
                            (logits_each_chunk && i + 1 == count) ||
                            absolute + 1 == tokens.size()});
        }
        const int32_t status = submit(rows);
        if (status != 0) return status;
    }
    return 0;
}

}  // namespace cke_oracle
