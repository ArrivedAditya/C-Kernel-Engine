"""Asset-free compiled checks for the rows submitted to llama.cpp."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER_DIR = ROOT / "version" / "v8" / "scripts"


def test_native_replay_chunk_rows_and_failure_propagation(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    assert compiler, "C++ compiler is required for the native oracle regression"
    source = tmp_path / "probe.cpp"
    source.write_text(
        r'''
#include "llama_token_replay_batches_v8.h"
#include <cassert>
#include <cstdint>
#include <limits>
#include <vector>

using cke_oracle::TokenRow;

int main(int argc, char ** argv) {
    assert(argc == 2);
    const std::string path = argv[1];
    {
        std::ofstream output(path, std::ios::binary);
        output << std::string(33, '1');
    }
    std::string contents;
    std::string error;
    assert(!cke_oracle::read_bounded_token_file(path, 32, contents, error));
    assert(error == "oversized --tokens-file");
    assert(contents.size() <= 32);
    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output << std::string(32, '2');
    }
    assert(cke_oracle::read_bounded_token_file(path, 32, contents, error));
    assert(contents == std::string(32, '2'));

    std::vector<int32_t> tokens;
    for (int i = 0; i < 10; ++i) tokens.push_back(100 + i);
    std::vector<size_t> sizes;
    size_t seen = 0;
    auto check = [&](const std::vector<TokenRow> & rows) -> int32_t {
        sizes.push_back(rows.size());
        for (const auto & row : rows) {
            assert(row.token == 100 + static_cast<int32_t>(seen));
            assert(row.position == 700 + static_cast<int32_t>(seen));
            assert(row.sequence_id == 0);
            assert(row.logits == (seen == tokens.size() - 1));
            ++seen;
        }
        return 0;
    };
    assert(cke_oracle::submit_token_batches(tokens, 700, 4, false, check) == 0);
    assert((sizes == std::vector<size_t>{4, 4, 2}));
    assert(seen == tokens.size());

    sizes.clear(); seen = 0;
    assert(cke_oracle::submit_token_batches(tokens, 700, 5, false, check) == 0);
    assert((sizes == std::vector<size_t>{5, 5}));

    int calls = 0;
    auto fail_second = [&](const std::vector<TokenRow> & rows) -> int32_t {
        ++calls;
        assert(rows.front().position == 700 + 4 * (calls - 1));
        return calls == 2 ? 37 : 0;
    };
    assert(cke_oracle::submit_token_batches(tokens, 700, 4, false, fail_second) == 37);
    assert(calls == 2);

    calls = 0;
    auto sequential = [&](const std::vector<TokenRow> & rows) -> int32_t {
        assert(rows.size() == 1 && rows[0].logits);
        assert(rows[0].position == 700 + calls);
        ++calls;
        return 0;
    };
    assert(cke_oracle::submit_token_batches(tokens, 700, 1, true, sequential) == 0);
    assert(calls == 10);

    assert(cke_oracle::submit_token_batches(tokens, 700, 0, false, check) == -1);
    assert(cke_oracle::submit_token_batches(tokens,
           std::numeric_limits<int32_t>::max() - 5, 4, false, check) == -1);
}
''',
        encoding="utf-8",
    )
    binary = tmp_path / "probe"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
         "-I", str(HEADER_DIR), str(source), "-o", str(binary)],
        check=True, capture_output=True, text=True,
    )
    subprocess.run([str(binary), str(tmp_path / "tokens.csv")],
                   check=True, capture_output=True, text=True)
