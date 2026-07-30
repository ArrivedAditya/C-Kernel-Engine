from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "compare_whisper_backends_v8.py"
SPEC = importlib.util.spec_from_file_location("whisper_benchmark", SCRIPT)
assert SPEC and SPEC.loader
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_backend_order_rotates_without_dropping_backends() -> None:
    backends = ["cke", "pytorch", "whisper_cpp"]
    assert benchmark.backend_order(0, backends) == backends
    assert benchmark.backend_order(1, backends) == [
        "pytorch", "whisper_cpp", "cke"
    ]
    assert benchmark.backend_order(2, backends) == [
        "whisper_cpp", "cke", "pytorch"
    ]


def test_backend_order_handles_empty_input() -> None:
    assert benchmark.backend_order(3, []) == []


def test_parse_whisper_cpp_timings_requires_total() -> None:
    text = """
whisper_print_timings:   encode time =   275.20 ms / 1 runs
whisper_print_timings:    total time =   539.12 ms
"""
    assert benchmark.parse_whisper_cpp_timings(text) == {
        "encode": 0.2752,
        "total": 0.53912,
    }


def test_parse_whisper_cpp_result_excludes_eot() -> None:
    payload = {
        "transcription": [
            {
                "text": " Hello.",
                "tokens": [
                    {"id": 2425, "text": " Hello"},
                    {"id": 13, "text": "."},
                    {"id": 50257, "text": "[_EOT_]"},
                ],
            }
        ]
    }
    assert benchmark.parse_whisper_cpp_result(payload) == {
        "text": " Hello.",
        "tokens": [2425, 13],
    }


def test_validate_results_fails_closed_on_token_or_text_change() -> None:
    base = {
        "backend": "cke",
        "repetition": 0,
        "tokens": [1, 2],
        "text": "ok",
    }
    same = {
        "backend": "pytorch",
        "repetition": 0,
        "tokens": [1, 2],
        "text": "ok",
    }
    changed = {
        "backend": "whisper_cpp",
        "repetition": 0,
        "tokens": [1, 3],
        "text": "ok",
    }
    assert benchmark.validate_results([base, same])["status"] == "pass"
    result = benchmark.validate_results([base, changed])
    assert result["status"] == "fail"
    assert result["mismatches"][0]["token_match"] is False


def test_summarize_uses_native_reference_and_median() -> None:
    runs = [
        {"backend": "cke", "wall_seconds": 11.0, "compute_seconds": 10.0},
        {"backend": "cke", "wall_seconds": 13.0, "compute_seconds": 12.0},
        {
            "backend": "whisper_cpp",
            "wall_seconds": 1.1,
            "compute_seconds": 1.0,
        },
        {
            "backend": "whisper_cpp",
            "wall_seconds": 1.3,
            "compute_seconds": 1.2,
        },
    ]
    result = benchmark.summarize(runs)
    assert result["reference_backend"] == "whisper_cpp"
    assert result["backends"]["cke"]["compute_seconds"] == 11.0
    assert result["backends"]["cke"]["compute_ratio_vs_reference"] == 10.0
