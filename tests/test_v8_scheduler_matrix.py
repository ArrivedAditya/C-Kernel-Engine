import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "bench_v8_scheduler_matrix",
    ROOT / "benchmarks" / "bench_v8_scheduler_matrix.py",
)
assert SPEC and SPEC.loader
MATRIX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MATRIX)


def test_text_result_rejects_successful_prefill_with_immediate_eos() -> None:
    status, reason = MATRIX.classify_text_result(
        {"returncode": 0},
        {"prompt_tokens": 19, "decode_tokens": 0},
        "",
        require_decode_count=True,
    )
    assert status == "fail"
    assert reason == "no_decoded_tokens"


def test_text_result_rejects_empty_reference_output() -> None:
    status, reason = MATRIX.classify_text_result(
        {"returncode": 0},
        {"prompt_tokens": 19, "decode_tokens": 0},
        "   ",
        require_decode_count=False,
    )
    assert status == "fail"
    assert reason == "empty_generated_text"


def test_text_result_accepts_nonempty_generation() -> None:
    status, reason = MATRIX.classify_text_result(
        {"returncode": 0},
        {"prompt_tokens": 19, "decode_tokens": 8},
        "Hello from the model.",
        require_decode_count=True,
    )
    assert status == "pass"
    assert reason == ""
