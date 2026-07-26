from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "benchmarks" / "bench_v8_qwen3vl_ocr.py"


def _load_benchmark():
    spec = importlib.util.spec_from_file_location("bench_v8_qwen3vl_ocr_contract", BENCHMARK_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BENCHMARK_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Qwen3VLOcrBenchmarkContractTests(unittest.TestCase):
    def test_xeon_speed_profile_uses_measured_gateup_thread_cap(self) -> None:
        benchmark = _load_benchmark()
        env = {"CK_SPEED_PROFILE": "qwen3vl_ocr_xeon_avx512"}

        benchmark._apply_qwen3vl_ocr_fast_defaults(env)

        self.assertEqual(env["CK_NUM_THREADS"], "20")
        self.assertEqual(env["CK_Q4K_GATEUP_SWIGLU_X16_THREAD_CAP"], "16")

    def test_explicit_gateup_thread_cap_is_preserved(self) -> None:
        benchmark = _load_benchmark()
        env = {
            "CK_SPEED_PROFILE": "qwen3vl_ocr_xeon_avx512",
            "CK_Q4K_GATEUP_SWIGLU_X16_THREAD_CAP": "12",
        }

        benchmark._apply_qwen3vl_ocr_fast_defaults(env)

        self.assertEqual(env["CK_Q4K_GATEUP_SWIGLU_X16_THREAD_CAP"], "12")

    def test_explicit_chat_template_reaches_ck_runner(self) -> None:
        benchmark = _load_benchmark()
        with tempfile.TemporaryDirectory() as td:
            image = Path(td) / "fixture.ppm"
            image.write_bytes(b"P6\n1 1\n255\n\x00\x00\x00")
            with mock.patch.object(benchmark, "_run", return_value=(1, "")):
                row = benchmark._run_one(
                    model="decoder.gguf",
                    mmproj="mmproj.gguf",
                    image=image,
                    prompt="Read the form.",
                    chat_template="qwen3vl",
                    threads=4,
                    max_tokens=8,
                    context_len=512,
                    image_min_tokens=64,
                    image_max_tokens=128,
                    force_compile=False,
                    force_convert=False,
                    bridge_runtime="decode-staged",
                    bridge_generation_mode="incremental-decode",
                    vision_activation_prefs=[],
                    profile_decoder=False,
                    timeout=60,
                )

        command = row["command"]
        index = command.index("--chat-template")
        self.assertEqual(command[index + 1], "qwen3vl")
        self.assertEqual(os.path.basename(command[1]), "ck_run_v8.py")


if __name__ == "__main__":
    unittest.main()
