from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/gemma4_bf16_dot_llama_avx2.json"
PROBE = ROOT / "tests/fixtures/gemma4_bf16_dot_probe.c"
U16_PTR = ctypes.POINTER(ctypes.c_uint16)
F32_PTR = ctypes.POINTER(ctypes.c_float)


def _input_bits(width: int) -> tuple[np.ndarray, np.ndarray]:
    i = np.arange(width, dtype=np.uint32)
    a = i * np.uint32(1664525) + np.uint32(1013904223)
    b = i * np.uint32(22695477) + np.uint32(1103515245)
    x = ((118 + (a >> 24) % 12) << 7) | (a & 127) | (((a >> 15) & 1) << 15)
    y = ((118 + (b >> 23) % 12) << 7) | (b & 127) | (((b >> 14) & 1) << 15)
    return x.astype(np.uint16), y.astype(np.uint16)


def _scalar_reference(x: np.ndarray, y: np.ndarray) -> np.float32:
    xf = (x.astype(np.uint32) << 16).view(np.float32)
    yf = (y.astype(np.uint32) << 16).view(np.float32)
    return np.float32(sum(float(np.float32(a * b)) for a, b in zip(xf, yf)))


class Gemma4Bf16ReductionOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if platform.machine().lower() not in ("x86_64", "amd64"):
            raise unittest.SkipTest("AVX2/scalar paired build requires x86_64")
        subprocess.run(["make", "--no-print-directory", "build/libckernel_engine.so"],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        cls._tmp = tempfile.TemporaryDirectory(prefix="ck_gemma_bf16_")
        cls.libs = {}
        engine = ROOT / "build/libckernel_engine.so"
        ctypes.CDLL(str(engine), mode=ctypes.RTLD_GLOBAL)
        for name, flags, expected_avx2 in (
            ("avx2", ["-march=haswell", "-mno-fma"], 1),
            ("scalar", ["-mno-avx", "-mno-avx2", "-mno-fma"], 0),
        ):
            path = Path(cls._tmp.name) / f"libgemma_{name}.so"
            subprocess.run([
                os.environ.get("CC", "gcc"), "-shared", "-fPIC", "-O3",
                "-fno-fast-math", *flags, "-I.", "-Iinclude",
                str(PROBE), "-Lbuild", "-lckernel_engine", "-lm", "-lpthread",
                "-Wl,-Bsymbolic", f"-Wl,-rpath,{engine.parent}", "-o", str(path),
            ], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
            lib = ctypes.CDLL(str(path))
            lib.ck_test_gemma4_compiled_avx2.restype = ctypes.c_int
            assert lib.ck_test_gemma4_compiled_avx2() == expected_avx2
            lib.ck_test_gemma4_bf16_dot.argtypes = [U16_PTR, U16_PTR, ctypes.c_int]
            lib.ck_test_gemma4_bf16_dot.restype = ctypes.c_float
            for symbol in ("gemma4_per_layer_prepare_forward",
                           "gemma4_per_layer_prepare_bf16_forward"):
                fn = getattr(lib, symbol)
                fn.argtypes = [F32_PTR, F32_PTR, ctypes.POINTER(ctypes.c_int32),
                               ctypes.c_void_p, U16_PTR, F32_PTR,
                               ctypes.c_int, ctypes.c_int, ctypes.c_int,
                               ctypes.c_int, ctypes.c_int, ctypes.c_float]
            cls.libs[name] = lib

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_pinned_oracle_vector_tails_and_scalar_fallback(self) -> None:
        fixture = json.loads(FIXTURE.read_text())
        self.assertEqual(fixture["oracle_source_revision"],
                         "19e28a27702117d8f2eb16b825b9a308111f67d9")
        distinct_reductions = 0
        for case in fixture["cases"]:
            width = case["width"]
            x, y = _input_bits(width)
            if width == 2560:
                self.assertEqual(hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest(),
                                 fixture["long_input_sha256"])
            if int(_scalar_reference(x, y).view(np.uint32)) != int(case["result_bits"], 16):
                distinct_reductions += 1
            for name, lib in self.libs.items():
                with self.subTest(width=width, isa=name):
                    result = np.float32(lib.ck_test_gemma4_bf16_dot(
                        x.ctypes.data_as(U16_PTR), y.ctypes.data_as(U16_PTR), width))
                    expected = (np.asarray([int(case["result_bits"], 16)], dtype=np.uint32)
                                .view(np.float32)[0] if name == "avx2"
                                else _scalar_reference(x, y))
                    self.assertEqual(int(result.view(np.uint32)),
                                     int(expected.view(np.uint32)))
        self.assertGreater(distinct_reductions, 0)

    def test_both_prepare_providers_with_nonzero_tokens_and_guards(self) -> None:
        fixture = json.loads(FIXTURE.read_text())
        for case in fixture["cases"]:
            width = case["width"]
            weights, hidden_bits = _input_bits(width)
            hidden = (hidden_bits.astype(np.uint32) << 16).view(np.float32).reshape(1, width)
            projection = np.zeros((1, 256, width), dtype=np.uint16)
            projection[0, 0] = weights
            gamma = np.ones(256, dtype=np.float32)
            gamma[0] = np.float32(1.125)
            token_ids = np.zeros(1, dtype=np.int32)
            bf16_token = np.zeros((1, 1, 256), dtype=np.uint16)
            bf16_token[0, 0, 0] = 0x3F00  # 0.5
            q5_token = np.zeros(176, dtype=np.uint8)
            q5_token[:2] = np.asarray([0x3C00], dtype="<u2").view(np.uint8)
            q5_token[4] = 1  # first subblock scale
            q5_token[48] = 1  # first low-nibble value
            for isa, lib in self.libs.items():
                dot = np.float32(lib.ck_test_gemma4_bf16_dot(
                    weights.ctypes.data_as(U16_PTR),
                    hidden_bits.ctypes.data_as(U16_PTR), width))
                projected = np.float32(dot * np.float32(1 / math.sqrt(width)))
                normed = np.float32(projected * np.float32(
                    1 / math.sqrt(float(projected) ** 2 / 256 + 1e-6)) * gamma[0])
                for symbol, token, token_value in (
                    ("gemma4_per_layer_prepare_forward", q5_token, 1.0),
                    ("gemma4_per_layer_prepare_bf16_forward", bf16_token, 0.5),
                ):
                    with self.subTest(width=width, isa=isa, provider=symbol):
                        guarded = np.full(258, np.float32(12345.0), dtype=np.float32)
                        out = guarded[1:-1]
                        getattr(lib, symbol)(
                            out.ctypes.data_as(F32_PTR), hidden.ctypes.data_as(F32_PTR),
                            token_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                            token.ctypes.data_as(ctypes.c_void_p),
                            projection.ctypes.data_as(U16_PTR), gamma.ctypes.data_as(F32_PTR),
                            1, 1, width, 256, 1, ctypes.c_float(1e-6))
                        expected = np.float32((16 * token_value + normed) *
                                              np.float32(0.7071067811865475))
                        if isa == "avx2":
                            frozen = case["prepare_q5" if token_value == 1.0 else "prepare_bf16"]
                            self.assertAlmostEqual(float(expected), frozen, delta=2e-6)
                            expected = np.float32(frozen)
                        self.assertAlmostEqual(float(out[0]), float(expected), delta=2e-5)
                        np.testing.assert_array_equal(out[1:], 0.0)
                        np.testing.assert_array_equal(guarded[[0, -1]], 12345.0)


if __name__ == "__main__":
    unittest.main()
