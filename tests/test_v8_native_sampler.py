from __future__ import annotations

import ctypes
import math
from pathlib import Path
import subprocess
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "version" / "v8" / "src" / "ck_sampler_v8.c"


class V8NativeSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory(prefix="ck_sampler_v8_")
        cls.library_path = Path(cls._temporary.name) / "libck_sampler_v8.so"
        subprocess.run(
            [
                "gcc",
                "-O3",
                "-fPIC",
                "-shared",
                "-I",
                str(ROOT / "include"),
                str(SOURCE),
                "-lm",
                "-o",
                str(cls.library_path),
            ],
            check=True,
            cwd=ROOT,
        )
        cls.library = ctypes.CDLL(str(cls.library_path))
        cls.sample = cls.library.ck_sample_top_p_v8
        cls.sample.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
        ]
        cls.sample.restype = ctypes.c_int

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def _sample(
        self,
        logits: list[float],
        *,
        temperature: float,
        top_p: float,
        random_value: float,
    ) -> int:
        values = (ctypes.c_float * len(logits))(*logits)
        return int(
            self.sample(values, len(logits), temperature, top_p, random_value)
        )

    def test_nonpositive_temperature_or_top_p_is_greedy(self) -> None:
        logits = [1.0, 4.0, 2.0]
        self.assertEqual(
            self._sample(logits, temperature=0.0, top_p=1.0, random_value=0.7),
            1,
        )
        self.assertEqual(
            self._sample(logits, temperature=1.0, top_p=0.0, random_value=0.7),
            1,
        )

    def test_top_p_one_samples_full_distribution_without_sorting(self) -> None:
        logits = [math.log(0.5), math.log(0.3), math.log(0.2)]
        self.assertEqual(
            self._sample(logits, temperature=1.0, top_p=1.0, random_value=0.1),
            0,
        )
        self.assertEqual(
            self._sample(logits, temperature=1.0, top_p=1.0, random_value=0.6),
            1,
        )
        self.assertEqual(
            self._sample(logits, temperature=1.0, top_p=1.0, random_value=0.9),
            2,
        )
        self.assertEqual(
            self._sample(
                [-1000.0, 0.0],
                temperature=1.0,
                top_p=1.0,
                random_value=0.0,
            ),
            1,
        )

    def test_nucleus_sort_is_probability_descending_and_tie_deterministic(self) -> None:
        logits = [math.log(0.2), math.log(0.5), math.log(0.3)]
        selected = {
            self._sample(logits, temperature=1.0, top_p=0.6, random_value=value)
            for value in (0.0, 0.25, 0.5, 0.75, 0.99)
        }
        self.assertEqual(selected, {1, 2})

        tied = [0.0, 0.0, -100.0]
        self.assertEqual(
            self._sample(tied, temperature=1.0, top_p=0.6, random_value=0.1),
            0,
        )
        self.assertEqual(
            self._sample(tied, temperature=1.0, top_p=0.6, random_value=0.9),
            1,
        )

    def test_invalid_random_value_fails_closed(self) -> None:
        logits = [1.0, 2.0]
        self.assertEqual(
            self._sample(logits, temperature=1.0, top_p=1.0, random_value=1.0),
            -1,
        )

    def test_production_vocabulary_paths_are_subquadratic(self) -> None:
        vocab_size = 248_320
        logits = (ctypes.c_float * vocab_size)(
            *(float((index % 257) - 128) / 32.0 for index in range(vocab_size))
        )

        started = time.perf_counter()
        full_result = self.sample(logits, vocab_size, 0.8, 1.0, 0.5)
        full_elapsed = time.perf_counter() - started
        self.assertGreaterEqual(full_result, 0)
        self.assertLess(full_elapsed, 2.0)

        uniform = (ctypes.c_float * vocab_size)(*([0.0] * vocab_size))
        started = time.perf_counter()
        nucleus_result = self.sample(uniform, vocab_size, 1.0, 0.999, 0.5)
        nucleus_elapsed = time.perf_counter() - started
        self.assertGreaterEqual(nucleus_result, 0)
        self.assertLess(nucleus_elapsed, 3.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
