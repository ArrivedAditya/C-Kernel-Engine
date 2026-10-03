"""Guard the mapped M=2 arithmetic and fail-closed generated batch selection."""

from __future__ import annotations

import copy
import ctypes
import json
import platform
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version/v8/scripts"))
from batch_decode_contract_v8 import resolve_two_row_batch_contract  # noqa: E402


def _fixture() -> tuple[list[dict], dict, dict]:
    buffers = [
        {"name": name, "size": size, "abs_offset": offset, "define": macro,
         "lifetime": lifetime, "mutable": True}
        for name, size, offset, macro, lifetime in (
            ("kv_cache", 64, 64, "A_KV_CACHE", "sequence"),
            ("embedded_input", 128, 128, "A_EMBEDDED_INPUT", "call"),
            ("residual", 128, 256, "A_RESIDUAL", "call"),
            ("q_scratch", 64, 384, "A_Q_SCRATCH", "call"),
            ("k_scratch", 32, 448, "A_K_SCRATCH", "call"),
            ("logits", 256, 512, "A_LOGITS", "call"),
        )
    ]
    layout = {"memory": {"arena": {"total_size": 768}, "activations": {"buffers": buffers}}}
    def arg(name: str, expr: str, ref: str | None = None) -> dict:
        row = {"name": name, "expr": expr}
        if ref:
            row["buffer_ref"] = ref
        return row
    ops = [
        {"op": "dense_embedding_lookup"},
        {"op": "residual_save", "args": [arg("dst", "(void*)(model->bump + A_RESIDUAL)", "residual")]},
        {"op": "attn_norm"},
        {"op": "q_proj", "layer": 0, "function": "gemv_q5_1_q8_1", "args": [
            arg("x", "(const float*)(model->bump + A_EMBEDDED_INPUT)", "embedded_input"),
            arg("y", "(float*)(model->bump + A_Q_SCRATCH)", "q_scratch"),
            arg("W", "(const void*)(model->bump + W_LAYER_0_WQ)"),
            arg("M", "16"), arg("K", "32")]},
        {"op": "k_proj", "layer": 0, "function": "gemv_q5_1_q8_1", "args": [
            arg("x", "(const float*)(model->bump + A_EMBEDDED_INPUT)", "embedded_input"),
            arg("y", "(float*)(model->bump + A_K_SCRATCH)", "k_scratch"),
            arg("W", "(const void*)(model->bump + W_LAYER_0_WK)"),
            arg("M", "8"), arg("K", "32")]},
        {"op": "attention"},
    ]
    return ops, layout, {"vocab_size": 64}


class BatchContractTests(unittest.TestCase):
    def test_admits_only_declared_kv_and_map_pair(self) -> None:
        ops, layout, config = _fixture()
        resolved = resolve_two_row_batch_contract(ops, layout, config)
        self.assertEqual(resolved["gemm_function"], "gemm_nt_q5_1_q8_1_m2")
        self.assertEqual(resolved["input_dim"], 32)
        for change in ("extra_state", "changed_weight", "changed_input", "reordered", "oversized_k"):
            bad_ops, bad_layout = copy.deepcopy(ops), copy.deepcopy(layout)
            if change == "extra_state":
                bad_layout["memory"]["activations"]["buffers"].append(
                    {"name": "unknown_state", "size": 16, "abs_offset": 704,
                     "lifetime": "sequence", "mutable": True})
            elif change == "changed_weight":
                bad_ops[3]["args"][2]["expr"] = "(const void*)0"
            elif change == "changed_input":
                bad_ops[4]["args"][0]["buffer_ref"] = "residual"
            elif change == "oversized_k":
                bad_ops[3]["args"][-1]["expr"] = "8224"
                bad_ops[4]["args"][-1]["expr"] = "8224"
                bad_layout["memory"]["activations"]["buffers"][1]["size"] = 8224 * 4
            else:
                bad_ops[3], bad_ops[4] = bad_ops[4], bad_ops[3]
            with self.subTest(change=change):
                self.assertIsNone(resolve_two_row_batch_contract(bad_ops, bad_layout, config))

    def test_m2_kernel_matches_two_independent_quantized_gemvs(self) -> None:
        source = ROOT / "src/kernels/gemm_kernels_q5_1_q8_1.c"
        modes = [[]]
        if platform.machine().lower() in {"x86_64", "amd64"}:
            modes.append(["-mavx2"])
        for flags in modes:
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as directory:
                library = Path(directory) / "libq51.so"
                subprocess.run(["cc", "-shared", "-fPIC", "-O2", *flags,
                                "-I", str(ROOT / "include"), str(source),
                                "-lm", "-o", str(library)], check=True,
                               capture_output=True, text=True)
                native = ctypes.CDLL(str(library))
                gemv = native.gemv_q5_1_q8_1
                gemv.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
                                 ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
                batch = native.gemm_nt_q5_1_q8_1_m2
                batch.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
                                  ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                                  ctypes.c_int, ctypes.c_int, ctypes.c_int]
                channels, width = 9, 64
                # Q5_1 blocks: fp16 scale=1, fp16 minimum=0, then packed bits.
                block = bytes((0x00, 0x3c, 0, 0, 0xaa, 0x55, 0x33, 0x77)) + bytes(range(16))
                weights = ctypes.create_string_buffer(block * (channels * width // 32))
                values = [((i * 17) % 41 - 20) / 13 for i in range(2 * width)]
                inputs = (ctypes.c_float * len(values))(*values)
                actual = (ctypes.c_float * (2 * channels))()
                batch(inputs, weights, None, actual, 2, channels, width)
                for row in range(2):
                    expected = (ctypes.c_float * channels)()
                    input_row = ctypes.cast(ctypes.byref(inputs, row * width * 4),
                                            ctypes.POINTER(ctypes.c_float))
                    gemv(expected, weights, input_row, channels, width)
                    self.assertEqual(bytes(expected), bytes(actual)[row * channels * 4:(row + 1) * channels * 4])


if __name__ == "__main__":
    unittest.main()
