#!/usr/bin/env python3
"""Composed Q8_0 activation/projection parity at vision-production width.

The default fixture is deterministic and stresses Q8 block maxima and rounding
thresholds at K=1152. Set V8_Q8_COMPOSED_REAL_DUMP to a Qwen3-VL X-ray
``dump.bin`` to run the same controls on selected real attention rows.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / os.environ.get("CK_BUILD_DIR", "build")
LLAMA_BIN = ROOT / "llama.cpp" / "build" / "bin"
K = 1152
ROWS = 8
PROJECTIONS = (
    ("qkv", 3456),
    ("attention_out", 1152),
    ("mlp_up", 4304),
)
QK8_0 = 32
BLOCK_BYTES = 34
ROW_BYTES = K // QK8_0 * BLOCK_BYTES


def _load_libraries() -> tuple[ctypes.CDLL, ctypes.CDLL, ctypes.CDLL]:
    ck_path = BUILD / "libckernel_engine.so"
    if not ck_path.exists():
        raise RuntimeError(f"missing CK library: {ck_path}")
    for name in ("libggml-base.so", "libggml.so", "libggml-cpu.so"):
        path = LLAMA_BIN / name
        if not path.exists():
            raise RuntimeError(f"missing llama.cpp library: {path}")
        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    return (
        ctypes.CDLL(str(ck_path)),
        ctypes.CDLL(str(LLAMA_BIN / "libggml-base.so")),
        ctypes.CDLL(str(LLAMA_BIN / "libggml-cpu.so")),
    )


def _bind(ck: ctypes.CDLL, base: ctypes.CDLL, cpu: ctypes.CDLL) -> None:
    f32p = ctypes.POINTER(ctypes.c_float)
    ck.quantize_row_q8_0.argtypes = [f32p, ctypes.c_void_p, ctypes.c_int]
    ck.quantize_row_q8_0.restype = None
    base.quantize_row_q8_0_ref.argtypes = [f32p, ctypes.c_void_p, ctypes.c_int64]
    base.quantize_row_q8_0_ref.restype = None
    ck.vec_dot_q8_0_q8_0.argtypes = [
        ctypes.c_int, f32p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    ck.vec_dot_q8_0_q8_0.restype = None
    cpu.ggml_vec_dot_q8_0_q8_0.argtypes = [
        ctypes.c_int, f32p, ctypes.c_size_t,
        ctypes.c_void_p, ctypes.c_size_t,
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
    ]
    cpu.ggml_vec_dot_q8_0_q8_0.restype = None
    cpu.ggml_cpu_init.argtypes = []
    cpu.ggml_cpu_init.restype = None
    cpu.ggml_cpu_init()


def _ptr_f32(a: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _vision_stress_rows() -> np.ndarray:
    r = np.arange(ROWS, dtype=np.float32)[:, None]
    c = np.arange(K, dtype=np.float32)[None, :]
    x = (
        np.sin(c * np.float32(0.017) + r * np.float32(0.31)) * np.float32(0.36)
        + np.cos(c * np.float32(0.0037) - r * np.float32(0.19)) * np.float32(0.11)
    ).astype(np.float32)
    for row in range(ROWS):
        for block in range(K // QK8_0):
            start = block * QK8_0
            peak = np.float32((1.0 + 0.07 * row + 0.003 * block) * (-1 if (row + block) & 1 else 1))
            x[row, start + ((row * 7 + block * 11) % QK8_0)] = peak
    return x


def _real_attention_rows(path: Path, layer: int) -> np.ndarray:
    scripts = ROOT / "version" / "v7" / "scripts"
    sys.path.insert(0, str(scripts))
    import parity_test  # type: ignore

    dumps = parity_test.read_dump_file(path)
    tensor = next(
        (np.asarray(d.data, dtype=np.float32) for d in dumps
         if d.layer_id == layer and d.op_name == "kqv_out"),
        None,
    )
    if tensor is None or tensor.size % K:
        raise RuntimeError(f"{path} has no layer-{layer} kqv_out with width {K}")
    matrix = tensor.reshape(-1, K)
    indices = np.array([0, 5, 41, 701, 952, matrix.shape[0] // 2, matrix.shape[0] - 2, matrix.shape[0] - 1])
    return np.ascontiguousarray(matrix[indices], dtype=np.float32)


def _quantize_rows(fn: object, x: np.ndarray) -> np.ndarray:
    out = np.empty((x.shape[0], ROW_BYTES), dtype=np.uint8)
    for row in range(x.shape[0]):
        fn(_ptr_f32(x[row]), ctypes.c_void_p(out[row].ctypes.data), K)
    return out


def _weight_rows(base: ctypes.CDLL, outputs: int) -> np.ndarray:
    c = np.arange(K, dtype=np.float32)
    weights = np.empty((outputs, ROW_BYTES), dtype=np.uint8)
    row = np.empty(K, dtype=np.float32)
    for n in range(outputs):
        row[:] = (
            np.sin(c * np.float32(0.011) + np.float32(n * 0.007)) * np.float32(0.14)
            + np.cos(c * np.float32(0.005) - np.float32(n * 0.013)) * np.float32(0.05)
        )
        base.quantize_row_q8_0_ref(
            _ptr_f32(row), ctypes.c_void_p(weights[n].ctypes.data), K,
        )
    return weights


def _project(ck: ctypes.CDLL, cpu: ctypes.CDLL, activations: np.ndarray,
             weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    outputs = weights.shape[0]
    ck_out = np.empty((activations.shape[0], outputs), dtype=np.float32)
    llama_out = np.empty_like(ck_out)
    for m in range(activations.shape[0]):
        a_ptr = ctypes.c_void_p(activations[m].ctypes.data)
        for n in range(outputs):
            w_ptr = ctypes.c_void_p(weights[n].ctypes.data)
            ck_value = ctypes.c_float()
            llama_value = ctypes.c_float()
            ck.vec_dot_q8_0_q8_0(K, ctypes.byref(ck_value), w_ptr, a_ptr)
            cpu.ggml_vec_dot_q8_0_q8_0(
                K, ctypes.byref(llama_value), 0, w_ptr, 0, a_ptr, 0, 1,
            )
            ck_out[m, n] = ck_value.value
            llama_out[m, n] = llama_value.value
    return ck_out, llama_out


def main() -> int:
    ck, base, cpu = _load_libraries()
    _bind(ck, base, cpu)
    dump = os.environ.get("V8_Q8_COMPOSED_REAL_DUMP")
    dump_layer = int(os.environ.get("V8_Q8_COMPOSED_LAYER", "0"))
    inputs = _real_attention_rows(Path(dump), dump_layer) if dump else _vision_stress_rows()
    source = (
        f"real_qwen3vl_layer{dump_layer}_attention"
        if dump else "deterministic_vision_boundary_stress"
    )

    ck_q8 = _quantize_rows(ck.quantize_row_q8_0, inputs)
    llama_q8 = _quantize_rows(base.quantize_row_q8_0_ref, inputs)
    quant_equal = bool(np.array_equal(ck_q8, llama_q8))

    projection_reports = []
    projection_passed = True
    for projection_name, outputs in PROJECTIONS:
        weights = _weight_rows(base, outputs)
        ck_proj, llama_proj = _project(ck, cpu, llama_q8, weights)
        bias = (
            np.sin(np.arange(outputs, dtype=np.float32) * np.float32(0.021))
            * np.float32(0.03)
        )
        ck_biased = ck_proj + bias
        llama_biased = llama_proj + bias
        before_exact = bool(np.array_equal(ck_proj, llama_proj))
        after_exact = bool(np.array_equal(ck_biased, llama_biased))
        projection_passed = projection_passed and before_exact and after_exact
        projection_reports.append({
            "name": projection_name,
            "shape": {"rows": ROWS, "outputs": outputs, "width": K},
            "bit_exact_before_bias": before_exact,
            "bit_exact_after_bias": after_exact,
            "different_values_before_bias": int(np.count_nonzero(ck_proj != llama_proj)),
            "different_values_after_bias": int(np.count_nonzero(ck_biased != llama_biased)),
            "max_abs_before_bias": float(
                np.max(np.abs(ck_proj.astype(np.float64) - llama_proj.astype(np.float64)), initial=0.0)
            ),
            "max_abs_after_bias": float(
                np.max(np.abs(ck_biased.astype(np.float64) - llama_biased.astype(np.float64)), initial=0.0)
            ),
            "compared_values": int(ck_proj.size),
        })

    perturbation = np.where(
        (np.arange(ROWS)[:, None] + np.arange(K)[None, :]) & 1,
        np.float32(5.0e-7),
        np.float32(-5.0e-7),
    )
    perturbed = np.ascontiguousarray(inputs + perturbation, dtype=np.float32)
    perturbed_q8 = _quantize_rows(base.quantize_row_q8_0_ref, perturbed)
    byte_changes = int(np.count_nonzero(llama_q8 != perturbed_q8))
    changed_blocks = int(np.count_nonzero(
        np.any(llama_q8.reshape(ROWS, -1, BLOCK_BYTES) != perturbed_q8.reshape(ROWS, -1, BLOCK_BYTES), axis=2)
    ))

    passed = quant_equal and projection_passed
    report = {
        "schema": "cke.q8_composed_llama_parity",
        "schema_version": 1,
        "status": "pass" if passed else "fail",
        "fixture_source": source,
        "shape": {"rows": ROWS, "width": K},
        "same_input_quantizer": {
            "byte_exact": quant_equal,
            "different_bytes": int(np.count_nonzero(ck_q8 != llama_q8)),
        },
        "same_quantized_input_projections": projection_reports,
        "attention_scale_input_sensitivity": {
            "perturbation_abs": 5.0e-7,
            "changed_blocks": changed_blocks,
            "changed_bytes": byte_changes,
            "total_blocks": ROWS * (K // QK8_0),
        },
    }
    report_path = BUILD / "q8_composed_llama_parity" / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
