"""
Q4_K x Q8_K matvec sanity test (decode-style).

Validates:
- Q8_K activation quantization (C)
- Q4_K weight dequantization (C)
- Q4_K x Q8_K matvec kernel vs dequantized FP32 reference
"""

import ctypes
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
UNITS = ROOT / "unittest"
SCRIPTS = ROOT / "scripts"
for path in (ROOT, UNITS, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from lib_loader import load_lib
from q4_k_quantize import quantize_q4_k_row
from test_utils import (
    TestReport, TestResult, get_cpu_info,
    print_system_info, time_function
)

QK_K = 256
BLOCK_Q4_K_SIZE = 144
BLOCK_Q8_K_SIZE = 4 + 256 + 32


def dequant_q8_k_row(blocks: bytes, k: int) -> np.ndarray:
    """Dequantize Q8_K blocks into FP32."""
    nb = k // QK_K
    out = np.zeros(k, dtype=np.float32)
    for b in range(nb):
        off = b * BLOCK_Q8_K_SIZE
        d = struct.unpack_from("<f", blocks, off)[0]
        qs = np.frombuffer(blocks, dtype=np.int8, count=QK_K, offset=off + 4)
        out[b * QK_K:(b + 1) * QK_K] = d * qs.astype(np.float32)
    return out


def dequant_q4_k_row_c(lib, row_bytes: bytes, k: int) -> np.ndarray:
    out = np.zeros(k, dtype=np.float32)
    buf = ctypes.create_string_buffer(row_bytes, len(row_bytes))
    lib.dequant_q4_k_row(
        ctypes.cast(buf, ctypes.c_void_p),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(k),
    )
    return out


def run_case(lib, m_out: int, k: int, tol: float, tol_fp32: float,
             warmup: int, iterations: int, pad_to_block: bool = False) -> int:
    k_used = k
    if pad_to_block:
        k_used = ((k + QK_K - 1) // QK_K) * QK_K

    x = (np.random.randn(k_used).astype(np.float32) * 0.1).astype(np.float32)
    w = (np.random.randn(m_out, k_used).astype(np.float32) * 0.1).astype(np.float32)
    if pad_to_block and k_used != k:
        x[k:] = 0.0
        w[:, k:] = 0.0

    # Quantize weights (Q4_K)
    row_bytes = (k_used // QK_K) * BLOCK_Q4_K_SIZE
    w_q4 = b"".join(quantize_q4_k_row(w[i]) for i in range(m_out))
    w_buf = ctypes.create_string_buffer(w_q4, len(w_q4))

    # Quantize activations (Q8_K) using C
    q8_blocks = k_used // QK_K
    x_q8 = ctypes.create_string_buffer(q8_blocks * BLOCK_Q8_K_SIZE)
    lib.quantize_row_q8_k(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(x_q8, ctypes.c_void_p),
        ctypes.c_int(k_used),
    )

    # Reference: dequantize and compute FP32 matvec
    x_deq = dequant_q8_k_row(x_q8.raw, k_used)
    w_deq = np.zeros((m_out, k_used), dtype=np.float32)
    for i in range(m_out):
        row = w_q4[i * row_bytes:(i + 1) * row_bytes]
        w_deq[i] = dequant_q4_k_row_c(lib, row, k_used)
    y_ref = w_deq @ x_deq
    w_t = torch.from_numpy(w)
    x_t = torch.from_numpy(x)
    y_fp32 = torch.matmul(w_t, x_t).numpy()

    # Kernel variants: gemv + gemm_nt
    w_ptr = ctypes.cast(w_buf, ctypes.c_void_p)
    x_q8_ptr = ctypes.cast(x_q8, ctypes.c_void_p)

    def run_gemv(fn_name: str) -> np.ndarray:
        out = np.zeros(m_out, dtype=np.float32)
        fn = getattr(lib, fn_name)
        fn(out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
           w_ptr,
           x_q8_ptr,
           ctypes.c_int(m_out),
           ctypes.c_int(k_used))
        return out

    y_ref_kernel = run_gemv("gemv_q4_k_q8_k_ref")
    y_avx2 = run_gemv("gemv_q4_k_q8_k_avx2")
    y_vnni = run_gemv("gemv_q4_k_q8_k_vnni")
    y_dispatch = run_gemv("gemv_q4_k_q8_k")

    y_gemm = np.zeros(m_out, dtype=np.float32)
    lib.gemm_nt_q4_k_q8_k(
        x_q8_ptr,
        w_ptr,
        None,
        y_gemm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(1),
        ctypes.c_int(m_out),
        ctypes.c_int(k_used),
    )

    # A complete four-row group uses the packed prefill reduction.
    y_continuation = np.zeros(m_out, dtype=np.float32)
    lib.gemv_q4_k_q8_k_prefill_order_parallel_dispatch(
        y_continuation.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        w_ptr, x_q8_ptr, None, ctypes.c_int(m_out), ctypes.c_int(k_used),
    )
    prefill_rows = ctypes.create_string_buffer(x_q8.raw * 4)
    y_prefill = np.zeros((4, m_out), dtype=np.float32)
    lib.gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch(
        ctypes.cast(prefill_rows, ctypes.c_void_p), w_ptr, None,
        y_prefill.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(4), ctypes.c_int(m_out), ctypes.c_int(k_used),
    )

    max_diff_ref = float(np.max(np.abs(y_ref_kernel - y_ref)))
    max_diff_avx2 = float(np.max(np.abs(y_avx2 - y_ref)))
    max_diff_vnni = float(np.max(np.abs(y_vnni - y_ref)))
    max_diff_dispatch = float(np.max(np.abs(y_dispatch - y_ref)))
    max_diff_gemm = float(np.max(np.abs(y_gemm - y_ref)))
    tol_fp32 = tol_fp32

    max_diff_fp32 = float(np.max(np.abs(y_gemm - y_fp32)))

    acc_report = TestReport(
        test_name="Q4_K x Q8_K Matvec (Decode)",
        dtype="q4_k/q8_k",
        shape=f"M={m_out}, K={k} (padded {k_used}), y=W@x" if pad_to_block else f"M={m_out}, K={k}, y=W@x",
        cpu_info=get_cpu_info()
    )
    acc_report.add_result(TestResult(
        name=f"gemv_ref vs dequant (M={m_out},K={k_used})",
        passed=max_diff_ref <= tol,
        max_diff=max_diff_ref,
        tolerance=tol
    ))
    acc_report.add_result(TestResult(
        name=f"gemv_avx2 vs dequant (M={m_out},K={k_used})",
        passed=max_diff_avx2 <= tol,
        max_diff=max_diff_avx2,
        tolerance=tol
    ))
    acc_report.add_result(TestResult(
        name=f"gemv_vnni vs dequant (M={m_out},K={k_used})",
        passed=max_diff_vnni <= tol,
        max_diff=max_diff_vnni,
        tolerance=tol
    ))
    acc_report.add_result(TestResult(
        name=f"gemv_dispatch vs dequant (M={m_out},K={k_used})",
        passed=max_diff_dispatch <= tol,
        max_diff=max_diff_dispatch,
        tolerance=tol
    ))
    acc_report.add_result(TestResult(
        name=f"gemm_nt vs dequant (M={m_out},K={k_used})",
        passed=max_diff_gemm <= tol,
        max_diff=max_diff_gemm,
        tolerance=tol
    ))
    acc_report.add_result(TestResult(
        name=f"gemm_nt vs FP32 (M={m_out},K={k_used})",
        passed=max_diff_fp32 <= tol_fp32,
        max_diff=max_diff_fp32,
        tolerance=tol_fp32
    ))
    acc_report.add_result(TestResult(
        name=f"continuation vs four-row prefill (M={m_out},K={k_used})",
        passed=np.array_equal(y_continuation, y_prefill[3]),
        max_diff=float(np.max(np.abs(y_continuation - y_prefill[3]))),
        tolerance=0.0,
    ))
    # Use distinct activations and both complete groups and residual rows.
    # This catches a continuation that always uses the packed reduction even
    # when the ordinary provider sends the last 1-3 rows through GEMV order.
    row_counts = [*range(1, 10)] + ([16, 17] if m_out >= 512 else [])
    varied_rows = []
    for row in range(max(row_counts)):
        values = np.ascontiguousarray(np.roll(x, row * 17) + row * 0.013,
                                      dtype=np.float32)
        quantized = ctypes.create_string_buffer(q8_blocks * BLOCK_Q8_K_SIZE)
        lib.quantize_row_q8_k(
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(quantized, ctypes.c_void_p), ctypes.c_int(k_used))
        varied_rows.append(quantized.raw)
    for rows in row_counts:
        inputs = ctypes.create_string_buffer(b"".join(varied_rows[:rows]))
        packed = rows - rows % 4
        for use_bias in (False, True):
            selected_bias = np.linspace(-0.2, 0.3, m_out, dtype=np.float32) if use_bias else None
            bias_pointer = (selected_bias.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                            if selected_bias is not None else None)
            expected = np.zeros((rows, m_out), dtype=np.float32)
            lib.gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch(
                ctypes.cast(inputs, ctypes.c_void_p), w_ptr, bias_pointer,
                expected.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(rows), ctypes.c_int(m_out), ctypes.c_int(k_used))
            actual = np.zeros_like(expected)
            for row in range(rows):
                row_bytes = varied_rows[row]
                row_input = ctypes.create_string_buffer(row_bytes)
                fn = (lib.gemv_q4_k_q8_k_prefill_order_parallel_dispatch
                      if row < packed else
                      lib.gemv_q4_k_q8_k_prefill_tail_parallel_dispatch)
                fn(actual[row].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                   w_ptr, ctypes.cast(row_input, ctypes.c_void_p), bias_pointer,
                   ctypes.c_int(m_out), ctypes.c_int(k_used))
            acc_report.add_result(TestResult(
                name=f"varied prefill rows={rows} bias={use_bias} (K={k_used})",
                passed=np.array_equal(actual, expected),
                max_diff=float(np.max(np.abs(actual - expected))), tolerance=0.0))
            if rows == 5 and not use_bias and k_used >= 1024:
                wrong = np.zeros(m_out, dtype=np.float32)
                tail_input = ctypes.create_string_buffer(varied_rows[4])
                lib.gemv_q4_k_q8_k_prefill_order_parallel_dispatch(
                    wrong.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    w_ptr, ctypes.cast(tail_input, ctypes.c_void_p), None,
                    ctypes.c_int(m_out), ctypes.c_int(k_used))
                acc_report.add_result(TestResult(
                    name=f"wrong packed reduction detected at tail (K={k_used})",
                    passed=not np.array_equal(wrong, expected[4]),
                    max_diff=float(np.max(np.abs(wrong - expected[4]))),
                    tolerance=0.0))
    bias = np.linspace(-0.2, 0.3, m_out, dtype=np.float32)
    bias_ptr = bias.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    y_with_bias = np.zeros(m_out, dtype=np.float32)
    y_prefill_with_bias = np.zeros((4, m_out), dtype=np.float32)
    lib.gemv_q4_k_q8_k_prefill_order_parallel_dispatch(
        y_with_bias.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        w_ptr, x_q8_ptr, bias_ptr, ctypes.c_int(m_out), ctypes.c_int(k_used),
    )
    lib.gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch(
        ctypes.cast(prefill_rows, ctypes.c_void_p), w_ptr, bias_ptr,
        y_prefill_with_bias.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(4), ctypes.c_int(m_out), ctypes.c_int(k_used),
    )
    acc_report.add_result(TestResult(
        name=f"biased continuation vs four-row prefill (M={m_out},K={k_used})",
        passed=np.array_equal(y_with_bias, y_prefill_with_bias[3]),
        max_diff=float(np.max(np.abs(y_with_bias - y_prefill_with_bias[3]))),
        tolerance=0.0,
    ))

    # Performance (microbench, small but useful for regressions)
    x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    y_ptr = y_gemm.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def quant_fn() -> None:
        lib.quantize_row_q8_k(x_ptr, x_q8_ptr, ctypes.c_int(k_used))

    def matvec_fn() -> None:
        lib.gemv_q4_k_q8_k(y_ptr, w_ptr, x_q8_ptr,
                           ctypes.c_int(m_out), ctypes.c_int(k_used))

    def end_to_end_fn() -> None:
        lib.quantize_row_q8_k(x_ptr, x_q8_ptr, ctypes.c_int(k_used))
        lib.gemv_q4_k_q8_k(y_ptr, w_ptr, x_q8_ptr,
                           ctypes.c_int(m_out), ctypes.c_int(k_used))

    def torch_matvec_fn() -> None:
        _ = torch.matmul(w_t, x_t)

    # Ensure the quant buffer is populated before timing matvec-only.
    quant_fn()

    q_time = time_function(quant_fn, warmup=warmup, iterations=iterations, name="Q8_K quantize")
    m_time = time_function(matvec_fn, warmup=warmup, iterations=iterations, name="Q4_K x Q8_K matvec")
    e_time = time_function(end_to_end_fn, warmup=warmup, iterations=iterations, name="End-to-end")
    pt_time = time_function(torch_matvec_fn, warmup=warmup, iterations=iterations, name="PyTorch FP32 matvec")

    perf_report = TestReport(
        test_name="Q4_K x Q8_K Matvec (Performance)",
        dtype="q4_k/q8_k",
        shape=f"M={m_out}, K={k} (padded {k_used}), y=W@x" if pad_to_block else f"M={m_out}, K={k}, y=W@x",
        cpu_info=get_cpu_info()
    )
    perf_report.add_result(TestResult(
        name=f"Q8_K quantize (M={m_out},K={k_used})",
        passed=True,
        max_diff=0.0,
        tolerance=0.0,
        kernel_time=q_time
    ))
    perf_report.add_result(TestResult(
        name=f"Q4_K x Q8_K gemv (M={m_out},K={k_used})",
        passed=True,
        max_diff=0.0,
        tolerance=0.0,
        pytorch_time=pt_time,
        kernel_time=m_time
    ))
    perf_report.add_result(TestResult(
        name=f"End-to-end (M={m_out},K={k_used})",
        passed=True,
        max_diff=0.0,
        tolerance=0.0,
        pytorch_time=pt_time,
        kernel_time=e_time
    ))

    acc_report.print_report()
    perf_report.print_report()

    if not acc_report.all_passed():
        return 1
    return 0


def main() -> int:
    print_system_info()

    try:
        lib = load_lib("libckernel_quant.so", "libckernel_engine.so")
    except Exception as exc:
        print(f"Warning: Could not load quantization library: {exc}")
        print("Run 'make libckernel_quant.so' first")
        return 0

    lib.quantize_row_q8_k.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    lib.quantize_row_q8_k.restype = None

    for name in (
        "gemv_q4_k_q8_k_ref",
        "gemv_q4_k_q8_k_avx2",
        "gemv_q4_k_q8_k_vnni",
        "gemv_q4_k_q8_k",
    ):
        fn = getattr(lib, name, None)
        if fn is not None:
            fn.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
            ]
            fn.restype = None

    lib.gemm_nt_q4_k_q8_k.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.gemm_nt_q4_k_q8_k.restype = None

    lib.gemv_q4_k_q8_k_prefill_order_parallel_dispatch.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int,
    ]
    lib.gemv_q4_k_q8_k_prefill_order_parallel_dispatch.restype = None
    lib.gemv_q4_k_q8_k_prefill_tail_parallel_dispatch.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
    ]
    lib.gemv_q4_k_q8_k_prefill_tail_parallel_dispatch.restype = None
    lib.gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    lib.gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch.restype = None

    lib.dequant_q4_k_row.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]
    lib.dequant_q4_k_row.restype = None

    np.random.seed(0)

    warmup = int(os.getenv("CK_TEST_WARMUP", "5"))
    iterations = int(os.getenv("CK_TEST_ITERS", "50"))
    tol = float(os.getenv("CK_TEST_TOL", "1e-2"))
    # Q4_K (4-bit) quantization has inherent error that accumulates in matmul
    # Max diff of 1-3 is typical for K=512-2048 due to quantization noise
    tol_fp32 = float(os.getenv("CK_TEST_TOL_FP32", "3.0"))

    sizes = [(16, 512), (64, 2048)]
    if os.getenv("CK_TEST_LARGE", "").lower() in ("1", "true", "yes"):
        sizes.append((128, 4096))

    status = 0
    for m_out, k in sizes:
        status |= run_case(lib, m_out, k, tol, tol_fp32, warmup, iterations)
    # Padded case for models with K not divisible by QK_K (e.g., 896 -> 1024).
    status |= run_case(lib, 16, 896, tol, tol_fp32, warmup, iterations, pad_to_block=True)
    status |= run_case(lib, 896, 1024, tol, tol_fp32, 1, 1)

    return status


if __name__ == "__main__":
    sys.exit(main())
