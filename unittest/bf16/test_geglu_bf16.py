"""
GeGLU BF16 forward/backward parity against PyTorch CPU.

The BF16 kernels use BF16 as storage and FP32 math/gradient accumulation for
backward. This test quantizes inputs/upstream gradients to BF16, converts those
quantized values back to FP32 for PyTorch autograd, and compares FP32 grads.
"""
import ctypes
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UNITS = ROOT / "unittest"
for path in (ROOT, UNITS):
    if str(path) not in sys.path:
        sys.path.append(str(path))

import numpy as np
import torch
import torch.nn.functional as F

from lib_loader import load_lib
from test_utils import TestReport, TestResult, get_cpu_info, max_diff, time_function, print_system_info
from bf16_utils import float32_to_bf16, bf16_to_float32, numpy_to_uint16_ptr

cpu = get_cpu_info()
if not cpu.avx512bf16:
    print("BF16 kernels require AVX-512 BF16; skipping this test on the current CPU.")
    sys.exit(0)

lib = load_lib("libckernel_engine.so")

lib.geglu_forward_bf16.argtypes = [
    ctypes.POINTER(ctypes.c_uint16),  # x [T, 2D]
    ctypes.POINTER(ctypes.c_uint16),  # out [T, D]
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),   # scratch [T*3D]
]
lib.geglu_forward_bf16.restype = None

lib.geglu_backward_bf16_mixed.argtypes = [
    ctypes.POINTER(ctypes.c_uint16),  # x [T, 2D]
    ctypes.POINTER(ctypes.c_uint16),  # d_out [T, D]
    ctypes.POINTER(ctypes.c_float),   # d_x [T, 2D]
    ctypes.c_int,
    ctypes.c_int,
]
lib.geglu_backward_bf16_mixed.restype = None


def run_tests(T=11, D=37, warmup=10, iterations=500):
    np.random.seed(3)
    x_np = (np.random.randn(T, 2 * D).astype(np.float32) * 0.75)
    d_out_np = (np.random.randn(T, D).astype(np.float32) * 0.25)
    x_bf = float32_to_bf16(x_np)
    d_out_bf = float32_to_bf16(d_out_np)
    out_bf = np.zeros((T, D), dtype=np.uint16)
    d_x = np.zeros((T, 2 * D), dtype=np.float32)
    scratch = np.zeros(T * 3 * D, dtype=np.float32)

    def c_forward():
        lib.geglu_forward_bf16(
            numpy_to_uint16_ptr(x_bf),
            numpy_to_uint16_ptr(out_bf),
            ctypes.c_int(T), ctypes.c_int(D),
            scratch.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )

    def c_backward():
        lib.geglu_backward_bf16_mixed(
            numpy_to_uint16_ptr(x_bf),
            numpy_to_uint16_ptr(d_out_bf),
            d_x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(T), ctypes.c_int(D),
        )

    x_ref = torch.from_numpy(bf16_to_float32(x_bf.copy())).to(torch.float32).requires_grad_(True)
    d_out_ref = torch.from_numpy(bf16_to_float32(d_out_bf.copy())).to(torch.float32)
    a, b = x_ref[:, :D], x_ref[:, D:]
    y_ref = F.gelu(a, approximate="tanh") * b
    y_ref.backward(d_out_ref)

    c_forward()
    c_backward()
    y_c = torch.from_numpy(bf16_to_float32(out_bf.copy()))
    d_x_c = torch.from_numpy(d_x.copy())
    y_diff = max_diff(y_c, y_ref.detach().to(torch.bfloat16).to(torch.float32))
    dx_diff = max_diff(d_x_c, x_ref.grad)

    report = TestReport(
        test_name="GeGLU Forward/Backward (BF16)",
        dtype="bf16/fp32",
        shape=f"T={T}, D={D}",
        cpu_info=get_cpu_info(),
    )
    report.add_result(TestResult(
        name="GeGLU Forward",
        passed=y_diff <= 3e-2,
        max_diff=y_diff,
        tolerance=3e-2,
        pytorch_time=None,
        kernel_time=time_function(c_forward, warmup=warmup, iterations=iterations, name="C GeGLU BF16"),
    ))
    report.add_result(TestResult(
        name="d_x_fp32_accum",
        passed=dx_diff <= 2e-4,
        max_diff=dx_diff,
        tolerance=2e-4,
        pytorch_time=None,
        kernel_time=time_function(c_backward, warmup=warmup, iterations=iterations, name="C GeGLU BF16 Backward"),
    ))
    return report


if __name__ == "__main__":
    print_system_info()
    report = run_tests()
    report.print_report()
    if not report.all_passed():
        sys.exit(1)
