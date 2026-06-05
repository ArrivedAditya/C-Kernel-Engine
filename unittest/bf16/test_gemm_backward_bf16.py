"""
BF16 linear/GEMM backward parity test.

The kernel uses BF16 storage for input, weight, and upstream gradient, while
accumulating d_input, d_weight, and d_bias in FP32. This matches the standard
mixed-precision training contract: BF16 activations/weights, FP32 gradients.
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
from test_utils import TestReport, TestResult, get_cpu_info, max_diff, print_system_info, time_function
from bf16_utils import float32_to_bf16, bf16_to_float32, numpy_to_uint16_ptr

cpu = get_cpu_info()
if not cpu.avx512bf16:
    print("BF16 kernels require AVX-512 BF16; skipping this test on the current CPU.")
    sys.exit(0)

lib = load_lib("libckernel_engine.so")

fptr = ctypes.POINTER(ctypes.c_float)
u16ptr = ctypes.POINTER(ctypes.c_uint16)

lib.gemm_backward_bf16_mixed.argtypes = [
    u16ptr,          # d_output [tokens, out_dim]
    u16ptr,          # input [tokens, in_dim]
    u16ptr,          # weight [out_dim, in_dim]
    fptr,            # d_input [tokens, in_dim]
    fptr,            # d_weight [out_dim, in_dim]
    fptr,            # d_bias [out_dim]
    ctypes.c_int,    # tokens
    ctypes.c_int,    # in_dim
    ctypes.c_int,    # out_dim
]
lib.gemm_backward_bf16_mixed.restype = None


def _ptr_f32(arr: np.ndarray) -> fptr:
    return np.ascontiguousarray(arr, dtype=np.float32).ctypes.data_as(fptr)


def run_tests(tokens=17, in_dim=33, out_dim=29, warmup=10, iterations=300):
    rng = np.random.default_rng(123)
    x_np = (rng.standard_normal((tokens, in_dim)).astype(np.float32) * 0.5)
    w_np = (rng.standard_normal((out_dim, in_dim)).astype(np.float32) * 0.25)
    dy_np = (rng.standard_normal((tokens, out_dim)).astype(np.float32) * 0.5)

    x_bf16 = float32_to_bf16(x_np)
    w_bf16 = float32_to_bf16(w_np)
    dy_bf16 = float32_to_bf16(dy_np)

    dx = np.zeros((tokens, in_dim), dtype=np.float32)
    dw = np.zeros((out_dim, in_dim), dtype=np.float32)
    db = np.zeros(out_dim, dtype=np.float32)

    x_ref = torch.from_numpy(bf16_to_float32(x_bf16)).to(dtype=torch.float32).requires_grad_(True)
    w_ref = torch.from_numpy(bf16_to_float32(w_bf16)).to(dtype=torch.float32).requires_grad_(True)
    dy_ref = torch.from_numpy(bf16_to_float32(dy_bf16)).to(dtype=torch.float32)

    def pytorch_ref():
        x = x_ref.detach().clone().requires_grad_(True)
        w = w_ref.detach().clone().requires_grad_(True)
        y = F.linear(x, w, bias=None)
        y.backward(dy_ref)
        return x.grad, w.grad, dy_ref.sum(dim=0)

    dx_ref, dw_ref, db_ref = pytorch_ref()

    def c_kernel():
        lib.gemm_backward_bf16_mixed(
            numpy_to_uint16_ptr(dy_bf16),
            numpy_to_uint16_ptr(x_bf16),
            numpy_to_uint16_ptr(w_bf16),
            _ptr_f32(dx),
            _ptr_f32(dw),
            _ptr_f32(db),
            ctypes.c_int(tokens),
            ctypes.c_int(in_dim),
            ctypes.c_int(out_dim),
        )

    c_kernel()

    report = TestReport(
        test_name="GEMM/Linear Backward (BF16 storage, FP32 grads)",
        dtype="bf16/fp32",
        shape=f"T={tokens}, K={in_dim}, N={out_dim}",
        cpu_info=get_cpu_info(),
    )

    dx_diff = max_diff(torch.from_numpy(dx.copy()), dx_ref)
    dw_diff = max_diff(torch.from_numpy(dw.copy()), dw_ref)
    db_diff = max_diff(torch.from_numpy(db.copy()), db_ref)

    report.add_result(TestResult(
        name="d_input",
        passed=dx_diff <= 2e-5,
        max_diff=dx_diff,
        tolerance=2e-5,
        pytorch_time=time_function(lambda: pytorch_ref()[0], warmup=warmup, iterations=iterations, name="PyTorch bwd"),
        kernel_time=time_function(c_kernel, warmup=warmup, iterations=iterations, name="C GEMM BF16 bwd"),
    ))
    report.add_result(TestResult(name="d_weight", passed=dw_diff <= 2e-5, max_diff=dw_diff, tolerance=2e-5))
    report.add_result(TestResult(name="d_bias", passed=db_diff <= 2e-5, max_diff=db_diff, tolerance=2e-5))
    return report


if __name__ == "__main__":
    print_system_info()
    report = run_tests()
    report.print_report()
    if not report.all_passed():
        sys.exit(1)
