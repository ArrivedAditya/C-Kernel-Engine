"""
Add kernel unit tests with PyTorch parity verification.

Tests element-wise addition kernels (for residual connections):
  - add_forward_bf16: y = a + b
  - add_scaled_forward_bf16: y = a + alpha * b
  - add_inplace_bf16: a += b
  - add_backward_bf16: gradient passthrough
  - add_forward_2d_bf16: 2D tensor version

Reports accuracy against PyTorch reference and timing.
"""
import ctypes
import os
import sys

import numpy as np
try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    print("TEST SKIPPED: PyTorch BF16 storage oracle is unavailable")
    sys.exit(0)

from lib_loader import load_lib
from test_utils import (
    TestReport, TestResult, get_cpu_info,
    max_diff, numpy_to_ptr, time_function, print_system_info
)


# Load the library
lib = load_lib("libckernel_add.so", "libckernel_engine.so")


# =============================================================================
# BF16 utility functions (match C implementation)
# =============================================================================

def float_to_bf16(x: np.ndarray) -> np.ndarray:
    """Use PyTorch's round-to-nearest-even BF16 storage as the oracle."""
    x_f32 = np.ascontiguousarray(x, dtype=np.float32)
    return torch.from_numpy(x_f32).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)


def bf16_to_float(x: np.ndarray) -> np.ndarray:
    """Convert bf16 (stored as uint16) back to float32."""
    x_u32 = x.astype(np.uint32) << 16
    return x_u32.view(np.float32)


# =============================================================================
# Function signatures
# =============================================================================

# void add_forward_bf16(const uint16_t *a, const uint16_t *b, uint16_t *y, size_t n)
lib.add_forward_bf16.argtypes = [
    ctypes.POINTER(ctypes.c_uint16),  # a
    ctypes.POINTER(ctypes.c_uint16),  # b
    ctypes.POINTER(ctypes.c_uint16),  # y
    ctypes.c_size_t,                   # n
]
lib.add_forward_bf16.restype = None

# void add_scaled_forward_bf16(const uint16_t *a, const uint16_t *b, uint16_t *y, float alpha, size_t n)
lib.add_scaled_forward_bf16.argtypes = [
    ctypes.POINTER(ctypes.c_uint16),  # a
    ctypes.POINTER(ctypes.c_uint16),  # b
    ctypes.POINTER(ctypes.c_uint16),  # y
    ctypes.c_float,                    # alpha
    ctypes.c_size_t,                   # n
]
lib.add_scaled_forward_bf16.restype = None

# void add_inplace_bf16(uint16_t *a, const uint16_t *b, size_t n)
lib.add_inplace_bf16.argtypes = [
    ctypes.POINTER(ctypes.c_uint16),  # a
    ctypes.POINTER(ctypes.c_uint16),  # b
    ctypes.c_size_t,                   # n
]
lib.add_inplace_bf16.restype = None

# void add_backward_bf16(const uint16_t *d_y, uint16_t *d_a, uint16_t *d_b, size_t n)
lib.add_backward_bf16.argtypes = [
    ctypes.POINTER(ctypes.c_uint16),  # d_y
    ctypes.POINTER(ctypes.c_uint16),  # d_a
    ctypes.POINTER(ctypes.c_uint16),  # d_b
    ctypes.c_size_t,                   # n
]
lib.add_backward_bf16.restype = None

# void add_forward_2d_bf16(const uint16_t *a, const uint16_t *b, uint16_t *y, int tokens, int dim, int aligned_dim)
lib.add_forward_2d_bf16.argtypes = [
    ctypes.POINTER(ctypes.c_uint16),  # a
    ctypes.POINTER(ctypes.c_uint16),  # b
    ctypes.POINTER(ctypes.c_uint16),  # y
    ctypes.c_int,                      # tokens
    ctypes.c_int,                      # dim
    ctypes.c_int,                      # aligned_dim
]
lib.add_forward_2d_bf16.restype = None

# FP32 versions for comparison
lib.add_forward_f32.argtypes = [
    ctypes.POINTER(ctypes.c_float),   # a
    ctypes.POINTER(ctypes.c_float),   # b
    ctypes.POINTER(ctypes.c_float),   # y
    ctypes.c_size_t,                   # n
]
lib.add_forward_f32.restype = None


def numpy_to_ptr_u16(arr: np.ndarray):
    """Get ctypes pointer to uint16 numpy array."""
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))


def numpy_to_ptr_f32(arr: np.ndarray):
    """Get ctypes pointer to float32 numpy array."""
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


# =============================================================================
# Test: Forward BF16 (y = a + b)
# =============================================================================

def test_forward_bf16(N=4096, warmup=10, iterations=100):
    """Test add_forward_bf16 against PyTorch."""
    print("\n" + "=" * 60)
    print("TEST: add_forward_bf16 (y = a + b)")
    print("=" * 60)

    np.random.seed(42)

    # Generate random test data
    a_f32 = np.random.randn(N).astype(np.float32)
    b_f32 = np.random.randn(N).astype(np.float32)

    # Convert to BF16
    a_bf16 = float_to_bf16(a_f32)
    b_bf16 = float_to_bf16(b_f32)
    y_bf16 = np.zeros(N, dtype=np.uint16)

    # Compare stored BF16 values, not the unrounded FP32 sum.
    a_bf16_f32 = bf16_to_float(a_bf16)
    b_bf16_f32 = bf16_to_float(b_bf16)

    # PyTorch reference (using actual BF16 values)
    a_torch = torch.from_numpy(a_bf16_f32)
    b_torch = torch.from_numpy(b_bf16_f32)
    ref = float_to_bf16((a_torch + b_torch).numpy())

    # C kernel
    lib.add_forward_bf16(
        numpy_to_ptr_u16(a_bf16),
        numpy_to_ptr_u16(b_bf16),
        numpy_to_ptr_u16(y_bf16),
        ctypes.c_size_t(N)
    )

    # Convert result back to float for comparison
    y_f32 = bf16_to_float(y_bf16)
    ref_f32 = bf16_to_float(ref)

    # Compute error
    diff = np.abs(y_f32 - ref_f32)
    max_err = np.max(diff)
    mean_err = np.mean(diff)
    rel_err = max_err / (np.max(np.abs(ref_f32)) + 1e-10)

    print(f"  Shape:     N={N}")
    print(f"  Max diff:  {max_err:.2e}")
    print(f"  Mean diff: {mean_err:.2e}")
    print(f"  Rel error: {rel_err:.2e}")

    # Timing
    def c_add():
        lib.add_forward_bf16(
            numpy_to_ptr_u16(a_bf16),
            numpy_to_ptr_u16(b_bf16),
            numpy_to_ptr_u16(y_bf16),
            ctypes.c_size_t(N)
        )

    def torch_add():
        return a_torch + b_torch

    # Warmup
    for _ in range(warmup):
        c_add()
        torch_add()

    # Time C kernel
    import time
    start = time.perf_counter()
    for _ in range(iterations):
        c_add()
    c_time = (time.perf_counter() - start) / iterations * 1e6  # microseconds

    # Time PyTorch
    start = time.perf_counter()
    for _ in range(iterations):
        torch_add()
    torch_time = (time.perf_counter() - start) / iterations * 1e6

    print(f"  C kernel:  {c_time:.2f} us")
    print(f"  PyTorch:   {torch_time:.2f} us")
    print(f"  Speedup:   {torch_time/c_time:.2f}x")

    # Pass/Fail
    passed = np.array_equal(y_bf16, ref)
    status = "PASS" if passed else "FAIL"
    print(f"\n  Status: {status}")

    return passed


# =============================================================================
# Test: Scaled Forward BF16 (y = a + alpha * b)
# =============================================================================

def test_scaled_forward_bf16(N=4096, alpha=0.5):
    """Test add_scaled_forward_bf16 against PyTorch."""
    print("\n" + "=" * 60)
    print(f"TEST: add_scaled_forward_bf16 (y = a + {alpha} * b)")
    print("=" * 60)

    np.random.seed(42)

    a_f32 = np.random.randn(N).astype(np.float32)
    b_f32 = np.random.randn(N).astype(np.float32)

    a_bf16 = float_to_bf16(a_f32)
    b_bf16 = float_to_bf16(b_f32)
    y_bf16 = np.zeros(N, dtype=np.uint16)

    a_bf16_f32 = bf16_to_float(a_bf16)
    b_bf16_f32 = bf16_to_float(b_bf16)

    # PyTorch reference
    ref = float_to_bf16(a_bf16_f32 + alpha * b_bf16_f32)

    # C kernel
    lib.add_scaled_forward_bf16(
        numpy_to_ptr_u16(a_bf16),
        numpy_to_ptr_u16(b_bf16),
        numpy_to_ptr_u16(y_bf16),
        ctypes.c_float(alpha),
        ctypes.c_size_t(N)
    )

    y_f32 = bf16_to_float(y_bf16)

    diff = np.abs(y_f32 - bf16_to_float(ref))
    max_err = np.max(diff)
    mean_err = np.mean(diff)

    print(f"  Shape:     N={N}, alpha={alpha}")
    print(f"  Max diff:  {max_err:.2e}")
    print(f"  Mean diff: {mean_err:.2e}")

    passed = np.array_equal(y_bf16, ref)
    status = "PASS" if passed else "FAIL"
    print(f"  Status: {status}")

    return passed


# =============================================================================
# Test: Inplace BF16 (a += b)
# =============================================================================

def test_inplace_bf16(N=4096):
    """Test add_inplace_bf16 against PyTorch."""
    print("\n" + "=" * 60)
    print("TEST: add_inplace_bf16 (a += b)")
    print("=" * 60)

    np.random.seed(42)

    a_f32 = np.random.randn(N).astype(np.float32)
    b_f32 = np.random.randn(N).astype(np.float32)

    a_bf16 = float_to_bf16(a_f32)
    b_bf16 = float_to_bf16(b_f32)

    a_bf16_f32 = bf16_to_float(a_bf16)
    b_bf16_f32 = bf16_to_float(b_bf16)

    # PyTorch reference
    ref = float_to_bf16(a_bf16_f32 + b_bf16_f32)

    # C kernel (modifies a in-place)
    a_bf16_copy = a_bf16.copy()
    lib.add_inplace_bf16(
        numpy_to_ptr_u16(a_bf16_copy),
        numpy_to_ptr_u16(b_bf16),
        ctypes.c_size_t(N)
    )

    y_f32 = bf16_to_float(a_bf16_copy)

    diff = np.abs(y_f32 - bf16_to_float(ref))
    max_err = np.max(diff)

    print(f"  Shape:     N={N}")
    print(f"  Max diff:  {max_err:.2e}")

    passed = np.array_equal(a_bf16_copy, ref)
    status = "PASS" if passed else "FAIL"
    print(f"  Status: {status}")

    return passed


# =============================================================================
# Test: Backward BF16 (gradient passthrough)
# =============================================================================

def test_backward_bf16(N=4096):
    """Test add_backward_bf16 - gradients should pass through unchanged."""
    print("\n" + "=" * 60)
    print("TEST: add_backward_bf16 (gradient passthrough)")
    print("=" * 60)

    np.random.seed(42)

    # For y = a + b:
    #   d_a = d_y (gradient passes through)
    #   d_b = d_y (gradient passes through)

    d_y_f32 = np.random.randn(N).astype(np.float32)
    d_y_bf16 = float_to_bf16(d_y_f32)

    d_a_bf16 = np.zeros(N, dtype=np.uint16)
    d_b_bf16 = np.zeros(N, dtype=np.uint16)

    # C kernel
    lib.add_backward_bf16(
        numpy_to_ptr_u16(d_y_bf16),
        numpy_to_ptr_u16(d_a_bf16),
        numpy_to_ptr_u16(d_b_bf16),
        ctypes.c_size_t(N)
    )

    # Both d_a and d_b should equal d_y
    d_a_f32 = bf16_to_float(d_a_bf16)
    d_b_f32 = bf16_to_float(d_b_bf16)
    d_y_actual = bf16_to_float(d_y_bf16)

    diff_a = np.abs(d_a_f32 - d_y_actual)
    diff_b = np.abs(d_b_f32 - d_y_actual)

    max_err_a = np.max(diff_a)
    max_err_b = np.max(diff_b)

    print(f"  Shape:        N={N}")
    print(f"  Max diff d_a: {max_err_a:.2e}")
    print(f"  Max diff d_b: {max_err_b:.2e}")

    # Should be exact (just a copy)
    passed = max_err_a == 0 and max_err_b == 0
    status = "PASS" if passed else "FAIL"
    print(f"  Status: {status}")

    return passed


# =============================================================================
# Test: 2D Forward BF16 (for transformer residuals)
# =============================================================================

def test_forward_2d_bf16(T=32, D=896, aligned_D=896):
    """Test add_forward_2d_bf16 with 2D tensor shapes."""
    print("\n" + "=" * 60)
    print(f"TEST: add_forward_2d_bf16 (shape [{T}, {D}])")
    print("=" * 60)

    np.random.seed(42)

    # Generate [T, aligned_D] shaped data
    a_f32 = np.random.randn(T, aligned_D).astype(np.float32)
    b_f32 = np.random.randn(T, aligned_D).astype(np.float32)

    a_bf16 = float_to_bf16(a_f32.flatten()).reshape(T, aligned_D)
    b_bf16 = float_to_bf16(b_f32.flatten()).reshape(T, aligned_D)
    y_bf16 = np.full((T, aligned_D), 0xDEAD, dtype=np.uint16)

    a_bf16_f32 = bf16_to_float(a_bf16.flatten()).reshape(T, aligned_D)
    b_bf16_f32 = bf16_to_float(b_bf16.flatten()).reshape(T, aligned_D)

    # PyTorch reference (only compare first D elements per row)
    ref = float_to_bf16(a_bf16_f32[:, :D] + b_bf16_f32[:, :D])

    # C kernel
    output_flat = y_bf16.ravel()
    assert np.shares_memory(output_flat, y_bf16), "C must write into the observed destination"
    lib.add_forward_2d_bf16(
        numpy_to_ptr_u16(a_bf16.ravel()),
        numpy_to_ptr_u16(b_bf16.ravel()),
        numpy_to_ptr_u16(output_flat),
        ctypes.c_int(T),
        ctypes.c_int(D),
        ctypes.c_int(aligned_D)
    )

    y_f32 = bf16_to_float(y_bf16[:, :D])

    diff = np.abs(y_f32 - bf16_to_float(ref))
    max_err = np.max(diff)
    mean_err = np.mean(diff)

    print(f"  Shape:     [{T}, {D}] (aligned: {aligned_D})")
    print(f"  Max diff:  {max_err:.2e}")
    print(f"  Mean diff: {mean_err:.2e}")

    passed = np.array_equal(y_bf16[:, :D], ref) and np.all(y_bf16[:, D:] == 0xDEAD)
    status = "PASS" if passed else "FAIL"
    print(f"  Status: {status}")

    return passed


def test_bf16_rounding_boundaries():
    """Finite ties must round correctly in scalar, SIMD, and SIMD-tail stores."""
    base_a = np.array([1.0, 1.0078125, -1.0, -1.0078125], dtype=np.float32)
    base_b = np.array([1 / 256, 1 / 256, -1 / 256, -1 / 256], dtype=np.float32)
    base_expected = np.array([0x3F80, 0x3F82, 0xBF80, 0xBF82], dtype=np.uint16)
    passed = True
    for n in (4, 16, 17, 32, 33, 65):
        a = np.tile(base_a, (n + 3) // 4)[:n]
        b = np.tile(base_b, (n + 3) // 4)[:n]
        a_bits = float_to_bf16(a)
        b_bits = float_to_bf16(b)
        expected = np.tile(base_expected, (n + 3) // 4)[:n]
        assert np.array_equal(float_to_bf16(a + b), expected)

        output = np.full(n, 0xDEAD, dtype=np.uint16)
        lib.add_forward_bf16(numpy_to_ptr_u16(a_bits), numpy_to_ptr_u16(b_bits),
                             numpy_to_ptr_u16(output), ctypes.c_size_t(n))
        inplace = a_bits.copy()
        lib.add_inplace_bf16(numpy_to_ptr_u16(inplace), numpy_to_ptr_u16(b_bits),
                             ctypes.c_size_t(n))
        passed &= np.array_equal(output, expected) and np.array_equal(inplace, expected)
    print(f"  BF16 tie-to-even stored output: {'PASS' if passed else 'FAIL'}")
    return passed


def test_bf16_backward_copy_bounds():
    """Copy only n elements, including at AVX-512 vector/tail boundaries."""
    guard = 8
    for n in (0, 1, 31, 32, 33, 63, 64, 65, 96):
        source = np.full(guard + n + 32, 0xA55A, dtype=np.uint16)
        source[guard:guard + n] = (np.arange(n, dtype=np.uint16) * 37 + 1)
        before_source = source.copy()
        outputs = [np.full_like(source, 0xDEAD) for _ in range(2)]
        lib.add_backward_bf16(
            numpy_to_ptr_u16(source[guard:]),
            numpy_to_ptr_u16(outputs[0][guard:]),
            numpy_to_ptr_u16(outputs[1][guard:]),
            ctypes.c_size_t(n),
        )
        if not np.array_equal(source, before_source):
            return False
        for output in outputs:
            if not (np.all(output[:guard] == 0xDEAD)
                    and np.array_equal(output[guard:guard + n], source[guard:guard + n])
                    and np.all(output[guard + n:] == 0xDEAD)):
                return False
    print("  BF16 backward vector/tail bounds: PASS")
    return True


def test_bf16_edge_shapes_and_negative_controls():
    """Zero length is a no-op; scalar and SIMD-tail lengths store exact values."""
    a = float_to_bf16(np.linspace(-1.0, 1.0, 33, dtype=np.float32))
    b = float_to_bf16(np.linspace(0.25, -0.25, 33, dtype=np.float32))
    expected = float_to_bf16(bf16_to_float(a) + bf16_to_float(b))
    for n in (0, 1, 17, 33):
        output = np.full(33, 0xDEAD, dtype=np.uint16)
        lib.add_forward_bf16(numpy_to_ptr_u16(a), numpy_to_ptr_u16(b),
                             numpy_to_ptr_u16(output), ctypes.c_size_t(n))
        if not (np.array_equal(output[:n], expected[:n])
                and np.all(output[n:] == 0xDEAD)):
            return False

    # A no-op kernel and a truncating store must both fail the same bitwise
    # assertion used above. Nonfinite payload semantics are a separate contract.
    a_tie = np.array([1.0078125], dtype=np.float32)
    b_tie = np.array([1 / 256], dtype=np.float32)
    rounded = float_to_bf16(a_tie + b_tie)
    truncated = ((a_tie + b_tie).view(np.uint32) >> 16).astype(np.uint16)
    stale = np.full(1, 0xDEAD, dtype=np.uint16)
    passed = (not np.array_equal(stale, rounded)
              and not np.array_equal(truncated, rounded))
    print(f"  BF16 stale-output/truncation controls: {'PASS' if passed else 'FAIL'}")
    return passed


# =============================================================================
# Test: FP32 version
# =============================================================================

def test_forward_f32(N=4096):
    """Test add_forward_f32 against PyTorch."""
    print("\n" + "=" * 60)
    print("TEST: add_forward_f32 (y = a + b)")
    print("=" * 60)

    np.random.seed(42)

    a_f32 = np.random.randn(N).astype(np.float32)
    b_f32 = np.random.randn(N).astype(np.float32)
    y_f32 = np.zeros(N, dtype=np.float32)

    # PyTorch reference
    ref = a_f32 + b_f32

    # C kernel
    lib.add_forward_f32(
        numpy_to_ptr_f32(a_f32),
        numpy_to_ptr_f32(b_f32),
        numpy_to_ptr_f32(y_f32),
        ctypes.c_size_t(N)
    )

    diff = np.abs(y_f32 - ref)
    max_err = np.max(diff)

    print(f"  Shape:     N={N}")
    print(f"  Max diff:  {max_err:.2e}")

    # Should be exact for FP32
    passed = max_err < 1e-6
    status = "PASS" if passed else "FAIL"
    print(f"  Status: {status}")

    return passed


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("ADD KERNEL UNIT TESTS")
    print("=" * 60)

    cpu_info = get_cpu_info()
    print(f"\nCPU: {cpu_info.model_name}")
    print(f"SIMD: {cpu_info.best_simd}")

    results = []

    # Run all tests
    results.append(("add_forward_bf16", test_forward_bf16()))
    results.append(("add_scaled_forward_bf16", test_scaled_forward_bf16()))
    results.append(("add_inplace_bf16", test_inplace_bf16()))
    results.append(("add_backward_bf16", test_backward_bf16()))
    results.append(("add_forward_2d_bf16", test_forward_2d_bf16()))
    results.append(("add_forward_2d_bf16_padded", test_forward_2d_bf16(T=4, D=37, aligned_D=48)))
    results.append(("add_forward_2d_bf16_small", test_forward_2d_bf16(T=1, D=1, aligned_D=4)))
    results.append(("bf16_rounding_boundaries", test_bf16_rounding_boundaries()))
    results.append(("bf16_backward_copy_bounds", test_bf16_backward_copy_bounds()))
    results.append(("bf16_edge_shapes_and_controls", test_bf16_edge_shapes_and_negative_controls()))
    results.append(("add_forward_f32", test_forward_f32()))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name:30s} {status}")

    print(f"\n  Total: {passed}/{total} passed")

    if passed == total:
        print("\n  ALL TESTS PASSED!")
        return 0
    else:
        print("\n  SOME TESTS FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
