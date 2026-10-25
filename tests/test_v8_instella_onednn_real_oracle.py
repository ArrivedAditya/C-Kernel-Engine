"""Optional pinned oneDNN arithmetic oracle for the routed BF16 expert provider.

The nightly row is SKIP when the pinned runtime is absent. Asset-equipped hosts
set CK_REQUIRE_ONEDNN_EXPERT_ORACLE=1 to make missing execution a failure.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(os.environ.get("CK_TEST_REPO_ROOT", Path(__file__).resolve().parents[1]))
FIXTURE = ROOT / "tests/fixtures/instella_onednn_expert"
THREADS = 8
SOURCE_HASH = "80afa71049cd69a3df32adcccb623b12cd7baa22"


def _provider_args(fn):
    fn.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 5 + [
        ctypes.c_void_p, ctypes.c_size_t,
    ]
    fn.restype = ctypes.c_int


def _run_child(library: Path, report_path: Path) -> None:
    import numpy as np
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(THREADS)
    rng = np.random.default_rng(930)
    h = i = 32
    experts, top_k, rows = 3, 2, 3
    hidden = rng.normal(0, 0.23, (rows, h)).astype(np.float32)
    gate = rng.normal(0, 0.19, (experts, i, h)).astype(np.float32)
    up = rng.normal(0, 0.17, (experts, i, h)).astype(np.float32)
    down = rng.normal(0, 0.21, (experts, h, i)).astype(np.float32)
    gate[:, 0, :] *= 3
    up[:, 0, :] *= 2
    down[:, :, 0] *= 4
    indices = np.array([[2, 0], [1, 2], [2, 1]], dtype=np.int32)
    routing = np.array(
        [[0.375, 0.625], [-0.875, 1.125], [0.499, 0.501]],
        dtype=np.float32,
    )

    def bf16_bits(values):
        return torch.from_numpy(values.copy()).to(torch.bfloat16).view(
            torch.uint16
        ).numpy().copy()

    gate_bits, up_bits, down_bits = (
        bf16_bits(gate), bf16_bits(up), bf16_bits(down)
    )
    model = ctypes.CDLL(str(library))
    loaded = Path(os.environ["CK_ONEDNN_LIBRARY"]).resolve()
    mappings = Path("/proc/self/maps").read_text()
    assert str(loaded) in mappings, "pinned oneDNN library was not loaded"
    run = model.ck_test_real_expert
    noop = model.ck_test_noop_expert
    _provider_args(run)
    _provider_args(noop)
    counter = ctypes.CDLL(None)
    counter.ck_alloc_counter_reset.restype = None
    counter.ck_alloc_counter_calls.restype = ctypes.c_ulonglong
    counter.ck_alloc_counter_bytes.restype = ctypes.c_ulonglong

    workspace_bytes = 8 * h + 16 * i
    raw = np.empty(workspace_bytes + 64, dtype=np.uint8)
    workspace = (raw.ctypes.data + 63) & ~63
    output = np.full((rows, h), -777, dtype=np.float32)

    def invoke(function, count=rows, weights=down_bits):
        return function(
            hidden.ctypes.data, indices.ctypes.data, routing.ctypes.data,
            gate_bits.ctypes.data, up_bits.ctypes.data, weights.ctypes.data,
            output.ctypes.data, count, h, i, experts, top_k,
            workspace, workspace_bytes,
        )

    counter.ck_alloc_counter_reset()
    assert invoke(noop) == 0
    baseline_calls = counter.ck_alloc_counter_calls()
    baseline_bytes = counter.ck_alloc_counter_bytes()
    assert baseline_calls == baseline_bytes == 0
    allocations = {}
    for phase in ("cold", "warm"):
        counter.ck_alloc_counter_reset()
        assert invoke(run) == 0
        allocations[phase] = {
            "malloc_calloc_realloc_calls": counter.ck_alloc_counter_calls(),
            "requested_bytes": counter.ck_alloc_counter_bytes(),
        }
    full_output = output.copy()

    x = torch.from_numpy(hidden.copy()).to(torch.bfloat16)
    g = torch.from_numpy(gate_bits.copy()).view(torch.bfloat16)
    u = torch.from_numpy(up_bits.copy()).view(torch.bfloat16)
    d = torch.from_numpy(down_bits.copy()).view(torch.bfloat16)

    def reference(grouped: bool):
        # DeepseekV3Experts groups assignments by expert, then slot, then row.
        acc = torch.zeros((rows, h), dtype=torch.bfloat16)
        for expert in range(experts):
            assignments = [
                (row, slot)
                for slot in range(top_k)
                for row in range(rows)
                if indices[row, slot] == expert
            ]
            batches = [assignments] if grouped else [
                [assignment] for assignment in assignments
            ]
            combined = torch.cat((g[expert], u[expert]), dim=0)
            for batch in batches:
                current = x[[row for row, _ in batch]]
                projection_gate, projection_up = F.linear(
                    current, combined
                ).chunk(2, dim=-1)
                activation = F.silu(projection_gate) * projection_up
                result = F.linear(activation, d[expert])
                for j, (row, slot) in enumerate(batch):
                    weight = torch.tensor(float(routing[row, slot]), dtype=torch.float32)
                    acc[row] += (result[j] * weight).to(torch.bfloat16)
        return acc.float().numpy()

    isolated = reference(False)
    grouped = reference(True)
    assert np.array_equal(full_output, isolated), "rowwise BF16 oracle mismatch"
    assert np.array_equal(full_output, grouped), "grouped multirow BF16 oracle mismatch"

    output.fill(-777)
    assert invoke(run, count=1) == 0
    assert np.array_equal(output[0], isolated[0]), "single-row BF16 oracle mismatch"

    altered = down_bits.copy()
    altered[2, :, 0] = np.uint16(0x4080)  # change a selected down-projection weight
    output.fill(-777)
    assert invoke(run, weights=altered) == 0
    changed = int(np.count_nonzero(output != full_output))
    assert changed > 0, "weight perturbation escaped the arithmetic gate"

    report = {
        "schema": "cke.instella_onednn_real_oracle.v1",
        "status": "PASS",
        "provider": "moe_swiglu_expert_forward_pytorch_onednn_3_12_bf16_storage_workspace",
        "onednn_source_hash": SOURCE_HASH,
        "loaded_onednn_sha256": hashlib.sha256(loaded.read_bytes()).hexdigest(),
        "loaded_onednn_backing_verified": True,
        "torch_version": torch.__version__,
        "omp_threads": THREADS,
        "max_cpu_isa": os.environ["DNNL_MAX_CPU_ISA"],
        "single_row_exact": h,
        "single_row_total": h,
        "rowwise_multirow_exact": full_output.size,
        "rowwise_multirow_total": full_output.size,
        "grouped_multirow_exact": full_output.size,
        "grouped_multirow_total": full_output.size,
        "perturbation_changed_elements": changed,
        "cke_staging_workspace_bytes": workspace_bytes,
        "backend_allocation_profile": {
            "scope": "glibc malloc/calloc/realloc requests during each three-row provider call; excludes aligned allocator APIs and retained cache",
            "ctypes_noop_calls": baseline_calls,
            "cold": allocations["cold"],
            "warm": allocations["warm"],
        },
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


class InstellaOneDNNRealOracleTests(unittest.TestCase):
    def test_real_backend_numerical_oracle(self):
        required = os.getenv("CK_REQUIRE_ONEDNN_EXPERT_ORACLE") == "1"
        library = Path(os.getenv("CK_ONEDNN_LIBRARY", ""))
        include_arg = os.getenv("CK_ONEDNN_INCLUDE", "")
        sleef = Path(os.getenv("CK_SLEEF_LIBRARY", ""))
        cpu = Path("/proc/cpuinfo")
        missing = []
        if not library.is_file():
            missing.append("CK_ONEDNN_LIBRARY")
        if not include_arg:
            missing.append("CK_ONEDNN_INCLUDE")
        if not sleef.is_file():
            missing.append("CK_SLEEF_LIBRARY")
        if not cpu.is_file() or "avx512_bf16" not in cpu.read_text():
            missing.append("AVX512_BF16 CPU")
        if platform.libc_ver()[0] != "glibc":
            missing.append("glibc allocator-profile interposer")
        try:
            import torch
            if not torch.__version__.startswith("2.13.0"):
                missing.append("PyTorch 2.13.0")
            elif not torch.backends.mkldnn.is_available():
                missing.append("PyTorch oneDNN")
        except ImportError:
            missing.append("PyTorch")
        if missing:
            reason = "pinned real backend unavailable: " + ", ".join(missing)
            if required:
                self.fail(reason)
            self.skipTest(reason)

        includes = shlex.split(include_arg)
        include_flags = [
            flag if flag.startswith("-I") else "-I" + flag for flag in includes
        ]
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            provider = tmp / "libreal_expert.so"
            counter = tmp / "liballoc_counter.so"
            compile_provider = [
                "cc", "-O2", "-shared", "-fPIC", "-fopenmp", "-DUSE_ONEDNN",
                "-fvisibility=hidden", "-ffunction-sections", "-fdata-sections",
                "-Wl,--gc-sections", "-I" + str(ROOT / "include"),
                *include_flags,
                str(ROOT / "src/kernels/axpy_kernels.c"),
                str(ROOT / "src/kernels/gemm_kernels_bf16.c"),
                str(ROOT / "src/kernels/swiglu_kernels.c"),
                str(FIXTURE / "real_provider_entry.c"),
                "-L" + str(library.parent), "-ldnnl",
                "-Wl,-rpath," + str(library.parent),
                "-lm", "-ldl", "-lpthread", "-o", str(provider),
            ]
            built = subprocess.run(compile_provider, capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            built = subprocess.run(
                ["cc", "-O2", "-shared", "-fPIC",
                 str(FIXTURE / "alloc_counter.c"), "-o", str(counter)],
                capture_output=True, text=True,
            )
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            report = tmp / "report.json"
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = str(THREADS)
            env["DNNL_MAX_CPU_ISA"] = "AVX512_CORE_BF16"
            env["LD_PRELOAD"] = str(counter)
            env["LD_LIBRARY_PATH"] = str(library.parent) + os.pathsep + env.get(
                "LD_LIBRARY_PATH", ""
            )
            result = subprocess.run(
                [sys.executable, str(Path(__file__)), "--child",
                 str(provider), str(report)],
                env=env, capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            payload = json.loads(report.read_text())
            self.assertEqual(payload["status"], "PASS")
            self.assertGreater(payload["perturbation_changed_elements"], 0)
            print("REAL_BACKEND_ORACLE " + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--child":
        _run_child(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        unittest.main()
