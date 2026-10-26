"""Pinned same-input replay of Instella's first multirow expert mismatch.

The optional Ryzen lane executes the current C provider against captured
full-width X-Ray operands and publisher output. Missing assets are SKIP unless
CK_REQUIRE_ONEDNN_GROUPING_ORACLE=1. The old M=1 reduction is a negative control.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/instella_onednn_expert"
LAYER = 2
THREADS = 8


def _child(provider: Path, bundle: Path, xray: Path, onednn: Path) -> None:
    import numpy as np
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(THREADS)
    prefix = f"layer_{LAYER:03d}_"

    def capture(side: str, name: str, shape: tuple[int, ...]) -> np.ndarray:
        values = np.fromfile(xray / side / f"{prefix}{name}.f32", dtype=np.float32)
        return np.ascontiguousarray(values.reshape(shape))

    hidden = capture("ck", "ffn_norm_full", (3, 2048))
    publisher_hidden = capture("pytorch", "ffn_norm_full", (3, 2048))
    assert np.array_equal(hidden, publisher_hidden), "expert inputs differ before replay"
    ids = capture("ck", "moe_selected_experts_full", (3, 6)).astype(np.int32)
    routes = capture("ck", "moe_routing_weights_full", (3, 6))
    publisher_ids = capture("pytorch", "moe_selected_experts_full", (3, 6)).astype(np.int32)
    publisher_routes = capture("pytorch", "moe_routing_weights_full", (3, 6))
    route_max_abs = 0.0
    for row in range(3):
        actual = sorted(zip(ids[row].tolist(), routes[row].tolist()))
        reference = sorted(zip(publisher_ids[row].tolist(), publisher_routes[row].tolist()))
        assert [expert for expert, _ in actual] == [expert for expert, _ in reference], \
            f"selected experts differ at row {row}"
        route_max_abs = max(route_max_abs, max(
            abs(a[1] - b[1]) for a, b in zip(actual, reference)
        ))
    assert route_max_abs < 1e-6, "routing weights differ beyond the captured boundary"
    expected = capture("pytorch", "moe_routed_output_full", (3, 2048))
    old_output = capture("ck", "moe_routed_output_full", (3, 2048))

    manifest = json.loads((bundle / "weights_manifest.json").read_text())
    entries = {item["name"]: item for item in manifest["entries"]}
    weights = {}
    for kind in ("gate", "up", "down"):
        entry = entries[f"layer.{LAYER}.moe_expert_{kind}"]
        assert entry["dtype"] == "bf16"
        weights[kind] = np.memmap(
            bundle / "weights.bump", dtype=np.uint16, mode="r",
            offset=entry["file_offset"], shape=tuple(entry["shape"]),
        )

    model = ctypes.CDLL(str(provider))
    function = model.ck_test_real_expert
    function.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 5 + [
        ctypes.c_void_p, ctypes.c_size_t,
    ]
    function.restype = ctypes.c_int
    assert str(onednn.resolve()) in Path("/proc/self/maps").read_text(), \
        "pinned oneDNN backing is not loaded"
    rows, h, i, experts, top_k = 3, 2048, 1408, 64, 6
    workspace_bytes = rows * (12 * h + 16 * i + 8) + 8 * i
    raw = np.empty(workspace_bytes + 64, dtype=np.uint8)
    workspace = (raw.ctypes.data + 63) & ~63
    output = np.full((rows, h), -999.0, dtype=np.float32)
    status = function(
        hidden.ctypes.data, ids.ctypes.data, routes.ctypes.data,
        weights["gate"].ctypes.data, weights["up"].ctypes.data,
        weights["down"].ctypes.data, output.ctypes.data,
        rows, h, i, experts, top_k, workspace, workspace_bytes,
    )
    assert status == 0, f"grouped provider rejected pinned inputs: {status}"
    publisher_operands_output = np.full((rows, h), -999.0, dtype=np.float32)
    status = function(
        hidden.ctypes.data, publisher_ids.ctypes.data,
        publisher_routes.ctypes.data, weights["gate"].ctypes.data,
        weights["up"].ctypes.data, weights["down"].ctypes.data,
        publisher_operands_output.ctypes.data, rows, h, i, experts, top_k,
        workspace, workspace_bytes,
    )
    assert status == 0, "grouped provider rejected publisher routing operands"

    def reference(grouped: bool) -> np.ndarray:
        inputs = torch.from_numpy(hidden.copy()).to(torch.bfloat16)
        result = torch.zeros((rows, h), dtype=torch.bfloat16)
        for expert in sorted(set(publisher_ids.flat)):
            assignments = [
                (row, slot) for slot in range(top_k) for row in range(rows)
                if publisher_ids[row, slot] == expert
            ]
            batches = [assignments] if grouped else [[item] for item in assignments]
            gate = torch.from_numpy(np.array(weights["gate"][expert])).view(torch.bfloat16)
            up = torch.from_numpy(np.array(weights["up"][expert])).view(torch.bfloat16)
            down = torch.from_numpy(np.array(weights["down"][expert])).view(torch.bfloat16)
            joined = torch.cat((gate, up), dim=0)
            for batch in batches:
                current = inputs[[row for row, _slot in batch]]
                projected_gate, projected_up = F.linear(current, joined).chunk(2, dim=-1)
                produced = F.linear(F.silu(projected_gate) * projected_up, down)
                for index, (row, slot) in enumerate(batch):
                    weight = torch.tensor(float(publisher_routes[row, slot]), dtype=torch.float32)
                    result[row] += (produced[index] * weight).to(torch.bfloat16)
        return result.float().numpy()

    grouped = reference(True)
    rowwise = reference(False)
    counts = {
        "input_exact": int(np.count_nonzero(hidden == publisher_hidden)),
        "grouped_c_vs_publisher_exact": int(np.count_nonzero(output == expected)),
        "grouped_c_on_publisher_operands_exact": int(np.count_nonzero(publisher_operands_output == expected)),
        "grouped_reference_vs_publisher_exact": int(np.count_nonzero(grouped == expected)),
        "rowwise_reference_vs_publisher_exact": int(np.count_nonzero(rowwise == expected)),
        "rowwise_reference_vs_old_ck_exact": int(np.count_nonzero(rowwise == old_output)),
        "total": int(expected.size),
    }
    print("INSTELLA_GROUPING_ORACLE " + json.dumps({
        "status": "PASS" if (
            counts["grouped_c_vs_publisher_exact"] == counts["total"]
            and counts["grouped_c_on_publisher_operands_exact"] == counts["total"]
            and counts["grouped_reference_vs_publisher_exact"] == counts["total"]
            and counts["rowwise_reference_vs_publisher_exact"] < counts["total"]
            and counts["rowwise_reference_vs_old_ck_exact"] == counts["total"]
        ) else "FAIL",
        "counts": counts,
        "oneDNN_sha256": hashlib.sha256(onednn.read_bytes()).hexdigest(),
        "captured_bundle_source_sha256": hashlib.sha256((bundle / "model_v8.c").read_bytes()).hexdigest(),
        "compiled_provider_source_sha256": os.environ["CK_COMPILED_PROVIDER_SOURCE_SHA256"],
        "tokens": [1, 5, 7], "layer": LAYER,
        "threads": THREADS, "max_cpu_isa": os.environ["DNNL_MAX_CPU_ISA"],
        "route_weight_max_abs_difference": route_max_abs,
    }, sort_keys=True))
    assert counts["grouped_c_vs_publisher_exact"] == counts["total"]
    assert counts["grouped_c_on_publisher_operands_exact"] == counts["total"]
    assert counts["grouped_reference_vs_publisher_exact"] == counts["total"]
    assert counts["rowwise_reference_vs_publisher_exact"] < counts["total"]
    assert counts["rowwise_reference_vs_old_ck_exact"] == counts["total"]


class InstellaOneDNNGroupingOracleTests(unittest.TestCase):
    def test_captured_three_row_expert_boundary(self) -> None:
        bundle = Path(os.getenv("CK_INSTELLA_ONEDNN_BUNDLE", ""))
        xray = Path(os.getenv("CK_INSTELLA_ONEDNN_GROUPING_XRAY", ""))
        onednn = Path(os.getenv("CK_ONEDNN_LIBRARY", ""))
        includes = os.getenv("CK_ONEDNN_INCLUDE", "")
        sleef = Path(os.getenv("CK_SLEEF_LIBRARY", ""))
        missing = []
        for name, path in (
            ("CK_INSTELLA_ONEDNN_BUNDLE", bundle / "weights.bump"),
            ("CK_INSTELLA_ONEDNN_GROUPING_XRAY", xray / "pytorch" / "layer_002_moe_selected_experts_full.f32"),
            ("CK_ONEDNN_LIBRARY", onednn),
            ("CK_SLEEF_LIBRARY", sleef),
        ):
            if not path.is_file():
                missing.append(name)
        if not includes:
            missing.append("CK_ONEDNN_INCLUDE")
        if not Path("/proc/cpuinfo").is_file() or "avx512_bf16" not in Path("/proc/cpuinfo").read_text():
            missing.append("AVX512_BF16 CPU")
        try:
            import torch
            if not torch.__version__.startswith("2.13.0"):
                missing.append("PyTorch 2.13.0")
        except ImportError:
            missing.append("PyTorch")
        if missing:
            reason = "pinned grouping replay unavailable: " + ", ".join(missing)
            if os.getenv("CK_REQUIRE_ONEDNN_GROUPING_ORACLE") == "1":
                self.fail(reason)
            self.skipTest(reason)

        include_flags = [
            part if part.startswith("-I") else "-I" + part
            for part in shlex.split(includes)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            provider = Path(temporary) / "libgrouped_expert.so"
            command = [
                "cc", "-O2", "-shared", "-fPIC", "-fopenmp", "-DUSE_ONEDNN",
                "-fvisibility=hidden", "-ffunction-sections", "-fdata-sections",
                "-Wl,--gc-sections", "-I" + str(ROOT / "include"), *include_flags,
                str(ROOT / "src/kernels/axpy_kernels.c"),
                str(ROOT / "src/kernels/gemm_kernels_bf16.c"),
                str(ROOT / "src/kernels/swiglu_kernels.c"),
                str(FIXTURE / "real_provider_entry.c"),
                "-L" + str(onednn.parent), "-ldnnl",
                "-Wl,-rpath," + str(onednn.parent),
                "-lm", "-ldl", "-lpthread", "-o", str(provider),
            ]
            built = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = str(THREADS)
            env["DNNL_MAX_CPU_ISA"] = "AVX512_CORE_BF16"
            env["CK_COMPILED_PROVIDER_SOURCE_SHA256"] = hashlib.sha256(
                (ROOT / "src/kernels/axpy_kernels.c").read_bytes()
            ).hexdigest()
            env["LD_LIBRARY_PATH"] = str(onednn.parent) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
            result = subprocess.run(
                [sys.executable, __file__, "--child", str(provider), str(bundle),
                 str(xray), str(onednn)], env=env, capture_output=True, text=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('"status": "PASS"', result.stdout)
            print(result.stdout.strip())


if __name__ == "__main__":
    if len(sys.argv) == 6 and sys.argv[1] == "--child":
        _child(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]))
    else:
        unittest.main()
