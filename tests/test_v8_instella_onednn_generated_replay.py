"""Asset-backed normal generated Instella replay for the candidate oneDNN map.

Set CK_INSTELLA_ONEDNN_BUNDLE and CK_INSTELLA_ONEDNN_CHECKPOINT to execute.
Numerical disagreement is a failing certification, never a successful fixture.
"""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(os.environ.get("CK_TEST_REPO_ROOT", Path(__file__).resolve().parents[1]))
PROVIDER = "moe_swiglu_expert_forward_pytorch_onednn_3_12_bf16_storage_checked"


class InstellaGeneratedOneDNNReplay(unittest.TestCase):
    def test_circuit_generated_three_token_execution_and_parity(self):
        bundle = Path(os.getenv("CK_INSTELLA_ONEDNN_BUNDLE", ""))
        checkpoint = Path(os.getenv("CK_INSTELLA_ONEDNN_CHECKPOINT", ""))
        required = os.getenv("CK_REQUIRE_ONEDNN_GENERATED_REPLAY") == "1"
        missing = []
        if not (bundle / "model_v8.c").is_file():
            missing.append("CK_INSTELLA_ONEDNN_BUNDLE")
        if not (checkpoint / "config.json").is_file():
            missing.append("CK_INSTELLA_ONEDNN_CHECKPOINT")
        if not Path(os.getenv("CK_SLEEF_LIBRARY", "")).is_file():
            missing.append("CK_SLEEF_LIBRARY")
        library = Path(os.getenv("CK_ONEDNN_LIBRARY", ""))
        if not library.is_file():
            missing.append("CK_ONEDNN_LIBRARY")
        cpu = Path("/proc/cpuinfo")
        if not cpu.is_file() or "avx512_bf16" not in cpu.read_text():
            missing.append("AVX512_BF16 CPU")
        if missing:
            reason = "pinned generated replay unavailable: " + ", ".join(missing)
            if required:
                self.fail(reason)
            self.skipTest(reason)

        prefill = json.loads((bundle / "lowered_prefill_call.json").read_text())
        provider_ops = [
            op for op in prefill["operations"] if op.get("function") == PROVIDER
        ]
        self.assertTrue(provider_ops, "normal prefill IR did not select the candidate map")
        self.assertIn(PROVIDER + "(", (bundle / "model_v8.c").read_text())
        self.assertTrue((bundle / "libckernel_engine.so").is_file())
        linked = subprocess.run(
            ["ldd", str(bundle / "libckernel_engine.so")],
            capture_output=True, text=True, check=True,
        ).stdout
        dnnl_lines = [line for line in linked.splitlines() if "libdnnl.so" in line]
        self.assertEqual(len(dnnl_lines), 1, linked)
        backing = Path(dnnl_lines[0].split("=>", 1)[1].split("(", 1)[0].strip())
        self.assertEqual(backing.resolve(), library.resolve(),
                         "generated engine does not load the pinned oneDNN library")

        with tempfile.TemporaryDirectory() as td:
            output = Path(td)
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "8"
            env["DNNL_MAX_CPU_ISA"] = "AVX512_CORE_BF16"
            env["LD_LIBRARY_PATH"] = str(library.parent) + os.pathsep + env.get(
                "LD_LIBRARY_PATH", ""
            )
            command = [
                sys.executable,
                str(ROOT / "version/v8/scripts/xray_decoder_pytorch_v8.py"),
                "--checkpoint", str(checkpoint),
                "--runtime", str(bundle),
                "--call-ir", str(bundle / "lowered_prefill_call.json"),
                "--token-ids", "1,5,7",
                "--output-dir", str(output),
                "--threads", "8",
                "--mla-replay-layer", "2",
            ]
            run = subprocess.run(
                command, cwd=ROOT, env=env, capture_output=True, text=True,
                timeout=600,
            )
            self.assertEqual(run.returncode, 0, run.stdout[-3000:] + run.stderr[-3000:])
            summary = json.loads((output / "xray_summary.json").read_text())
            import numpy as np

            comparisons = {}
            for layer, boundary in ((1, "layer_out"), (2, "moe_routed_output"), (2, "layer_out")):
                stem = f"layer_{layer:03d}_{boundary}_full.f32"
                actual = np.fromfile(output / "ck" / stem, dtype=np.float32)
                expected = np.fromfile(output / "pytorch" / stem, dtype=np.float32)
                self.assertEqual(actual.shape, expected.shape)
                self.assertTrue(np.isfinite(actual).all())
                self.assertTrue(np.isfinite(expected).all())
                comparisons[f"layer_{layer}_{boundary}"] = {
                    "exact": int(np.count_nonzero(actual == expected)),
                    "total": int(actual.size),
                }
            numerical_pass = all(
                item["exact"] == item["total"] for item in comparisons.values()
            ) and summary.get("numerical_status") == "pass"
            print("GENERATED_ONEDNN_REPLAY " + json.dumps({
                "generated_execution": "pass",
                "numerical_parity": "pass" if numerical_pass else "fail",
                "comparisons": comparisons,
                "full_model_numerical_status": summary.get("numerical_status"),
                "token_ids": [1, 5, 7],
                "omp_threads": 8,
                "max_cpu_isa": env["DNNL_MAX_CPU_ISA"],
                "onednn_library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            }, sort_keys=True))
            self.assertTrue(numerical_pass, "generated execution completed but numerical parity failed")


if __name__ == "__main__":
    unittest.main()
