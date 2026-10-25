"""Asset-free lifecycle guard for the opt-in pinned oneDNN expert map."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "version/v8/kernel_maps/moe_swiglu_expert_forward_pytorch_onednn_3_12_bf16_storage.json"


class InstellaOneDNNExpertContractTests(unittest.TestCase):
    def test_map_requires_pinned_runtime_and_planner_workspace(self):
        data = json.loads(MAP.read_text())
        self.assertEqual(data["constraints"]["external_runtime"], "onednn_3_12_openmp")
        self.assertEqual(
            data["constraints"]["source_hash"],
            "80afa71049cd69a3df32adcccb623b12cd7baa22",
        )
        self.assertEqual(data["scratch"][0]["size_bytes"], "8 * H + 16 * I")
        self.assertEqual(data["scratch"][0]["alignment"], 64)
        self.assertEqual(
            data["impl"]["function"],
            "moe_swiglu_expert_forward_pytorch_onednn_3_12_bf16_storage_checked",
        )
        self.assertIn("-DUSE_ONEDNN", data["impl"]["variants"][0]["compile_flags"])

    def test_native_routing_and_failed_admission_leave_output_untouched(self):
        source = ROOT / "tests/fixtures/instella_onednn_expert/contract_probe.c"
        kernel = ROOT / "src/kernels/axpy_kernels.c"
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "probe"
            proc = subprocess.run(
                [
                    "cc", "-O2", "-ffunction-sections", "-fdata-sections",
                    "-I", str(ROOT / "include"), "-Wl,--gc-sections",
                    str(source), str(kernel), "-o", str(binary), "-lm",
                ],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            run = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("PASS:", run.stdout)


if __name__ == "__main__":
    unittest.main()
