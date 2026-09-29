"""Independent PyTorch oracle and bounds for FP32 adaptive LayerNorm."""

import ctypes
import json
import math
import os
import platform
import shlex
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/tts/adaptive_layer_norm_torch.json"
FLOAT = ctypes.c_float
POINTER = ctypes.POINTER(FLOAT)


def flat(value):
    if isinstance(value, list):
        return [item for part in value for item in flat(part)]
    return [value]


class AdaptiveLayerNormOracleTest(unittest.TestCase):
    def assert_pinned_torch(self, torch):
        expected = os.environ.get("CKE_EXPECTED_TORCH_VERSION")
        if expected:
            self.assertEqual(torch.__version__.split("+", 1)[0], expected)
            self.assertIsNone(torch.version.cuda, "nightly oracle must use CPU wheels")
        expected_mkl = os.environ.get("CKE_EXPECTED_MKL_CBWR")
        if expected_mkl:
            self.assertEqual(os.environ.get("MKL_CBWR"), expected_mkl)
            if not torch.backends.mkl.is_available():
                self.skipTest("requested historical MKL reference environment is unavailable")

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        functions = []
        for optimization in ("-O0", "-O3"):
            library = Path(cls.temp.name) / f"libaudio_adaln_{optimization}.so"
            subprocess.run([
                "cc", "-std=c11", optimization, "-Wall", "-Wextra",
                "-Werror", "-pedantic", "-shared", "-fPIC", "-I",
                str(ROOT / "include"),
                str(ROOT / "src/kernels/audio_adaptive_layer_norm.c"),
                "-o", str(library), "-lm",
            ], check=True)
            function = ctypes.CDLL(str(library)).audio_adaptive_layer_norm_f32
            function.argtypes = [POINTER, ctypes.c_size_t] * 6 + [
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_size_t, ctypes.c_size_t, FLOAT,
            ]
            function.restype = ctypes.c_int
            functions.append(function)
        cls.fn, cls.optimized = functions

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def args_for(self, case):
        tokens, channels, style_dim = (case[key] for key in
                                        ("tokens", "channels", "style_dim"))
        input_stride, output_stride = channels + 3, channels + 2
        source = [777.0] * (tokens * input_stride)
        for row, values in enumerate(case["input"]):
            source[row * input_stride:row * input_stride + channels] = values
        values = [source, case["style"], case["projection_weight"],
                  case["projection_bias"], [-999.0] * (tokens * output_stride),
                  [13.0] * (2 * channels)]
        args = []
        for data in values:
            entries = flat(data)
            args.extend(((FLOAT * len(entries))(*entries), len(entries)))
        args[11] *= ctypes.sizeof(FLOAT)
        return args + [tokens, channels, style_dim, input_stride,
                       output_stride, case["epsilon"]]

    def assert_matches(self, case, tolerance=5e-6, *, comparison="native-vs-fixture", backend="2.8.0-fixture"):
        args = self.args_for(case)
        previous = None
        worst = (0.0, None)
        for _ in range(2):
            self.assertEqual(self.fn(*args), 0)
            for row, expected_row in enumerate(case["output"]):
                offset = row * args[-2]
                for channel, expected in enumerate(expected_row):
                    actual = args[8][offset + channel]
                    self.assertTrue(math.isfinite(actual), (row, channel))
                    self.assertTrue(math.isfinite(expected), (row, channel, "oracle"))
                    error = abs(actual - expected)
                    if error > worst[0]:
                        worst = (error, (row, channel, actual, expected))
                self.assertEqual(list(args[8][offset + len(expected_row):
                                               offset + args[-2]]),
                                 [-999.0] * (args[-2] - len(expected_row)))
            valid = [args[8][row * args[-2] + channel]
                     for row in range(case["tokens"])
                     for channel in range(case["channels"])]
            if previous is not None:
                self.assertEqual(valid, previous)
            previous = valid
        record = self.case_evidence(case, comparison, backend, worst[0], tolerance)
        self.assertEqual(record["status"], "pass", worst)
        print(f"adaptive_layer_norm shape={case['tokens']}x{case['channels']} "
              f"style={case['style_dim']} max_abs={worst[0]:.9g} "
              f"worst={worst[1]} tolerance={tolerance:.9g}")

    def test_committed_torch_fixture(self):
        fixture = json.loads(FIXTURE.read_text())
        for case in fixture["cases"]:
            with self.subTest(shape=(case["tokens"], case["channels"],
                                     case["style_dim"])):
                self.assert_matches(case)

    def test_optimized_matches_scalar(self):
        for case in json.loads(FIXTURE.read_text())["cases"]:
            scalar = self.args_for(case)
            optimized = self.args_for(case)
            self.assertEqual(self.fn(*scalar), 0)
            self.assertEqual(self.optimized(*optimized), 0)
            for row in range(case["tokens"]):
                for channel in range(case["channels"]):
                    index = row * scalar[-2] + channel
                    self.assertLessEqual(abs(scalar[8][index] - optimized[8][index]),
                                         1e-6, (row, channel))

    def test_rejection_preserves_output(self):
        case = json.loads(FIXTURE.read_text())["cases"][1]
        for capacity_index in (1, 3, 5, 7, 9, 11):
            with self.subTest(capacity_index=capacity_index):
                args = self.args_for(case)
                args[capacity_index] = 0
                self.assertEqual(self.fn(*args), -3)
                self.assertEqual(list(args[8]), [-999.0] * len(args[8]))
        for index, invalid in ((12, 0), (13, 0), (14, 0),
                               (15, 0), (16, 0), (17, 0.0),
                               (15, ctypes.c_size_t(-1).value),
                               (16, ctypes.c_size_t(-1).value)):
            with self.subTest(index=index, invalid=invalid):
                args = self.args_for(case)
                args[index] = invalid
                self.assertEqual(self.fn(*args), -2)
                self.assertEqual(list(args[8]), [-999.0] * len(args[8]))
        for buffer_index in (0, 2, 4, 6):
            args = self.args_for(case)
            args[buffer_index][0] = math.nan
            self.assertEqual(self.fn(*args), -4)
            self.assertEqual(list(args[8]), [-999.0] * len(args[8]))
        args = self.args_for(case)
        args[0] = None
        self.assertEqual(self.fn(*args), -1)
        args = self.args_for(case)
        args[2][0] = 3.402823466e38
        args[4][0] = 3.402823466e38
        self.assertEqual(self.fn(*args), -4)
        self.assertEqual(list(args[8]), [-999.0] * len(args[8]))
        args = self.args_for(case)
        for index in range(args[5]):
            args[4][index] = 0.0
        gamma_index = max(range(case["channels"]),
                          key=lambda channel: case["input"][0][channel])
        args[6][gamma_index] = 3.402823466e38
        args[6][case["channels"] + gamma_index] = 3.402823466e38
        self.assertEqual(self.fn(*args), -4)
        self.assertEqual(list(args[8]), [-999.0] * len(args[8]))

    def test_exact_capacity_and_large_geometry_rejection(self):
        case = json.loads(FIXTURE.read_text())["cases"][0]
        args = self.args_for(case)
        args[1] = (case["tokens"] - 1) * args[15] + case["channels"]
        args[9] = (case["tokens"] - 1) * args[16] + case["channels"]
        self.assertEqual(self.fn(*args), 0)
        args = self.args_for(case)
        args[12] = (1 << (ctypes.sizeof(ctypes.c_int) * 8 - 1)) - 1
        args[15] = ctypes.c_size_t(-1).value
        self.assertEqual(self.fn(*args), -2)
        self.assertEqual(list(args[8]), [-999.0] * len(args[8]))

    def capture_live_cases(self):
        if hasattr(type(self), "_live_cases"):
            return type(self)._live_cases
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as exc:
            self.skipTest(f"live PyTorch oracle dependency unavailable: {exc}")
        self.assert_pinned_torch(torch)
        fixture = json.loads(FIXTURE.read_text())
        stage_cases = []
        comparisons = []
        for case in fixture["cases"]:
            style = torch.tensor(case["style"], dtype=torch.float32)
            weight = torch.tensor(case["projection_weight"], dtype=torch.float32)
            bias = torch.tensor(case["projection_bias"], dtype=torch.float32)
            x = torch.tensor(case["input"], dtype=torch.float32)
            projection = F.linear(style, weight, bias)
            gamma, beta = projection.chunk(2)
            normalized = F.layer_norm(
                x, (case["channels"],), eps=case["epsilon"])
            oracle = (1 + gamma) * normalized + beta
            live_values = flat(oracle.tolist())
            fixture_values = flat(case["output"])
            self.assertEqual(len(live_values), len(fixture_values))
            worst = (0.0, None)
            for index, (actual, expected) in enumerate(zip(
                    live_values, fixture_values)):
                self.assertTrue(math.isfinite(actual), (index, actual))
                self.assertTrue(math.isfinite(expected), (index, expected))
                error = abs(actual - expected)
                if error > worst[0]:
                    worst = (error, (index, actual, expected))
            stage_cases.append({
                "shape": [case["tokens"], case["channels"], case["style_dim"]],
                "projection": projection.tolist(),
                "normalized": normalized.tolist(),
                "output": oracle.tolist(),
                "fixture_vs_live_max_abs": worst[0],
                "fixture_vs_live_worst": worst[1],
            })
            comparisons.append((case, oracle.tolist(), worst))
        report_path = os.environ.get("CKE_ADALN_STAGE_REPORT")
        if report_path:
            capability = getattr(torch.backends.cpu, "get_cpu_capability", None)
            report = {
                "schema": "cke.tts_adaln_reference_stages.v1",
                "provider": "audio_adaptive_layer_norm_f32",
                "numerical_contract": "audio_adaptive_layer_norm_style_linear_fp32",
                "fixture_torch": fixture["torch_version"],
                "live_torch": torch.__version__,
                "torch_git_revision": torch.version.git_version,
                "torch_build_configuration": torch.__config__.show(),
                "python": platform.python_version(),
                "machine": platform.machine(),
                "host_node": platform.node(),
                "processor": platform.processor(),
                "torch_cpu_capability": capability() if callable(capability) else None,
                "aten_cpu_capability_env": os.environ.get("ATEN_CPU_CAPABILITY"),
                "dispatch_identity_verified": False,
                "mkl_cbwr_env": os.environ.get("MKL_CBWR"),
                "torch_mkl_available": torch.backends.mkl.is_available(),
                "torch_threads": torch.get_num_threads(),
                "mkldnn_enabled": torch.backends.mkldnn.enabled,
                "cases": stage_cases,
            }
            path = Path(report_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2) + "\n")
        type(self)._live_cases = (torch, fixture, comparisons)
        return type(self)._live_cases

    def case_evidence(self, case, comparison, backend, maximum, tolerance, *, finite=True):
        settings = {
            "mkl_cbwr_requested": os.environ.get("MKL_CBWR"),
            "aten_cpu_capability_requested": os.environ.get("ATEN_CPU_CAPABILITY"),
            "threads": sys.modules["torch"].get_num_threads() if "torch" in sys.modules else None,
            "dispatch_identity": "NOT_VERIFIED",
        }
        torch = sys.modules.get("torch")
        oracle_identity = {
            "git_revision": torch.version.git_version,
            "build_configuration": torch.__config__.show(),
            "cpu_capability_reported": torch.backends.cpu.get_cpu_capability(),
        } if torch is not None and backend == torch.__version__ else None
        shape = [case["tokens"], case["channels"], case["style_dim"]]
        record = {
            "case_id": f"audio_adaptive_layer_norm_f32.{comparison}.{backend}.{os.environ.get('CKE_ORACLE_LANE', 'native-dispatch')}.{shape}",
            "name": comparison, "configuration": str(shape),
            "provider": "audio_adaptive_layer_norm_f32", "dtype": "fp32",
            "direction": "inference", "shape": shape, "oracle": "pytorch",
            "backend_version": backend, "execution_settings": settings,
            "oracle_build_identity": oracle_identity,
            "consumers": ["kokoro"], "evidence_kind": "numerical",
            "max_diff": maximum if finite else None, "tolerance": tolerance,
            "status": "pass" if finite and maximum <= tolerance else "fail",
            "reproduction_command": " ".join([
                *(shlex.quote(f"{key}={os.environ[key]}") for key in
                  ("MKL_CBWR", "ATEN_CPU_CAPABILITY", "CKE_EXPECTED_TORCH_VERSION", "CKE_ORACLE_LANE") if key in os.environ),
                "python", "-m", "unittest", shlex.quote(self.id())]),
        }
        print("CKE_NUMERICAL_CASE " + json.dumps(record, sort_keys=True))
        return record

    def test_fixture_vs_live_oracle(self):
        torch, fixture, comparisons = self.capture_live_cases()
        # Reproduction is distinct from native arithmetic; never refresh silently.
        for case, live_output, worst in comparisons:
            with self.subTest(shape=(case["tokens"], case["channels"], case["style_dim"])):
                record = self.case_evidence(case, "fixture-vs-live", torch.__version__,
                                            worst[0], 1e-7)
                self.assertEqual(record["status"], "pass",
                                 f"fixture torch={fixture['torch_version']} live={torch.__version__} "
                                 f"worst={worst[1]}")

    def test_live_torch_oracle(self):
        torch, _fixture, comparisons = self.capture_live_cases()
        # All native comparisons execute even when fixture reproduction fails.
        for case, live_output, _worst in comparisons:
            with self.subTest(shape=(case["tokens"], case["channels"], case["style_dim"])):
                self.assert_matches({**case, "output": live_output},
                                    comparison="native-vs-live", backend=torch.__version__)

    def test_live_production_geometry(self):
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as exc:
            self.skipTest(f"live PyTorch oracle dependency unavailable: {exc}")
        self.assert_pinned_torch(torch)
        tokens, channels, style_dim = 24, 512, 128
        torch.manual_seed(4428128)
        x = torch.randn(tokens, channels) * 0.2
        style = torch.randn(style_dim) * 0.2
        weight = torch.randn(2 * channels, style_dim) * 0.04
        bias = torch.randn(2 * channels) * 0.03
        gamma, beta = F.linear(style, weight, bias).chunk(2)
        output = (1 + gamma) * F.layer_norm(x, (channels,), eps=1e-5) + beta
        case = {"tokens": tokens, "channels": channels,
                "style_dim": style_dim, "input": x.tolist(),
                "style": style.tolist(),
                "projection_weight": weight.tolist(),
                "projection_bias": bias.tolist(),
                "output": output.tolist(), "epsilon": 1e-5}
        self.assert_matches(case, comparison="native-vs-live-production", backend=torch.__version__)


if __name__ == "__main__":
    unittest.main()
