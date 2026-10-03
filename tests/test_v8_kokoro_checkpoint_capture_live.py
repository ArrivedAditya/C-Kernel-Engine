"""Regenerate and compare the pinned Kokoro predictor operation checkpoints."""

import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / "version/v8/tts/reference"


def main() -> int:
    model_dir = os.environ.get("CKE_KOKORO_MODEL_DIR")
    if not model_dir:
        print("TEST SKIPPED: set CKE_KOKORO_MODEL_DIR to the pinned Kokoro-82M snapshot")
        return 0
    try:
        import kokoro  # noqa: F401
        import misaki  # noqa: F401
        import numpy  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:
        print(f"TEST SKIPPED: pinned Kokoro oracle dependency unavailable: {exc}")
        return 0

    expected = json.loads((REFERENCE_DIR / "fixture_manifest.json").read_text())
    with tempfile.TemporaryDirectory(
        prefix="cke-kokoro-checkpoints-", dir=os.environ.get("CKE_KOKORO_TEST_TMPDIR")
    ) as temporary:
        subprocess.run([
            sys.executable, str(REFERENCE_DIR / "capture_kokoro_v1.py"),
            "--model-dir", str(Path(model_dir).resolve()),
            "--output-dir", temporary,
        ], check=True, cwd=ROOT)
        actual = json.loads((Path(temporary) / "manifest.json").read_text())
        for field in ("status", "assets", "phonemes", "input_ids", "voice_row_index",
                      "fine_predictor_hooks", "waveform"):
            if actual.get(field) != expected.get(field):
                raise AssertionError(f"reference manifest mismatch: {field}")
        if actual["environment"]["packages"] != expected["environment"]["packages"]:
            raise AssertionError("pinned oracle package versions changed")
        if not set(expected["tensors"]).issubset(actual["tensors"]):
            missing = sorted(set(expected["tensors"]) - set(actual["tensors"]))
            raise AssertionError(f"baseline checkpoint keys missing: {missing}")
        for name, record in expected["tensors"].items():
            if actual["tensors"][name] != record:
                raise AssertionError(f"checkpoint changed: {name}")
            if not (Path(temporary) / record["file"]).is_file():
                raise AssertionError(f"checkpoint file missing: {name}")
        for index in (0, 2, 4):
            stem = f"predictor_text_encoder_lstms_{index}"
            batch_sizes = numpy.load(Path(temporary) / f"{stem}_input_0_1.npy")
            sorted_indices = numpy.load(Path(temporary) / f"{stem}_input_0_2.npy")
            unsorted_indices = numpy.load(Path(temporary) / f"{stem}_input_0_3.npy")
            if (batch_sizes.shape != (36,) or not numpy.all(batch_sizes == 1) or
                    sorted_indices.tolist() != [0] or unsorted_indices.tolist() != [0]):
                raise AssertionError(f"unsupported packed batch-one semantics: {stem}")
        checkpoint = torch.load(
            Path(model_dir) / "kokoro-v1_0.pth", map_location="cpu", weights_only=True
        )["predictor"]
        library = Path(temporary) / "libaudio_adaptive_layer_norm.so"
        subprocess.run([
            "cc", "-std=c11", "-O3", "-Wall", "-Wextra", "-Werror",
            "-shared", "-fPIC", "-I", str(ROOT / "include"),
            str(ROOT / "src/kernels/audio_adaptive_layer_norm.c"),
            "-o", str(library), "-lm",
        ], check=True)
        pointer = ctypes.POINTER(ctypes.c_float)
        function = ctypes.CDLL(str(library)).audio_adaptive_layer_norm_f32
        function.argtypes = [pointer, ctypes.c_size_t] * 6 + [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float,
        ]
        function.restype = ctypes.c_int
        max_errors = {}
        for index in (1, 3, 5):
            stem = f"predictor_text_encoder_lstms_{index}"
            x = numpy.ascontiguousarray(numpy.load(Path(temporary) / f"{stem}_input_0.npy"))
            style = numpy.ascontiguousarray(numpy.load(Path(temporary) / f"{stem}_input_1.npy"))
            target = numpy.load(Path(temporary) / f"{stem}_output.npy")
            prefix = f"module.text_encoder.lstms.{index}.fc"
            weight = numpy.ascontiguousarray(checkpoint[f"{prefix}.weight"].numpy())
            bias = numpy.ascontiguousarray(checkpoint[f"{prefix}.bias"].numpy())
            output = numpy.empty_like(x)
            scratch = numpy.empty(2 * x.shape[-1], dtype=numpy.float32)
            token_count, channels = x.shape[1:]
            arguments = (
                x.reshape(-1), style.reshape(-1), weight.reshape(-1),
                bias.reshape(-1), output.reshape(-1), scratch,
            )
            call = []
            for array in arguments:
                call.extend((array.ctypes.data_as(pointer),
                             array.nbytes if array is scratch else array.size))
            status = function(*call, token_count, channels, style.size,
                              channels, channels, ctypes.c_float(1e-5))
            if status != 0 or not numpy.isfinite(output).all():
                raise AssertionError(f"adaptive norm {index}: status={status} or nonfinite output")
            error = numpy.abs(output - target)
            worst = numpy.unravel_index(int(error.argmax()), error.shape)
            maximum = float(error[worst])
            if maximum > 5e-6:
                raise AssertionError(
                    f"adaptive norm {index}: max_abs={maximum} at {worst}, "
                    f"candidate={output[worst]}, oracle={target[worst]}"
                )
            max_errors[str(index)] = {
                "max_abs": maximum, "worst_index": [int(part) for part in worst],
            }
        print(json.dumps({
            "status": "PASS",
            "oracle": "pinned Kokoro/PyTorch CPU",
            "operation_hooks": len(actual["fine_predictor_hooks"]),
            "tensor_captures": len(actual["tensors"]),
            "waveform_frames": actual["waveform"]["frames"],
            "waveform_pcm16_sha256": actual["waveform"]["pcm16_sha256"],
            "adaptive_layer_norm_max_errors": max_errors,
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
