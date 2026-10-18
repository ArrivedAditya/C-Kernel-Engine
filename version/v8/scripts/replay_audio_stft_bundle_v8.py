"""Replay a private checked-STFT operation packet against CKE and PyTorch.

The packet supplies data and scalar ABI values, never code or tolerances. Validate
its complete physical geometry before loading the recorded library or calling it.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from version.v8.scripts.xray_operation_bundle_v8 import BundleError, load_bundle, sha256


ROOT = Path(__file__).resolve().parents[3]
MAP = ROOT / "version/v8/kernel_maps/audio_stft_mag_phase_checked_f32.json"
FLOAT = ctypes.POINTER(ctypes.c_float)
SCALARS = ("samples_capacity", "window_capacity", "table_capacity",
           "output_capacity", "output_stride", "scratch_capacity",
           "samples_count", "n_fft", "hop", "frames")


def validate_stft_bundle(path: Path):
    manifest, tensors = load_bundle(path)
    provider = json.loads(MAP.read_text(encoding="utf-8"))
    capability = provider["numerical_capabilities"][0]
    if (manifest["operation"] != provider["op"] or
            manifest["provider"] != provider["impl"]["function"] or
            manifest["contract"] != capability["contract_id"]):
        raise BundleError("selected STFT provider or numerical contract differs from map")
    context = manifest["context"]
    if set(context) != set(SCALARS) | {"reference_identity", "state"}:
        raise BundleError("STFT packet has incomplete or unexpected scalar arguments")
    if context["state"] != "stateless":
        raise BundleError("STFT operation has no recurrent or RNG state")
    identity = context["reference_identity"]
    if (not isinstance(identity, dict) or
            set(identity) != {"model_pin", "code_pin", "asset_sha256"} or
            any(not isinstance(identity[key], str) or not identity[key]
                for key in ("model_pin", "code_pin")) or
            not isinstance(identity["asset_sha256"], dict) or
            not identity["asset_sha256"] or
            any(not isinstance(name, str) or not name or
                not isinstance(digest, str) or
                re.fullmatch(r"[0-9a-f]{64}", digest) is None
                for name, digest in identity["asset_sha256"].items())):
        raise BundleError("STFT packet lacks reference identity")
    if any(type(context[key]) is not int or context[key] <= 0 for key in SCALARS):
        raise BundleError("STFT scalar arguments must be positive integers")
    n = context["samples_count"]
    fft = context["n_fft"]
    hop = context["hop"]
    frames = context["frames"]
    stride = context["output_stride"]
    bins = fft // 2 + 1
    if (fft < 2 or fft % 2 or n <= fft // 2 or
            frames != n // hop + 1 or stride < frames or
            context["samples_capacity"] < n or
            context["window_capacity"] != fft or
            context["table_capacity"] != bins * fft or
            context["output_capacity"] != 2 * bins * stride or
            context["scratch_capacity"] < 2 * bins * frames):
        raise BundleError("STFT packet has inconsistent length, stride or capacity")
    expected = {
        "samples": (context["samples_capacity"],),
        "reference_samples": (context["samples_capacity"],), "window": (fft,),
        "cos_table": (bins, fft), "sin_table": (bins, fft),
        "cke_output": (2 * bins, stride),
        "reference_output": (2 * bins, frames),
    }
    if set(tensors) != set(expected):
        raise BundleError("STFT packet requires full operands and both outputs")
    for role, shape in expected.items():
        if tensors[role].shape != shape or tensors[role].dtype != np.float32:
            raise BundleError(f"{role} layout or dtype differs from STFT ABI")
    axes = {"samples": ["sample"], "reference_samples": ["sample"],
            "window": ["tap"], "cos_table": ["bin", "tap"],
            "sin_table": ["bin", "tap"],
            "cke_output": ["channel", "physical_frame"],
            "reference_output": ["channel", "valid_frame"]}
    if any(manifest["tensors"][role]["axes"] != labels
           for role, labels in axes.items()):
        raise BundleError("STFT packet physical layout labels differ from ABI")
    for role in ("samples", "reference_samples", "window", "cos_table",
                 "sin_table", "cke_output", "reference_output"):
        if not np.isfinite(tensors[role]).all():
            raise BundleError(f"{role} has nonfinite values")
    return manifest, tensors, context


def _native(library, tensors, context, samples):
    call = library.audio_stft_mag_phase_checked_f32
    call.argtypes = [FLOAT, ctypes.c_size_t, FLOAT, ctypes.c_size_t,
                     FLOAT, FLOAT, ctypes.c_size_t, FLOAT, ctypes.c_size_t,
                     ctypes.c_size_t, FLOAT, ctypes.c_size_t] + [ctypes.c_size_t] * 4
    call.restype = ctypes.c_int
    output = np.full(tensors["cke_output"].shape, -91., dtype=np.float32)
    scratch = np.empty(context["scratch_capacity"], dtype=np.float32)
    pointer = lambda array: array.ctypes.data_as(FLOAT)
    status = call(pointer(samples), context["samples_capacity"],
                  pointer(tensors["window"]), context["window_capacity"],
                  pointer(tensors["cos_table"]), pointer(tensors["sin_table"]),
                  context["table_capacity"], pointer(output),
                  context["output_capacity"], context["output_stride"],
                  pointer(scratch), context["scratch_capacity"],
                  context["samples_count"], context["n_fft"],
                  context["hop"], context["frames"])
    if status:
        raise BundleError(f"checked STFT provider rejected verified packet: {status}")
    return output


def _comparison(actual, expected, maximum):
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        return {"status": "FAIL", "reason": "nonfinite output"}
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    worst = np.unravel_index(int(error.argmax()), error.shape)
    return {"status": "PASS" if float(error[worst]) <= maximum else "FAIL",
            "max_abs": float(error[worst]), "worst_index": list(map(int, worst)),
            "actual": float(actual[worst]), "reference": float(expected[worst]),
            "rmse": float(np.sqrt(np.mean(error * error))), "ceiling": maximum}


def replay(path: Path, *, runtime: Path | None = None, torch_module=None):
    manifest, tensors, context = validate_stft_bundle(path)
    # The caller chooses executable code. Recorded bytes establish identity,
    # but a self-declared packet hash is not a trust decision.
    if runtime is None:
        raise BundleError("trusted runtime path must be supplied separately")
    runtime = runtime.resolve()
    if not runtime.is_file() or sha256(runtime) != manifest["runtime"]["sha256"]:
        raise BundleError("chosen runtime differs from packet artifact hash")
    library = ctypes.CDLL(str(runtime))
    native_connected = _native(library, tensors, context, tensors["samples"])
    frames = context["frames"]
    captured = tensors["cke_output"]
    fidelity = {"status": "PASS" if np.array_equal(
        native_connected.view(np.uint32), captured.view(np.uint32)) else "FAIL"}
    native_reference = _native(library, tensors, context, tensors["reference_samples"])
    result = {
        "provider": manifest["provider"], "contract": manifest["contract"],
        "runtime_sha256": manifest["runtime"]["sha256"],
        "reference_identity": context["reference_identity"],
        "capture_reproduced": fidelity,
        "upstream_source": _comparison(tensors["samples"][:context["samples_count"]],
            tensors["reference_samples"][:context["samples_count"]], 1e-5),
        "cke_on_reference_inputs": _comparison(native_reference[:, :frames],
            tensors["reference_output"], 1e-5),
        "connected_model": _comparison(captured[:, :frames],
            tensors["reference_output"], 1e-5),
        "same_input_oracle": {"status": "NOT_TESTED", "reason": "pinned PyTorch 2.8 unavailable"},
    }
    if torch_module is not None:
        torch = torch_module
        if not torch.__version__.startswith("2.8."):
            raise BundleError("same-input oracle requires pinned PyTorch 2.8")
        torch.set_num_threads(1)
        window = torch.from_numpy(tensors["window"].copy())
        def oracle(samples):
            spectrum = torch.stft(torch.from_numpy(samples.copy()),
                context["n_fft"], context["hop"], context["n_fft"],
                window=window, center=True, pad_mode="reflect",
                normalized=False, return_complex=True)
            return np.ascontiguousarray(np.concatenate((torch.abs(spectrum).numpy(),
                torch.angle(spectrum).numpy()), axis=0))
        connected_oracle = oracle(tensors["samples"][:context["samples_count"]])
        reference_oracle = oracle(tensors["reference_samples"][:context["samples_count"]])
        result["same_input_oracle"] = _comparison(native_connected[:, :frames],
            connected_oracle, 1e-5)
        bins = context["n_fft"] // 2 + 1
        result["same_input_magnitude"] = _comparison(
            native_connected[:bins, :frames], connected_oracle[:bins], 1e-5)
        result["same_input_raw_phase"] = _comparison(
            native_connected[bins:, :frames], connected_oracle[bins:], 1e-5)
        result["cke_on_reference_inputs_oracle"] = _comparison(
            native_reference[:, :frames], reference_oracle, 1e-5)
        result["reference_input_raw_phase"] = _comparison(
            native_reference[bins:, :frames], reference_oracle[bins:], 1e-5)
        result["reference_capture_reproduced"] = _comparison(reference_oracle,
            tensors["reference_output"], 1e-5)
        result["torch_version"] = torch.__version__
        result["torch_git_version"] = torch.version.git_version
        result["torch_cpu_capability"] = torch.backends.cpu.get_cpu_capability()
        result["torch_num_threads"] = torch.get_num_threads()
        result["torch_config_sha256"] = hashlib.sha256(
            torch.__config__.show().encode()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument("--runtime", required=True, type=Path,
                        help="explicitly selected trusted library, hash-matched to packet")
    parser.add_argument("--live-pytorch", action="store_true",
                        help="require pinned PyTorch 2.8 same-input comparisons")
    args = parser.parse_args()
    torch = None
    if args.live_pytorch:
        import torch as torch_module
        torch = torch_module
    print(json.dumps(replay(args.packet, runtime=args.runtime,
                            torch_module=torch), indent=2,
                     sort_keys=True))


if __name__ == "__main__":
    main()
