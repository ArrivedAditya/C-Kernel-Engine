"""Asset-backed Kokoro weight export and reference-waveform invariant gate."""

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]
EXPORTER = ROOT / "version/v8/tts/export_kokoro_bump.py"
SPEC = importlib.util.spec_from_file_location("kokoro_bump_export", EXPORTER)
assert SPEC and SPEC.loader
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def main() -> int:
    source = os.environ.get("CKE_KOKORO_MODEL_DIR")
    if not source:
        print("TEST SKIPPED: set CKE_KOKORO_MODEL_DIR to the pinned Kokoro-82M snapshot")
        return 0

    import numpy as np
    import torch
    from kokoro import KModel
    from torch.nn.utils.parametrize import remove_parametrizations

    model_dir = Path(source).resolve()
    with tempfile.TemporaryDirectory(
        prefix="cke-kokoro-bump-", dir=os.environ.get("CKE_KOKORO_TEST_TMPDIR")
    ) as temporary:
        output_dir = Path(temporary)
        manifest = exporter.export(model_dir, output_dir)
        if exporter.verify_bundle(output_dir) != manifest:
            raise AssertionError("exported BUMP bundle verification mismatch")
        if len(manifest["entries"]) != 602:
            raise AssertionError("expected 599 effective model tensors and three voice tensors")
        coverage = manifest["provenance"]["source_tensor_coverage"]
        if coverage != {
            "pass": True,
            "raw_checkpoint_tensors": 548,
            "consumed_raw_tensors": 548,
            "synthesized_instance_norm_affine_tensors": 140,
        }:
            raise AssertionError(f"unexpected source tensor coverage: {coverage}")

        torch.set_grad_enabled(False)
        torch.set_num_threads(1)
        torch.manual_seed(exporter.PIN["fixture"]["torch_seed"])
        model = KModel(
            repo_id=exporter.PIN["model"]["repository"],
            config=str(model_dir / "config.json"),
            model=str(model_dir / "kokoro-v1_0.pth"),
        ).eval()
        voice = torch.load(model_dir / "voices/af_heart.pt", map_location="cpu", weights_only=True)[
            exporter.REFERENCE["voice_row_index"]
        ]
        if voice.ndim == 1:
            voice = voice.unsqueeze(0)
        inference_rng_state = torch.get_rng_state()
        before = model(
            exporter.REFERENCE["phonemes"], voice,
            speed=exporter.PIN["fixture"]["speed"], return_output=True,
        ).audio.detach().cpu().numpy().reshape(-1).copy()
        for module in model.modules():
            if hasattr(module, "parametrizations") and hasattr(module.parametrizations, "weight"):
                remove_parametrizations(module, "weight", leave_parametrized=True)
        state = model.state_dict()
        by_name = {entry["name"]: entry for entry in manifest["entries"]}
        for source_name, value in state.items():
            name = exporter.canonical_name(source_name)
            entry = by_name[name]
            mapped = np.memmap(
                output_dir / "weights.bump", mode="r", dtype="<f4",
                offset=entry["file_offset"], shape=tuple(entry["shape"]),
            )
            if not np.array_equal(mapped, value.detach().cpu().numpy()):
                raise AssertionError(f"imported tensor mismatch: {name}")
        for name, expected in (
            ("voice.fixed.full", voice.numpy().reshape(-1)),
            ("voice.fixed.decoder", voice.numpy().reshape(-1)[:128]),
            ("voice.fixed.predictor", voice.numpy().reshape(-1)[128:]),
        ):
            entry = by_name[name]
            mapped = np.memmap(
                output_dir / "weights.bump", mode="r", dtype="<f4",
                offset=entry["file_offset"], shape=tuple(entry["shape"]),
            )
            if not np.array_equal(mapped, expected):
                raise AssertionError(f"imported voice tensor mismatch: {name}")
        torch.set_rng_state(inference_rng_state)
        after = model(
            exporter.REFERENCE["phonemes"], voice,
            speed=exporter.PIN["fixture"]["speed"], return_output=True,
        ).audio.detach().cpu().numpy().reshape(-1)
        if not np.array_equal(before, after):
            raise AssertionError("effective-weight materialization changed upstream inference")
        buffer = io.BytesIO()
        np.save(buffer, after, allow_pickle=False)
        waveform_hash = hashlib.sha256(buffer.getvalue()).hexdigest()
        expected_hash = exporter.REFERENCE["tensors"]["waveform_f32"]["sha256"]
        if waveform_hash != expected_hash:
            raise AssertionError(f"upstream waveform hash {waveform_hash} != pinned {expected_hash}")
        print(json.dumps({
            "status": "PASS",
            "oracle": "pinned Kokoro/PyTorch CPU",
            "effective_tensors": len(state),
            "weight_norm_transforms": manifest["provenance"]["effective_weight_transforms"],
            "synthesized_instance_norm_affine_tensors": coverage[
                "synthesized_instance_norm_affine_tensors"
            ],
            "waveform_frames": len(after),
            "waveform_npy_sha256": waveform_hash,
            "bump_sha256": exporter.sha256_file(output_dir / "weights.bump"),
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
