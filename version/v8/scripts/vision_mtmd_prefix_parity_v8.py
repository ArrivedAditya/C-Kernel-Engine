#!/usr/bin/env python3
"""Compare a generated CKE image prefix with independently encoded llama.cpp MTMD output."""

from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
PROBE_SOURCE = ROOT / "version/v8/tools/llama_mtmd_prefix_probe_v8.cpp"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_identity(path: Path) -> dict[str, str | int]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "size_bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def validate_cke_report(report: dict, image: Path) -> tuple[Path, int, int]:
    if report.get("status") != "ok" or report.get("prefix_source") != "encoder":
        raise ValueError("CKE report must describe successful generated encoder execution")
    encoder = report.get("encoder_report")
    if not isinstance(encoder, dict) or encoder.get("image_source") != "file":
        raise ValueError("CKE report lacks file-backed encoder evidence")
    if Path(str(encoder.get("image_path") or "")).resolve() != image.resolve():
        raise ValueError("CKE image path does not match requested image")
    if not re.fullmatch(r"[0-9a-f]{64}", str(encoder.get("image_sha256") or "")):
        raise ValueError("CKE report lacks image-byte provenance")
    if sha256(image) != encoder["image_sha256"]:
        raise ValueError("CKE image bytes changed since capture")
    if not re.fullmatch(r"[0-9a-f]{64}", str(encoder.get("decoded_rgb8_sha256") or "")):
        raise ValueError("CKE report lacks decoded RGB8 provenance")
    tokens = report.get("prefix_tokens")
    dim = report.get("prefix_embed_dim")
    if type(tokens) is not int or type(dim) is not int or tokens <= 0 or dim <= 0:
        raise ValueError("CKE prefix geometry is invalid")
    if tokens != encoder.get("prefix_tokens") or dim != encoder.get("embed_dim"):
        raise ValueError("CKE bridge and encoder prefix extents disagree")
    path = Path(str(report.get("prefix_dump_path") or ""))
    if not path.is_file() or path.stat().st_size != tokens * dim * 4:
        raise ValueError("CKE prefix dump is missing or has the wrong extent")
    if sha256(path) != report.get("prefix_dump_sha256"):
        raise ValueError("CKE prefix dump hash is missing or stale")
    runtime = report.get("encoder_runtime") or {}
    library = Path(str(runtime.get("so_path") or ""))
    if not library.is_file():
        raise ValueError("CKE generated encoder library is missing")
    if sha256(library) != encoder.get("model_library_sha256"):
        raise ValueError("CKE generated encoder library changed since capture")
    return path, tokens, dim


def compare_prefixes(cke_path: Path, oracle_path: Path, tokens: int, dim: int) -> dict:
    expected_bytes = tokens * dim * 4
    if cke_path.stat().st_size != expected_bytes or oracle_path.stat().st_size != expected_bytes:
        raise ValueError("CKE and oracle prefix extents disagree")
    cke = array("f")
    oracle = array("f")
    cke.frombytes(cke_path.read_bytes())
    oracle.frombytes(oracle_path.read_bytes())
    if sys.byteorder != "little":
        cke.byteswap()
        oracle.byteswap()
    if not all(math.isfinite(value) for value in cke) or not all(math.isfinite(value) for value in oracle):
        raise ValueError("prefix contains nonfinite values")
    differences = [abs(left - right) for left, right in zip(cke, oracle)]
    worst = max(range(len(differences)), key=differences.__getitem__)
    rmse = math.sqrt(math.fsum(delta * delta for delta in differences) / len(differences))
    return {
        "tokens": tokens,
        "embed_dim": dim,
        "rmse": rmse,
        "max_abs": differences[worst],
        "worst_token": worst // dim,
        "worst_channel": worst % dim,
        "cke_worst_value": cke[worst],
        "oracle_worst_value": oracle[worst],
    }


def _run(command: list[str], *, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {result.stderr[-3000:]}")
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cke-report", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mmproj", type=Path, required=True)
    parser.add_argument("--llama-root", type=Path, required=True)
    parser.add_argument("--llama-build", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-rmse", type=float, required=True)
    parser.add_argument("--max-abs", type=float, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "prefix-parity.json"
    output.unlink(missing_ok=True)
    if not all(math.isfinite(value) and value >= 0 for value in (args.max_rmse, args.max_abs)):
        parser.error("numerical thresholds must be finite and nonnegative")
    image = args.image.resolve(strict=True)
    report = json.loads(args.cke_report.read_text(encoding="utf-8"))
    cke_path, tokens, dim = validate_cke_report(report, image)
    llama_root = args.llama_root.resolve(strict=True)
    build = args.llama_build.resolve(strict=True)
    revision = _run(["git", "-C", str(llama_root), "rev-parse", "HEAD"]).strip()
    pinned = _run(["git", "-C", str(ROOT), "ls-tree", "HEAD", "llama.cpp"]).split()
    if len(pinned) != 4 or pinned[2] != revision:
        raise ValueError("llama.cpp oracle revision does not match CKE gitlink")
    probe = args.output_dir / "llama_mtmd_prefix_probe_v8"
    oracle_path = args.output_dir / "oracle-prefix.f32"
    oracle_rgb_path = args.output_dir / "oracle-decoded.rgb8"
    _run([
        "g++", "-std=c++17", "-O2", "-Wall", "-Wextra",
        "-I", str(llama_root / "include"),
        "-I", str(llama_root / "tools/mtmd"),
        "-I", str(llama_root / "ggml/include"),
        str(PROBE_SOURCE), "-L", str(build), f"-Wl,-rpath,{build}",
        "-lmtmd", "-lllama", "-o", str(probe),
    ])
    env = {**os.environ, "LD_LIBRARY_PATH": str(build)}
    linked = _run(["ldd", str(probe)], env=env)
    library_hashes = {}
    library_paths = {}
    for name in ("libmtmd", "libllama", "libggml", "libggml-base", "libggml-cpu"):
        match = re.search(rf"^\s*{name}\.so\S* => (\S+)", linked, re.MULTILINE)
        if not match or Path(match.group(1)).resolve().parent != build:
            raise ValueError(f"oracle {name} did not resolve to the pinned build")
        library_paths[name] = Path(match.group(1))
        library_hashes[name] = sha256(library_paths[name])
    oracle_path.unlink(missing_ok=True)
    oracle_rgb_path.unlink(missing_ok=True)
    model_hash = sha256(args.model)
    mmproj_hash = sha256(args.mmproj)
    probe_hash = sha256(probe)
    probe_output = _run([
        str(probe), str(args.model), str(args.mmproj), str(image),
        str(oracle_path), str(oracle_rgb_path),
    ], env=env)
    if f"tokens={tokens} dim={dim}" not in probe_output:
        raise ValueError("oracle visual prefix geometry does not match CKE")
    if sha256(args.model) != model_hash or sha256(args.mmproj) != mmproj_hash:
        raise ValueError("oracle model or vision projector changed during execution")
    if sha256(probe) != probe_hash or any(
        sha256(library_paths[name]) != digest for name, digest in library_hashes.items()
    ):
        raise ValueError("oracle helper or library changed during execution")
    if sha256(oracle_rgb_path) != report["encoder_report"]["decoded_rgb8_sha256"]:
        raise ValueError("CKE and oracle decoded RGB8 pixels disagree")
    metrics = compare_prefixes(cke_path, oracle_path, tokens, dim)
    passed = metrics["rmse"] <= args.max_rmse and metrics["max_abs"] <= args.max_abs
    result = {
        "status": "pass" if passed else "fail",
        "lane": "independent_mtmd_encoder_prefix",
        "input_provenance": "independently_decoded_and_preprocessed_from_same_p6",
        "metrics": metrics,
        "thresholds": {"max_rmse": args.max_rmse, "max_abs": args.max_abs},
        "decoder_prefix_exports": {
            "contract": "cke.decoder_prefix_f32.v1",
            "tokens": tokens,
            "row_dim": dim,
            "ck": artifact_identity(cke_path),
            "llama": artifact_identity(oracle_path),
        },
        "artifact_identity": {
            "cke_bridge_report": artifact_identity(args.cke_report),
            "cke_model_library": artifact_identity(Path(report["encoder_runtime"]["so_path"])),
            "image": artifact_identity(image),
            "oracle_model": artifact_identity(args.model),
            "oracle_mmproj": artifact_identity(args.mmproj),
            "oracle_probe": artifact_identity(probe),
            **{name: artifact_identity(path) for name, path in library_paths.items()},
        },
        "provenance": {
            "image_sha256": sha256(image),
            "cke_decoded_rgb8_sha256": report["encoder_report"]["decoded_rgb8_sha256"],
            "oracle_decoded_rgb8_sha256": sha256(oracle_rgb_path),
            "cke_prefix_sha256": sha256(cke_path),
            "cke_encoder_library_sha256": sha256(Path(report["encoder_runtime"]["so_path"])),
            "oracle_prefix_sha256": sha256(oracle_path),
            "oracle_model_sha256": model_hash,
            "oracle_mmproj_sha256": mmproj_hash,
            "oracle_probe_sha256": probe_hash,
            "oracle_library_sha256": library_hashes,
            "oracle_revision": revision,
        },
        "limitations": [
            "CKE library identity is bridge-reported; loaded-library attestation is not established",
            "P6 input and CKE Python preprocessing do not certify native image-file deployment",
            "Prefix parity does not certify decoder logits or image-task quality",
        ],
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "metrics": metrics, "report": str(output)}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
