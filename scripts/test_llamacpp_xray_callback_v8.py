#!/usr/bin/env python3
"""Check llama.cpp's public tensor callback without changing model execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import struct
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def new_capture_run(output: Path) -> tuple[Path, Path]:
    output.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=output))
    capture = run_dir / "capture"
    capture.mkdir()
    return run_dir, capture


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def loaded_oracle_libraries(helper: Path, llama_root: Path) -> dict[str, Path]:
    result = subprocess.run(["ldd", str(helper)], capture_output=True, text=True, check=True)
    libraries = {}
    for line in result.stdout.splitlines():
        fields = line.strip().split()
        if fields and fields[0].startswith(("libllama.so", "libggml")) and len(fields) >= 3 and fields[1] == "=>":
            library = Path(fields[2]).resolve(strict=True)
            if not library.is_relative_to(llama_root.resolve(strict=True)):
                raise RuntimeError(f"loaded oracle library is outside the selected oracle: {library}")
            libraries[fields[0]] = library
    if not any(name.startswith("libllama.so") for name in libraries) or not any(
        name.startswith("libggml-cpu.so") for name in libraries
    ):
        raise RuntimeError("the replay helper did not resolve libllama and libggml-cpu")
    return libraries


def validate_capture(capture_dir: Path, name: str) -> dict[str, object]:
    records = [json.loads(line) for line in (capture_dir / "index.json").read_text().splitlines() if line]
    matching = [record for record in records if record.get("base_name") == name]
    if len(matching) != 1:
        raise RuntimeError(f"expected one {name} capture, found {len(matching)}")
    record = matching[0]
    count = record.get("elem_count")
    nbytes = record.get("nbytes")
    shape = record.get("shape")
    if (record.get("dtype") != 0 or not isinstance(shape, list) or not shape
            or any(type(extent) is not int or extent <= 0 for extent in shape)
            or math.prod(shape) != count or not isinstance(count, int) or count <= 0
            or not isinstance(nbytes, int) or nbytes != count * 4):
        raise RuntimeError(f"invalid FP32 capture extent: {record}")
    tensor = capture_dir / f"{record['name']}.bin"
    if tensor.stat().st_size != nbytes:
        raise RuntimeError(f"capture size differs from metadata: {tensor}")
    if not all(math.isfinite(value) for (value,) in struct.iter_unpack("f", tensor.read_bytes())):
        raise RuntimeError(f"capture contains nonfinite values: {tensor}")
    return {"name": record["name"], "elements": count, "bytes": nbytes, "sha256": sha256(tensor)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--llama-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capture-name", default="attn_norm-0")
    parser.add_argument("--expected-commit", help="Oracle SHA; defaults to this checkout's gitlink")
    args = parser.parse_args()
    model = args.model.resolve(strict=True)
    llama_root = args.llama_root.resolve(strict=True)
    actual_commit = subprocess.check_output(["git", "-C", str(llama_root), "rev-parse", "HEAD"], text=True).strip()
    expected_commit = args.expected_commit
    if not expected_commit:
        row = subprocess.check_output(["git", "ls-tree", "HEAD", "llama.cpp"], cwd=ROOT, text=True).split()
        if len(row) < 3:
            raise RuntimeError("this checkout has no pinned llama.cpp gitlink")
        expected_commit = row[2]
    if actual_commit != expected_commit:
        raise RuntimeError(f"selected llama.cpp commit {actual_commit} does not match {expected_commit}")
    output = args.output_dir.resolve()
    run_dir, capture = new_capture_run(output)

    os.environ["CK_LLAMA_CPP_ROOT"] = str(llama_root)
    sys.path.insert(0, str(ROOT / "version" / "v8" / "scripts"))
    from compare_first_token_logits_v8 import ensure_llama_helper

    helper = ensure_llama_helper()
    libraries = loaded_oracle_libraries(helper, llama_root)
    results = []
    for enabled in (False, True):
        logits = run_dir / ("with_capture.bin" if enabled else "without_capture.bin")
        command = [str(helper), "--model", str(model), "--prompt", "Hello", "--ctx", "256",
                   "--threads", "1", "--logits-out", str(logits)]
        if enabled:
            command.extend(["--dump-dir", str(capture), "--dump-names", args.capture_name])
        process = subprocess.run(command, capture_output=True, text=True)
        if process.returncode != 0:
            raise RuntimeError(f"replay failed ({process.returncode}): {process.stderr[-2000:]}")
        if logits.stat().st_size <= 0 or logits.stat().st_size % 4:
            raise RuntimeError(f"invalid logits output: {logits}")
        results.append(sha256(logits))
    if results[0] != results[1]:
        raise RuntimeError("capture changed oracle logits")

    report = {
        "status": "pass",
        "llama_cpp_commit": actual_commit,
        "model_sha256": sha256(model),
        "helper_sha256": sha256(helper),
        "run_dir": str(run_dir),
        "loaded_oracle_libraries": {
            name: {"path": str(path), "sha256": sha256(path)} for name, path in libraries.items()
        },
        "logits_sha256": results[0],
        "capture": validate_capture(capture, args.capture_name),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
