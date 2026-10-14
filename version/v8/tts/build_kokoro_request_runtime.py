#!/usr/bin/env python3
"""Build a bounded Kokoro native request runtime from the pinned BUMP bundle.

This is offline compiler/import tooling. The emitted C entry owns the neural
schedule; the C host binds phoneme IDs, voice conditioning, and request noise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from build_kokoro_waveform_circuit import build_circuit
from export_kokoro_bump import verify_bundle
from export_kokoro_request_package import DEFAULT_BINDINGS, package_request, sha256_file


ROOT = Path(__file__).resolve().parents[3]
COMPILER = ROOT / "version/v8/scripts/build_ir_v8.py"
CODEGEN = ROOT / "version/v8/scripts/codegen_v8.py"
HOST = ROOT / "version/v8/tts/native_kokoro_wav_host.c"


def compiler_manifest(bump_dir: Path, tokens: int,
                      bindings_path: Path = DEFAULT_BINDINGS) -> dict:
    """Make virtual graph-weight metadata; payloads stay in the source BUMP."""
    graph = build_circuit(tokens)
    bindings = json.loads(bindings_path.read_text())
    if bindings.get("schema") != "cke.kokoro.request_weight_bindings.v1":
        raise ValueError("unsupported request weight binding schema")
    source_manifest_path = bump_dir / "weights_manifest.json"
    source_bump_path = bump_dir / "weights.bump"
    if (sha256_file(source_manifest_path) != bindings["source_manifest_sha256"] or
            sha256_file(source_bump_path) != bindings["source_bump_sha256"]):
        raise ValueError("BUMP artifact does not match pinned request bindings")
    source = verify_bundle(bump_dir)
    by_name = {entry["name"]: entry for entry in source["entries"]}
    if len(by_name) != len(source["entries"]):
        raise ValueError("duplicate source weight name")
    entries = []
    offset = 64
    for name, rule in sorted(bindings["weights"].items()):
        names = rule["source"]
        if not isinstance(names, list) or len(names) not in (1, 2):
            raise ValueError(f"invalid source count: {name}")
        try:
            parts = [by_name[source_name] for source_name in names]
        except KeyError as exc:
            raise ValueError(f"missing source weight: {exc.args[0]}") from exc
        if any(part["dtype"] != "fp32" for part in parts):
            raise ValueError(f"unsupported source dtype: {name}")
        if len(parts) == 2:
            if (rule.get("transform") != "concatenate-forward-reverse" or
                    parts[0]["shape"] != parts[1]["shape"]):
                raise ValueError(f"invalid recurrent direction pairing: {name}")
            shape = [2, *parts[0]["shape"]]
        else:
            if rule.get("transform"):
                raise ValueError(f"unexpected single-source transform: {name}")
            shape = parts[0]["shape"]
        size = sum(part["size"] for part in parts)
        elements = 1
        for dimension in shape:
            if not isinstance(dimension, int) or dimension <= 0:
                raise ValueError(f"invalid source geometry: {name}")
            elements *= dimension
        if size != elements * 4:
            raise ValueError(f"source byte size contradicts shape: {name}")
        entries.append({"name": name, "dtype": "fp32", "shape": shape,
                        "size": size, "file_offset": offset,
                        "sha256": rule["sha256"]})
        offset = (offset + size + 63) & ~63
    config = {
        "model": graph["name"], "arch": graph["name"],
        "num_layers": 1, "embed_dim": 128, "num_heads": 1,
        "num_kv_heads": 1, "head_dim": 128, "intermediate_size": 256,
        "context_length": tokens, "max_seq_len": tokens,
        "vocab_size": 178, "T": tokens, "C": 128,
        "epsilon": 1e-12, "speed": 1.0,
        "activation_buffer_dtypes": {
            "word_ids": "i32", "type_ids": "i32",
            "runtime_values": "i32", "runtime_valid_extent": "i32",
        },
    }
    return {"config": config, "entries": entries, "quant_summary": {},
            "template": graph}


def _run(argv: list[str], *, log: Path | None = None) -> None:
    result = subprocess.run(argv, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    if log is not None:
        log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(argv)}\n"
                           f"{result.stdout[-4000:]}")


def build_runtime(bump_dir: Path, output: Path, tokens: int,
                  bindings_path: Path = DEFAULT_BINDINGS) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    manifest = compiler_manifest(bump_dir, tokens, bindings_path)
    manifest_path = output / "compiler_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    _run([sys.executable, str(COMPILER), "--manifest", str(manifest_path),
          "--mode", "prefill", "--output", str(output / "ir1.json"),
          "--layout-output", str(output / "layout.json"),
          "--call-output", str(output / "call.json")], log=output / "build-ir.log")
    calls = json.loads((output / "call.json").read_text())
    if calls.get("errors") or not calls.get("operations") or not calls.get("entry"):
        raise RuntimeError("unexpected incomplete generated Kokoro graph")
    _run([sys.executable, str(CODEGEN), "--ir", str(output / "call.json"),
          "--layout", str(output / "layout.json"),
          "--output", str(output / "generated.c")], log=output / "codegen.log")
    sys.path.insert(0, str(ROOT / "version/v8/scripts"))
    import build_ir_v8  # canonical registry for selected provider sources
    registry = {item["id"]: item for item in
                build_ir_v8.load_kernel_registry()["kernels"]}
    sources = sorted({source for op in calls["operations"]
                      for source in registry[op["call_abi"]["kernel_id"]]
                      ["impl"]["sources"]})
    entry = manifest["template"]["native_entry"]["function"]
    (output / "exports.map").write_text(
        "{ global: " + entry + "; local: *; };\n")
    generated = output / "generated.c"
    _run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-pedantic",
          "-fsyntax-only", str(generated), "-I", str(ROOT / "include")],
         log=output / "syntax.log")
    library = output / "generated.so"
    compile_command = ["cc", "-std=c11", "-O2", "-Wall", "-Wextra",
        "-Werror", "-shared", "-fPIC", "-ffp-contract=off",
        "-ffunction-sections", "-fdata-sections", "-Wl,--gc-sections",
        "-Wl,--no-undefined", "-Wl,--version-script=" + str(output / "exports.map"),
        str(generated), *[str(ROOT / path) for path in sources],
        "-I", str(ROOT / "include"), "-lm", "-lpthread", "-ldl",
        "-o", str(library)]
    _run(compile_command, log=output / "compile.log")
    library_sha256 = sha256_file(library)
    for source, name in ((HOST, "host.c"),
            (ROOT / "src/kernels/audio_noise_counter.c", "audio_noise_counter.c"),
            (ROOT / "src/kernels/checked_float_buffer.h", "checked_float_buffer.h"),
            (ROOT / "include/ckernel_tts.h", "ckernel_tts.h")):
        shutil.copy2(source, output / name)
    host_command = ["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
        "-pedantic", f"-DCK_TTS_COMPILED_TOKEN_COUNT={tokens}",
        f'-DCK_TTS_EXPECTED_LIBRARY_SHA256="{library_sha256}"',
        "-I", str(output), str(output / "host.c"),
        str(output / "audio_noise_counter.c"), "-L", str(output),
        "-l:generated.so", "-Wl,-rpath,$ORIGIN", "-lm", "-ldl", "-lcrypto",
        "-o", str(output / "native-kokoro-wav")]
    _run(host_command, log=output / "host-compile.log")
    package = package_request(output / "layout.json", bump_dir,
                              output / "request.cktts", bindings_path)
    report = {"schema": "cke.kokoro.native_request_build.v1",
              "phoneme_count": tokens, "generated_operations": len(calls["operations"]),
              "selected_providers": sorted({op["call_abi"]["kernel_id"]
                                            for op in calls["operations"]}),
              "compiler_manifest_sha256": sha256_file(manifest_path),
              "generated_c_sha256": sha256_file(generated),
              "library_sha256": library_sha256,
              "host_sha256": sha256_file(output / "native-kokoro-wav"),
              "package_sha256": package["package_sha256"],
              "source_sha256": {path: sha256_file(ROOT / path) for path in sources},
              "weight_offset_contract": "compiler-planning offsets only; verified "
                  "source BUMP payloads are placed by the CKTTS003 exporter",
              "compile_command": compile_command,
              "host_compile_command": host_command,
              "scope": "exact phoneme count, supplied IDs and voice bank; no text frontend"}
    (output / "build-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bump-dir", required=True, type=Path)
    parser.add_argument("--tokens", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build_runtime(args.bump_dir, args.output_dir, args.tokens), indent=2))


if __name__ == "__main__":
    main()
