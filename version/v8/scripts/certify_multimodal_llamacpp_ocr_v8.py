#!/usr/bin/env python3
"""Compare verified CKE OCR cases with independent llama.cpp image execution.

This is an answer/ground-truth lane. It does not establish identical prompt token
IDs, encoder-prefix parity, or decoder-logit parity.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from typing import Any

from certify_multimodal_ocr_corpus_v8 import (
    DEFAULT_PROMPT,
    _accepted_values,
    _build_prompt,
    _case_config,
    _extract_json_object,
    _load_resumed,
    _load_samples,
    _max_new_tokens,
    _normalized,
    _score,
    _sha256_bytes,
    _sha256_file,
    _sha256_json,
    _write_json,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "cke.independent_llamacpp_ocr_comparison"
LIBRARIES = ("libllama-common", "libmtmd", "libllama", "libggml", "libggml-base", "libggml-cpu")


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected an object: {path}")
    return value


def _oracle_identity(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    build = root / "build" / "bin"
    cli = build / "llama-mtmd-cli"
    if not cli.is_file():
        raise FileNotFoundError(cli)
    source = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    pin = subprocess.run(
        ["git", "-C", str(ROOT), "ls-tree", "HEAD", "llama.cpp"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    if len(pin) != 4 or pin[2] != source:
        raise ValueError("llama.cpp source does not match the CKE gitlink")
    linked = subprocess.run(["ldd", str(cli)], capture_output=True, text=True, check=True).stdout
    libraries: dict[str, dict[str, str]] = {}
    for name in LIBRARIES:
        match = re.search(rf"^\s*{re.escape(name)}\.so\S* => (\S+)", linked, re.MULTILINE)
        if not match:
            raise ValueError(f"oracle CLI does not resolve {name}")
        path = Path(match.group(1)).resolve(strict=True)
        if path.parent != build.resolve():
            raise ValueError(f"oracle {name} is not loaded from the pinned build")
        libraries[name] = {"path": str(path), "sha256": _sha256_file(path)}
    return {
        "source_commit": source,
        "cli": {"path": str(cli.resolve()), "sha256": _sha256_file(cli)},
        "libraries": libraries,
    }


def _check_oracle_identity(identity: dict[str, Any]) -> None:
    for item in (identity["cli"], *identity["libraries"].values()):
        if _sha256_file(Path(item["path"])) != item["sha256"]:
            raise ValueError("oracle binary or loaded library changed during execution")


def _oracle_template_identity(gguf_path: Path) -> dict[str, str]:
    from run_multimodal_bridge_v8 import _read_gguf_metadata

    template = _read_gguf_metadata(gguf_path, {"tokenizer.chat_template"}, strict=True).get(
        "tokenizer.chat_template"
    )
    if not isinstance(template, str) or not template:
        raise ValueError("decoder GGUF has no chat template for the Jinja oracle lane")
    return {"source": "decoder_gguf_metadata", "sha256": _sha256_bytes(template.encode("utf-8"))}


def _field_comparison(
    truth: dict[str, Any], cke: dict[str, Any] | None, oracle: dict[str, Any] | None,
) -> dict[str, int | bool]:
    shared = both_correct = cke_only_correct = oracle_only_correct = 0
    for key, expected in truth.items():
        accepted = {_normalized(value) for value in _accepted_values(expected)}
        cke_present = isinstance(cke, dict) and key in cke
        oracle_present = isinstance(oracle, dict) and key in oracle
        cke_value = _normalized(cke[key]) if cke_present else None
        oracle_value = _normalized(oracle[key]) if oracle_present else None
        shared += int(cke_present and oracle_present and cke_value == oracle_value)
        cke_correct = cke_present and cke_value in accepted
        oracle_correct = oracle_present and oracle_value in accepted
        both_correct += int(cke_correct and oracle_correct)
        cke_only_correct += int(cke_correct and not oracle_correct)
        oracle_only_correct += int(oracle_correct and not cke_correct)
    return {
        "expected_fields": len(truth),
        "agreed_present_fields": shared,
        "both_correct": both_correct,
        "cke_only_correct": cke_only_correct,
        "oracle_only_correct": oracle_only_correct,
        "raw_json_equal": isinstance(cke, dict) and isinstance(oracle, dict) and cke == oracle,
    }


def _oracle_command(
    identity: dict[str, Any], config: dict[str, Any], image: Path, prompt: str,
    max_new_tokens: int, threads: int, flash_attention: str,
) -> list[str]:
    return [
        identity["cli"]["path"],
        "-m", config["decoder_gguf"]["path"],
        "-mm", config["encoder_gguf"]["path"],
        "--image", str(image), "-p", prompt,
        "-n", str(max_new_tokens), "-c", str(config["context_len"]),
        "-t", str(threads), "-tb", str(threads),
        "--temp", "0", "--top-p", "1", "--repeat-penalty", "1",
        "--jinja", "--no-warmup", "--no-mmproj-offload", "-ngl", "0",
        "--flash-attn", flash_attention,
    ]


def _run_oracle_command(
    command: list[str], raw_path: Path, stderr_path: Path, timeout_sec: float,
) -> tuple[float, int]:
    started = time.perf_counter()
    with raw_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        raw_path.chmod(0o600)
        stderr_path.chmod(0o600)
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired as exc:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise TimeoutError(f"llama.cpp exceeded {timeout_sec:g}s per-case limit") from exc
    return time.perf_counter() - started, returncode


def _case_identity(global_hash: str, cke: dict[str, Any]) -> dict[str, Any]:
    return {
        "global_config_sha256": global_hash,
        "cke_case_config": cke["case_config"],
        "cke_output_sha256": cke["output_sha256"],
        "cke_trace_sha256": cke["token_trace_sha256"],
    }


def _resumed_case(
    path: Path, raw_path: Path, expected: dict[str, Any], truth: dict[str, Any],
    cke: dict[str, Any], oracle_identity: dict[str, Any],
) -> dict[str, Any] | None:
    if not path.is_file() or not raw_path.is_file():
        return None
    try:
        row = _read_object(path)
        raw = raw_path.read_text(encoding="utf-8")
        parsed = _extract_json_object(raw)
        if (
            row.get("status") == "complete"
            and row.get("case_identity") == expected
            and row.get("oracle_identity") == oracle_identity
            and row.get("oracle_output_sha256") == _sha256_bytes(raw.encode("utf-8"))
            and row.get("oracle_parsed_output") == parsed
            and row.get("oracle_metrics") == _score(truth, parsed)
            and row.get("cke_metrics") == cke["metrics"]
            and row.get("comparison") == _field_comparison(truth, cke["parsed_output"], parsed)
            and isinstance(row.get("oracle_wall_sec"), (int, float))
            and math.isfinite(row["oracle_wall_sec"])
            and row["oracle_wall_sec"] >= 0
        ):
            return row
    except (OSError, UnicodeError, ValueError, KeyError, json.JSONDecodeError):
        pass
    return None


def _aggregate(rows: list[dict[str, Any]], selected: int) -> dict[str, Any]:
    complete = [row for row in rows if row.get("status") == "complete"]
    return {
        "selected": selected,
        "completed": len(complete),
        "errors": selected - len(complete),
        "cke_exact_fields": sum(row["cke_metrics"]["exact_fields"] for row in complete),
        "oracle_exact_fields": sum(row["oracle_metrics"]["exact_fields"] for row in complete),
        "expected_fields": sum(row["comparison"]["expected_fields"] for row in complete),
        "agreed_present_fields": sum(row["comparison"]["agreed_present_fields"] for row in complete),
        "both_correct": sum(row["comparison"]["both_correct"] for row in complete),
        "raw_json_equal_cases": sum(bool(row["comparison"]["raw_json_equal"]) for row in complete),
        "oracle_wall_sec": sum(row["oracle_wall_sec"] for row in complete),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cke-output-dir", type=Path, required=True)
    parser.add_argument("--llama-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--require-selected-images", type=int)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--flash-attention", choices=("on", "off", "auto"), default="auto")
    parser.add_argument("--oracle-timeout-sec", type=float, default=1800.0)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    args = parser.parse_args(argv)
    if (args.threads < 1 or args.start_index < 1 or args.limit is not None and args.limit < 1
            or not math.isfinite(args.oracle_timeout_sec) or args.oracle_timeout_sec <= 0):
        parser.error("threads, start-index, limit, and oracle timeout must be positive and finite")
    manifest = args.manifest.resolve(strict=True)
    cke_dir = args.cke_output_dir.resolve(strict=True)
    config = _read_object(cke_dir / "config.json")
    config_hash = config.get("config_sha256")
    if not isinstance(config_hash, str) or config_hash != _sha256_json({
        key: value for key, value in config.items() if key != "config_sha256"
    }):
        raise ValueError("CKE run configuration is malformed or stale")
    if config.get("source_mode") != "gguf":
        parser.error("the llama.cpp image lane requires the same GGUF model and projector as CKE")
    if config.get("manifest_sha256") != _sha256_file(manifest):
        raise ValueError("CKE result uses a different corpus manifest")
    for role in ("decoder", "encoder"):
        source = config[f"{role}_gguf"]
        if _sha256_file(Path(source["path"])) != source["sha256"]:
            raise ValueError(f"{role} GGUF changed since the CKE run")
    samples = [row for row in _load_samples(manifest) if row["index"] >= args.start_index]
    if args.limit is not None:
        samples = samples[:args.limit]
    if not samples:
        parser.error("no images selected")
    if args.require_selected_images is not None:
        if len(samples) != args.require_selected_images or len({row["image_sha256"] for row in samples}) != len(samples):
            parser.error("selected corpus extent or distinct-image count is incorrect")
    oracle_identity = _oracle_identity(args.llama_root)
    oracle_template = _oracle_template_identity(Path(config["decoder_gguf"]["path"]))
    global_config = {
        "schema": SCHEMA, "schema_version": 1,
        "manifest_sha256": config["manifest_sha256"],
        "cke_config_sha256": config["config_sha256"],
        "model_sha256": config["decoder_gguf"]["sha256"],
        "projector_sha256": config["encoder_gguf"]["sha256"],
        "oracle_identity": oracle_identity,
        "threads": args.threads, "flash_attention": args.flash_attention,
        "oracle_timeout_sec": args.oracle_timeout_sec,
        "cke_chat_template": config["chat_template"],
        "cke_thinking_mode": config["thinking_mode"],
        "oracle_template_source": "GGUF Jinja via pinned llama-mtmd-cli",
        "oracle_template": oracle_template,
        "oracle_thinking_mode": "enabled_by_pinned_cli_default",
    }
    global_hash = _sha256_json(global_config)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    output.chmod(0o700)
    _write_json(output / "config.json", global_config)
    rows: list[dict[str, Any]] = []
    for sample in samples:
        case_dir = output / f"image{sample['index']:02d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        case_dir.chmod(0o700)
        if sample["prompt"] is not None:
            prompt = sample["prompt"]
        elif config["prompt_template_sha256"] == _sha256_bytes(DEFAULT_PROMPT.encode("utf-8")):
            prompt = _build_prompt(DEFAULT_PROMPT, sample["truth"])
        else:
            raise ValueError("custom CKE prompt template cannot be reconstructed from its hash")
        cke_path = cke_dir / f"image{sample['index']:02d}" / "case_result.json"
        max_tokens = _max_new_tokens(config["max_new_tokens"], sample)
        expected_cke = _case_config(config["config_sha256"], sample, prompt, max_tokens)
        cke = _load_resumed(cke_path, expected_cke, config, sample["truth"])
        if cke is None or cke.get("prompt") != prompt:
            rows.append({"status": "incomplete", "reason": "CKE case is absent or stale"})
            print("[case incomplete] CKE evidence is absent or stale")
            if not args.continue_on_failure:
                break
            continue
        expected = _case_identity(global_hash, cke)
        result_path = case_dir / "case_result.json"
        raw_path = case_dir / "llama_raw.txt"
        resumed = None if args.force_rerun else _resumed_case(
            result_path, raw_path, expected, sample["truth"], cke, oracle_identity,
        )
        if resumed is not None:
            rows.append(resumed)
            print("[case resumed] independent oracle evidence verified")
            continue
        raw_path.unlink(missing_ok=True)
        try:
            _check_oracle_identity(oracle_identity)
            command = _oracle_command(
                oracle_identity, config, sample["image"], prompt, max_tokens,
                args.threads, args.flash_attention,
            )
            wall_sec, returncode = _run_oracle_command(
                command, raw_path, case_dir / "llama_stderr.txt", args.oracle_timeout_sec,
            )
            if returncode:
                raise RuntimeError(f"llama.cpp exited {returncode}")
            if _sha256_file(sample["image"]) != sample["image_sha256"]:
                raise ValueError("image bytes changed during oracle execution")
            _check_oracle_identity(oracle_identity)
            raw = raw_path.read_text(encoding="utf-8")
            parsed = _extract_json_object(raw)
            oracle_metrics = _score(sample["truth"], parsed)
            row = {
                "schema": SCHEMA, "schema_version": 1, "status": "complete",
                "case_identity": expected, "oracle_identity": oracle_identity,
                "oracle_output_sha256": _sha256_bytes(raw.encode("utf-8")),
                "oracle_parsed_output": parsed, "oracle_metrics": oracle_metrics,
                "cke_metrics": cke["metrics"],
                "comparison": _field_comparison(sample["truth"], cke["parsed_output"], parsed),
                "oracle_wall_sec": wall_sec,
                "input_contract": "same_image_bytes_prompt_text_gguf_and_generation_budget",
                "chat_rendering_alignment": "not_tested",
                "token_id_alignment": "not_tested",
                "numerical_parity": "not_tested",
            }
            _write_json(result_path, row)
            rows.append(row)
            print("[case complete] independent image answers scored locally")
        except Exception as exc:
            rows.append({"status": "timeout" if isinstance(exc, TimeoutError) else "error",
                         "reason": str(exc)})
            _write_json(result_path, rows[-1])
            print("[case error] inspect private result and oracle log")
            if not args.continue_on_failure:
                break
    _write_json(output / "summary.json", {
        "schema": SCHEMA, "schema_version": 1,
        "status": "complete" if len(rows) == len(samples) and all(row.get("status") == "complete" for row in rows) else "incomplete",
        "scope": "independent_image_to_answer_task_comparison",
        "numerical_parity": "not_tested",
        "chat_rendering_alignment": "not_tested",
        "token_id_alignment": "not_tested",
        "config_sha256": global_hash,
        "aggregate": _aggregate(rows, len(samples)),
    })
    return 0 if len(rows) == len(samples) and all(row.get("status") == "complete" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
