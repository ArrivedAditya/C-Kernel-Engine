#!/usr/bin/env python3
from __future__ import annotations

"""Certify production-formatted text prompts against a pinned llama.cpp oracle."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path[:0] = [str(SCRIPT_DIR), str(REPO_ROOT / "scripts")]

from compare_multitoken_logits_v8 import run_multitoken_trajectory_parity  # type: ignore  # noqa: E402
from gguf_tokenizer import GGUFTokenizer  # type: ignore  # noqa: E402
from run_multimodal_bridge_v8 import (  # type: ignore  # noqa: E402
    _encode_prompt_segment,
    _format_prompt_with_chat_contract,
    _resolve_decoder_chat_contract,
)


CORRUPTION_MARKERS = (
    "\\uFFFD",
    "\ufffd",
    "\u00c3",
    "\u00c2",
    "\u00e2\u20ac",
    "\u00f0\u0178",
    "\ufffd\u0141",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prompt_set(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("prompt set schema_version must be 1")
    stages = [int(value) for value in payload.get("stages", [])]
    if not stages or stages != sorted(set(stages)) or any(value <= 0 for value in stages):
        raise ValueError("prompt stages must be unique, positive, and increasing")
    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("prompt set must contain prompts")
    ids = [str(row.get("id", "")) for row in prompts]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("prompt IDs must be non-empty and unique")
    return payload


def format_and_tokenize_prompts(
    prompt_set: dict[str, Any], gguf_path: Path
) -> list[dict[str, Any]]:
    tokenizer = GGUFTokenizer.from_gguf(str(gguf_path))
    contract = _resolve_decoder_chat_contract(
        gguf_path, chat_template_mode=str(prompt_set.get("chat_template_mode", "auto"))
    )
    rows: list[dict[str, Any]] = []
    for source in prompt_set["prompts"]:
        formatted = _format_prompt_with_chat_contract(
            str(source["text"]),
            contract,
            thinking_mode=str(prompt_set.get("thinking_mode", "auto")),
        )
        tokens = _encode_prompt_segment(tokenizer, formatted, add_bos=True)
        expected = [int(value) for value in source.get("tokens", [])]
        if tokens != expected:
            raise ValueError(
                f"production prompt tokens changed for {source['id']}: "
                f"expected={expected} actual={tokens}"
            )
        rows.append({**source, "formatted": formatted, "tokens": tokens})
    return rows


def decoded_text_is_clean(text: str) -> bool:
    return not any(marker in text for marker in CORRUPTION_MARKERS)


def report_satisfies_stage(report: dict[str, Any], stage: int) -> bool:
    if not bool(report.get("pass")):
        return False
    if report.get("matched_stop_token") is not None:
        return True
    return len(report.get("steps", [])) >= int(stage)


def reusable_report_path(output_dir: Path, prompt_id: str, stages: list[int], stage: int) -> Path | None:
    for previous_stage in reversed([value for value in stages if value < stage]):
        candidate = output_dir / f"{prompt_id}-{previous_stage}.json"
        if not candidate.exists():
            continue
        report = json.loads(candidate.read_text(encoding="utf-8"))
        if report_satisfies_stage(report, stage):
            return candidate
    return None


def xray_handoff(
    model_dir: Path, gguf_path: Path, parity_report: Path, output_root: Path, threads: int
) -> str:
    capture_root = output_root / f"{parity_report.stem}-xray"
    return (
        "python3 version/v8/scripts/xray_text_recurrent_v8.py "
        f"--model-dir {model_dir} --gguf {gguf_path} "
        f"--parity-report {parity_report} --capture-root {capture_root} "
        f"--output {capture_root / 'report.json'} --ctx-len 1034 "
        f"--threads {threads} --ck-prefill-mode hybrid"
    )


def git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--prompt-set", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ctx-len", type=int, default=1034)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    gguf_path = args.gguf.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_set = load_prompt_set(args.prompt_set.resolve())
    prompts = format_and_tokenize_prompts(prompt_set, gguf_path)

    llama_root_value = os.environ.get("CK_LLAMA_CPP_ROOT", "").strip()
    llama_root = Path(llama_root_value).resolve() if llama_root_value else None
    expected_llama_commit = str(prompt_set.get("llama_cpp_commit", ""))
    actual_llama_commit = git_head(llama_root) if llama_root else ""
    if not args.dry_run and actual_llama_commit != expected_llama_commit:
        raise RuntimeError(
            f"llama.cpp oracle commit mismatch: expected={expected_llama_commit} "
            f"actual={actual_llama_commit or 'unavailable'}"
        )

    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "pass",
        "prompt_set": str(prompt_set["name"]),
        "stages": [int(value) for value in prompt_set["stages"]],
        "prompts": [],
        "provenance": {
            "cke_commit": git_head(REPO_ROOT),
            "llama_cpp_commit": actual_llama_commit,
            "gguf_sha256": sha256_file(gguf_path),
            "engine_sha256": sha256_file(model_dir / "libckernel_engine.so"),
            "model_runtime_sha256": sha256_file(model_dir / "libmodel.so"),
            "threads": int(args.threads),
        },
    }
    if args.dry_run:
        summary["prompts"] = prompts
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0

    tokenizer = GGUFTokenizer.from_gguf(str(gguf_path))
    stop_tokens = {int(value) for value in prompt_set.get("stop_token_ids", [])}
    for stage in summary["stages"]:
        for prompt in prompts:
            report_path = output_dir / f"{prompt['id']}-{stage}.json"
            if report_path.exists():
                existing = json.loads(report_path.read_text(encoding="utf-8"))
                if report_satisfies_stage(existing, int(stage)):
                    continue
            reusable = reusable_report_path(
                output_dir, str(prompt["id"]), summary["stages"], int(stage)
            )
            if reusable is not None:
                report = json.loads(reusable.read_text(encoding="utf-8"))
                report["certified_stage"] = int(stage)
                report["reused_eos_report"] = str(reusable)
                report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
                continue
            report = run_multitoken_trajectory_parity(
                model_dir=model_dir,
                gguf_path=gguf_path,
                prompt_tokens=[int(value) for value in prompt["tokens"]],
                max_new_tokens=int(stage),
                ctx_len=int(args.ctx_len),
                top_k=int(args.top_k),
                threads=int(args.threads),
                llama_no_repack=False,
                stop_token_ids=stop_tokens,
            )
            generated = report["final_prefix"][len(prompt["tokens"]) :]
            decoded = tokenizer.decode(generated, skip_special=True)
            report["prompt_id"] = str(prompt["id"])
            report["prompt_text"] = str(prompt["text"])
            report["formatted_prompt"] = str(prompt["formatted"])
            report["decoded_text"] = decoded
            report["utf8_clean"] = decoded_text_is_clean(decoded)
            report["xray_handoff"] = xray_handoff(
                model_dir, gguf_path, report_path, output_dir, int(args.threads)
            )
            report["pass"] = bool(report["pass"] and report["utf8_clean"])
            report["status"] = "pass" if report["pass"] else "fail"
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            if not report["pass"]:
                summary["status"] = "fail"
                summary["first_failure"] = {
                    "prompt_id": prompt["id"],
                    "stage": int(stage),
                    "report": str(report_path),
                    "xray_handoff": report["xray_handoff"],
                }
                (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
                print(json.dumps(summary, indent=2))
                return 3

    for prompt in prompts:
        final_path = output_dir / f"{prompt['id']}-{summary['stages'][-1]}.json"
        final_report = json.loads(final_path.read_text(encoding="utf-8"))
        summary["prompts"].append(
            {
                "id": prompt["id"],
                "status": final_report["status"],
                "steps": len(final_report.get("steps", [])),
                "matched_stop_token": final_report.get("matched_stop_token"),
                "utf8_clean": bool(final_report.get("utf8_clean")),
                "report": str(final_path),
            }
        )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
