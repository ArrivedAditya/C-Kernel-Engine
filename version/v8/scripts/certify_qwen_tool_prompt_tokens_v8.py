#!/usr/bin/env python3
"""Reproduce exact Qwen3.8 tool-prompt IDs through a generated v8 runtime.

The fixture is synthetic: it contains no captured Qwen Code instructions. It
exercises the same long Jinja/tool-definition, special-token, and indentation
boundaries that exposed silent native-tokenizer truncation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from server.live import _render_with_chat_templates
from server.session_v8 import SessionV8


GGUF_SHA256 = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
TEMPLATE_SHA256 = "ab34ce52b89d4a0fed64dc6a71070c20b22ec084ec6980f50c49e9f81588919d"
# Pinned against llama-tokenize on the GGUF above.
PROMPT_SHA256 = "c9c3a13d7c813804db0a6ff4690a2cd2e35ff5525e6124432bb4d4c7a3ed2e14"
TOKEN_IDS_SHA256 = "d0c078c4a7c4aedbc95dd12aafe2e4900db175639e2ee952e60d235d2b254685"
TOKEN_COUNT = 6688


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixture_prompt(template: str) -> str:
    class Tool:
        def __init__(self, name: str, description: str) -> None:
            self.name = name
            self.description = description

        def model_dump(self) -> dict:
            return {
                "type": "function",
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Absolute path  - preserve spaces."},
                    },
                    "required": ["file_path"],
                },
            }

    instructions = "Follow the advertised tool names exactly.\n" + "".join(
        f"Rule {index:04d}: read_file is read-only.  - Keep indentation, Unicode café, and trailing newlines intact.\n"
        for index in range(190)
    )
    tools = [
        Tool("read_file", "Read an exact file.  - Preserve UTF-8 and whitespace. " * 45),
        Tool("edit", "Replace exact source text.  - Never normalize whitespace. " * 45),
    ]
    body = SimpleNamespace(instructions=instructions, tools=tools, tool_choice="auto")
    rendered = _render_with_chat_templates(
        None,
        {"tool_use": template},
        [{"role": "user", "content": "Use read_file to inspect /tmp/cke-tool-probe.txt."}],
        body,
    )
    if rendered is None:
        raise RuntimeError("tool template did not render")
    return rendered


def reference_ids(executable: Path, gguf: Path, prompt: bytes) -> list[int]:
    with tempfile.TemporaryDirectory(prefix="cke-qwen-tool-tokens-") as temp:
        prompt_path = Path(temp) / "prompt.txt"
        prompt_path.write_bytes(prompt)
        result = subprocess.run(
            [str(executable), "-m", str(gguf), "-f", str(prompt_path),
             "--ids", "--no-bos", "--no-escape", "--log-disable"],
            check=True, capture_output=True, text=True,
        )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip().startswith("[")]
    if not lines:
        raise RuntimeError("llama-tokenize did not emit token IDs")
    ids = ast.literal_eval(lines[-1])
    if not isinstance(ids, list) or not all(isinstance(item, int) for item in ids):
        raise RuntimeError("invalid independent token-ID output")
    return ids


def ids_sha256(ids: list[int]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(ids)}i", *ids)).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--llama-tokenize", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    template_path = args.runtime / "additional_chat_templates" / "tool_use.jinja"
    template_bytes = template_path.read_bytes()
    template_hash = hashlib.sha256(template_bytes).hexdigest()
    if template_hash != TEMPLATE_SHA256:
        raise RuntimeError(f"unexpected tool template sha256: {template_hash}")
    gguf_hash = sha256_file(args.gguf)
    if gguf_hash != GGUF_SHA256:
        raise RuntimeError(f"unexpected GGUF sha256: {gguf_hash}")
    prompt = fixture_prompt(template_bytes.decode("utf-8")).encode("utf-8")
    prompt_hash = hashlib.sha256(prompt).hexdigest()
    if len(prompt) <= 16384:
        raise RuntimeError("fixture no longer exercises the long-segment boundary")

    oracle = reference_ids(args.llama_tokenize, args.gguf, prompt)
    session = SessionV8.open(args.runtime, context_length=262144, num_threads=1)
    try:
        count = session.count_tokens(prompt.decode("utf-8"))
        import ctypes
        output = (ctypes.c_int32 * count)()
        written = session.lib.ck_session_v8_encode(session.session, prompt, output, count)
        if written != count:
            raise RuntimeError(f"generated tokenizer count/write mismatch: {count}/{written}")
        actual = list(output)
    finally:
        session.close()

    first_difference = next(
        (index for index, (left, right) in enumerate(zip(actual, oracle)) if left != right),
        None,
    )
    if first_difference is None and len(actual) != len(oracle):
        first_difference = min(len(actual), len(oracle))
    passed = (
        len(actual) == len(oracle)
        and first_difference is None
        and prompt_hash == PROMPT_SHA256
        and len(oracle) == TOKEN_COUNT
        and ids_sha256(oracle) == TOKEN_IDS_SHA256
    )
    report = {
        "schema": "cke.qwen_tool_prompt_token_parity.v1",
        "passed": passed,
        "prompt_bytes": len(prompt),
        "prompt_sha256": prompt_hash,
        "template_sha256": template_hash,
        "gguf_sha256": gguf_hash,
        "expected_token_count": TOKEN_COUNT,
        "oracle_token_count": len(oracle),
        "generated_token_count": len(actual),
        "oracle_ids_sha256": ids_sha256(oracle),
        "generated_ids_sha256": ids_sha256(actual),
        "first_difference": first_difference,
        "generated_library_sha256": sha256_file(args.runtime / "libmodel.so"),
        "engine_library_sha256": sha256_file(args.runtime / "libckernel_engine.so"),
        "tokenizer_library_sha256": sha256_file(args.runtime / "libckernel_tokenizer.so"),
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
