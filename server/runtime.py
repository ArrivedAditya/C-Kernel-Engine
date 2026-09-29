"""Runtime artifact helpers for ``server/`` (no FastAPI here).

The server resolves the generated capacity (``layout_decode.json``) and the
canonical ``chat_template.jinja`` sidecar without importing the scripts tree.
``weights_manifest.json`` is never consulted for prompt rendering. Path
constants are shared with ``server.session_v8`` (single source of truth).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from .session_v8 import BUILD_DIR, PROJECT_ROOT, SESSION_LIB_PATH
from .serving_bundle import load_resolved_serving, resolved_templates


def ensure_native_session_lib() -> None:
    """Build ``libck_session_v8.so`` if it is not already present."""
    if SESSION_LIB_PATH.is_file():
        return
    subprocess.run(["make", "ck-session-v8"], cwd=str(PROJECT_ROOT), check=True)


def resolve_runtime_context_length(run_dir: Path, requested: int | None) -> int | None:
    """Return the generated plan's capacity, guarding oversized requests."""
    layout_path = Path(run_dir) / "layout_decode.json"
    try:
        payload = json.loads(layout_path.read_text(encoding="utf-8"))
        value = payload.get("config", {}).get("context_length")
    except (OSError, UnicodeDecodeError, ValueError, AttributeError):
        value = None
    planned = value if isinstance(value, int) and value > 0 else None
    if requested is not None:
        if planned is not None and requested > planned:
            raise ValueError(
                f"requested context length {requested} exceeds generated runtime "
                f"capacity {planned}; rebuild with --context-len {requested}"
            )
        return requested
    return planned


def resolve_runtime_vision_capability(run_dir: Path) -> bool:
    """Whether the generated runtime has a vision encoder.

    Reads ``layout_decode.json`` config for the vision path markers used by
    the bridge contract (patch counts plus projector dims). Text-only
    bundles lack these keys and report ``False``; unreadable manifests
    fail closed to ``False``.
    """
    layout_path = Path(run_dir) / "layout_decode.json"
    try:
        payload = json.loads(layout_path.read_text(encoding="utf-8"))
        config = payload.get("config", {})
    except (OSError, UnicodeDecodeError, ValueError, AttributeError):
        return False
    if not isinstance(config, dict):
        return False

    def _positive(*keys: str) -> bool:
        for key in keys:
            try:
                value = int(config.get(key) or 0)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return True
        return False

    return _positive("vision_num_patches", "vision_merged_tokens") and _positive(
        "projector_out_dim", "projection_dim", "projector_total_out_dim"
    )


def load_manifest_templates(
    run_dir: Path,
) -> tuple[str | None, dict[str, str] | None, dict[str, Any] | None]:
    """Load (chat_template, chat_templates, chat_contract) for a run dir.

    Resolved bundles use only their validated serving.json and packaged assets.
    Legacy bundles use the canonical chat_template.jinja sidecar.
    ``weights_manifest.json`` / ``config.json`` are never read here, and no
    chat contract is loaded from disk (always ``None``) — prompt rendering
    is pure Jinja from the sidecar.
    """
    run_dir = Path(run_dir)
    resolved = load_resolved_serving(run_dir)
    if resolved is not None:
        return resolved_templates(run_dir, resolved)
    chat_template: str | None = None
    chat_templates: dict[str, str] | None = None
    chat_contract: dict[str, Any] | None = None
    sidecar = run_dir / "chat_template.jinja"
    if sidecar.is_file():
        try:
            txt = sidecar.read_bytes().decode("utf-8")
        except (OSError, UnicodeError) as exc:
            raise ValueError(f"cannot read native chat template {sidecar}: {exc}") from exc
        if not txt.strip():
            raise ValueError(f"native chat template is empty: {sidecar}")
        chat_template = txt
    additional_dir = run_dir / "additional_chat_templates"
    if additional_dir.is_dir():
        collected: dict[str, str] = {}
        for jinja_file in additional_dir.glob("*.jinja"):
            try:
                txt = jinja_file.read_bytes().decode("utf-8")
            except (OSError, UnicodeError) as exc:
                raise ValueError(f"cannot read chat template {jinja_file}: {exc}") from exc
            if not txt.strip():
                raise ValueError(f"chat template is empty: {jinja_file}")
            collected[jinja_file.stem] = txt
        if collected:
            chat_templates = collected
    return chat_template, chat_templates, chat_contract


def load_tool_protocol(
    run_dir: Path,
    chat_template: str | None,
    chat_templates: dict[str, str] | None,
) -> str | None:
    """Load an explicit tool wire protocol bound to the selected Jinja source.

    A template mentioning tools is not a capability declaration. The optional
    sidecar is operator-authored and deliberately separate from executed
    certification evidence.
    """
    resolved = load_resolved_serving(Path(run_dir))
    if resolved is not None:
        expected_chat, expected_variants, _ = resolved_templates(Path(run_dir), resolved)
        if chat_template != expected_chat or chat_templates != expected_variants:
            raise ValueError("template override conflicts with resolved serving bundle")
        return resolved["output_protocol"]
    path = Path(run_dir) / "tool_protocol.json"
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read tool protocol sidecar {path}: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema") != "cke.v8.tool_protocol.v1":
        raise ValueError(f"invalid tool protocol sidecar schema: {path}")
    protocol = document.get("protocol")
    if protocol not in {
        "tagged_json", "bare_json", "qwen_xml", "qwen_code_xml", "qwen_code_xml_raw_v2",
    }:
        raise ValueError(f"unsupported tool protocol {protocol!r} in {path}")
    selected = None
    if isinstance(chat_templates, dict):
        for key in ("tool_use", "tools", "default"):
            value = chat_templates.get(key)
            if isinstance(value, str) and value.strip():
                selected = value
                break
    if selected is None:
        selected = chat_template
    if not isinstance(selected, str) or not selected.strip():
        raise ValueError(f"tool protocol sidecar {path} has no selected chat template")
    actual = hashlib.sha256(selected.encode("utf-8")).hexdigest()
    if document.get("template_sha256") != actual:
        raise ValueError(
            f"tool protocol sidecar {path} does not match selected chat template "
            f"sha256:{actual}"
        )
    return protocol
