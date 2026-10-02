"""Validate resolved serving data without consulting repository model profiles."""
from __future__ import annotations

import hashlib
import ctypes
import json
import os
import re
from pathlib import Path
from typing import Any

PROTOCOLS = frozenset({"none", "tagged_json", "bare_json", "qwen_xml", "qwen_code_xml", "qwen_code_xml_raw_v2"})
SCHEMA = "cke.resolved_serving.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contract_identity(document: dict[str, Any]) -> str:
    payload = {key: value for key, value in document.items() if key != "identity"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def bundle_path(run_dir: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("serving asset path must be bundle-relative")
    path = run_dir / relative
    if not path.resolve().is_relative_to(run_dir.resolve()):
        raise ValueError(f"serving asset escapes bundle: {relative}")
    return path


def load_resolved_serving(run_dir: Path, *, document: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Return validated data; an invalid present sidecar never falls back."""
    run_dir = Path(run_dir)
    sidecar = run_dir / "serving.json"
    if document is None and not sidecar.exists() and not sidecar.is_symlink():
        return None  # Explicit legacy bundle support, not circuit certification.
    try:
        doc = document if document is not None else json.loads(sidecar.read_bytes())
        if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
            raise ValueError("invalid resolved serving schema")
        if doc.get("identity") != contract_identity(doc):
            raise ValueError("resolved serving identity mismatch")
        if doc.get("renderer") != "jinja-chat-v1":
            raise ValueError("unsupported serving renderer")
        if doc.get("output_protocol") not in PROTOCOLS:
            raise ValueError("unsupported serving output protocol")
        if doc.get("input_modalities") != ["text"]:
            raise ValueError("this resolved serving path supports text only")
        assets = doc.get("assets")
        if not isinstance(assets, dict) or not assets:
            raise ValueError("missing serving asset inventory")
        required = {"circuit", "profile", "chat", "tools", "config.json", "layout_decode.json", "libmodel.so", "libckernel_engine.so", "libckernel_tokenizer.so", "weights.bump", "weights_manifest.map"}
        if not required.issubset(assets):
            raise ValueError("incomplete serving asset inventory")
        for name, item in assets.items():
            if not isinstance(item, dict):
                raise ValueError(f"invalid serving asset: {name}")
            path = bundle_path(run_dir, item.get("path"))
            if name.endswith((".json", ".so", ".bump", ".model", ".map")) and name not in {"circuit", "profile", "chat", "tools"} and item.get("path") != name:
                raise ValueError(f"serving runtime asset has wrong binding: {name}")
            if sha256_file(path) != item.get("sha256"):
                raise ValueError(f"stale serving asset: {name}")
        circuit = json.loads(bundle_path(run_dir, assets["circuit"]["path"]).read_bytes())
        profile = json.loads(bundle_path(run_dir, assets["profile"]["path"]).read_bytes())
        config = json.loads(bundle_path(run_dir, assets["config.json"]["path"]).read_bytes())
        layout = json.loads(bundle_path(run_dir, assets["layout_decode.json"]["path"]).read_bytes())
        declaration = circuit.get("serving", {})
        if declaration.get("schema") != "cke.circuit_serving.v1" or declaration.get("profile_ref") != doc.get("profile_ref"):
            raise ValueError("serving circuit/profile ownership mismatch")
        if config.get("model") is not None and config["model"] != circuit.get("name"):
            raise ValueError("serving circuit does not match runtime configuration")
        if config.get("model") is None:
            manifest_asset = assets.get("weights_manifest.json")
            if manifest_asset is None:
                raise ValueError("serving circuit has no compiled manifest identity")
            manifest = json.loads(bundle_path(run_dir, manifest_asset["path"]).read_bytes())
            if manifest.get("model") != circuit.get("name"):
                raise ValueError("serving circuit does not match compiled manifest model")
        if profile.get("schema") != "cke.serving_profile.v1" or profile.get("id") != doc.get("profile_id"):
            raise ValueError("serving profile identity mismatch")
        variant = profile.get("variants", {}).get(doc.get("variant"))
        if not isinstance(variant, dict) or variant.get("output_protocol") != doc["output_protocol"]:
            raise ValueError("serving variant/protocol mismatch")
        if profile.get("renderer") != doc["renderer"] or profile.get("input_modalities") != doc["input_modalities"]:
            raise ValueError("serving profile capabilities mismatch")
        capacity = layout.get("config", {}).get("context_length")
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0 or capacity != doc.get("context_capacity"):
            raise ValueError("serving context capacity mismatch")
        for role in ("chat", "tools"):
            if not bundle_path(run_dir, assets[role]["path"]).read_bytes().decode("utf-8").strip():
                raise ValueError(f"empty serving template: {role}")
        return doc
    except (OSError, UnicodeError, TypeError, KeyError, AttributeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid resolved serving bundle {sidecar}: {exc}") from exc


def resolved_templates(run_dir: Path, doc: dict[str, Any]) -> tuple[str, dict[str, str], None]:
    assets = doc["assets"]
    chat = bundle_path(run_dir, assets["chat"]["path"]).read_bytes().decode("utf-8")
    tools = bundle_path(run_dir, assets["tools"]["path"]).read_bytes().decode("utf-8")
    return chat, {"tool_use": tools}, None


def resolved_renderer_tokens(run_dir: Path, doc: dict[str, Any]) -> dict[str, str]:
    """Read publisher Jinja token variables only from verified bundle assets.

    The caller must first validate ``doc`` with ``load_resolved_serving``. A
    missing variable stays undefined so a template that requires it fails
    rather than rendering an invented delimiter.
    """
    variables: dict[str, str] = {}

    def bind(key: str, value: str) -> None:
        if not value:
            raise ValueError(f"invalid publisher token variable: {key}")
        if key in variables and variables[key] != value:
            raise ValueError(f"conflicting publisher token variable: {key}")
        variables[key] = value

    for name in ("special_tokens_map.json", "tokenizer_config.json"):
        asset = doc["assets"].get(name)
        if asset is None:
            continue
        payload = json.loads(bundle_path(Path(run_dir), asset["path"]).read_bytes())
        if not isinstance(payload, dict):
            raise ValueError(f"invalid publisher token metadata: {name}")
        for key, value in payload.items():
            if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*_token", key):
                continue
            if isinstance(value, dict):
                value = value.get("content")
            if value is None or not isinstance(value, str):
                # An unsupported metadata shape cannot supply a Jinja scalar;
                # templates requiring it still fail under StrictUndefined.
                continue
            bind(key, value)

    # A GGUF conversion can retain special-token IDs in its verified manifest
    # without an HF tokenizer_config.json. Resolve their text only when the
    # exact ID is present in the bundled tokenizer.json added-token table.
    manifest_asset = doc["assets"].get("weights_manifest.json")
    tokenizer_asset = doc["assets"].get("tokenizer.json")
    if manifest_asset is not None and tokenizer_asset is not None:
        manifest = json.loads(bundle_path(Path(run_dir), manifest_asset["path"]).read_bytes())
        tokenizer = json.loads(bundle_path(Path(run_dir), tokenizer_asset["path"]).read_bytes())
        if not isinstance(manifest, dict) or not isinstance(tokenizer, dict):
            raise ValueError("invalid bundled GGUF token metadata")
        special = manifest.get("special_tokens", {})
        added = tokenizer.get("added_tokens", [])
        if not isinstance(special, dict) or not isinstance(added, list):
            raise ValueError("invalid bundled GGUF special-token metadata")
        by_id: dict[int, str] = {}
        for item in added:
            if not isinstance(item, dict):
                continue
            token_id, content = item.get("id"), item.get("content")
            if (not isinstance(token_id, int) or isinstance(token_id, bool)
                    or not isinstance(content, str)):
                continue
            if token_id in by_id and by_id[token_id] != content:
                raise ValueError(f"conflicting bundled tokenizer text for ID {token_id}")
            by_id[token_id] = content
        for name in ("bos", "eos", "unk", "pad"):
            token_id = special.get(f"{name}_token_id")
            if isinstance(token_id, int) and not isinstance(token_id, bool):
                value = by_id.get(token_id)
                if value is not None:
                    bind(f"{name}_token", value)
    return variables


def verify_loaded_libraries(doc: dict[str, Any], *, maps_text: str | None = None) -> None:
    """Bind file validation to actual Linux native dependencies after session open."""
    if maps_text is None:
        try:
            maps_text = Path("/proc/self/maps").read_text()
        except OSError as exc:
            raise ValueError("resolved serving requires Linux loaded-library identity inspection") from exc
    paths = set()
    for line in maps_text.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            paths.add(fields[5])
    for name in ("libmodel.so", "libckernel_engine.so", "libckernel_tokenizer.so"):
        loaded = {Path(path) for path in paths if Path(path.removesuffix(" (deleted)")).name == name}
        if not loaded:
            raise ValueError(f"required serving library is not loaded: {name}")
        expected = doc["assets"][name]["sha256"]
        for path in loaded:
            try:
                matches = sha256_file(path) == expected
            except OSError as exc:
                raise ValueError(f"cannot identify loaded serving library: {path}") from exc
            if not matches:
                raise ValueError(f"loaded serving library identity mismatch: {name}")


def loaded_serving_identity(
    doc: dict[str, Any], *, model: str, session_library: ctypes.CDLL,
) -> dict[str, Any]:
    """Describe a verified loaded bundle without publishing filesystem paths.

    Call only after ``verify_loaded_libraries`` succeeds on the open session.
    The session ABI library is outside the BUMP bundle, so identify its loaded
    file separately instead of implying that the bundle pins it.
    """
    try:
        symbol_address = ctypes.cast(session_library.ck_session_v8_open, ctypes.c_void_p).value
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("cannot identify loaded native session symbol") from exc
    if not symbol_address:
        raise ValueError("cannot identify loaded native session symbol")
    try:
        maps = Path("/proc/self/maps").read_text()
    except OSError as exc:
        raise ValueError("cannot inspect loaded native session mapping") from exc
    candidates = []
    selected = []
    for row in maps.splitlines():
        fields = row.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        path = fields[5]
        if Path(path.removesuffix(" (deleted)")).name != "libck_session_v8.so":
            continue
        start, end = (int(part, 16) for part in fields[0].split("-", 1))
        entry = (fields, path)
        candidates.append(entry)
        if start <= symbol_address < end and "x" in fields[1]:
            selected.append(entry)
    if len(selected) != 1 or not candidates:
        raise ValueError("loaded native session symbol has no unique executable mapping")
    fields, path = selected[0]
    if path.endswith(" (deleted)") or any(
        item[0][3:5] != fields[3:5] for item in candidates if item[1] == path
    ):
        raise ValueError("loaded native session mapping is deleted or ambiguous")
    library = Path(path)
    try:
        with library.open("rb") as stream:
            backing = os.fstat(stream.fileno())
            device = f"{os.major(backing.st_dev):02x}:{os.minor(backing.st_dev):02x}"
            if fields[3].lower() != device or int(fields[4]) != backing.st_ino:
                raise ValueError("loaded native session backing file was replaced")
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
            current = library.stat()
            if current.st_dev != backing.st_dev or current.st_ino != backing.st_ino:
                raise ValueError("loaded native session backing file was replaced")
    except OSError as exc:
        raise ValueError("cannot read loaded native session backing file") from exc
    return {
        "schema": "cke.loaded_serving_identity.v1",
        "model": model,
        "serving_identity": doc["identity"],
        "profile_id": doc["profile_id"],
        "variant": doc["variant"],
        "output_protocol": doc["output_protocol"],
        "context_capacity": doc["context_capacity"],
        "assets_sha256": {
            name: asset["sha256"] for name, asset in sorted(doc["assets"].items())
        },
        "session_library_sha256": digest.hexdigest(),
    }
