#!/usr/bin/env python3
"""Resolve circuit-owned serving profiles into an offline deployment bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

V8_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V8_ROOT.parents[1]))
from server.serving_bundle import PROTOCOLS, SCHEMA, contract_identity, load_resolved_serving, sha256_file


def _json(path: Path) -> dict:
    doc = json.loads(path.read_bytes())
    if not isinstance(doc, dict):
        raise ValueError(f"expected an object: {path}")
    return doc


def _source_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("profile reference must be relative")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"missing or escaping serving profile/asset: {relative}")
    return path


def _publisher_templates(run_dir: Path) -> dict[str, tuple[bytes, str]]:
    found: dict[str, tuple[bytes, str]] = {}

    def add(name: str, value: bytes, source: str) -> None:
        if not value.decode("utf-8").strip():
            raise ValueError(f"empty publisher template: {source}")
        if name in found and found[name][0] != value:
            raise ValueError(f"conflicting publisher template {name}: {source}")
        found[name] = (value, source)

    native = run_dir / "chat_template.jinja"
    if native.is_file():
        add("default", native.read_bytes(), "chat_template.jinja")
    # Legacy additional_chat_templates can contain operator-installed overrides.
    # Do not relabel them as publisher variants without an imported declaration.
    tokenizer = run_dir / "tokenizer_config.json"
    if tokenizer.is_file():
        templates = _json(tokenizer).get("chat_template")
        if isinstance(templates, str):
            add("default", templates.encode("utf-8"), "tokenizer_config.json:chat_template")
        elif isinstance(templates, dict):
            for name, value in templates.items():
                if not isinstance(name, str) or not isinstance(value, str):
                    raise ValueError("invalid named publisher template")
                add(name, value.encode("utf-8"), f"tokenizer_config.json:{name}")
        elif isinstance(templates, list):
            for item in templates:
                if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not isinstance(item.get("template"), str):
                    raise ValueError("invalid publisher template variant")
                add(item["name"], item["template"].encode("utf-8"), f"tokenizer_config.json:{item['name']}")
        elif templates is not None:
            raise ValueError("invalid publisher chat_template")
    if "default" not in found:
        raise ValueError("publisher default template is missing")
    return found


def resolve_serving_bundle(run_dir: Path, circuit_path: Path, *, variant: str | None = None, v8_root: Path = V8_ROOT) -> dict:
    run_dir, circuit_path, v8_root = Path(run_dir), Path(circuit_path), Path(v8_root)
    circuit = _json(circuit_path)
    declaration = circuit.get("serving")
    if not isinstance(declaration, dict) or declaration.get("schema") != "cke.circuit_serving.v1":
        raise ValueError("circuit has no supported serving declaration")
    profile_ref = declaration.get("profile_ref")
    profile_path = _source_path(v8_root, profile_ref)
    profile = _json(profile_path)
    if profile.get("schema") != "cke.serving_profile.v1" or profile.get("renderer") != "jinja-chat-v1" or profile.get("input_modalities") != ["text"]:
        raise ValueError("unsupported serving profile contract")
    selected = variant or declaration.get("default_variant")
    spec = profile.get("variants", {}).get(selected)
    if not isinstance(spec, dict) or spec.get("output_protocol") not in PROTOCOLS:
        raise ValueError(f"unsupported serving variant: {selected}")
    config = _json(run_dir / "config.json")
    if config.get("model") != circuit.get("name"):
        raise ValueError("circuit does not match runtime configuration")
    publisher = _publisher_templates(run_dir)
    assets: dict[str, dict] = {}

    def store(name: str, data: bytes, provenance: str, suffix: str) -> None:
        digest = hashlib.sha256(data).hexdigest()
        relative = f"serving/assets/{digest}{suffix}"
        path = run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != data:
            raise ValueError("conflicting content-addressed serving asset")
        if not path.exists():
            path.write_bytes(data)
        assets[name] = {"path": relative, "sha256": digest, "source": provenance}

    store("circuit", circuit_path.read_bytes(), "circuit-template", ".json")
    store("profile", profile_path.read_bytes(), profile_ref, ".json")
    for role in ("chat", "tools"):
        binding = spec.get(role)
        if not isinstance(binding, dict):
            raise ValueError(f"missing template binding: {role}")
        if binding.get("source") == "publisher":
            name = binding.get("variant")
            if name not in publisher and "fallback_variant" in binding:
                name = binding["fallback_variant"]
            if name not in publisher:
                raise ValueError(f"missing publisher template variant: {name}")
            data, source = publisher[name]
        elif binding.get("source") == "profile_asset":
            path = _source_path(profile_path.parent, binding.get("path"))
            data, source = path.read_bytes(), f"{profile_ref}:{binding['path']}"
        else:
            raise ValueError(f"unsupported template source: {role}")
        if not data.decode("utf-8").strip():
            raise ValueError(f"empty selected template: {role}")
        store(role, data, source, ".jinja")
    required = ("config.json", "layout_decode.json", "libmodel.so", "libckernel_engine.so", "libckernel_tokenizer.so", "weights.bump", "weights_manifest.map")
    optional = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model", "tiktoken.model", "generation_config.json")
    for name in required + optional:
        path = run_dir / name
        if name in required or path.is_file():
            assets[name] = {"path": name, "sha256": sha256_file(path), "source": "generated-or-imported-bundle"}
    capacity = _json(run_dir / "layout_decode.json").get("config", {}).get("context_length")
    document = {"schema": SCHEMA, "profile_id": profile.get("id"), "profile_ref": profile_ref, "variant": selected,
                "renderer": profile["renderer"], "output_protocol": spec["output_protocol"],
                "input_modalities": profile["input_modalities"], "context_capacity": capacity,
                "assets": assets, "certification": "NOT_TESTED"}
    document["identity"] = contract_identity(document)
    load_resolved_serving(run_dir, document=document)
    # Publish last: an interrupted asset write cannot publish a new contract.
    sidecar = run_dir / "serving.json"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=run_dir,
                                     prefix=".serving-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(document, indent=2) + "\n")
    try:
        temporary.replace(sidecar)
    finally:
        temporary.unlink(missing_ok=True)
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--circuit", required=True, type=Path)
    parser.add_argument("--variant", default=None)
    args = parser.parse_args()
    doc = resolve_serving_bundle(args.run, args.circuit, variant=args.variant)
    print(f"Resolved serving bundle {doc['identity']} ({doc['variant']}); certification NOT_TESTED")


if __name__ == "__main__":
    main()
