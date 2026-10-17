"""Regression tests: serving must never treat a local source artifact as its
disposable build/run directory.

ck_serve_runtime_v8._resolve_run_dir used to return the local GGUF path
itself for local inputs, so `cks-v8-run serve model.gguf` crashed with
FileExistsError when the runtime build tried to mkdir the source file, and
source directories risked absorbing build output. Resolution now mirrors the
CLI run path: build output goes under the cache; only an already-compiled
runtime directory is used in place.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "version" / "v8" / "scripts"))

import ck_run_v8
import ck_serve_runtime_v8


def _resolve(model: str) -> Path:
    return ck_serve_runtime_v8._resolve_run_dir(model, None)


def test_local_gguf_resolves_to_cache_not_source(tmp_path: Path) -> None:
    source_dir = tmp_path / "models with spaces"
    source_dir.mkdir()
    gguf = source_dir / "tiny model.gguf"
    gguf.write_bytes(b"GGUF fake")
    before = sorted(p.name for p in source_dir.iterdir())

    run_dir = _resolve(str(gguf))

    assert run_dir != gguf
    assert run_dir != source_dir
    assert run_dir.parent == ck_run_v8.CACHE_DIR
    assert run_dir.name == gguf.stem
    # Resolution must not touch the source artifact or its directory.
    assert gguf.read_bytes() == b"GGUF fake"
    assert sorted(p.name for p in source_dir.iterdir()) == before


def test_local_gguf_resolution_is_repeatable(tmp_path: Path) -> None:
    gguf = tmp_path / "repeat.gguf"
    gguf.write_bytes(b"GGUF fake")
    first = _resolve(str(gguf))
    second = _resolve(str(gguf))
    assert first == second


def test_compiled_runtime_dir_is_used_in_place(tmp_path: Path) -> None:
    runtime = tmp_path / "compiled-runtime"
    runtime.mkdir()
    for name in ("weights.bump", "weights_manifest.json", "config.json"):
        (runtime / name).write_bytes(b"x")
    assert _resolve(str(runtime)) == runtime.resolve()


def test_source_dir_without_runtime_gets_build_subdir(tmp_path: Path) -> None:
    source_dir = tmp_path / "hf-checkout"
    source_dir.mkdir()
    run_dir = _resolve(str(source_dir))
    assert run_dir == source_dir / ".ck_build_v8"


def test_explicit_run_dir_is_honored(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit run dir"
    assert ck_serve_runtime_v8._resolve_run_dir("model", str(explicit)) == explicit.resolve()
