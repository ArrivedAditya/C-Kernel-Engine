#!/usr/bin/env python3
"""Export a bounded, dictionary-only native English Kokoro frontend.

The source lexicon is pinned Misaki US gold. Export chooses its DEFAULT entry
when a word has context-dependent pronunciations; this is a separate, limited
pronunciation contract, not full Misaki G2P parity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

from build_kokoro_request_runtime import ROOT


MISAKI_COMMIT = "fba1236595f2d2bf21d414ba6e57d25256afada3"
KOKORO_ASSET_COMMIT = "e8a90b41091c3c5b70375c47cc959799920fa4d6"
MISAKI_US_GOLD_SHA256 = "dc414872a49a28ae6c141463d502fd945f3b2fde040484fdc47d00cc4612686f"
KOKORO_CONFIG_SHA256 = "5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f"
SOURCE = ROOT / "version/v8/tts/native_kokoro_lexicon_text.c"
WORD = re.compile(r"[a-z]+(?:'[a-z]+)*\Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def build_frontend(config_path: Path, lexicon_path: Path, output: Path,
                   *, config_sha256: str = KOKORO_CONFIG_SHA256,
                   lexicon_sha256: str = MISAKI_US_GOLD_SHA256) -> dict:
    if output.is_symlink() or (output.exists() and
            (not output.is_dir() or any(output.iterdir()))):
        raise ValueError("output directory must be empty")
    if sha256(config_path) != config_sha256 or sha256(lexicon_path) != lexicon_sha256:
        raise ValueError("frontend source assets do not match pinned hashes")
    vocab = json.loads(config_path.read_text())["vocab"]
    source = json.loads(lexicon_path.read_text())
    for symbol in (" ", ".", ",", "!", "?"):
        if not isinstance(vocab.get(symbol), int):
            raise ValueError(f"missing Kokoro punctuation symbol {symbol!r}")
    entries = {}
    for word, value in source.items():
        if len(word) > 64 or not WORD.fullmatch(word):
            continue
        phonemes = (value if isinstance(value, str) else
                    value.get("DEFAULT") if isinstance(value, dict) else None)
        if not isinstance(phonemes, str) or not phonemes:
            continue
        ids = [vocab.get(char) for char in phonemes]
        if len(ids) > 96 or any(not isinstance(item, int) or item <= 0 or
                                item >= 178 for item in ids):
            continue
        entries[word] = ",".join(map(str, ids))
    if not entries:
        raise ValueError("pinned lexicon has no supported words")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.staging-",
                                     dir=output.parent) as temporary:
        stage = Path(temporary) / "frontend"
        stage.mkdir()
        sidecar = stage / "us-gold.cklex"
        sidecar.write_text("CKLEX1\n" + "".join(
            f"{word}\t{ids}\n" for word, ids in sorted(entries.items())),
            encoding="utf-8")
        sidecar_hash = sha256(sidecar)
        header = stage / "frontend_identity.h"
        header.write_text(
            "/* Generated from pinned Misaki US gold and Kokoro config. */\n"
            f'#define CK_TTS_LEXICON_SHA256 "{sidecar_hash}"\n'
            f'#define CK_TTS_SPACE_ID {vocab[" "]}\n'
            f'#define CK_TTS_DOT_ID {vocab["."]}\n'
            f'#define CK_TTS_COMMA_ID {vocab[","]}\n'
            f'#define CK_TTS_EXCLAMATION_ID {vocab["!"]}\n'
            f'#define CK_TTS_QUESTION_ID {vocab["?"]}\n')
        binary = stage / "native-kokoro-text"
        command = ["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                   "-pedantic", "-I", str(stage), str(SOURCE), "-lcrypto",
                   "-o", str(binary)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"native frontend compilation failed: {result.stderr}")
        report = {
            "schema": "cke.kokoro.native_lexicon_text.v1",
            "source_misaki_commit": MISAKI_COMMIT,
            "source_misaki_lexicon_url": f"https://github.com/hexgrad/misaki/blob/{MISAKI_COMMIT}/misaki/data/us_gold.json",
            "source_misaki_license": "Apache-2.0",
            "source_kokoro_asset_commit": KOKORO_ASSET_COMMIT,
            "source_kokoro_config_url": f"https://huggingface.co/hexgrad/Kokoro-82M/blob/{KOKORO_ASSET_COMMIT}/config.json",
            "source_lexicon_sha256": lexicon_sha256,
            "source_config_sha256": config_sha256,
            "exported_lexicon_sha256": sidecar_hash,
            "frontend_binary_sha256": sha256(binary),
            "word_count": len(entries),
            "scope": "US gold DEFAULT pronunciation, ASCII words and basic punctuation; "
                     "unsupported words/forms rejected; model supports only bundled ID counts",
        }
        (stage / "frontend_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        if output.is_symlink() or (output.exists() and
                (not output.is_dir() or any(output.iterdir()))):
            raise ValueError("output directory must be empty")
        stage.replace(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--misaki-us-gold", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build_frontend(args.config, args.misaki_us_gold,
                                    args.output_dir), indent=2))


if __name__ == "__main__":
    main()
