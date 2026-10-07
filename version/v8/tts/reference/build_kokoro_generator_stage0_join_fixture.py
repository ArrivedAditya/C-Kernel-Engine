#!/usr/bin/env python3
"""Package the direct pinned full-model first source/main generator join."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / 'tests/fixtures/tts/kokoro_generator_stage0_join_pinned.npz'


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    manifest_path = args.capture_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['environment']['packages']['torch'] != '2.8.0+cpu':
        parser.error('join fixture requires pinned PyTorch 2.8')
    captures = {
        'join': 'decoder_generator_stage0_join',
        'main': 'decoder_generator_ups_0',
        'source': 'decoder_generator_noise_res_0',
    }
    arrays = {}
    for name, key in captures.items():
        item = manifest['tensors'][key]
        path = args.capture_dir / item['file']
        if sha256(path.read_bytes()) != item['sha256']:
            parser.error(f'capture checksum mismatch: {key}')
        arrays[name] = np.load(path, allow_pickle=False).astype(np.float32)
        if arrays[name].shape != (1, 256, 2060):
            parser.error(f'{key}: unexpected shape {arrays[name].shape}')
    if not np.array_equal(arrays['main'] + arrays['source'], arrays['join']):
        parser.error('pinned source/main sum does not reproduce direct join hook')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, join=arrays['join'])
    metadata = {
        'scope': 'direct pinned full-model first generator source/main join',
        'model_pin': manifest['pin']['model']['revision'],
        'code_pin': manifest['pin']['reference_code']['kokoro'],
        'oracle': 'pinned full KModel first resblock input prehook',
        'environment': manifest['environment']['packages'],
        'asset_sha256': manifest['assets'],
        'capture_manifest_sha256': sha256(manifest_path.read_bytes()),
        'source_tensor_sha256': {name: manifest['tensors'][key]['sha256']
                                 for name, key in captures.items()},
        'array_sha256': {'join': sha256(arrays['join'].tobytes())},
        'fixture_sha256': sha256(args.output.read_bytes()),
        'input_text': manifest['graphemes'],
        'valid_frames': 2060,
    }
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
