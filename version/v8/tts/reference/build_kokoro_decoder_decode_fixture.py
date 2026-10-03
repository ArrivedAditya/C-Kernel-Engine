#!/usr/bin/env python3
"""Package direct pinned decoder block hooks and effective BUMP weights."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = ROOT / 'tests/fixtures/tts'


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--block-index', type=int, choices=range(4), required=True)
    parser.add_argument('--capture-dir', type=Path, required=True)
    parser.add_argument('--bundle-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    index = args.block_index
    prefix = f'waveform_decoder.decode.{index}.'
    stem = f'decoder_decode_{index}'
    capture_names = (f'{stem}_input_0', f'{stem}_norm1',
        f'{stem}_conv1', f'{stem}_norm2', f'{stem}_conv2',
        f'{stem}_conv1x1', stem)
    if index == 3:
        capture_names += (f'{stem}_pool', f'{stem}_upsample')
    output = args.output or OUTPUT_DIR / f'kokoro_decoder_decode{index}_pinned.npz'
    capture = json.loads((args.capture_dir / 'manifest.json').read_text())
    bundle = json.loads((args.bundle_dir / 'weights_manifest.json').read_text())
    if capture['pin']['model']['revision'] != bundle['pin']['model']['revision']:
        parser.error('capture and BUMP model revisions differ')
    if capture['assets'] != bundle['provenance']['source_asset_sha256']:
        parser.error('capture and BUMP source assets differ')
    if capture['environment']['packages']['torch'] != '2.8.0+cpu':
        parser.error('direct decoder capture requires pinned PyTorch 2.8 oracle')
    tensors = {}
    for name in capture_names:
        entry = capture['tensors'][name]
        path = args.capture_dir / entry['file']
        if sha256(path.read_bytes()) != entry['sha256']:
            parser.error(f'capture checksum mismatch: {name}')
        tensors[name] = np.load(path, allow_pickle=False).astype(np.float32)
    weight_names = tuple(sorted(entry['name'] for entry in bundle['entries']
        if entry['name'].startswith(prefix)))
    expected_count = 15 if index == 3 else 13
    if len(weight_names) != expected_count:
        parser.error(f'expected {expected_count} decode[{index}] weights, got {len(weight_names)}')
    entries = {entry['name']: entry for entry in bundle['entries']}
    with (args.bundle_dir / 'weights.bump').open('rb') as stream:
        for name in weight_names:
            entry = entries[name]
            stream.seek(entry['file_offset'])
            data = stream.read(entry['size'])
            if sha256(data) != entry['sha256']:
                parser.error(f'effective weight checksum mismatch: {name}')
            tensors[name] = np.frombuffer(data, '<f4').copy().reshape(entry['shape'])
    # The pinned Conv1d shortcut has bias=False; declare the checked ABI's
    # required bias pointer as an explicit exact-zero tensor in the bundle.
    output_channels = 512 if index == 3 else 1024
    tensors[f'{prefix}conv1x1.cke_zero_bias'] = \
        np.zeros(output_channels, np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **tensors)
    metadata = {
        'scope': f'direct pinned full-model decode[{index}] hooks and effective weights',
        'model_pin': capture['pin']['model']['revision'],
        'code_pin': capture['pin']['reference_code']['kokoro'],
        'oracle': 'pinned full KModel forward hooks',
        'environment': capture['environment']['packages'],
        'asset_sha256': capture['assets'],
        'capture_manifest_sha256': sha256(
            (args.capture_dir / 'manifest.json').read_bytes()),
        'source_tensor_sha256': {name: capture['tensors'][name]['sha256']
                                 for name in capture_names},
        'effective_weight_sha256': {name: entries[name]['sha256']
                                    for name in weight_names},
        'synthetic_zero_bias': f'{prefix}conv1x1.cke_zero_bias',
        'array_sha256': {name: sha256(value.tobytes())
                         for name, value in tensors.items()},
        'fixture_sha256': sha256(output.read_bytes()),
        'valid_frames': 206 if index == 3 else 103,
    }
    output.with_suffix('.json').write_text(
        json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
