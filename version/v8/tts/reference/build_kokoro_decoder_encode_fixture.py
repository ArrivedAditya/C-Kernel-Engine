#!/usr/bin/env python3
"""Package pinned direct decoder-encode hooks and effective BUMP weights."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / 'tests/fixtures/tts/kokoro_decoder_encode_pinned.npz'
CAPTURE_NAMES = (
    'decoder_style', 'decoder_encode_norm1', 'decoder_encode_conv1',
    'decoder_encode_norm2', 'decoder_encode_conv2',
    'decoder_encode_conv1x1', 'decoder_encode',
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', type=Path, required=True)
    parser.add_argument('--bundle-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    capture = json.loads((args.capture_dir / 'manifest.json').read_text())
    bundle = json.loads((args.bundle_dir / 'weights_manifest.json').read_text())
    if capture['pin']['model']['revision'] != bundle['pin']['model']['revision']:
        parser.error('capture and BUMP model revisions differ')
    if capture['assets'] != bundle['provenance']['source_asset_sha256']:
        parser.error('capture and BUMP source assets differ')
    if capture['environment']['packages']['torch'] != '2.8.0+cpu':
        parser.error('decoder direct capture requires pinned PyTorch 2.8 oracle')
    tensors = {}
    for name in CAPTURE_NAMES:
        entry = capture['tensors'][name]
        path = args.capture_dir / entry['file']
        if sha256(path.read_bytes()) != entry['sha256']:
            parser.error(f'capture checksum mismatch: {name}')
        tensors[name] = np.load(path, allow_pickle=False).astype(np.float32)
    weight_names = tuple(sorted(entry['name'] for entry in bundle['entries']
        if entry['name'].startswith('waveform_decoder.encode.')))
    if len(weight_names) != 13:
        parser.error(f'expected 13 effective decoder encode weights, got {len(weight_names)}')
    entries = {entry['name']: entry for entry in bundle['entries']}
    with (args.bundle_dir / 'weights.bump').open('rb') as stream:
        for name in weight_names:
            entry = entries[name]
            stream.seek(entry['file_offset'])
            data = stream.read(entry['size'])
            if sha256(data) != entry['sha256']:
                parser.error(f'effective weight checksum mismatch: {name}')
            tensors[name] = np.frombuffer(data, '<f4').copy().reshape(entry['shape'])
    # The reference Conv1d shortcut has bias=False. The checked CKE Conv1D ABI
    # accepts a bias pointer, so declare an exact zero tensor in the bundle.
    tensors['waveform_decoder.encode.conv1x1.cke_zero_bias'] = \
        np.zeros(1024, np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **tensors)
    metadata = {
        'scope': 'direct pinned full-model decoder encode hooks and effective weights',
        'model_pin': capture['pin']['model']['revision'],
        'code_pin': capture['pin']['reference_code']['kokoro'],
        'oracle': 'pinned full KModel forward hooks',
        'environment': capture['environment']['packages'],
        'asset_sha256': capture['assets'],
        'capture_manifest_sha256': sha256(
            (args.capture_dir / 'manifest.json').read_bytes()),
        'source_tensor_sha256': {name: capture['tensors'][name]['sha256']
                                 for name in CAPTURE_NAMES},
        'effective_weight_sha256': {name: entries[name]['sha256']
                                    for name in weight_names},
        'synthetic_zero_bias': 'waveform_decoder.encode.conv1x1.cke_zero_bias',
        'array_sha256': {name: sha256(value.tobytes())
                         for name, value in tensors.items()},
        'fixture_sha256': sha256(args.output.read_bytes()),
        'valid_frames': 103,
    }
    args.output.with_suffix('.json').write_text(
        json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
