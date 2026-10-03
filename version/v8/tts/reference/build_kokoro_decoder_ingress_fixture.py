#!/usr/bin/env python3
"""Package direct pinned decoder hooks and effective BUMP weights for PR tests."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / 'tests/fixtures/tts/kokoro_decoder_ingress_pinned.npz'
CAPTURE_NAMES = (
    'decoder_input_0', 'decoder_F0_conv', 'decoder_N_conv',
    'decoder_asr_res', 'decoder_encode',
)
WEIGHT_NAMES = (
    'waveform_decoder.F0_conv.weight', 'waveform_decoder.F0_conv.bias',
    'waveform_decoder.N_conv.weight', 'waveform_decoder.N_conv.bias',
    'waveform_decoder.asr_res.0.weight',
    'waveform_decoder.asr_res.0.bias',
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
    entries = {entry['name']: entry for entry in bundle['entries']}
    with (args.bundle_dir / 'weights.bump').open('rb') as stream:
        for name in WEIGHT_NAMES:
            entry = entries[name]
            stream.seek(entry['file_offset'])
            data = stream.read(entry['size'])
            if sha256(data) != entry['sha256']:
                parser.error(f'effective weight checksum mismatch: {name}')
            tensors[name] = np.frombuffer(data, '<f4').copy().reshape(entry['shape'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **tensors)
    metadata = {
        'scope': 'direct pinned full-model decoder ingress hooks and effective weights',
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
                                    for name in WEIGHT_NAMES},
        'array_sha256': {name: sha256(value.tobytes())
                         for name, value in tensors.items()},
        'fixture_sha256': sha256(args.output.read_bytes()),
        'valid_frames': 103, 'upsampled_frames': 206,
    }
    args.output.with_suffix('.json').write_text(
        json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
