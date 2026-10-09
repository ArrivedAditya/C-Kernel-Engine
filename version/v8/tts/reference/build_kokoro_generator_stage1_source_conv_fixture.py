#!/usr/bin/env python3
"""Package direct pinned second source-convolution output and effective weights."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / 'tests/fixtures/tts/kokoro_generator_stage1_source_conv_pinned.npz'


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', type=Path, required=True)
    parser.add_argument('--bundle-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    capture_path = args.capture_dir / 'manifest.json'
    capture = json.loads(capture_path.read_text())
    bundle = json.loads((args.bundle_dir / 'weights_manifest.json').read_text())
    previous_path = ROOT / 'tests/fixtures/tts/kokoro_generator_stage1_ingress_pinned.npz'
    previous = json.loads(previous_path.with_suffix('.json').read_text())
    if (capture['pin']['model']['revision'] != bundle['pin']['model']['revision'] or
        capture['assets'] != bundle['provenance']['source_asset_sha256'] or
        capture['environment']['packages']['torch'] != '2.8.0+cpu'):
        parser.error('capture and BUMP identities or pinned PyTorch version differ')
    if (previous['model_pin'] != capture['pin']['model']['revision'] or
        previous['code_pin'] != capture['pin']['reference_code']['kokoro'] or
        previous['asset_sha256'] != capture['assets']):
        parser.error('stage-one fixture belongs to a different experiment')
    arrays = {}
    names = {
        'output': 'decoder_generator_noise_convs_1',
        'stft_input': 'generator_source_magnitude',
        'stft_phase': 'generator_source_phase',
    }
    for label, name in names.items():
        item = capture['tensors'][name]
        path = args.capture_dir / item['file']
        if sha256(path.read_bytes()) != item['sha256']:
            parser.error(f'capture checksum mismatch: {name}')
        arrays[label] = np.load(path, allow_pickle=False).astype(np.float32)
    if arrays['output'].shape != (1, 128, 12361):
        parser.error(f'unexpected source output shape: {arrays["output"].shape}')
    if arrays['stft_input'].shape != (1, 11, 12361):
        parser.error('unexpected magnitude shape')
    if arrays['stft_phase'].shape != (1, 11, 12361):
        parser.error('unexpected phase shape')
    entries = {item['name']: item for item in bundle['entries']}
    keys = {'weight': 'waveform_decoder.generator.noise_convs.1.weight',
            'bias': 'waveform_decoder.generator.noise_convs.1.bias'}
    with (args.bundle_dir / 'weights.bump').open('rb') as stream:
        for label, key in keys.items():
            item = entries[key]
            stream.seek(item['file_offset'])
            data = stream.read(item['size'])
            if sha256(data) != item['sha256']:
                parser.error(f'weight checksum mismatch: {key}')
            arrays[label] = np.frombuffer(data, '<f4').copy().reshape(item['shape'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    metadata = {
        'scope': 'direct pinned full-model second source convolution',
        'model_pin': capture['pin']['model']['revision'],
        'code_pin': capture['pin']['reference_code']['kokoro'],
        'asset_sha256': capture['assets'],
        'environment': capture['environment']['packages'],
        'oracle': 'pinned full KModel noise_convs.1 hook',
        'capture_manifest_sha256': sha256(capture_path.read_bytes()),
        'stage1_fixture_sha256': sha256(previous_path.read_bytes()),
        'source_tensor_sha256': {
            label: capture['tensors'][name]['sha256']
            for label, name in names.items()},
        'effective_weight_sha256': {
            key: entries[key]['sha256'] for key in keys.values()},
        'array_sha256': {
            label: sha256(value.tobytes()) for label, value in arrays.items()},
        'fixture_sha256': sha256(args.output.read_bytes()),
        'valid_frames': 12361,
        'input_text': capture['graphemes']}
    args.output.with_suffix('.json').write_text(
        json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
