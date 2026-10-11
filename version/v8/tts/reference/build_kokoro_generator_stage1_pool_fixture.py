#!/usr/bin/env python3
"""Package one pinned final activation and the second-stage main weights."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / 'tests/fixtures/tts/kokoro_generator_stage1_pool_pinned.npz'


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
    previous_path = ROOT / 'tests/fixtures/tts/kokoro_generator_stage1_join_pinned.npz'
    previous = json.loads(previous_path.with_suffix('.json').read_text())
    for label, a, b in (
        ('model', capture['pin']['model']['revision'], previous['model_pin']),
        ('code', capture['pin']['reference_code']['kokoro'], previous['code_pin']),
        ('assets', capture['assets'], previous['asset_sha256'])):
        if a != b:
            parser.error(f'join fixture {label} identity differs')
    if (capture['pin']['model']['revision'] != bundle['pin']['model']['revision']
            or capture['assets'] != bundle['provenance']['source_asset_sha256']
            or capture['environment']['packages']['torch'] != '2.8.0+cpu'):
        parser.error('capture, BUMP or pinned PyTorch identity differs')
    stem = 'decoder_generator_stage1_final_activation'
    item = capture['tensors'][stem]
    path = args.capture_dir / item['file']
    if sha256(path.read_bytes()) != item['sha256']:
        parser.error('final activation capture checksum mismatch')
    arrays = {'final_activation': np.load(path, allow_pickle=False).astype(np.float32)}
    if arrays['final_activation'].shape != (1, 128, 12361):
        parser.error('unexpected final activation shape')
    entries = {item['name']: item for item in bundle['entries']}
    names = {}
    for block in (3, 4, 5):
        prefix = f'waveform_decoder.generator.resblocks.{block}'
        for pair in range(3):
            for side in (1, 2):
                adain = f'{prefix}.adain{side}.{pair}'
                for suffix in ('fc.weight', 'fc.bias', 'norm.weight', 'norm.bias'):
                    key = f'{adain}.{suffix}'
                    names[key] = key
                alpha = f'{prefix}.alpha{side}.{pair}'
                names[alpha + '.channel'] = alpha
                for suffix in ('weight', 'bias'):
                    key = f'{prefix}.convs{side}.{pair}.{suffix}'
                    names[key] = key
    with (args.bundle_dir / 'weights.bump').open('rb') as stream:
        for canonical, raw in names.items():
            entry = entries[raw]
            stream.seek(entry['file_offset'])
            data = stream.read(entry['size'])
            if sha256(data) != entry['sha256']:
                parser.error(f'weight checksum mismatch: {raw}')
            tensor = np.frombuffer(data, '<f4').copy().reshape(entry['shape'])
            arrays[canonical] = tensor.reshape(128).copy() if \
                canonical.endswith('.channel') else tensor
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    metadata = {
        'scope': 'pinned full-model second-stage final activation and new weights',
        'model_pin': capture['pin']['model']['revision'],
        'code_pin': capture['pin']['reference_code']['kokoro'],
        'asset_sha256': capture['assets'],
        'environment': capture['environment']['packages'],
        'oracle': 'pinned full KModel conv_post pre-hook',
        'capture_manifest_sha256': sha256(capture_path.read_bytes()),
        'stage1_join_fixture_sha256': sha256(previous_path.read_bytes()),
        'source_tensor_sha256': item['sha256'],
        'effective_weight_sha256': {
            raw: entries[raw]['sha256'] for raw in names.values()},
        'transformations': {
            'alpha': 'Each [1,128,1] alpha is stored as contiguous [128]'},
        'array_sha256': {
            label: sha256(value.tobytes()) for label, value in arrays.items()},
        'fixture_sha256': sha256(args.output.read_bytes()),
        'valid_frames': 12361,
        'input_text': capture['graphemes']}
    args.output.with_suffix('.json').write_text(
        json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
