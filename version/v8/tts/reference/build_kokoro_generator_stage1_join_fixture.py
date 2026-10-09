#!/usr/bin/env python3
"""Package pinned second source residual/join checkpoints and only new weights."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / 'tests/fixtures/tts/kokoro_generator_stage1_join_pinned.npz'
PREFIX = 'waveform_decoder.generator.noise_res.1'


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
    previous_path = ROOT / 'tests/fixtures/tts/kokoro_generator_stage1_source_conv_pinned.npz'
    previous = json.loads(previous_path.with_suffix('.json').read_text())
    for key, left, right in (
        ('model', capture['pin']['model']['revision'], previous['model_pin']),
        ('code', capture['pin']['reference_code']['kokoro'], previous['code_pin']),
        ('assets', capture['assets'], previous['asset_sha256']),
    ):
        if left != right:
            parser.error(f'previous source fixture {key} differs')
    if (capture['pin']['model']['revision'] != bundle['pin']['model']['revision']
            or capture['assets'] != bundle['provenance']['source_asset_sha256']
            or capture['environment']['packages']['torch'] != '2.8.0+cpu'):
        parser.error('capture, BUMP, or pinned PyTorch identity differs')
    captures = {
        'source_residual': 'decoder_generator_noise_res_1',
        'join': 'decoder_generator_stage1_join',
    }
    arrays = {}
    for label, stem in captures.items():
        item = capture['tensors'][stem]
        path = args.capture_dir / item['file']
        if sha256(path.read_bytes()) != item['sha256']:
            parser.error(f'capture checksum mismatch: {stem}')
        arrays[label] = np.load(path, allow_pickle=False).astype(np.float32)
        if arrays[label].shape != (1, 128, 12361):
            parser.error(f'{stem}: unexpected shape {arrays[label].shape}')
    entries = {item['name']: item for item in bundle['entries']}
    names = {}
    for pair in range(3):
        for side in (1, 2):
            adain = f'{PREFIX}.adain{side}.{pair}'
            for suffix in ('fc.weight', 'fc.bias', 'norm.weight', 'norm.bias'):
                key = f'{adain}.{suffix}'
                names[key] = key
            alpha = f'{PREFIX}.alpha{side}.{pair}'
            names[alpha + '.channel'] = alpha
            for suffix in ('weight', 'bias'):
                key = f'{PREFIX}.convs{side}.{pair}.{suffix}'
                names[key] = key
    with (args.bundle_dir / 'weights.bump').open('rb') as stream:
        for canonical, raw in names.items():
            item = entries[raw]
            stream.seek(item['file_offset'])
            data = stream.read(item['size'])
            if sha256(data) != item['sha256']:
                parser.error(f'effective weight checksum mismatch: {raw}')
            tensor = np.frombuffer(data, '<f4').copy().reshape(item['shape'])
            arrays[canonical] = tensor.reshape(128).copy() if \
                canonical.endswith('.channel') else tensor
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    metadata = {
        'scope': 'direct pinned full-model second source residual and join',
        'model_pin': capture['pin']['model']['revision'],
        'code_pin': capture['pin']['reference_code']['kokoro'],
        'asset_sha256': capture['assets'],
        'environment': capture['environment']['packages'],
        'oracle': 'pinned full KModel noise_res.1 hook and resblocks.3 pre-hook',
        'capture_manifest_sha256': sha256(capture_path.read_bytes()),
        'stage1_source_fixture_sha256': sha256(previous_path.read_bytes()),
        'source_tensor_sha256': {
            label: capture['tensors'][stem]['sha256']
            for label, stem in captures.items()},
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
