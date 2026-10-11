#!/usr/bin/env python3
"""Pack pinned final spectral-head captures without duplicating upstream assets."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / 'tests/fixtures/tts/kokoro_spectral_head_pinned.npz'
TENSORS = {
    'conv_post': 'decoder_generator_conv_post',
    'magnitude': 'istft_magnitude',
    'phase': 'istft_phase',
    'waveform': 'decoder_generator',
}


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
    previous_path = ROOT / 'tests/fixtures/tts/kokoro_generator_stage1_pool_pinned.npz'
    previous = json.loads(previous_path.with_suffix('.json').read_text())
    if (capture['pin']['model']['revision'] != previous['model_pin'] or
        capture['pin']['reference_code']['kokoro'] != previous['code_pin'] or
        capture['assets'] != previous['asset_sha256'] or
        capture['assets'] != bundle['provenance']['source_asset_sha256'] or
        capture['environment']['packages']['torch'] != '2.8.0+cpu'):
        parser.error('capture, previous fixture and BUMP identities differ')
    arrays = {}
    source_hashes = {}
    for name, capture_name in TENSORS.items():
        item = capture['tensors'][capture_name]
        path = args.capture_dir / item['file']
        if sha256(path.read_bytes()) != item['sha256']:
            parser.error(f'capture checksum mismatch: {capture_name}')
        arrays[name] = np.load(path, allow_pickle=False).astype(np.float32)
        source_hashes[capture_name] = item['sha256']
    expected_shapes = {
        'conv_post': (1, 22, 12361),
        'magnitude': (1, 11, 12361),
        'phase': (1, 11, 12361),
        'waveform': (1, 1, 61800),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            parser.error(f'{name} has unexpected shape {arrays[name].shape}')
    entries = {item['name']: item for item in bundle['entries']}
    with (args.bundle_dir / 'weights.bump').open('rb') as stream:
        for suffix in ('weight', 'bias'):
            key = f'waveform_decoder.generator.conv_post.{suffix}'
            item = entries[key]
            stream.seek(item['file_offset'])
            data = stream.read(item['size'])
            if sha256(data) != item['sha256']:
                parser.error(f'weight checksum mismatch: {key}')
            arrays[suffix] = np.frombuffer(data, '<f4').copy().reshape(item['shape'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    metadata = {
        'scope': 'final convolution, spectral transform and waveform direct-model hooks',
        'model_pin': previous['model_pin'], 'code_pin': previous['code_pin'],
        'asset_sha256': previous['asset_sha256'],
        'environment': capture['environment']['packages'],
        'oracle': 'pinned PyTorch 2.8 full-model hooks and TorchSTFT.inverse',
        'capture_manifest_sha256': sha256(capture_path.read_bytes()),
        'previous_fixture_sha256': sha256(previous_path.read_bytes()),
        'source_tensor_sha256': source_hashes,
        'effective_weight_sha256': {
            f'waveform_decoder.generator.conv_post.{suffix}':
            entries[f'waveform_decoder.generator.conv_post.{suffix}']['sha256']
            for suffix in ('weight', 'bias')},
        'array_sha256': {
            name: sha256(value.tobytes()) for name, value in arrays.items()},
        'fixture_sha256': sha256(args.output.read_bytes()),
        'valid_frames': 12361, 'sample_count': 61800,
        'input_text': capture['graphemes'],
    }
    args.output.with_suffix('.json').write_text(
        json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
