#!/usr/bin/env python3
"""Package direct pinned stage-one generator checkpoints and effective weights."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / 'tests/fixtures/tts/kokoro_generator_stage1_ingress_pinned.npz'


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
    if capture['pin']['model']['revision'] != bundle['pin']['model']['revision']:
        parser.error('capture and BUMP model revisions differ')
    if capture['assets'] != bundle['provenance']['source_asset_sha256']:
        parser.error('capture and BUMP source assets differ')
    if capture['environment']['packages']['torch'] != '2.8.0+cpu':
        parser.error('stage-one fixture requires pinned PyTorch 2.8')
    earlier = ROOT / 'tests/fixtures/tts/kokoro_generator_stage0_pool_pinned.json'
    earlier_meta = json.loads(earlier.read_text())
    if (earlier_meta['model_pin'] != capture['pin']['model']['revision'] or
        earlier_meta['code_pin'] != capture['pin']['reference_code']['kokoro'] or
        earlier_meta['asset_sha256'] != capture['assets']):
        parser.error('stage-zero fixture belongs to a different experiment')
    captures = {
        'activation': 'decoder_generator_stage1_activation',
        'deconv': 'decoder_generator_ups_1',
        'reflection': 'decoder_generator_reflection_pad'}
    shapes = {'activation': (1, 256, 2060),
              'deconv': (1, 128, 12360),
              'reflection': (1, 128, 12361)}
    arrays = {}
    for label, key in captures.items():
        item = capture['tensors'][key]
        source = args.capture_dir / item['file']
        if sha256(source.read_bytes()) != item['sha256']:
            parser.error(f'capture checksum mismatch: {key}')
        arrays[label] = np.load(source, allow_pickle=False).astype(np.float32)
        if arrays[label].shape != shapes[label]:
            parser.error(f'{key}: unexpected shape {arrays[label].shape}')
    entries = {item['name']: item for item in bundle['entries']}
    weights = {
        'weight': 'waveform_decoder.generator.ups.1.weight',
        'bias': 'waveform_decoder.generator.ups.1.bias'}
    with (args.bundle_dir / 'weights.bump').open('rb') as stream:
        for label, key in weights.items():
            item = entries[key]
            stream.seek(item['file_offset'])
            data = stream.read(item['size'])
            if sha256(data) != item['sha256']:
                parser.error(f'effective weight checksum mismatch: {key}')
            arrays[label] = np.frombuffer(data, '<f4').copy().reshape(item['shape'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    metadata = {
        'scope': 'direct pinned full-model second generator ingress',
        'model_pin': capture['pin']['model']['revision'],
        'code_pin': capture['pin']['reference_code']['kokoro'],
        'asset_sha256': capture['assets'],
        'environment': capture['environment']['packages'],
        'oracle': 'pinned full KModel ups.1 and reflection_pad hooks',
        'capture_manifest_sha256': sha256(capture_path.read_bytes()),
        'stage0_fixture_sha256': sha256(earlier.with_suffix('.npz').read_bytes()),
        'source_tensor_sha256': {
            label: capture['tensors'][key]['sha256']
            for label, key in captures.items()},
        'effective_weight_sha256': {
            key: entries[key]['sha256'] for key in weights.values()},
        'array_sha256': {
            label: sha256(value.tobytes()) for label, value in arrays.items()},
        'fixture_sha256': sha256(args.output.read_bytes()),
        'valid_input_frames': 2060,
        'valid_deconv_frames': 12360,
        'valid_reflected_frames': 12361,
        'input_text': capture['graphemes']}
    args.output.with_suffix('.json').write_text(
        json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
