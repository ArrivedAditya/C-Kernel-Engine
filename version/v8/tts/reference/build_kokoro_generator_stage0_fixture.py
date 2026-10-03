#!/usr/bin/env python3
"""Package the direct first-generator-stage hook and effective BUMP weights."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / 'tests/fixtures/tts/kokoro_generator_stage0_pinned.npz'
CAPTURE_NAME = 'decoder_generator_ups_0'
WEIGHT_NAMES = (
    'waveform_decoder.generator.ups.0.weight',
    'waveform_decoder.generator.ups.0.bias',
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', type=Path, required=True)
    parser.add_argument('--bundle-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--omit-weights', action='store_true',
                        help='Keep alternate utterance captures without duplicate weights')
    args = parser.parse_args()
    capture_path = args.capture_dir / 'manifest.json'
    capture = json.loads(capture_path.read_text())
    bundle = json.loads((args.bundle_dir / 'weights_manifest.json').read_text())
    if capture['pin']['model']['revision'] != bundle['pin']['model']['revision']:
        parser.error('capture and BUMP model revisions differ')
    if capture['assets'] != bundle['provenance']['source_asset_sha256']:
        parser.error('capture and BUMP source assets differ')
    if capture['environment']['packages']['torch'] != '2.8.0+cpu':
        parser.error('first-generator hook requires pinned PyTorch 2.8 oracle')
    entry = capture['tensors'][CAPTURE_NAME]
    source = args.capture_dir / entry['file']
    if sha256(source.read_bytes()) != entry['sha256']:
        parser.error('generator checkpoint checksum mismatch')
    tensors = {CAPTURE_NAME: np.load(source, allow_pickle=False),
               'word_ids': np.asarray(capture['input_ids'], dtype=np.int32)}
    for name in ('decoder_style', 'predictor_style'):
        item = capture['tensors'][name]
        path = args.capture_dir / item['file']
        if sha256(path.read_bytes()) != item['sha256']:
            parser.error(f'conditioning checksum mismatch: {name}')
        tensors[name] = np.load(path, allow_pickle=False).astype(np.float32)
    entries = {item['name']: item for item in bundle['entries']}
    with (args.bundle_dir / 'weights.bump').open('rb') as stream:
        for name in WEIGHT_NAMES:
            item = entries[name]
            stream.seek(item['file_offset'])
            data = stream.read(item['size'])
            if sha256(data) != item['sha256']:
                parser.error(f'effective weight checksum mismatch: {name}')
            if not args.omit_weights:
                tensors[name] = np.frombuffer(data, '<f4').copy().reshape(item['shape'])
    if (tensors[CAPTURE_NAME].ndim != 3 or
            tensors[CAPTURE_NAME].shape[:2] != (1, 256) or
            tensors['word_ids'].shape != (36,) or
            tensors[CAPTURE_NAME].shape[2] % 10):
        parser.error('unexpected direct first-generator checkpoint shape')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **tensors)
    metadata = {
        'scope': 'direct pinned full-model generator.ups[0] hook and effective weights',
        'model_pin': capture['pin']['model']['revision'],
        'code_pin': capture['pin']['reference_code']['kokoro'],
        'oracle': 'pinned full KModel forward hook',
        'environment': capture['environment']['packages'],
        'asset_sha256': capture['assets'],
        'capture_manifest_sha256': sha256(capture_path.read_bytes()),
        'source_tensor_sha256': {CAPTURE_NAME: entry['sha256']},
        'effective_weight_sha256': {name: entries[name]['sha256']
                                    for name in WEIGHT_NAMES},
        'weights_included': not args.omit_weights,
        'array_sha256': {name: sha256(value.tobytes())
                         for name, value in tensors.items()},
        'fixture_sha256': sha256(args.output.read_bytes()),
        'input_text': capture['graphemes'],
        'valid_input_frames': tensors[CAPTURE_NAME].shape[2] // 10,
        'valid_output_frames': tensors[CAPTURE_NAME].shape[2],
    }
    args.output.with_suffix('.json').write_text(
        json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
