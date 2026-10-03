#!/usr/bin/env python3
"""Package additional pinned utterances for connected decoder parity."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
TENSOR_NAMES = ('predictor_style', 'decoder_style', 'predicted_duration',
                *(f'decoder_decode_{i}' for i in range(4)),
                *(f'decoder_decode_3_{part}' for part in
                  ('norm1', 'pool', 'conv1', 'norm2', 'conv2',
                   'upsample', 'conv1x1')))


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z][a-z0-9_]*', args.name):
        parser.error('name must be a lowercase fixture identifier')
    output = args.output or (ROOT / 'tests/fixtures/tts' /
        f'kokoro_decoder_utterance_{args.name}_pinned.npz')
    capture = json.loads((args.capture_dir / 'manifest.json').read_text())
    if capture['environment']['packages']['torch'] != '2.8.0+cpu':
        parser.error('additional utterance requires pinned PyTorch 2.8 oracle')
    if len(capture['input_ids']) != 36 or capture['voice_row_index'] != 33:
        parser.error('utterance must fit the declared 36-token voice geometry')
    arrays = {'word_ids': np.asarray(capture['input_ids'], np.int32)}
    for name in TENSOR_NAMES:
        entry = capture['tensors'][name]
        path = args.capture_dir / entry['file']
        if sha256(path.read_bytes()) != entry['sha256']:
            parser.error(f'capture checksum mismatch: {name}')
        arrays[name] = np.load(path, allow_pickle=False).astype(np.float32)
    durations = arrays['predicted_duration'].ravel().astype(np.int64)
    valid = int(durations.sum())
    if valid < 1 or valid > 128 or arrays['decoder_decode_3'].shape != (1, 512, 2*valid):
        parser.error('oracle frames exceed declared decoder capacity')
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    meta = {
        'scope': 'additional direct pinned full-model decoder utterance',
        'input_text': capture['input_text'],
        'phonemes': capture['phonemes'],
        'model_pin': capture['pin']['model']['revision'],
        'code_pin': capture['pin']['reference_code']['kokoro'],
        'environment': capture['environment']['packages'],
        'asset_sha256': capture['assets'],
        'capture_manifest_sha256': sha256(
            (args.capture_dir / 'manifest.json').read_bytes()),
        'source_tensor_sha256': {name: capture['tensors'][name]['sha256']
                                 for name in TENSOR_NAMES},
        'array_sha256': {name: sha256(value.tobytes())
                         for name, value in arrays.items()},
        'fixture_sha256': sha256(output.read_bytes()),
        'valid_frames': valid, 'upsampled_frames': 2*valid,
    }
    output.with_suffix('.json').write_text(json.dumps(meta, indent=2) + '\n')


if __name__ == '__main__':
    main()
