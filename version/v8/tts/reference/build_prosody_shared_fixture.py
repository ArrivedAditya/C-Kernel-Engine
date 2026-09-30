"""Package pinned shared-prosody LSTM weights and an independent capture.

This development tool reads a verified BUMP export and PyTorch checkpoint.
It does not execute the model or participate in native deployment.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
REFERENCE = ROOT / 'version/v8/tts/reference/fixture_manifest.json'
OUTPUT = ROOT / 'tests/fixtures/tts/kokoro_prosody_shared_pinned.npz'
PREFIX = 'duration_prosody.shared'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-dir', type=Path, required=True)
    parser.add_argument('--capture-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    manifest = json.loads((args.bundle_dir / 'weights_manifest.json').read_text())
    entries = {item['name']: item for item in manifest['entries']}
    bump = (args.bundle_dir / 'weights.bump').read_bytes()
    arrays = {}
    origins = {}
    for kind in ('weight_ih', 'weight_hh', 'bias_ih', 'bias_hh'):
        directions = []
        for suffix in ('l0', 'l0_reverse'):
            name = f'{PREFIX}.{kind}_{suffix}'
            entry = entries[name]
            start = entry['file_offset']
            payload = bump[start:start + entry['size']]
            if hashlib.sha256(payload).hexdigest() != entry['sha256']:
                raise ValueError(f'BUMP payload mismatch: {name}')
            directions.append(np.frombuffer(payload, dtype='<f4').reshape(entry['shape']))
            origins[name] = entry['sha256']
        arrays[kind] = np.ascontiguousarray(np.stack(directions))
    reference = json.loads(REFERENCE.read_text())['tensors']['predictor_shared_0']
    source = args.capture_dir / reference['file']
    if hashlib.sha256(source.read_bytes()).hexdigest() != reference['sha256']:
        raise ValueError('prosody checkpoint hash mismatch')
    arrays['output'] = np.ascontiguousarray(
        np.load(source, allow_pickle=False)[0])
    if arrays['output'].shape != (103, 512):
        raise ValueError('unexpected prosody checkpoint shape')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    metadata = {
        'scope': 'shared prosody BiLSTM checkpoint; downstream F0/N not included',
        'model_pin': manifest['pin']['model']['revision'],
        'oracle': 'pinned PyTorch predictor_shared_0 checkpoint',
        'oracle_sha256': reference['sha256'],
        'weight_payload_sha256': origins,
        'array_sha256': {name: hashlib.sha256(value.tobytes()).hexdigest()
                         for name, value in arrays.items()},
        'fixture_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
        'frame_count': 103,
        'frame_capacity': 128,
    }
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
