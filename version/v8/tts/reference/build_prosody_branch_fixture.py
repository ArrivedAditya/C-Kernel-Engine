"""Package direct F0/noise block and projection hooks from pinned KModel."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / 'tests/fixtures/tts/kokoro_prosody_branch_model_pinned.npz'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    manifest_path = args.capture_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['status'] != 'full_oracle_captured':
        raise ValueError('direct full-model reference was not captured')
    if manifest['environment']['packages']['torch'] != '2.8.0+cpu':
        raise ValueError('direct capture must use pinned PyTorch 2.8.0+cpu')
    expected = [f'predictor.{branch}.{index}'
                for branch in ('F0', 'N') for index in range(3)]
    if manifest.get('prosody_branch_hooks') != expected:
        raise ValueError('missing or reordered direct prosody hooks')
    tensors = {}
    for branch in ('F0', 'N'):
        for suffix in ('0', '1', '2', 'proj'):
            name = f'predictor_{branch}_{suffix}'
            record = manifest['tensors'][name]
            path = args.capture_dir / record['file']
            if sha256(path) != record['sha256']:
                raise ValueError(f'direct capture hash mismatch: {name}')
            array = np.load(path, allow_pickle=False)
            expected_shape = ([1, 512, 103] if suffix == '0' else
                [1, 256, 206] if suffix in ('1', '2') else [1, 1, 206])
            if list(array.shape) != expected_shape or array.dtype != np.float32:
                raise ValueError(f'direct capture geometry mismatch: {name}')
            if not np.isfinite(array).all():
                raise ValueError(f'direct capture has nonfinite values: {name}')
            tensors[name] = array[0].copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **tensors)
    metadata = {'scope': 'direct pinned KModel F0/N blocks and projections',
        'model_pin': manifest['pin']['model']['revision'],
        'code_pin': manifest['pin']['reference_code']['kokoro'],
        'voice': manifest['pin']['fixture']['voice'],
        'oracle': 'pinned full KModel forward hooks',
        'environment': manifest['environment']['packages'],
        'asset_sha256': manifest['assets'],
        'capture_manifest_sha256': sha256(manifest_path),
        'source_tensor_sha256': {name: manifest['tensors'][name]['sha256']
                                for name in tensors},
        'array_sha256': {name: hashlib.sha256(value.tobytes()).hexdigest()
                         for name, value in tensors.items()},
        'fixture_sha256': sha256(args.output),
        'valid_frames': 103, 'upsampled_frames': 206}
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
