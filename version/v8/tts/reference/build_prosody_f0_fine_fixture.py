"""Package direct full-model F0 checkpoints for connected drift diagnosis."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from capture_kokoro_v1 import PROSODY_F0_FINE_HOOKS

ROOT = Path(__file__).resolve().parents[4]
PINNED = ROOT / 'version/v8/tts/reference/fixture_manifest.json'
OUTPUT = ROOT / 'tests/fixtures/tts/kokoro_prosody_f0_fine_pinned.npz'
EXTRA = ('bert_encoder', 'predictor_text_encoder',
         'predictor_shared_input', 'predictor_shared_0')
NAMES = EXTRA + tuple(name.replace('.', '_') for name in PROSODY_F0_FINE_HOOKS)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    manifest_path = args.capture_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    pinned = json.loads(PINNED.read_text())
    if (manifest.get('status') != 'full_oracle_captured' or
            manifest.get('prosody_f0_fine_hooks') != list(PROSODY_F0_FINE_HOOKS) or
            manifest.get('bert_attention_implementation') != 'sdpa' or
            manifest.get('bert_attention_override') is not False or
            manifest.get('pin') != pinned.get('pin') or
            manifest.get('assets') != pinned.get('assets') or
            manifest.get('environment', {}).get('packages') !=
            pinned.get('environment', {}).get('packages') or
            manifest.get('input_ids') != pinned.get('input_ids') or
            manifest.get('tensors', {}).get('waveform_f32', {}).get('sha256') !=
            pinned.get('tensors', {}).get('waveform_f32', {}).get('sha256')):
        raise ValueError('fine F0 capture differs from pinned full-model experiment')
    tensors = {}
    for name in NAMES:
        record = manifest['tensors'][name]
        path = args.capture_dir / record['file']
        if sha256(path) != record['sha256']:
            raise ValueError(f'capture hash mismatch: {name}')
        value = np.load(path, allow_pickle=False)
        if value.dtype != np.float32 or value.shape[0] != 1 or not np.isfinite(value).all():
            raise ValueError(f'invalid tensor: {name}')
        tensors[name] = value[0].copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **tensors)
    metadata = {
        'scope': 'direct pinned KModel shared-scan input and F0 inner modules',
        'model_pin': manifest['pin']['model']['revision'],
        'code_pin': manifest['pin']['reference_code']['kokoro'],
        'voice': manifest['pin']['fixture']['voice'],
        'oracle': 'pinned full KModel forward and pre-forward hooks',
        'bert_attention_implementation': 'sdpa',
        'environment': manifest['environment']['packages'],
        'asset_sha256': manifest['assets'],
        'capture_waveform_sha256': manifest['tensors']['waveform_f32']['sha256'],
        'source_tensor_sha256': {name: manifest['tensors'][name]['sha256']
                                 for name in NAMES},
        'array_sha256': {name: hashlib.sha256(value.tobytes()).hexdigest()
                         for name, value in tensors.items()},
        'fixture_sha256': sha256(args.output),
        'valid_frames': 103, 'upsampled_frames': 206,
    }
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
