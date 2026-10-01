"""Independent PyTorch execution of Kokoro's first F0/noise residual blocks.

Starts from the pinned PyTorch 2.8 shared-LSTM capture and the independently
executed first AdaIN fixture. This is an operation-level oracle, not yet a
full-model branch capture or deployed execution path.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / 'tests/fixtures/tts/kokoro_prosody_first_block_pinned.npz'
NORM = ROOT / 'tests/fixtures/tts/kokoro_prosody_norm_pinned.npz'
SHARED = ROOT / 'tests/fixtures/tts/kokoro_prosody_shared_pinned.npz'
DURATION = ROOT / 'tests/fixtures/tts/kokoro_duration_predictor_pinned.npz'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    manifest = json.loads((args.bundle_dir / 'weights_manifest.json').read_text())
    entries = {item['name']: item for item in manifest['entries']}
    bump = (args.bundle_dir / 'weights.bump').read_bytes()
    with np.load(NORM) as archive:
        norm = {name: archive[name].copy() for name in archive.files}
    with np.load(SHARED) as archive:
        shared = archive['output'].copy()
    with np.load(DURATION) as archive:
        style = archive['predictor_style'].copy()
    arrays, origins = {}, {}

    def weight(name):
        entry = entries[name]
        payload = bump[entry['file_offset']:entry['file_offset'] + entry['size']]
        digest = hashlib.sha256(payload).hexdigest()
        if digest != entry['sha256']:
            raise ValueError(f'BUMP payload mismatch: {name}')
        origins[name] = digest
        return np.frombuffer(payload, dtype='<f4').reshape(entry['shape']).copy()

    for branch in ('F0', 'N'):
        prefix = f'duration_prosody.{branch}.0'
        weights = {name: weight(f'{prefix}.{name}') for name in (
            'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias',
            'norm2.fc.weight', 'norm2.fc.bias', 'norm2.norm.weight',
            'norm2.norm.bias')}
        for name, value in weights.items():
            arrays[f'{branch}_{name.replace(".", "_")}'] = value
        with torch.no_grad():
            affine = F.linear(torch.from_numpy(style),
                torch.from_numpy(weights['norm2.fc.weight']),
                torch.from_numpy(weights['norm2.fc.bias']))
            arrays[f'{branch}_norm2_affine'] = affine.numpy().copy()
            for length in (36, 72, 103):
                original = shared if length == 103 else norm[f'length{length}_shared']
                initial = (norm[f'{branch}_output'] if length == 103 else
                           norm[f'length{length}_{branch}_output'])
                x = torch.from_numpy(initial.copy())[None]
                activated = F.leaky_relu(x, .2)
                conv1 = F.conv1d(activated,
                    torch.from_numpy(weights['conv1.weight']),
                    torch.from_numpy(weights['conv1.bias']), padding=1)
                normalized = F.instance_norm(conv1,
                    weight=torch.from_numpy(weights['norm2.norm.weight']),
                    bias=torch.from_numpy(weights['norm2.norm.bias']),
                    use_input_stats=True, eps=1e-5)
                normalized = ((1 + affine[:512, None]) * normalized +
                              affine[512:, None])
                activated2 = F.leaky_relu(normalized, .2)
                conv2 = F.conv1d(activated2,
                    torch.from_numpy(weights['conv2.weight']),
                    torch.from_numpy(weights['conv2.bias']), padding=1)
                original = torch.from_numpy(original.T.copy())
                result = (conv2[0] + original) * (2 ** -.5)
                key = f'length{length}_{branch}'
                for stage, value in (('act0', activated[0]),
                                     ('conv0', conv1[0]),
                                     ('norm1', normalized[0]),
                                     ('act1', activated2[0]),
                                     ('conv1', conv2[0]),
                                     ('block0', result)):
                    arrays[f'{key}_{stage}'] = value.numpy().copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    metadata = {'scope': 'first F0/N residual blocks; no second/third block or final projection',
        'model_pin': manifest['pin']['model']['revision'],
        'oracle': 'PyTorch F.leaky_relu, F.conv1d, F.instance_norm, F.linear, residual sum',
        'oracle_version': torch.__version__,
        'norm_fixture_sha256': hashlib.sha256(NORM.read_bytes()).hexdigest(),
        'shared_fixture_sha256': hashlib.sha256(SHARED.read_bytes()).hexdigest(),
        'weight_payload_sha256': origins,
        'array_sha256': {name: hashlib.sha256(value.tobytes()).hexdigest()
                         for name, value in arrays.items()},
        'fixture_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
        'frame_capacity': 128, 'tested_valid_frames': [36, 72, 103]}
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
