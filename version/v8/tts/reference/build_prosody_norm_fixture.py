"""Capture the first F0/noise AdaIN stages from pinned effective BUMP weights.

The input is the separately pinned PyTorch 2.8 shared-LSTM checkpoint. The
normalization outputs are independently executed PyTorch operations; this is
development oracle tooling, not deployed model scheduling.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / 'tests/fixtures/tts/kokoro_prosody_norm_pinned.npz'
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
    arrays = {}
    origins = {}
    with np.load(SHARED) as archive:
        shared = archive['output'].copy()
    with np.load(DURATION) as archive:
        style = archive['predictor_style'].copy()
        predictor = archive['predictor_output'].copy()
    source = torch.from_numpy(shared.T.copy())[None]
    conditioning = torch.from_numpy(style)[None]
    with np.load(SHARED) as archive:
        scan_weights = {name: archive[name].copy() for name in
            ('weight_ih', 'weight_hh', 'bias_ih', 'bias_hh')}
    scan = torch.nn.LSTM(640, 256, batch_first=True, bidirectional=True).eval()
    with torch.no_grad():
        for kind, values in scan_weights.items():
            for direction, suffix in enumerate(('', '_reverse')):
                getattr(scan, f'{kind}_l0{suffix}').copy_(
                    torch.from_numpy(values[direction]))
        for frames_per_token in (1, 2):
            expanded = np.repeat(predictor, frames_per_token, axis=0)
            length = expanded.shape[0]
            output, _ = scan(torch.from_numpy(expanded)[None])
            arrays[f'length{length}_shared'] = output[0].numpy().copy()
    for branch in ('F0', 'N'):
        prefix = f'duration_prosody.{branch}.0.norm1'
        weights = {}
        for kind in ('fc.weight', 'fc.bias', 'norm.weight', 'norm.bias'):
            name = f'{prefix}.{kind}'
            entry = entries[name]
            start = entry['file_offset']
            payload = bump[start:start + entry['size']]
            if hashlib.sha256(payload).hexdigest() != entry['sha256']:
                raise ValueError(f'BUMP payload mismatch: {name}')
            weights[kind] = np.frombuffer(payload, dtype='<f4').reshape(
                entry['shape']).copy()
            arrays[f'{branch}_{kind.replace(".", "_")}'] = weights[kind]
            origins[name] = entry['sha256']
        with torch.no_grad():
            affine = F.linear(conditioning,
                torch.from_numpy(weights['fc.weight']),
                torch.from_numpy(weights['fc.bias']))[0]
            normalized = F.instance_norm(source,
                weight=torch.from_numpy(weights['norm.weight']),
                bias=torch.from_numpy(weights['norm.bias']),
                use_input_stats=True, eps=1e-5)[0]
            output = ((1 + affine[:512, None]) * normalized +
                      affine[512:, None])
        arrays[f'{branch}_style_affine'] = affine.numpy().copy()
        arrays[f'{branch}_output'] = output.numpy().copy()
        for length in (36, 72):
            with torch.no_grad():
                input_value = torch.from_numpy(
                    arrays[f'length{length}_shared'].T.copy())[None]
                normalized = F.instance_norm(input_value,
                    weight=torch.from_numpy(weights['norm.weight']),
                    bias=torch.from_numpy(weights['norm.bias']),
                    use_input_stats=True, eps=1e-5)[0]
                output = ((1 + affine[:512, None]) * normalized +
                          affine[512:, None])
            arrays[f'length{length}_{branch}_output'] = output.numpy().copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    metadata = {
        'scope': 'first F0/N norm from pinned shared-LSTM output; no convolution or full branch',
        'model_pin': manifest['pin']['model']['revision'],
        'oracle': 'PyTorch F.linear then F.instance_norm and style affine',
        'oracle_version': torch.__version__,
        'shared_fixture_sha256': hashlib.sha256(SHARED.read_bytes()).hexdigest(),
        'duration_fixture_sha256': hashlib.sha256(DURATION.read_bytes()).hexdigest(),
        'weight_payload_sha256': origins,
        'array_sha256': {name: hashlib.sha256(value.tobytes()).hexdigest()
                         for name, value in arrays.items()},
        'fixture_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
        'valid_frames': int(shared.shape[0]), 'frame_capacity': 128,
    }
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
