#!/usr/bin/env python3
"""Package direct full-model checkpoints for Kokoro's first source block."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / 'tests/fixtures/tts/kokoro_source_residual_block0_pinned.npz'
PREFIX = 'waveform_decoder.generator.noise_res.0'


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', type=Path, required=True)
    parser.add_argument('--bundle-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    manifest_path = args.capture_dir / 'manifest.json'
    capture = json.loads(manifest_path.read_text())
    bundle = json.loads((args.bundle_dir / 'weights_manifest.json').read_text())
    if capture['pin']['model']['revision'] != bundle['pin']['model']['revision']:
        parser.error('capture and BUMP model revisions differ')
    if capture['assets'] != bundle['provenance']['source_asset_sha256']:
        parser.error('capture and BUMP source assets differ')
    if capture['environment']['packages']['torch'] != '2.8.0+cpu':
        parser.error('block fixture requires pinned PyTorch 2.8')
    captures = {}
    for pair in (1, 2):
        stem = f'decoder_generator_noise_res_0'
        captures.update({
            f'pair{pair}_style_c1': f'{stem}_adain1_{pair}_fc',
            f'pair{pair}_conv_c1': f'{stem}_convs1_{pair}',
            f'pair{pair}_conv_c2': f'{stem}_convs2_{pair}',
            f'pair{pair}_output': f'{stem}_pair{pair}_output' if pair == 1
                else 'decoder_generator_noise_res_0',
        })
    arrays = {}
    for label, key in captures.items():
        item = capture['tensors'][key]
        source = args.capture_dir / item['file']
        if sha256(source.read_bytes()) != item['sha256']:
            parser.error(f'capture checksum mismatch: {key}')
        arrays[label] = np.load(source, allow_pickle=False).astype(np.float32)
    weights = {}
    for pair in (1, 2):
        for side in (1, 2):
            stem = f'{PREFIX}.adain{side}.{pair}'
            for name, suffix in (('style_weight', '.fc.weight'),
                                 ('style_bias', '.fc.bias'),
                                 ('norm_weight', '.norm.weight'),
                                 ('norm_bias', '.norm.bias')):
                weights[f'pair{pair}_{name}_c{side}'] = stem + suffix
            weights[f'pair{pair}_alpha_c{side}'] = \
                f'{PREFIX}.alpha{side}.{pair}'
            weights[f'pair{pair}_conv_weight_c{side}'] = \
                f'{PREFIX}.convs{side}.{pair}.weight'
            weights[f'pair{pair}_conv_bias_c{side}'] = \
                f'{PREFIX}.convs{side}.{pair}.bias'
    entries = {item['name']: item for item in bundle['entries']}
    with (args.bundle_dir / 'weights.bump').open('rb') as stream:
        for label, key in weights.items():
            item = entries[key]
            stream.seek(item['file_offset'])
            data = stream.read(item['size'])
            if sha256(data) != item['sha256']:
                parser.error(f'effective weight checksum mismatch: {key}')
            tensor = np.frombuffer(data, '<f4').copy().reshape(item['shape'])
            arrays[label] = tensor.reshape(256).copy() if '_alpha_' in label \
                else tensor
    for pair in (1, 2):
        for name in ('conv_c1', 'conv_c2', 'output'):
            key = f'pair{pair}_{name}'
            if arrays[key].shape != (1, 256, 2060):
                parser.error(f'{key}: unexpected shape {arrays[key].shape}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    metadata = {
        'scope': 'direct pinned full-model first source residual block',
        'model_pin': capture['pin']['model']['revision'],
        'code_pin': capture['pin']['reference_code']['kokoro'],
        'oracle': 'pinned full KModel forward and pre-convolution hooks',
        'environment': capture['environment']['packages'],
        'asset_sha256': capture['assets'],
        'capture_manifest_sha256': sha256(manifest_path.read_bytes()),
        'source_tensor_sha256': {label: capture['tensors'][key]['sha256']
                                 for label, key in captures.items()},
        'effective_weight_sha256': {key: entries[key]['sha256']
                                    for key in weights.values()},
        'transformations': {'alpha': 'Each [1,256,1] alpha is stored as '
                             'canonical contiguous [256], without value changes'},
        'array_sha256': {name: sha256(value.tobytes())
                         for name, value in arrays.items()},
        'fixture_sha256': sha256(args.output.read_bytes()),
        'input_text': capture['graphemes'],
        'valid_frames': 2060,
    }
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
