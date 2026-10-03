#!/usr/bin/env python3
"""Package pinned first source-residual pair outputs and effective weights."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / 'tests/fixtures/tts/kokoro_source_residual_first_pair_pinned.npz'
PREFIX = 'waveform_decoder.generator.noise_res.0'
CAPTURES = {
    'style1': 'decoder_generator_noise_res_0_adain2_0_fc',
    'norm1': 'decoder_generator_noise_res_0_adain2_0',
    'snake1': 'decoder_generator_noise_res_0_snake1',
    'conv1': 'decoder_generator_noise_res_0_convs2_0',
    'pair0': 'decoder_generator_noise_res_0_pair0_output',
}
WEIGHTS = {
    'style_weight': f'{PREFIX}.adain2.0.fc.weight',
    'style_bias': f'{PREFIX}.adain2.0.fc.bias',
    'norm_weight': f'{PREFIX}.adain2.0.norm.weight',
    'norm_bias': f'{PREFIX}.adain2.0.norm.bias',
    'alpha': f'{PREFIX}.alpha2.0',
    'conv_weight': f'{PREFIX}.convs2.0.weight',
    'conv_bias': f'{PREFIX}.convs2.0.bias',
}


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
        parser.error('first-pair fixture requires pinned PyTorch 2.8')
    arrays = {}
    for label, key in CAPTURES.items():
        item = capture['tensors'][key]
        source = args.capture_dir / item['file']
        if sha256(source.read_bytes()) != item['sha256']:
            parser.error(f'capture checksum mismatch: {key}')
        arrays[label] = np.load(source, allow_pickle=False).astype(np.float32)
    entries = {item['name']: item for item in bundle['entries']}
    with (args.bundle_dir / 'weights.bump').open('rb') as stream:
        for label, key in WEIGHTS.items():
            item = entries[key]
            stream.seek(item['file_offset'])
            data = stream.read(item['size'])
            if sha256(data) != item['sha256']:
                parser.error(f'effective weight checksum mismatch: {key}')
            tensor = np.frombuffer(data, '<f4').copy().reshape(item['shape'])
            arrays[label] = tensor.reshape(256).copy() if label == 'alpha' else tensor
    expected = {'style1': (1, 512), 'norm1': (1, 256, 2060),
                'snake1': (1, 256, 2060), 'conv1': (1, 256, 2060),
                'pair0': (1, 256, 2060), 'style_weight': (512, 128),
                'style_bias': (512,), 'norm_weight': (256,),
                'norm_bias': (256,), 'alpha': (256,),
                'conv_weight': (256, 256, 7), 'conv_bias': (256,)}
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            parser.error(f'{name}: shape {arrays[name].shape} != {shape}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    metadata = {
        'scope': 'direct pinned full-model first source residual pair',
        'model_pin': capture['pin']['model']['revision'],
        'code_pin': capture['pin']['reference_code']['kokoro'],
        'oracle': 'pinned full KModel forward and pre-convolution hooks',
        'environment': capture['environment']['packages'],
        'asset_sha256': capture['assets'],
        'capture_manifest_sha256': sha256(manifest_path.read_bytes()),
        'source_tensor_sha256': {label: capture['tensors'][key]['sha256']
                                 for label, key in CAPTURES.items()},
        'effective_weight_sha256': {key: entries[key]['sha256']
                                    for key in WEIGHTS.values()},
        'transformations': {'alpha': f'{PREFIX}.alpha2.0 [1,256,1] '
                             '→ canonical contiguous [256]; no value change'},
        'array_sha256': {name: sha256(value.tobytes())
                         for name, value in arrays.items()},
        'fixture_sha256': sha256(args.output.read_bytes()),
        'input_text': capture['graphemes'],
        'valid_frames': 2060,
    }
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
