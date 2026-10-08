#!/usr/bin/env python3
"""Package direct full-model checkpoints for the first-stage generator pool."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / 'tests/fixtures/tts/kokoro_generator_stage0_pool_pinned.npz'


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

    captures = {'input_join': 'decoder_generator_stage0_join'}
    for block in (1, 2):
        for pair in range(3):
            stem = f'decoder_generator_resblocks_{block}'
            captures.update({
                f'block{block}_pair{pair}_conv_c1': f'{stem}_convs1_{pair}',
                f'block{block}_pair{pair}_conv_c2': f'{stem}_convs2_{pair}',
                f'block{block}_pair{pair}_output': f'{stem}_pair{pair}_output'
                    if pair < 2 else stem,
            })
    arrays = {}
    for label, key in captures.items():
        item = capture['tensors'][key]
        source = args.capture_dir / item['file']
        if sha256(source.read_bytes()) != item['sha256']:
            parser.error(f'capture checksum mismatch: {key}')
        arrays[label] = np.load(source, allow_pickle=False).astype(np.float32)
        if arrays[label].shape != (1, 256, 2060):
            parser.error(f'{key}: unexpected shape {arrays[label].shape}')

    weight_names = {}
    for block in (1, 2):
        prefix = f'waveform_decoder.generator.resblocks.{block}'
        for pair in range(3):
            for side in (1, 2):
                stem = f'{prefix}.adain{side}.{pair}'
                label_prefix = f'block{block}_pair{pair}'
                for label, suffix in (('style_weight', '.fc.weight'),
                                      ('style_bias', '.fc.bias'),
                                      ('norm_weight', '.norm.weight'),
                                      ('norm_bias', '.norm.bias')):
                    weight_names[f'{label_prefix}_{label}_c{side}'] = stem + suffix
                weight_names[f'{label_prefix}_alpha_c{side}'] = \
                    f'{prefix}.alpha{side}.{pair}'
                weight_names[f'{label_prefix}_conv_weight_c{side}'] = \
                    f'{prefix}.convs{side}.{pair}.weight'
                weight_names[f'{label_prefix}_conv_bias_c{side}'] = \
                    f'{prefix}.convs{side}.{pair}.bias'
    entries = {item['name']: item for item in bundle['entries']}
    with (args.bundle_dir / 'weights.bump').open('rb') as stream:
        for label, key in weight_names.items():
            item = entries[key]
            stream.seek(item['file_offset'])
            data = stream.read(item['size'])
            if sha256(data) != item['sha256']:
                parser.error(f'effective weight checksum mismatch: {key}')
            tensor = np.frombuffer(data, '<f4').copy().reshape(item['shape'])
            arrays[label] = tensor.reshape(256).copy() if '_alpha_' in label \
                else tensor
    first_path = ROOT / 'tests/fixtures/tts/kokoro_generator_main_resblock0_pinned.npz'
    first_meta = json.loads(first_path.with_suffix('.json').read_text())
    if (first_meta['model_pin'] != capture['pin']['model']['revision'] or
        first_meta['code_pin'] != capture['pin']['reference_code']['kokoro'] or
        first_meta['asset_sha256'] != capture['assets']):
        parser.error('first-block fixture belongs to a different pinned experiment')
    first = np.load(first_path)
    np.testing.assert_array_equal(first['input_join'], arrays['input_join'])
    for label, key in (('block0_output', 'pair2_output'),):
        arrays[label] = first[key]
    arrays['stage0_mean'] = ((arrays['block0_output'] +
        arrays['block1_pair2_output'] + arrays['block2_pair2_output']) /
        np.float32(3.0)).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    metadata = {
        'scope': 'direct pinned full-model first-stage generator residual pool',
        'model_pin': capture['pin']['model']['revision'],
        'code_pin': capture['pin']['reference_code']['kokoro'],
        'oracle': 'pinned full KModel convolution and pair-boundary hooks; mean reconstructed from three captured block outputs',
        'environment': capture['environment']['packages'],
        'asset_sha256': capture['assets'],
        'capture_manifest_sha256': sha256(manifest_path.read_bytes()),
        'first_block_fixture_sha256': sha256(first_path.read_bytes()),
        'source_tensor_sha256': {label: capture['tensors'][key]['sha256']
                                 for label, key in captures.items()},
        'effective_weight_sha256': {key: entries[key]['sha256']
                                    for key in weight_names.values()},
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
