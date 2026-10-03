#!/usr/bin/env python3
"""Package direct pinned Kokoro harmonic-source and STFT checkpoints."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / 'tests/fixtures/tts/kokoro_source_pinned.npz'
CAPTURES = {
    'f0': 'decoder_input_1',
    'gaussian': 'generator_harmonic_gaussian',
    'sine_waves': 'decoder_generator_m_source_l_sin_gen_0',
    'source': 'generator_source_samples',
    'window': 'generator_stft_window',
    'magnitude': 'generator_source_magnitude',
    'phase': 'generator_source_phase',
    'first_source_conv': 'decoder_generator_noise_convs_0',
}
WEIGHTS = {
    'linear_weight': 'waveform_decoder.generator.m_source.l_linear.weight',
    'linear_bias': 'waveform_decoder.generator.m_source.l_linear.bias',
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
        parser.error('source fixture requires pinned PyTorch 2.8 oracle')
    arrays = {}
    for name, key in CAPTURES.items():
        entry = capture['tensors'][key]
        source = args.capture_dir / entry['file']
        if sha256(source.read_bytes()) != entry['sha256']:
            parser.error(f'capture checksum mismatch: {key}')
        arrays[name] = np.load(source, allow_pickle=False).astype(np.float32)
    entries = {item['name']: item for item in bundle['entries']}
    with (args.bundle_dir / 'weights.bump').open('rb') as stream:
        for name, key in WEIGHTS.items():
            entry = entries[key]
            stream.seek(entry['file_offset'])
            data = stream.read(entry['size'])
            if sha256(data) != entry['sha256']:
                parser.error(f'weight checksum mismatch: {key}')
            arrays[name] = np.frombuffer(data, '<f4').copy().reshape(entry['shape'])
    if (arrays['f0'].shape != (1, 206) or
            arrays['gaussian'].shape != (1, 61800, 9) or
            arrays['sine_waves'].shape != (1, 61800, 9) or
            arrays['source'].shape != (1, 61800) or
            arrays['window'].shape != (20,) or
            arrays['magnitude'].shape != (1, 11, 12361) or
            arrays['phase'].shape != (1, 11, 12361) or
            arrays['first_source_conv'].shape != (1, 256, 2060)):
        parser.error('unexpected source capture geometry')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    metadata = {
        'scope': 'direct pinned full-model harmonic-source and STFT hooks',
        'model_pin': capture['pin']['model']['revision'],
        'code_pin': capture['pin']['reference_code']['kokoro'],
        'oracle': 'pinned full KModel forward hooks and captured random draws',
        'environment': capture['environment']['packages'],
        'asset_sha256': capture['assets'],
        'capture_manifest_sha256': sha256(manifest_path.read_bytes()),
        'effective_weight_sha256': {key: entries[key]['sha256'] for key in WEIGHTS.values()},
        'array_sha256': {name: sha256(value.tobytes()) for name, value in arrays.items()},
        'fixture_sha256': sha256(args.output.read_bytes()),
        'input_text': capture['graphemes'],
        'valid_f0_frames': 206,
        'valid_source_samples': 61800,
        'valid_stft_frames': 12361,
    }
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
