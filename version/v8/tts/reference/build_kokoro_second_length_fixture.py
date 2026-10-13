#!/usr/bin/env python3
"""Reduce a pinned full-model Kokoro capture to the short WAV regression inputs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / 'tests/fixtures/tts/kokoro_waveform_hello_8_pinned.npz'
TENSORS = {
    'predictor_style': 'predictor_style',
    'decoder_style': 'decoder_style',
    'gaussian': 'generator_harmonic_gaussian',
    'window': 'generator_stft_window',
    'durations': 'predicted_duration',
    'f0': 'predictor_F0_proj',
    'waveform': 'waveform_f32',
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    capture = args.capture.resolve()
    manifest_path = capture / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if (manifest['status'] != 'full_oracle_captured' or
        manifest['input_text'] != 'Hello.' or
        manifest['phonemes'] != 'həlˈO.' or
        len(manifest['input_ids']) != 8 or
        manifest['voice_row_index'] != 5):
        raise ValueError('unexpected short Kokoro capture identity')
    tensors = {'word_ids': np.asarray(manifest['input_ids'], dtype=np.int32)}
    for name, label in TENSORS.items():
        record = manifest['tensors'][label]
        path = capture / record['file']
        if sha256(path) != record['sha256']:
            raise ValueError(f'capture tensor checksum mismatch: {label}')
        tensors[name] = np.load(path, allow_pickle=False)
    if (tensors['gaussian'].shape != (1, 32400, 9) or
        tensors['waveform'].shape != (32400,) or
        tensors['durations'].shape != (8,)):
        raise ValueError('unexpected short Kokoro tensor geometry')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **tensors)
    metadata = {
        'schema': 'cke.kokoro.short_waveform_fixture.v1',
        'scope': 'Hello. 8 phoneme IDs; direct pinned full-model capture',
        'model_pin': manifest['pin']['model'],
        'code_pin': manifest['pin']['reference_code'],
        'asset_sha256': manifest['assets'],
        'environment': manifest['environment']['packages'],
        'input_text': manifest['input_text'],
        'phonemes': manifest['phonemes'],
        'voice_row_index': manifest['voice_row_index'],
        'input_ids': manifest['input_ids'],
        'source_manifest_sha256': sha256(manifest_path),
        'fixture_sha256': sha256(args.output),
        'tensors': {name: {'shape': list(value.shape), 'dtype': str(value.dtype)}
                    for name, value in tensors.items()},
        'reproduction_command':
            'python version/v8/tts/reference/capture_kokoro_v1.py '
            '--model-dir /path/to/pinned/kokoro --output-dir /tmp/hello '
            "--text 'Hello.'; python "
            'version/v8/tts/reference/build_kokoro_second_length_fixture.py '
            '--capture /tmp/hello',
    }
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
