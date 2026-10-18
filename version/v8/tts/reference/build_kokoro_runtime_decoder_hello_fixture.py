#!/usr/bin/env python3
"""Package the pinned eight-ID decoder and first generator capture."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / 'tests/fixtures/tts/kokoro_runtime_decoder_hello_8_pinned.npz'
NAMES = ('predictor_style', 'decoder_style', 'predicted_duration',
         'decoder_decode_0', 'decoder_decode_1', 'decoder_decode_2',
         'decoder_decode_3', 'decoder_generator_ups_0',
         'generator_harmonic_gaussian', 'generator_source_samples',
         'decoder_input_1', 'generator_stft_window',
         'generator_source_magnitude', 'generator_source_phase',
         'decoder_generator_noise_convs_0',
         'decoder_generator_noise_res_0',
         'decoder_generator_stage0_join',
         'decoder_generator_resblocks_0',
         'decoder_generator_resblocks_1',
         'decoder_generator_resblocks_2',
         'decoder_generator_stage1_activation',
         'decoder_generator_ups_1',
         'decoder_generator_reflection_pad',
         'decoder_generator_noise_convs_1',
         'decoder_generator_noise_res_1',
         'decoder_generator_stage1_join',
         'decoder_generator_stage1_final_activation',
         'decoder_generator_conv_post', 'istft_magnitude',
         'istft_phase', 'waveform_f32')
WAVEFORM_ONLY = ('predictor_style', 'decoder_style',
                 'generator_harmonic_gaussian',
                 'decoder_generator_stage1_join',
                 'decoder_generator_stage1_final_activation',
                 'decoder_generator_conv_post', 'istft_magnitude',
                 'istft_phase', 'waveform_f32')


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True,
                        help='output of capture_kokoro_v1.py --text Hello.')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--text', default='Hello.')
    parser.add_argument('--tokens', type=int, default=8)
    parser.add_argument('--waveform-only', action='store_true',
                        help='retain only the connected waveform checkpoints')
    args = parser.parse_args()
    manifest_path = args.capture / 'manifest.json'
    capture = json.loads(manifest_path.read_text())
    if (args.tokens < 3 or args.tokens > 40 or
            capture['input_text'] != args.text or
            len(capture['input_ids']) != args.tokens or
            capture['voice_row_index'] != args.tokens - 3
            or capture['status'] != 'full_oracle_captured'):
        raise ValueError('expected the pinned full-model capture and voice row')
    arrays = {'word_ids': np.asarray(capture['input_ids'], dtype=np.int32)}
    for name in WAVEFORM_ONLY if args.waveform_only else NAMES:
        tensor = capture['tensors'][name]
        path = args.capture / tensor['file']
        if sha256(path.read_bytes()) != tensor['sha256']:
            raise ValueError(f'capture tensor changed: {name}')
        arrays[name] = np.load(path, allow_pickle=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    meta = {
        'schema': ('cke.kokoro.runtime_decoder_hello_8.v11'
                   if args.tokens == 8 and args.text == 'Hello.' else
                   f'cke.kokoro.runtime_waveform_{args.tokens}.v1'),
        'scope': f'direct pinned full-model {args.tokens}-ID decoder through waveform',
        'model_pin': capture['pin']['model']['revision'],
        'code_pin': capture['pin']['reference_code']['kokoro'],
        'asset_sha256': capture['assets'],
        'environment': capture['environment']['packages'],
        'voice_row_index': capture['voice_row_index'],
        'valid_frames': (int(arrays['decoder_decode_0'].shape[-1])
                         if not args.waveform_only else
                         int(arrays['waveform_f32'].size // 600)),
        'upsampled_frames': (int(arrays['decoder_decode_3'].shape[-1])
                             if not args.waveform_only else
                             int(arrays['waveform_f32'].size // 300)),
        'source_manifest_sha256': sha256(manifest_path.read_bytes()),
        'array_sha256': {name: sha256(value.tobytes())
                         for name, value in arrays.items()},
        'fixture_sha256': sha256(args.output.read_bytes()),
        'reproduction_command':
            'python version/v8/tts/reference/capture_kokoro_v1.py '
            f"--model-dir /path/to/pinned/kokoro --output-dir /tmp/capture --text '{args.text}'; "
            'python version/v8/tts/reference/build_kokoro_runtime_decoder_hello_fixture.py '
            f"--capture /tmp/capture --text '{args.text}' --tokens {args.tokens} "
            f"{'--waveform-only ' if args.waveform_only else ''}--output {args.output}",
    }
    args.output.with_suffix('.json').write_text(json.dumps(meta, indent=2) + '\n')


if __name__ == '__main__':
    main()
