"""Extract exact pinned effective remaining prosody weights for CI replay."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / 'tests/fixtures/tts/kokoro_prosody_second_block_weights_pinned.npz'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    manifest = json.loads((args.bundle_dir / 'weights_manifest.json').read_text())
    entries = {entry['name']: entry for entry in manifest['entries']}
    bump = (args.bundle_dir / 'weights.bump').read_bytes()
    arrays = {}
    origins = {}
    for branch in ('F0', 'N'):
        prefixes = (f'duration_prosody.{branch}.1.',
                    f'duration_prosody.{branch}.2.',
                    f'duration_prosody.{branch}_proj.')
        for name, entry in entries.items():
            if not name.startswith(prefixes):
                continue
            payload = bump[entry['file_offset']:entry['file_offset'] + entry['size']]
            if hashlib.sha256(payload).hexdigest() != entry['sha256']:
                raise ValueError(f'BUMP payload mismatch: {name}')
            arrays[name] = np.frombuffer(payload, '<f4').reshape(entry['shape']).copy()
            origins[name] = {'source_sha256': entry['sha256'],
                             'source_shape': entry['shape'],
                             'transform': 'identity'}
        # Conv1D's checked ABI has an explicit bias. Kokoro's shortcut has no
        # bias, so the importer declares an exact zero tensor in the bundle.
        name = prefixes[0] + 'conv1x1.cke_zero_bias'
        arrays[name] = np.zeros(256, np.float32)
        origins[name] = {'source_sha256': None, 'source_shape': None,
                         'transform': 'synthesize_exact_zero_bias_for_biasless_conv1d'}
    if len(arrays) != 60:
        raise ValueError(f'expected 60 remaining-prosody tensors, found {len(arrays)}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    metadata = {'scope': 'pinned effective Kokoro second/third F0/N block and projection weights',
        'model_pin': manifest['pin']['model']['revision'],
        'bundle_manifest_sha256': hashlib.sha256(
            (args.bundle_dir / 'weights_manifest.json').read_bytes()).hexdigest(),
        'origins': origins,
        'array_sha256': {name: hashlib.sha256(value.tobytes()).hexdigest()
                         for name, value in arrays.items()},
        'fixture_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
