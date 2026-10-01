"""Independent PyTorch 2.8 nearest and depthwise ConvTranspose1d cases."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / 'tests/fixtures/tts/kokoro_prosody_upsample_torch28.npz'
BRANCH = ROOT / 'tests/fixtures/tts/kokoro_prosody_branch_model_pinned.npz'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not torch.__version__.startswith('2.8.0'):
        raise ValueError('this fixture requires pinned PyTorch 2.8')
    torch.set_num_threads(1)
    manifest = json.loads((args.bundle_dir / 'weights_manifest.json').read_text())
    entries = {entry['name']: entry for entry in manifest['entries']}
    bump = (args.bundle_dir / 'weights.bump').read_bytes()
    def weight(name):
        entry = entries[name]
        payload = bump[entry['file_offset']:entry['file_offset'] + entry['size']]
        if hashlib.sha256(payload).hexdigest() != entry['sha256']:
            raise ValueError(f'BUMP weight mismatch: {name}')
        return np.frombuffer(payload, dtype='<f4').reshape(entry['shape']).copy()
    arrays = {}
    rng = np.random.default_rng(784)
    with np.load(BRANCH) as direct:
        model_input = direct['predictor_F0_0'].copy()
    for index, (channels, frames) in enumerate(((1, 1), (3, 7), (512, 103))):
        source = (model_input if index == 2 else
                  rng.normal(size=(channels, frames)).astype(np.float32))
        kernel = (weight('duration_prosody.F0.1.pool.weight') if index == 2 else
                  rng.normal(size=(channels, 1, 3)).astype(np.float32))
        bias = (weight('duration_prosody.F0.1.pool.bias') if index == 2 else
                rng.normal(size=(channels,)).astype(np.float32))
        with torch.no_grad():
            value = torch.from_numpy(source.copy())[None]
            nearest = F.interpolate(value, scale_factor=2, mode='nearest')
            transposed = F.conv_transpose1d(value,
                torch.from_numpy(kernel), torch.from_numpy(bias),
                stride=2, padding=1, output_padding=1, groups=channels)
        for name, array in (('input', source), ('weight', kernel), ('bias', bias),
                            ('nearest', nearest[0].numpy().copy()),
                            ('transposed', transposed[0].numpy().copy())):
            arrays[f'case{index}_{name}'] = array
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    metadata = {'oracle': 'PyTorch F.interpolate nearest and F.conv_transpose1d depthwise',
        'oracle_version': torch.__version__,
        'model_pin': manifest['pin']['model']['revision'],
        'direct_branch_fixture_sha256': hashlib.sha256(BRANCH.read_bytes()).hexdigest(),
        'shapes': [[1, 1], [3, 7], [512, 103]],
        'geometry': {'scale': 2, 'kernel': 3, 'stride': 2,
                     'padding': 1, 'output_padding': 1},
        'array_sha256': {name: hashlib.sha256(value.tobytes()).hexdigest()
                         for name, value in arrays.items()},
        'fixture_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
