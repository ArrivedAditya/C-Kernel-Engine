#!/usr/bin/env python3
"""Capture independent pinned PyTorch SineGen cases for bounded source geometry."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from kokoro.istftnet import SineGen
import kokoro.istftnet as istftnet


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / 'tests/fixtures/tts/audio_harmonic_source_cases_pytorch28.npz'
CASES = (
    ('single_voiced', [120.], 4, 2, 104),
    ('uv_transitions', [0., 5., 120., 0., 240.], 4, 3, 205),
    ('longer_mixed', [80., 80., 0., 100., 180., 0., 200.], 6, 4, 306),
    ('long_unvoiced', [0., 0., 0., 440., 330., 20., 0., 15., 220.], 8, 2, 407),
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    if torch.__version__ != '2.8.0+cpu':
        raise RuntimeError(f'pinned PyTorch 2.8.0+cpu required, got {torch.__version__}')
    torch.set_num_threads(1)
    arrays = {}
    cases = []
    original_randn_like = torch.randn_like
    try:
        for name, values, upsample, harmonics, seed in CASES:
            f0 = np.asarray(values, np.float32)
            gaussian = np.random.default_rng(seed).standard_normal(
                (1, len(f0) * upsample, harmonics)).astype(np.float32)
            gaussian_tensor = torch.from_numpy(gaussian)
            draws = []
            def fixed_gaussian(input_value, *args, **kwargs):
                if tuple(input_value.shape) != tuple(gaussian_tensor.shape):
                    raise RuntimeError(f'unexpected SineGen Gaussian shape for {name}')
                draws.append(1)
                return gaussian_tensor.clone()
            torch.randn_like = fixed_gaussian
            torch.manual_seed(seed)
            model = SineGen(24000, upsample, harmonic_num=harmonics - 1,
                            voiced_threshold=10)
            expanded = torch.from_numpy(np.repeat(f0, upsample)).view(1, -1, 1)
            with torch.no_grad():
                waveform, uv, _ = model(expanded)
            if len(draws) != 1:
                raise RuntimeError(f'expected one captured Gaussian draw for {name}')
            arrays[f'{name}_f0'] = f0
            arrays[f'{name}_gaussian'] = gaussian
            arrays[f'{name}_output'] = waveform.numpy().copy()
            arrays[f'{name}_uv'] = uv.numpy().copy()
            cases.append({'name': name, 'frames': len(f0),
                          'upsample': upsample, 'harmonics': harmonics,
                          'seed': seed})
    finally:
        torch.randn_like = original_randn_like
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT, **arrays)
    metadata = {
        'oracle': 'pinned kokoro.istftnet.SineGen.forward',
        'torch': torch.__version__,
        'reference_code_sha256': sha256(Path(istftnet.__file__).read_bytes()),
        'cases': cases,
        'array_sha256': {key: sha256(value.tobytes()) for key, value in arrays.items()},
        'fixture_sha256': sha256(OUTPUT.read_bytes()),
    }
    OUTPUT.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
