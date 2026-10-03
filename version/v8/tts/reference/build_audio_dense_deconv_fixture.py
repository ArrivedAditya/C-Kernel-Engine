#!/usr/bin/env python3
"""Commit small independent PyTorch ConvTranspose1D reference cases."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / 'tests/fixtures/tts/audio_dense_deconv_torch_reference.npz'
CASES = (
    (1, 1, 1, 4, 2, 1, 0),
    (3, 5, 4, 5, 2, 1, 0),
    (5, 3, 3, 20, 10, 5, 0),
    (7, 4, 3, 12, 6, 3, 0),
    (2, 3, 5, 3, 2, 1, 1),
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    seed = 1937
    rng = np.random.default_rng(seed)
    arrays = {}
    for index, (inputs, outputs, frames, kernel, stride, padding,
                output_padding) in enumerate(CASES):
        source = (rng.standard_normal((inputs, frames)) * .2).astype(np.float32)
        weight = (rng.standard_normal((inputs, outputs, kernel)) * .2).astype(np.float32)
        bias = (rng.standard_normal(outputs) * .1).astype(np.float32)
        result = functional.conv_transpose1d(
            torch.from_numpy(source)[None], torch.from_numpy(weight),
            torch.from_numpy(bias), stride=stride, padding=padding,
            output_padding=output_padding)[0].detach().numpy().copy()
        for name, value in (('input', source), ('weight', weight),
                            ('bias', bias), ('output', result)):
            arrays[f'case{index}_{name}'] = value
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT, **arrays)
    metadata = {
        'oracle': 'torch.nn.functional.conv_transpose1d',
        'oracle_version': torch.__version__,
        'numpy_version': np.__version__,
        'seed': seed,
        'torch_threads': torch.get_num_threads(),
        'mkldnn_enabled': torch.backends.mkldnn.enabled,
        'generator_sha256': digest(Path(__file__).read_bytes()),
        'cases': CASES,
        'fixture_sha256': digest(OUTPUT.read_bytes()),
        'array_sha256': {name: digest(value.tobytes())
                         for name, value in arrays.items()},
    }
    OUTPUT.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(OUTPUT)


if __name__ == '__main__':
    main()
