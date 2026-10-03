#!/usr/bin/env python3
"""Create independent pinned PyTorch fixtures for checked dilated Conv1D."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / 'tests/fixtures/tts/audio_dilated_conv_pytorch28_reference.npz'
GEOMETRIES = ((3, 4, 7, 3, 2), (17, 19, 31, 5, 3),
              (256, 256, 17, 7, 5))


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    if torch.__version__ != '2.8.0+cpu':
        raise SystemExit(f'expected pinned PyTorch 2.8.0+cpu, got {torch.__version__}')
    torch.set_num_threads(1)
    rng = np.random.default_rng(72531)
    arrays = {}
    cases = []
    for index, (inputs, outputs, frames, kernel, dilation) in enumerate(GEOMETRIES):
        source = rng.normal(0, .3, (inputs, frames)).astype(np.float32)
        weight = rng.normal(0, .1, (outputs, inputs, kernel)).astype(np.float32)
        bias = rng.normal(0, .03, outputs).astype(np.float32)
        padding = dilation * (kernel - 1) // 2
        result = F.conv1d(torch.from_numpy(source[None]),
                          torch.from_numpy(weight), torch.from_numpy(bias),
                          padding=padding, dilation=dilation)[0].numpy().copy()
        for name, value in (('input', source), ('weight', weight),
                            ('bias', bias), ('output', result)):
            arrays[f'case{index}_{name}'] = value
        cases.append({'input_channels': inputs, 'output_channels': outputs,
                      'frames': frames, 'kernel': kernel,
                      'dilation': dilation, 'padding': padding})
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT, **arrays)
    metadata = {
        'oracle': 'independent torch.nn.functional.conv1d',
        'torch': torch.__version__, 'numpy': np.__version__,
        'seed': 72531, 'cases': cases,
        'array_sha256': {name: sha256(value.tobytes())
                         for name, value in arrays.items()},
        'fixture_sha256': sha256(OUTPUT.read_bytes()),
    }
    OUTPUT.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
