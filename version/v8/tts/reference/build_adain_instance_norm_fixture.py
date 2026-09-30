"""Create an independent PyTorch oracle for channelwise AdaIN inference."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / 'tests/fixtures/tts/adain_instance_norm_torch.npz'
CASES = ((1, 2), (2, 5), (3, 7), (8, 16), (512, 103))
EPSILON = 1e-5


def main():
    rng = np.random.default_rng(20921)
    arrays = {}
    for index, (channels, frames) in enumerate(CASES):
        x = rng.normal(0, .25, (channels, frames)).astype(np.float32)
        if index == 1:
            x[:] = 0.
        weight = rng.normal(1, .1, channels).astype(np.float32)
        bias = rng.normal(0, .1, channels).astype(np.float32)
        style = rng.normal(0, .2, 2 * channels).astype(np.float32)
        with torch.no_grad():
            native = F.instance_norm(
                torch.from_numpy(x)[None], weight=torch.from_numpy(weight),
                bias=torch.from_numpy(bias), use_input_stats=True,
                eps=EPSILON)[0]
            output = (1 + torch.from_numpy(style[:channels, None])) * native
            output = output + torch.from_numpy(style[channels:, None])
        for name, value in (('input', x), ('norm_weight', weight),
                            ('norm_bias', bias), ('style_affine', style),
                            ('output', output.numpy())):
            arrays[f'case{index}_{name}'] = np.ascontiguousarray(value)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT, **arrays)
    metadata = {
        'oracle': 'torch.nn.functional.instance_norm plus explicit style affine',
        'backend_version': torch.__version__, 'device': 'cpu',
        'epsilon': EPSILON,
        'shapes': [list(shape) for shape in CASES],
        'array_sha256': {name: hashlib.sha256(value.tobytes()).hexdigest()
                         for name, value in arrays.items()},
        'fixture_sha256': hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
    }
    OUTPUT.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
