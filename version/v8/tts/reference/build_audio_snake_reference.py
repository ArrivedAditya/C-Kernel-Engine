"""Create committed independent PyTorch 2.8 Snake inference fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'tests/fixtures/tts/audio_snake_pytorch28_reference.npz'


def main() -> None:
    if not torch.__version__.startswith('2.8.'):
        raise RuntimeError('fixture requires pinned PyTorch 2.8')
    rng = np.random.default_rng(3371)
    arrays: dict[str, np.ndarray] = {}
    for index, (channels, frames, scale) in enumerate(((3, 7, 2.0),
                                                        (17, 31, 3.0),
                                                        (256, 129, 0.7))):
        data = rng.normal(0, scale, (channels, frames)).astype(np.float32)
        alpha = rng.uniform(0.15, 3.0, channels).astype(np.float32)
        if index == 0:
            data[0, :5] = [-3, -1, 0, 1, 3]
            alpha[:] = [1., 0.25, -2.]
        x = torch.from_numpy(data.copy())
        a = torch.from_numpy(alpha.copy())[:, None]
        result = (x + (1 / a) * (torch.sin(a * x) ** 2)).numpy()
        arrays[f'case{index}_input'] = data
        arrays[f'case{index}_alpha'] = alpha
        arrays[f'case{index}_output'] = result.copy()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **arrays)
    meta = {
        'schema': 'cke.audio.snake.pytorch_reference.v1',
        'formula': 'x + (1 / alpha) * sin(alpha * x)^2',
        'torch': torch.__version__,
        'torch_git_version': torch.version.git_version,
        'fixture_sha256': hashlib.sha256(OUT.read_bytes()).hexdigest(),
        'array_sha256': {name: hashlib.sha256(value.tobytes()).hexdigest()
                         for name, value in arrays.items()},
        'generator': 'version/v8/tts/reference/build_audio_snake_reference.py',
    }
    OUT.with_suffix('.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(OUT)


if __name__ == '__main__':
    main()
