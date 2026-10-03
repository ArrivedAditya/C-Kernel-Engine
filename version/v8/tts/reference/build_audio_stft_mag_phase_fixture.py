#!/usr/bin/env python3
"""Capture a small independent PyTorch magnitude/phase STFT reference."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'tests/fixtures/tts/audio_stft_mag_phase_torch_reference.npz'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    samples = np.random.default_rng(871).normal(0, .05, 60).astype(np.float32)
    spectrum = torch.stft(torch.from_numpy(samples), 20, 5, 20,
        window=torch.hann_window(20, periodic=True), return_complex=True)
    arrays = {'samples': samples,
              'magnitude': torch.abs(spectrum).numpy(),
              'phase': torch.angle(spectrum).numpy()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **arrays)
    metadata = {
        'oracle': 'torch.stft; torch.abs; torch.angle',
        'torch': torch.__version__,
        'n_fft': 20,
        'hop': 5,
        'center': True,
        'pad_mode': 'reflect',
        'window': 'periodic Hann',
        'array_sha256': {name: digest(value.tobytes()) for name, value in arrays.items()},
        'fixture_sha256': digest(OUT.read_bytes()),
    }
    OUT.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
