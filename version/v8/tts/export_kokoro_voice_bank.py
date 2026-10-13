#!/usr/bin/env python3
"""Convert the pinned af_heart PyTorch voice table to the native CKVOICE1 ABI.

This is import tooling. Native inference reads CKVOICE1 without Python or Torch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile

import numpy as np


MANIFEST = Path(__file__).resolve().parent / 'reference/fixture_manifest.json'


def export_voice_bank(source: Path, destination: Path) -> dict:
    expected = json.loads(MANIFEST.read_text())['assets']['voices/af_heart.pt']
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f'af_heart source hash mismatch: {actual} != {expected}')
    import torch  # Oracle/import dependency; never needed by the native host.

    tensor = torch.load(source, map_location='cpu', weights_only=True)
    if (not isinstance(tensor, torch.Tensor) or tensor.ndim == 0 or
            tensor.shape[-1] != 256):
        raise ValueError('expected a tensor with 256 style values per row')
    values = tensor.detach().cpu().numpy().reshape(-1, 256)
    if not 8 <= values.shape[0] <= 4096 or not np.isfinite(values).all():
        raise ValueError('invalid af_heart row count or nonfinite style')
    payload = np.asarray(values, dtype='<f4', order='C').tobytes()
    packed = b'CKVOICE1' + struct.pack('<II', values.shape[0], 256) + payload
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent,
                                         prefix=destination.name + '.',
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(packed)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {
        'schema': 'cke.kokoro.native_voice_bank.v1',
        'voice': 'af_heart',
        'source_sha256': actual,
        'native_sha256': hashlib.sha256(packed).hexdigest(),
        'rows': int(values.shape[0]),
        'columns': 256,
        'dtype': 'fp32-le',
        'decoder_style': 'columns 0:128',
        'predictor_style': 'columns 128:256',
        'selection': 'row = compiled phoneme ID count - 3',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='pinned voices/af_heart.pt')
    parser.add_argument('destination', type=Path, help='output CKVOICE1 file')
    args = parser.parse_args()
    print(json.dumps(export_voice_bank(args.source, args.destination), indent=2))


if __name__ == '__main__':
    main()
