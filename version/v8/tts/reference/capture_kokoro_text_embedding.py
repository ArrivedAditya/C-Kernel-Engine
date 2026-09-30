#!/usr/bin/env python3
"""Pin the first acoustic text-encoder operation from Kokoro/PyTorch 2.8.

Offline oracle tooling only. Deployed execution consumes the exported table and
IDs through CKE's native provider; Python does not schedule the model there.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'version/v8/tts'))
import export_kokoro_bump as exporter

MODEL_DIR = Path('/data/cke/models/kokoro-82m-e8a90b4')
OUTPUT = ROOT / 'tests/fixtures/tts/kokoro_text_embedding_pinned.npz'


def digest(data):
    return hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()


def capture(model_dir, output):
    import torch
    from kokoro.model import KModel
    from kokoro.modules import AdaLayerNorm
    versions = {name: importlib.metadata.version(name)
                for name in ('kokoro', 'misaki', 'torch')}
    for name, actual in versions.items():
        expected = exporter.REFERENCE['environment']['packages'][name]
        if actual != expected:
            raise RuntimeError(f'{name} {actual} differs from pinned {expected}')
    assets = exporter.verify_pinned_assets(model_dir)
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    model = KModel(repo_id='hexgrad/Kokoro-82M',
        config=str(model_dir/'config.json'),
        model=str(model_dir/'kokoro-v1_0.pth')).eval()
    source = ROOT/'tests/fixtures/tts/kokoro_encoder_pinned.npz'
    ids = np.load(source)['word_ids'].astype(np.int32)
    tokens = torch.from_numpy(ids.astype(np.int64))
    weight = model.text_encoder.embedding.weight.detach().cpu().numpy().copy()
    expected = model.text_encoder.embedding(tokens).T.contiguous().cpu().numpy().copy()
    if weight.shape != (178, 512) or expected.shape != (512, 36):
        raise RuntimeError('pinned embedding geometry changed')
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, ids=ids, table=weight, expected=expected)
    manifest = {
        'schema': 'cke.kokoro.text_embedding_fixture.v1',
        'scope': 'text encoder embedding only; no convolution or waveform',
        'model_pin': exporter.PIN['model'],
        'reference_code': exporter.PIN['reference_code'],
        'torch': torch.__version__, 'oracle_packages': versions,
        'oracle_source_sha256': {
            'kokoro.model': exporter.sha256_file(Path(inspect.getfile(KModel))),
            'kokoro.modules': exporter.sha256_file(Path(inspect.getfile(AdaLayerNorm)))},
        'package_source_commit_match': 'NOT_TESTED', 'tokens': 36,
        'source_fixture_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'asset_sha256': assets,
        'arrays_sha256': {name: digest(array) for name,array in
            [('ids',ids),('table',weight),('expected',expected)]},
        'oracle': 'model.text_encoder.embedding(ids).T contiguous',
    }
    manifest['fixture_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'fixture':str(output),'sha256':manifest['fixture_sha256']}))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model-dir',type=Path,default=MODEL_DIR)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    capture(args.model_dir,args.output)

if __name__=='__main__': main()
