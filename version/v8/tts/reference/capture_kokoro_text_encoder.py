#!/usr/bin/env python3
"""Capture Kokoro's complete all-valid acoustic text encoder as an offline oracle."""
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

DEFAULT_OUTPUT = ROOT / 'tests/fixtures/tts/kokoro_text_encoder_pinned.npz'


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def capture(model_dir, output):
    import torch
    from torch.nn.utils.parametrize import remove_parametrizations
    from kokoro.model import KModel
    from kokoro.modules import AdaLayerNorm

    packages = {name: importlib.metadata.version(name)
                for name in ('kokoro', 'misaki', 'torch')}
    for name, actual in packages.items():
        expected = exporter.REFERENCE['environment']['packages'][name]
        if actual != expected:
            raise RuntimeError(f'{name} {actual} differs from pinned {expected}')
    assets = exporter.verify_pinned_assets(model_dir)
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    model = KModel(repo_id=exporter.PIN['model']['repository'],
        config=str(model_dir/'config.json'),
        model=str(model_dir/'kokoro-v1_0.pth')).eval()
    ids_fixture = ROOT/'tests/fixtures/tts/kokoro_encoder_pinned.npz'
    ids = np.load(ids_fixture)['word_ids'].astype(np.int32)
    token_ids = torch.from_numpy(ids.astype(np.int64)).unsqueeze(0)
    arrays = {'ids': ids}
    x = model.text_encoder.embedding(token_ids).transpose(1,2)
    arrays['embedding'] = x.squeeze(0).contiguous().numpy().copy()
    for index, block in enumerate(model.text_encoder.cnn):
        conv = block[0]
        remove_parametrizations(conv, 'weight', leave_parametrized=True)
        arrays[f'conv{index}_weight'] = conv.weight.detach().cpu().numpy().copy()
        arrays[f'conv{index}_bias'] = conv.bias.detach().cpu().numpy().copy()
        arrays[f'norm{index}_gamma'] = block[1].gamma.detach().cpu().numpy().copy()
        arrays[f'norm{index}_beta'] = block[1].beta.detach().cpu().numpy().copy()
        x = conv(x)
        arrays[f'conv{index}_output'] = x.squeeze(0).contiguous().numpy().copy()
        x = block[1](x)
        arrays[f'norm{index}_output'] = x.squeeze(0).contiguous().numpy().copy()
        x = block[2](x)
        arrays[f'act{index}_output'] = x.squeeze(0).contiguous().numpy().copy()
        x = block[3](x)
    lstm = model.text_encoder.lstm
    for kind in ('weight_ih','weight_hh','bias_ih','bias_hh'):
        arrays[f'lstm_{kind}'] = np.stack([
            getattr(lstm,f'{kind}_l0{suffix}').detach().cpu().numpy()
            for suffix in ('','_reverse')],axis=0).astype(np.float32)
    packed = x.transpose(1,2)
    packed_sequence = torch.nn.utils.rnn.pack_padded_sequence(
        packed,torch.tensor([len(ids)]),batch_first=True,enforce_sorted=False)
    result_sequence,_ = lstm(packed_sequence)
    result,_ = torch.nn.utils.rnn.pad_packed_sequence(
        result_sequence,batch_first=True)
    arrays['lstm_input'] = packed.squeeze(0).contiguous().numpy().copy()
    arrays['lstm_output_token_major'] = result.squeeze(0).contiguous().numpy().copy()
    arrays['text_features'] = result.transpose(1,2).squeeze(0).contiguous().numpy().copy()
    direct = model.text_encoder(token_ids, torch.tensor([len(ids)]),
                                torch.zeros((1,len(ids)),dtype=torch.bool))
    if not torch.equal(result.transpose(1,2),direct):
        raise RuntimeError('manual all-valid checkpoint sequence differs from model forward')
    with np.load(ROOT/'tests/fixtures/tts/kokoro_duration_predictor_pinned.npz') as duration:
        counts = duration['durations'].astype(np.int32)
    arrays['durations'] = counts
    arrays['text_expanded'] = torch.repeat_interleave(
        torch.from_numpy(arrays['text_features']),
        torch.from_numpy(counts.astype(np.int64)),dim=1).numpy().copy()
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,**arrays)
    meta = {'schema':'cke.kokoro.text_encoder_fixture.v1',
        'scope':'all-valid 36-token acoustic text encoder and duration expansion; no prosody or waveform',
        'model_pin':exporter.PIN['model'],
        'reference_code':exporter.PIN['reference_code'],
        'oracle_packages':packages,
        'oracle_source_sha256':{
            'kokoro.model':exporter.sha256_file(Path(inspect.getfile(KModel))),
            'kokoro.modules':exporter.sha256_file(Path(inspect.getfile(AdaLayerNorm)))},
        'package_source_commit_match':'NOT_TESTED',
        'asset_sha256':assets,
        'source_ids_fixture_sha256':hashlib.sha256(ids_fixture.read_bytes()).hexdigest(),
        'arrays_sha256':{name:digest(value) for name,value in arrays.items()},
        'array_shapes':{name:list(value.shape) for name,value in arrays.items()},
        'oracle':'Kokoro 0.9.4 PyTorch 2.8 effective weight-normalized Conv1D, LayerNorm, LeakyReLU(0.2), zero-state BiLSTM',
        'fixture_sha256':hashlib.sha256(output.read_bytes()).hexdigest()}
    output.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps({'fixture':str(output),'sha256':meta['fixture_sha256'],
                      'text_shape':list(arrays['text_features'].shape)}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    capture(args.model_dir,args.output)


if __name__=='__main__': main()
