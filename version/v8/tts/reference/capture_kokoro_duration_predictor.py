#!/usr/bin/env python3
"""Offline pinned oracle and canonical effective weights for Kokoro duration.

The deployed native graph never imports this module or runs PyTorch.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import numpy as np
import torch
from kokoro import KModel

ROOT = Path(__file__).resolve().parents[4]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path)
    parser.add_argument('--bump-dir', type=Path)
    args = parser.parse_args()
    pin = json.loads((ROOT / 'version/v8/tts/reference/fixture_manifest.json').read_text())
    for name, expected in pin['assets'].items():
        if digest(args.model_dir / name) != expected:
            parser.error(f'pinned asset mismatch: {name}')
    for package in ('kokoro', 'torch', 'transformers'):
        if importlib.metadata.version(package) != pin['environment']['packages'][package]:
            parser.error(f'pinned dependency mismatch: {package}')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    model = KModel(repo_id=pin['pin']['model']['repository'],
        config=str(args.model_dir / 'config.json'),
        model=str(args.model_dir / 'kokoro-v1_0.pth')).eval()
    model.bert.config._attn_implementation = 'eager'
    ids = np.load(ROOT / 'tests/fixtures/tts/bert_embedding_pinned.npz')['ids']
    if len(ids) != len(pin['input_ids']) or not np.array_equal(ids, pin['input_ids']):
        raise RuntimeError('phoneme IDs disagree with pinned reference')
    voice_pack = torch.load(args.model_dir / 'voices/af_heart.pt', map_location='cpu', weights_only=True)
    row = int(pin['voice_row_index'])
    voice = voice_pack[row].detach().float().reshape(1, -1)
    if voice.shape != (1, 256):
        raise RuntimeError('selected af_heart conditioning row has wrong shape')
    style = voice[:, 128:]
    tensors = {'word_ids': ids.astype(np.int32), 'predictor_style': style.numpy().reshape(-1).copy()}
    origins = {}
    handles = []

    def capture(name, module, packed=False):
        def hook(_module, input_args, result):
            source = input_args[0]
            value = result[0] if isinstance(result, tuple) else result
            if packed:
                source, value = source.data, value.data
            tensors[name + '_input'] = source.detach().numpy().reshape(len(ids), -1).copy()
            tensors[name + '_output'] = value.detach().numpy().reshape(len(ids), -1).copy()
        handles.append(module.register_forward_hook(hook))

    predictor = model.predictor
    for index, slot in enumerate((0, 2, 4)):
        capture(f'scan{index}', predictor.text_encoder.lstms[slot], packed=True)
        capture(f'norm{index}', predictor.text_encoder.lstms[slot+1])
    capture('head_scan', predictor.lstm)
    capture('duration_logits', predictor.duration_proj)
    try:
        with torch.no_grad():
            token_ids = torch.from_numpy(ids.astype(np.int64)).reshape(1, -1)
            encoded = model.bert(token_ids, attention_mask=torch.ones_like(token_ids))
            features = model.bert_encoder(encoded)
            lengths = torch.tensor([len(ids)])
            mask = torch.zeros((1, len(ids)), dtype=torch.bool)
            d = predictor.text_encoder(features.transpose(-1, -2), style, lengths, mask)
            x, _ = predictor.lstm(d)
            logits = predictor.duration_proj(x)
            durations = torch.round(torch.sigmoid(logits).sum(-1)).clamp(min=1).long()
    finally:
        for handle in handles:
            handle.remove()
    tensors['encoder_features'] = features.detach().numpy().reshape(len(ids), 512).copy()
    tensors['predictor_output'] = d.detach().numpy().reshape(len(ids), 640).copy()
    tensors['duration_logits'] = logits.detach().numpy().reshape(len(ids), 50).copy()
    tensors['durations'] = durations.numpy().astype(np.int32).reshape(-1)
    tensors['expanded_frames'] = np.asarray([int(durations.sum())], np.int32)
    for index, slot in enumerate((0, 2, 4, None)):
        module = predictor.lstm if slot is None else predictor.text_encoder.lstms[slot]
        prefix = 'duration_prosody.head_scan' if slot is None else f'duration_prosody.text_encoder.scan{index}'
        source = 'duration_prosody.lstm' if slot is None else f'duration_prosody.text_encoder.lstms.{slot}'
        for kind in ('weight_ih', 'weight_hh', 'bias_ih', 'bias_hh'):
            names = [f'{source}.{kind}_l0', f'{source}.{kind}_l0_reverse']
            value = np.stack([getattr(module, f'{kind}_l0{suffix}').detach().numpy()
                              for suffix in ('', '_reverse')], axis=0).astype(np.float32)
            name = f'weight__{prefix}__{kind}'
            tensors[name] = value
            origins[name] = {'source_name': ','.join(names), 'transform': 'stack_direction_axis_0'}
    for index, slot in enumerate((1, 3, 5)):
        module = predictor.text_encoder.lstms[slot].fc
        prefix = f'duration_prosody.text_encoder.norm{index}'
        for kind in ('weight', 'bias'):
            name = f'weight__{prefix}__projection_{kind}'
            tensors[name] = getattr(module, kind).detach().numpy().copy()
            origins[name] = {'source_name': f'duration_prosody.text_encoder.lstms.{slot}.fc.{kind}',
                             'transform': 'identity'}
    for kind in ('weight', 'bias'):
        name = f'weight__duration_prosody.duration_head__{kind}'
        tensors[name] = getattr(predictor.duration_proj.linear_layer, kind).detach().numpy().copy()
        origins[name] = {'source_name': f'duration_prosody.duration_proj.linear_layer.{kind}',
                         'transform': 'identity'}
    if args.reference_dir:
        for name, expected in (
                ('durations', 'predicted_duration.npy'),
                ('predictor_style', 'predictor_style.npy')):
            reference = np.load(args.reference_dir / expected).reshape(-1)
            if not np.array_equal(tensors[name].reshape(-1), reference):
                raise RuntimeError(f'pinned independent {name} capture disagrees')
    if args.bump_dir:
        manifest = json.loads((args.bump_dir / 'weights_manifest.json').read_text())
        if manifest['pin']['model'] != pin['pin']['model']:
            raise RuntimeError('full BUMP model pin disagrees')
        entries = {item['name']: item for item in manifest['entries']}
        with (args.bump_dir / 'weights.bump').open('rb') as stream:
            for name, origin in origins.items():
                source_names = origin['source_name'].split(',')
                values = tensors[name]
                for index, source_name in enumerate(source_names):
                    entry = entries[source_name]
                    stream.seek(entry['file_offset'])
                    raw = stream.read(entry['size'])
                    if hashlib.sha256(raw).hexdigest() != entry['sha256']:
                        raise RuntimeError(f'full BUMP source hash mismatch: {source_name}')
                    selected = values if len(source_names) == 1 else values[index]
                    expected = np.frombuffer(raw, np.float32).reshape(entry['shape'])
                    if not np.array_equal(selected, expected):
                        raise RuntimeError(f'effective imported tensor mismatch: {source_name}')
            voice_entry = entries['voice.fixed.predictor']
            stream.seek(voice_entry['file_offset'])
            raw = stream.read(voice_entry['size'])
            if (hashlib.sha256(raw).hexdigest() != voice_entry['sha256'] or
                    not np.array_equal(tensors['predictor_style'], np.frombuffer(raw, np.float32))):
                raise RuntimeError('selected BUMP voice style disagrees')
    if tensors['expanded_frames'][0] <= 0 or any(
            not np.isfinite(v).all() for k,v in tensors.items() if v.dtype.kind == 'f'):
        raise RuntimeError('invalid oracle duration capture')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **tensors)
    metadata = {
        'schema': 'cke.kokoro.duration_predictor.oracle.v1',
        'scope': 'pinned all-valid 36-token phoneme encoder through duration logits and checked frame count',
        'model_pin': pin['pin']['model'], 'asset_hashes': pin['assets'],
        'voice': 'af_heart', 'voice_row_index': row,
        'conditioning_selection': pin['pin']['fixture']['condition_selection'],
        'dependencies': {p: importlib.metadata.version(p) for p in ('kokoro','torch','transformers')},
        'torch_build_configuration': torch.__config__.show(),
        'torch_cpu_capability_reported': torch.backends.cpu.get_cpu_capability(),
        'dispatch_identity': 'NOT_VERIFIED', 'threads': 1,
        'weights': origins, 'shapes': {k: list(v.shape) for k,v in tensors.items()},
        'fixture_sha256': digest(args.output), 'capture_source_sha256': digest(Path(__file__)),
        'generated_waveform': 'NOT_TESTED', 'complete_native_duration': 'NOT_TESTED'}
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'tensors': len(tensors), 'duration_frames': int(tensors['expanded_frames'][0]),
                      'fixture_sha256': metadata['fixture_sha256']}))


if __name__ == '__main__':
    main()
