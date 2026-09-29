#!/usr/bin/env python3
"""Offline first shared ALBERT-layer capture from the pinned Kokoro model.

Python supplies import/oracle evidence only; it is not deployed model scheduling.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import numpy as np
import torch
from kokoro import KModel

ROOT = Path(__file__).resolve().parents[4]
PREFIX = 'encoder.albert_layer_groups.0.albert_layers.0'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--primitives-output', type=Path)
    parser.add_argument('--bump-dir', type=Path, help='optional pinned full-export identity')
    args = parser.parse_args()
    reference = json.loads((ROOT / 'version/v8/tts/reference/fixture_manifest.json').read_text())
    for name, expected in reference['assets'].items():
        if sha(args.model_dir / name) != expected:
            parser.error(f'pinned asset hash mismatch: {name}')
    for package in ('kokoro', 'torch', 'transformers'):
        if importlib.metadata.version(package) != reference['environment']['packages'][package]:
            parser.error(f'pinned dependency mismatch: {package}')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    model = KModel(repo_id=reference['pin']['model']['repository'],
                   config=str(args.model_dir / 'config.json'),
                   model=str(args.model_dir / 'kokoro-v1_0.pth')).eval()
    model.bert.config._attn_implementation = 'eager'
    layer = dict(model.bert.named_modules())[PREFIX]
    if model.bert.config.hidden_act != 'gelu_new':
        raise RuntimeError(f'unsupported pinned activation: {model.bert.config.hidden_act}')
    tensors = {}
    handles = []
    modules = {
        'embedding': model.bert.embeddings,
        'projection': model.bert.encoder.embedding_hidden_mapping_in,
        'query': layer.attention.query,
        'key': layer.attention.key,
        'value': layer.attention.value,
        'attention_projection': layer.attention.dense,
        'attention_norm': layer.attention.LayerNorm,
        'ffn_linear': layer.ffn,
        'ffn_projection': layer.ffn_output,
        'layer_output': layer.full_layer_layer_norm,
    }
    for name, module in modules.items():
        def hook(_module, inputs, output, name=name):
            # ALBERT reuses this module twelve times. Certify only the first invocation.
            if f'{name}_expected' not in tensors:
                if inputs:
                    tensors[f'{name}_input'] = inputs[0].detach().cpu().numpy().copy().reshape(36, -1)
                tensors[f'{name}_expected'] = output.detach().cpu().numpy().copy().reshape(36, -1)
        handles.append(module.register_forward_hook(hook))
    ids = np.load(ROOT / 'tests/fixtures/tts/bert_embedding_pinned.npz')['ids']
    try:
        with torch.no_grad():
            model.bert(torch.from_numpy(ids).long().reshape(1, -1),
                       attention_mask=torch.ones((1, len(ids)), dtype=torch.long))
    finally:
        for handle in handles:
            handle.remove()
    for name, tensor in layer.state_dict().items():
        if name.startswith(('attention.query.', 'attention.key.', 'attention.value.')):
            continue  # Already certified and stored in the pinned Q/K/V fixture.
        tensors['weight__' + name.replace('.', '__')] = tensor.detach().cpu().numpy().copy()
    old = np.load(ROOT / 'tests/fixtures/tts/kokoro_attention_context_pinned.npz')['context_expected']
    if not np.allclose(tensors['attention_projection_input'], old, rtol=0, atol=1e-7):
        raise RuntimeError('first attention context differs from pinned earlier capture')
    if any(not np.isfinite(t).all() for t in tensors.values()):
        raise RuntimeError('nonfinite oracle capture')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **tensors)
    bundle_hash = None
    if args.bump_dir:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import export_kokoro_bump as exporter
        manifest = exporter.verify_bundle(args.bump_dir)
        if manifest['pin']['model'] != reference['pin']['model'] or manifest['provenance']['source_asset_sha256'] != reference['assets']:
            raise RuntimeError('full BUMP export pin/assets do not match reference')
        bundle_hash = sha(args.bump_dir / 'weights_manifest.json')
    meta = {
        'schema': 'cke.kokoro.first_albert_layer.oracle.v1',
        'scope': 'first complete shared ALBERT layer; all 36 tokens valid; eval/dropout disabled',
        'model_pin': reference['pin']['model'], 'asset_hashes': reference['assets'],
        'dependencies': {p: importlib.metadata.version(p) for p in ('kokoro', 'torch', 'transformers')},
        'torch_git_revision': torch.version.git_version, 'torch_build_configuration': torch.__config__.show(),
        'torch_cpu_capability_reported': torch.backends.cpu.get_cpu_capability(),
        'dispatch_identity': 'NOT_VERIFIED', 'threads': 1, 'attention_implementation': 'eager',
        'activation': model.bert.config.hidden_act, 'epsilon': model.bert.config.layer_norm_eps,
        'bump_bundle_manifest_sha256': bundle_hash,
        'fixture_sha256': sha(args.output), 'capture_source_sha256': sha(Path(__file__)),
        'shapes': {name: list(t.shape) for name, t in tensors.items()},
        'complete_native_layer': 'NOT_TESTED', 'generated_waveform': 'NOT_TESTED',
    }
    args.output.with_suffix('.json').write_text(json.dumps(meta, indent=2) + '\n')
    if args.primitives_output:
        keys = ['attention_norm_input', 'attention_norm_expected',
                'layer_output_input', 'layer_output_expected',
                'weight__attention__LayerNorm__weight', 'weight__attention__LayerNorm__bias',
                'weight__full_layer_layer_norm__weight', 'weight__full_layer_layer_norm__bias',
                'ffn_linear_expected', 'ffn_projection_input']
        args.primitives_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.primitives_output, **{key: tensors[key] for key in keys})
        primitive_meta = dict(meta, scope='LayerNorm/tanh GELU operation captures with oracle-supplied inputs',
                              fixture_sha256=sha(args.primitives_output),
                              shapes={key: list(tensors[key].shape) for key in keys})
        args.primitives_output.with_suffix('.json').write_text(json.dumps(primitive_meta, indent=2) + '\n')
    print(json.dumps({'tensors': len(tensors), 'activation': meta['activation'],
                      'epsilon': meta['epsilon'], 'fixture_sha256': meta['fixture_sha256']}))


if __name__ == '__main__':
    main()
