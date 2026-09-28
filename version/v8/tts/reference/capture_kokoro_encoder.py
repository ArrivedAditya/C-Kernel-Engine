#!/usr/bin/env python3
"""Pinned offline oracle for twelve shared ALBERT calls and phoneme projection."""
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


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokens', type=int, default=36)
    args = parser.parse_args()
    tokens = args.tokens
    if not 2 <= tokens <= 512:
        parser.error('tokens must be between 2 and 512')
    pin = json.loads((ROOT / 'version/v8/tts/reference/fixture_manifest.json').read_text())
    for name, expected in pin['assets'].items():
        if sha(args.model_dir / name) != expected:
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
    config = model.bert.config
    if (config.num_hidden_layers, config.num_hidden_groups, config.inner_group_num,
        config.hidden_act) != (12, 1, 1, 'gelu_new'):
        raise RuntimeError('unsupported pinned encoder configuration')
    layer = model.bert.encoder.albert_layer_groups[0].albert_layers[0]
    tensors = {}
    handles = []
    counts = {}
    modules = {'query_output': layer.attention.query, 'key_output': layer.attention.key,
               'value_output': layer.attention.value,
               'attention_projection_output': layer.attention.dense,
               'attention_norm_output': layer.attention.LayerNorm,
               'ffn_linear_output': layer.ffn, 'ffn_projection_output': layer.ffn_output,
               'albert_layer_output': layer.full_layer_layer_norm}
    inputs = {'attention_projection_output': 'attention_context_output',
              'attention_norm_output': 'attention_residual_output',
              'ffn_projection_output': 'ffn_gelu_output',
              'albert_layer_output': 'layer_residual_output'}
    for name, module in modules.items():
        def hook(_module, args, output, name=name):
            invocation = counts.get(name, 0)
            counts[name] = invocation + 1
            prefix = f'l{invocation:02d}_'
            tensors[prefix + name] = output.detach().cpu().numpy().copy().reshape(tokens, -1)
            if name in inputs:
                tensors[prefix + inputs[name]] = args[0].detach().cpu().numpy().copy().reshape(tokens, -1)
        handles.append(module.register_forward_hook(hook))
    for name, module in {'embedding_output': model.bert.embeddings,
                         'projection_output': model.bert.encoder.embedding_hidden_mapping_in,
                         'phoneme_features': model.bert_encoder}.items():
        def hook(_module, args, output, name=name):
            if name in tensors:
                raise RuntimeError('unexpected repeated prefix/projection')
            tensors[name] = output.detach().cpu().numpy().copy().reshape(tokens, -1)
        handles.append(module.register_forward_hook(hook))
    ids = np.load(ROOT / 'tests/fixtures/tts/bert_embedding_pinned.npz')['ids']
    if tokens != len(ids):
        ids = np.resize(ids[1:-1], tokens).astype(np.int32)
        ids[0] = ids[-1] = 0
    try:
        with torch.no_grad():
            encoded = model.bert(torch.from_numpy(ids).long().reshape(1, -1),
                                 attention_mask=torch.ones((1, len(ids)), dtype=torch.long))[0]
            model.bert_encoder(encoded)
    finally:
        for handle in handles:
            handle.remove()
    if set(counts.values()) != {12}:
        raise RuntimeError(f'shared layer invocation mismatch: {counts}')
    for name, value in model.bert_encoder.state_dict().items():
        tensors[f'weight__phoneme_projection__{name}'] = value.detach().cpu().numpy().copy()
    if len(tensors) != 149 or any(not np.isfinite(x).all() for x in tensors.values()):
        raise RuntimeError('missing or nonfinite encoder capture')
    tensors['word_ids'] = ids.astype(np.int32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **tensors)
    metadata = {
        'schema': 'cke.kokoro.phoneme_encoder.oracle.v1',
        'scope': f'{tokens} fixed phoneme IDs; all tokens valid; 12 shared layers and 768-to-512 projection',
        'tokens': tokens,
        'model_pin': pin['pin']['model'], 'asset_hashes': pin['assets'],
        'dependencies': {p: importlib.metadata.version(p) for p in ('kokoro', 'torch', 'transformers')},
        'torch_git_revision': torch.version.git_version,
        'torch_build_configuration': torch.__config__.show(),
        'torch_cpu_capability_reported': torch.backends.cpu.get_cpu_capability(),
        'dispatch_identity': 'NOT_VERIFIED', 'threads': 1,
        'attention_implementation': 'eager', 'activation': config.hidden_act,
        'epsilon': config.layer_norm_eps, 'shared_layer_invocations': counts,
        'fixture_sha256': sha(args.output), 'capture_source_sha256': sha(Path(__file__)),
        'shapes': {k: list(v.shape) for k, v in tensors.items()},
        'complete_native_encoder': 'NOT_TESTED', 'generated_waveform': 'NOT_TESTED',
    }
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'tensors': len(tensors), 'fixture_sha256': metadata['fixture_sha256']}))


if __name__ == '__main__':
    main()
