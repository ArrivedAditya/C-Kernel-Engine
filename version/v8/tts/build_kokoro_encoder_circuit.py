#!/usr/bin/env python3
"""Author the fixed-input Kokoro encoder circuit; never schedule model execution.

The bounded fixture expands twelve shared ALBERT invocations into declared
edges. Distinct capture buffers preserve every invocation for X-Ray. The
generic compiler consumes the resulting circuit without model-name branches.
"""
import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'tests/fixtures/tts/kokoro_first_albert_layer_generated_circuit.json'
OUTPUT = ROOT / 'version/v8/circuits/kokoro_phoneme_encoder_bounded.json'


def build_circuit(tokens=36):
    if isinstance(tokens, bool) or not isinstance(tokens, int) or not 2 <= tokens <= 512:
        raise ValueError('encoder tokens must be an integer between 2 and 512')
    graph = json.loads(SOURCE.read_text())
    graph['name'] = 'kokoro_phoneme_encoder_bounded'
    graph['native_entry']['function'] = 'ck_kokoro_phoneme_encoder'
    graph['contract']['runtime_invariants'].update(
        supported_tokens=tokens, all_tokens_valid=True, shared_layer_invocations=12,
        complete_waveform=False)
    block = graph['block_types']['phoneme_encoder']
    projection = block['body']['ops'][0]
    layer_ops = block['body']['ops'][1:] + block['footer']
    exports = graph['semantic_checkpoints']['exports']
    original_buffers = graph['activation_buffers']
    graph['activation_buffers'] = {name: copy.deepcopy(original_buffers[name])
                                   for name in ('word_ids', 'type_ids', 'embedding_output', 'projection_output')}
    header = block['header'] + [projection]
    header_exports = {op['id']: copy.deepcopy(exports[op['id']]) for op in header}
    for item in header_exports.values():
        item['section'] = 'header'
    new_exports = header_exports
    body = []
    layer_buffers = sorted(set(original_buffers) - set(graph['activation_buffers']))
    for invocation in range(12):
        names = {name: f'l{invocation:02d}_{name}' for name in layer_buffers}
        names['projection_output'] = ('projection_output' if invocation == 0
                                      else f'l{invocation-1:02d}_albert_layer_output')
        for name in layer_buffers:
            graph['activation_buffers'][names[name]] = copy.deepcopy(original_buffers[name])
        for original in layer_ops:
            op = copy.deepcopy(original)
            op['id'] = f'l{invocation:02d}_{original["id"]}'
            for edges in op['graph_slots'].values():
                for port, tensor in edges.items():
                    edges[port] = names[tensor]
            body.append(op)
            export = copy.deepcopy(exports[original['id']])
            export.update(section='body', template_op_id=op['id'])
            for point in export['checkpoints']:
                point['producer'] = op['id']
                point['tensor'] = names[point['tensor']]
                point['id'] = point['id'].replace('.albert0.', f'.albert{invocation}.')
            new_exports[op['id']] = export
    final = copy.deepcopy(projection)
    final.update(id='phoneme_projection')
    final['weight_refs'] = {kind: f'phoneme_projection.{kind}' for kind in ('weight', 'bias')}
    final['graph_slots'] = {'inputs': {'input': 'l11_albert_layer_output'},
                            'outputs': {'output': 'phoneme_features'}}
    final['params'] = {'M': 36, 'K': 768, 'N': 512, 'call_constants': {
        'linear_input_elements': 36*768, 'linear_input_stride': 768,
        'linear_weight_elements': 512*768, 'linear_weight_stride': 768,
        'linear_bias_elements': 512, 'linear_output_elements': 36*512,
        'linear_output_stride': 512, 'linear_rows': 36,
        'linear_input_channels': 768, 'linear_output_channels': 512}}
    graph['activation_buffers']['phoneme_features'] = {'shape': [36, 512]}
    graph['activation_bindings'] = {name: name for name in graph['activation_buffers']}
    block.update(header=header, body={'type': 'dense', 'ops': body}, footer=[final])
    export = copy.deepcopy(exports[projection['id']])
    export.update(section='footer', template_op_id=final['id'])
    export['checkpoints'][0].update(id='kokoro.phoneme_encoder.features',
                                   producer=final['id'], tensor='phoneme_features')
    new_exports[final['id']] = export
    graph['semantic_checkpoints']['exports'] = new_exports
    for request in graph['required_numerical_contracts'].values():
        request['phases']['prefill']['evidence'] = 'tests/test_v8_kokoro_generated_encoder.py'
    # Storage capacities and per-call dimensions are authored together. This
    # specializes a graph; it does not claim runtime-changing valid lengths.
    for buffer in graph['activation_buffers'].values():
        buffer['shape'][0] = tokens
    for key in ('id_elements', 'tokens'):
        graph['runtime_constants'][key] = tokens
    graph['runtime_constants']['output_elements'] = tokens * 128
    for op in header + body + [final]:
        params = op.get('params', {})
        if op['op'] == 'audio_scaled_residual_add':
            params['N'] = params['elements'] = tokens * 768
        for key in ('M', 'T', 'Q'):
            if key in params:
                params[key] = params[key] // 36 * tokens
        for name, value in params.get('call_constants', {}).items():
            if name in ('linear_rows', 'norm_rows', 'gelu_rows', 'attention_tokens'):
                params['call_constants'][name] = tokens
            elif name.endswith(('_input_elements', '_output_elements', '_scratch_elements',
                               '_query_elements', '_key_elements', '_value_elements')):
                params['call_constants'][name] = value // 36 * tokens
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--tokens', type=int, default=36)
    args = parser.parse_args()
    # This artifact is the declared schedule consumed by normal compilation.
    contents = json.dumps(build_circuit(args.tokens), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != contents:
            raise SystemExit('encoder circuit differs from authoring source')
    else:
        args.output.write_text(contents)


if __name__ == '__main__':
    main()
