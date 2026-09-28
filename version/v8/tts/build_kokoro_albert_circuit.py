#!/usr/bin/env python3
"""Canonical ALBERT declarations shared by encoder authoring and tests.

This module authors shapes and edges; the normal compiler executes the graph.
"""
import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFINITION = Path(__file__).with_name('kokoro_albert_block.json')
FIXTURE = ROOT / 'tests/fixtures/tts/kokoro_first_albert_layer_generated_circuit.json'


def specialize(graph, tokens):
    """Declare capacities from dimensions, never from a fixture's token count."""
    if isinstance(tokens, bool) or not isinstance(tokens, int) or not 2 <= tokens <= 512:
        raise ValueError('encoder tokens must be an integer between 2 and 512')
    for buffer in graph['activation_buffers'].values():
        buffer['shape'][0] = tokens
    runtime = graph['runtime_constants']
    runtime.update(id_elements=tokens, tokens=tokens,
                   output_elements=tokens * runtime['output_stride'])
    block = graph['block_types']['phoneme_encoder']
    for op in block['header'] + block['body']['ops'] + block['footer']:
        p = op.get('params', {})
        c = p.get('call_constants', {})
        if op['op'] == 'linear_rows_checked':
            p['M'] = c['linear_rows'] = tokens
            c['linear_input_elements'] = tokens * c['linear_input_stride']
            c['linear_output_elements'] = tokens * c['linear_output_stride']
        elif op['op'] == 'attention_full_token_major_checked':
            p['T'] = c['attention_tokens'] = tokens
            elements = tokens * p['H'] * p['D']
            for port in ('query', 'key', 'value', 'output'):
                c[f'attention_{port}_elements'] = elements
        elif op['op'] in ('layernorm', 'gelu'):
            prefix = 'norm' if op['op'] == 'layernorm' else 'gelu'
            p['M'] = c[f'{prefix}_rows'] = tokens
            c[f'{prefix}_input_elements'] = tokens * c[f'{prefix}_input_stride']
            c[f'{prefix}_output_elements'] = tokens * c[f'{prefix}_output_stride']
            # Checked LayerNorm stages outputs plus mean/rstd for every row.
            p['Q'] = c[f'{prefix}_scratch_elements'] = tokens * (p['C'] + (2 if prefix == 'norm' else 0))
        elif op['op'] == 'audio_scaled_residual_add':
            tensor = op['graph_slots']['inputs']['residual']
            width = graph['activation_buffers'][tensor]['shape'][1]
            p['N'] = p['elements'] = tokens * width
        elif op['op'] != 'embedding_three_table_layer_norm':
            raise ValueError(f'undeclared shape specialization for {op["op"]}')
    return graph


def build_circuit(tokens=36):
    return specialize(copy.deepcopy(json.loads(DEFINITION.read_text())), tokens)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=FIXTURE)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    contents = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != contents:
            raise SystemExit('first-layer circuit differs from canonical authoring source')
    else:
        args.output.write_text(contents)


if __name__ == '__main__':
    main()
