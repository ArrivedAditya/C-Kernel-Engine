"""Pinned duration fixtures and BUMP setup shared by generated-graph tests.

This module prepares model assets for tests; generated C executes the graph.
"""
import ctypes
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from tests.v8_kokoro_fixture_support import load_first_layer_fixtures

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'version/v8/tts'))
import export_kokoro_bump as exporter


def prepare_duration_fixture(root, circuit):
    fixture = SimpleNamespace()
    encoder_path = ROOT / 'tests/fixtures/tts/kokoro_encoder_pinned.npz'
    duration_path = ROOT / 'tests/fixtures/tts/kokoro_duration_predictor_pinned.npz'
    fixture.encoder = dict(np.load(encoder_path))
    fixture.duration = dict(np.load(duration_path))
    fixture.encoder_meta = json.loads(encoder_path.with_suffix('.json').read_text())
    fixture.duration_meta = json.loads(duration_path.with_suffix('.json').read_text())
    for path, meta in ((encoder_path, fixture.encoder_meta),
                       (duration_path, fixture.duration_meta)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta['fixture_sha256']:
            raise RuntimeError(f'fixture SHA-256 mismatch: {path}')
    if (fixture.duration_meta['model_pin'] != fixture.encoder_meta['model_pin'] or
        fixture.duration_meta['asset_hashes'] != fixture.encoder_meta['asset_hashes']):
        raise RuntimeError('duration and encoder oracle pins disagree')
    tensors = load_first_layer_fixtures(fixture)
    for kind in ('weight', 'bias'):
        tensors[f'phoneme_projection.{kind}'] = fixture.encoder[
            f'weight__phoneme_projection__{kind}']
    origins = {name: {'source_name': name, 'transform': 'identity'}
               for name in tensors}
    for name, value in fixture.duration.items():
        if name.startswith('weight__'):
            canonical = name.removeprefix('weight__').replace('__', '.')
            tensors[canonical] = value
            origins[canonical] = fixture.duration_meta['weights'][name]
    fixture.bundle = exporter.write_bundle(root, tensors, origins,
        {'n_token': 178, 'hidden_dim': 512, 'plbert': {
            'intermediate_size': 2048, 'max_position_embeddings': 512,
            'num_attention_heads': 12}},
        {'source': 'pinned independent effective-weight fixture with explicit recurrent direction stacking'})
    exporter.verify_bundle(root)
    fixture.bump = (root / 'weights.bump').read_bytes()
    fixture.entries = {entry['name']: entry for entry in fixture.bundle['entries']}
    template = json.loads(Path(circuit).read_text())
    fixture.source = {'config': {
        'model': template['name'], 'arch': template['name'],
        'num_layers': 1, 'embed_dim': 128, 'num_heads': 1,
        'num_kv_heads': 1, 'head_dim': 128, 'intermediate_size': 256,
        'context_length': 36, 'max_seq_len': 36, 'vocab_size': 178,
        'T': 36, 'C': 128, 'epsilon': 1e-12, 'speed': 1.0,
        'activation_buffer_dtypes': {
            'word_ids': 'i32', 'type_ids': 'i32',
            'runtime_values': 'i32', 'runtime_valid_extent': 'i32'}},
        'entries': fixture.bundle['entries'], 'quant_summary': {},
        'template': template}
    return fixture


def populated_duration_arena(layout, entries, bump, encoder, duration):
    size = layout['memory']['arena']['total_size']
    raw = (ctypes.c_uint8 * (size + 63))()
    arena = (ctypes.c_uint8 * size).from_buffer(raw,
        (-ctypes.addressof(raw)) & 63)
    for item in layout['memory']['weights']['entries']:
        entry = entries[item['name']]
        start = entry['file_offset']; end = start + entry['size']
        data = bump[start:end]
        if hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise RuntimeError(f'BUMP payload mismatch: {item["name"]}')
        arena[item['abs_offset']:item['abs_offset']+len(data)] = data
    buffers = {item['name']: item for item in
               layout['memory']['activations']['buffers']}
    for name, item in buffers.items():
        dtype = (np.int32 if name in
            ('word_ids', 'type_ids', 'runtime_values', 'runtime_valid_extent')
            else np.float32)
        np.ndarray((item['size']//4,), dtype, buffer=arena,
                   offset=item['abs_offset'])[:] = -999 if dtype == np.int32 else -777.
    for name, value, dtype in (
            ('word_ids', encoder['word_ids'], np.int32),
            ('type_ids', np.zeros(36, np.int32), np.int32),
            ('predictor_style', duration['predictor_style'], np.float32)):
        np.ndarray((len(value),), dtype, buffer=arena,
                   offset=buffers[name]['abs_offset'])[:] = value
    return arena
