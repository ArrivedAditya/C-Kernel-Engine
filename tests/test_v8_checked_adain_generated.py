"""Model-neutral AdaIN graph through normal v8 lowering and generated C."""
import ctypes
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.v8_checked_graph_test_support import compile_native_graph


class CheckedAdaINGeneratedTest(unittest.TestCase):
    def test_style_conditioning_and_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weight = np.array([1.25, .75], np.float32)
            bias = np.array([.1, -.2], np.float32)
            entries = []
            payload = weight.tobytes() + bias.tobytes()
            for name, offset, data in (
                    ('fixture.norm_weight', 0, weight),
                    ('fixture.norm_bias', weight.nbytes, bias)):
                entries.append({'name': name, 'dtype': 'fp32',
                    'shape': [2], 'file_offset': offset, 'size': data.nbytes,
                    'sha256': hashlib.sha256(data.tobytes()).hexdigest()})
            operation = {'id': 'normalize', 'op': 'audio_adain_instance_norm',
                'kernel': 'audio_adain_instance_norm_f32',
                'returns_status': True,
                'weight_refs': {'norm_weight': 'fixture.norm_weight',
                                'norm_bias': 'fixture.norm_bias'},
                'params': {'C': 2, 'T': 3, 'normalization_epsilon': 1e-5,
                    'call_constants': {
                        'adain_input_elements': 10, 'adain_input_stride': 5,
                        'adain_norm_weight_elements': 2,
                        'adain_norm_bias_elements': 2,
                        'adain_style_affine_elements': 4,
                        'adain_output_elements': 10, 'adain_output_stride': 5,
                        'adain_channels': 2, 'adain_frames': 3}},
                'graph_slots': {
                    'inputs': {'input': 'external:features',
                               'style_affine': 'external:style_affine'},
                    'outputs': {'output': 'normalized'}}}
            circuit = {'version': 3, 'name': 'synthetic_checked_adain',
                'family': 'bounded_graph', 'checked_native_entry': True,
                'contract': {'runtime_invariants': {'inference_only': True,
                    'production_kernel_heap_allocation': False}},
                'activation_buffers': {
                    'features': {'shape': [2, 5]},
                    'style_affine': {'shape': [4]},
                    'normalized': {'shape': [2, 5]}},
                'activation_bindings': {name: name for name in
                    ('features', 'style_affine', 'normalized')},
                'native_entry': {'function': 'ck_synthetic_checked_adain',
                    'params': [{'c_type': 'uint8_t *', 'name': 'arena'},
                               {'c_type': 'size_t', 'name': 'arena_bytes'}],
                    'arena': {'pointer': 'arena', 'bytes': 'arena_bytes'}},
                'sequence': ['component'], 'block_types': {'component': {
                    'sequence': ['header', 'body', 'footer'], 'header': [],
                    'body': {'type': 'dense', 'ops': [operation]}, 'footer': []}}}
            path = root / 'circuit.json'
            path.write_text(json.dumps(circuit))
            source = {'config': {'model': circuit['name'],
                'arch': circuit['name'], 'num_layers': 1, 'embed_dim': 2,
                'num_heads': 1, 'num_kv_heads': 1, 'head_dim': 2,
                'intermediate_size': 2, 'context_length': 3,
                'max_seq_len': 3, 'vocab_size': 1},
                'entries': entries, 'quant_summary': {}, 'template': circuit}
            layout, calls, _library, _loaded, function = compile_native_graph(
                root, source, path)
            self.assertEqual(calls['errors'], [])
            self.assertEqual([item['function'] for item in calls['operations']],
                             ['audio_adain_instance_norm_f32'])
            size = layout['memory']['arena']['total_size']
            raw = (ctypes.c_uint8 * (size + 63))()
            arena = (ctypes.c_uint8 * size).from_buffer(
                raw, (-ctypes.addressof(raw)) & 63)
            weights = {item['name']: item for item in
                       layout['memory']['weights']['entries']}
            for name, offset, data in (
                    ('fixture.norm_weight', 0, weight),
                    ('fixture.norm_bias', weight.nbytes, bias)):
                item = weights[name]
                arena[item['abs_offset']:item['abs_offset'] + data.nbytes] = \
                    payload[offset:offset + data.nbytes]
            buffers = {item['name']: item for item in
                       layout['memory']['activations']['buffers']}
            def view(name, shape):
                return np.ndarray(shape, np.float32, buffer=arena,
                                  offset=buffers[name]['abs_offset'])
            features = view('features', (2, 5))
            style = view('style_affine', (4,))
            result = view('normalized', (2, 5))
            features[:] = -99.
            features[:, :3] = [[1., 2., 4.], [-2., 0., 2.]]
            style[:] = [.2, -.1, .3, -.4]
            result[:] = -77.
            self.assertEqual(function(arena, len(arena)), 0)
            mean = features[:, :3].mean(axis=1, keepdims=True, dtype=np.float64)
            variance = ((features[:, :3] - mean) ** 2).mean(
                axis=1, keepdims=True, dtype=np.float64)
            expected = (features[:, :3] - mean) / np.sqrt(variance + 1e-5)
            expected = ((expected.astype(np.float32) * weight[:, None] +
                         bias[:, None]) * (1 + style[:2, None]) +
                        style[2:, None])
            np.testing.assert_allclose(result[:, :3], expected,
                                       rtol=0, atol=1e-6)
            self.assertTrue(np.all(result[:, 3:] == -77.))
            result[:] = -77.
            style[0] = np.nan
            self.assertNotEqual(function(arena, len(arena)), 0)
            self.assertTrue(np.all(result == -77.))


if __name__ == '__main__':
    unittest.main()
