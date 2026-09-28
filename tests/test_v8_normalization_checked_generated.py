"""Model-independent residual -> checked normalization -> activation graph."""
import ctypes
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests import test_v8_checked_op_constants as fixture_support
from tests.v8_checked_graph_test_support import compile_native_graph


class CheckedNormalizationGeneratedTest(unittest.TestCase):
    def compile(self, mutate=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        tensors = {'fixture.gamma': np.ones(5, 'f'), 'fixture.beta': np.zeros(5, 'f')}
        bundle = fixture_support.write_synthetic_weights(root, tensors)
        ops = [
            {'id': 'sum', 'op': 'audio_scaled_residual_add', 'kernel': 'audio_scaled_residual_add_f32',
             'returns_status': True, 'params': {'N': 10, 'elements': 10, 'scale': 1.0},
             'graph_slots': {'inputs': {'residual': 'external:left', 'branch': 'external:right'},
                             'outputs': {'output': 'sum_output'}}},
            {'id': 'normalize', 'op': 'layernorm', 'kernel': 'layernorm_rows_checked_f32',
             'returns_status': True, 'weight_refs': {'gamma': 'fixture.gamma', 'beta': 'fixture.beta'},
             'params': {'M': 2, 'C': 5, 'Q': 14, 'call_constants': {
                 'norm_input_elements': 10, 'norm_input_stride': 5,
                 'norm_gamma_elements': 5, 'norm_beta_elements': 5,
                 'norm_output_elements': 14, 'norm_output_stride': 7,
                 'norm_scratch_elements': 14, 'norm_rows': 2, 'norm_channels': 5}},
             'graph_slots': {'inputs': {'input': 'sum_output'}, 'outputs': {'output': 'normalized'}}},
            {'id': 'activate', 'op': 'gelu', 'kernel': 'gelu_rows_tanh_checked_f32',
             'returns_status': True,
             'params': {'M': 2, 'C': 5, 'Q': 10, 'call_constants': {
                 'gelu_input_elements': 14, 'gelu_input_stride': 7,
                 'gelu_output_elements': 16, 'gelu_output_stride': 8,
                 'gelu_scratch_elements': 10, 'gelu_rows': 2, 'gelu_channels': 5}},
             'graph_slots': {'inputs': {'input': 'normalized'}, 'outputs': {'output': 'activated'}}},
        ]
        template = {
            'version': 3, 'name': 'checked_normalization_fixture', 'family': 'bounded_graph',
            'checked_native_entry': True,
            'contract': {'runtime_invariants': {'inference_only': True, 'production_kernel_heap_allocation': False}},
            'activation_buffers': {name: {'shape': shape} for name, shape in (
                ('left', [2, 5]), ('right', [2, 5]), ('sum_output', [2, 5]),
                ('normalized', [2, 7]), ('activated', [2, 8]))},
            'activation_bindings': {name: name for name in ('left', 'right', 'sum_output', 'normalized', 'activated')},
            'native_entry': {'function': 'ck_checked_normalization_fixture',
                             'params': [{'c_type': 'uint8_t *', 'name': 'arena'},
                                        {'c_type': 'size_t', 'name': 'arena_bytes'}],
                             'arena': {'pointer': 'arena', 'bytes': 'arena_bytes'}},
            'sequence': ['component'], 'block_types': {'component': {
                'sequence': ['header', 'body', 'footer'], 'header': [ops[0]],
                'body': {'type': 'dense', 'ops': [ops[1]]}, 'footer': [ops[2]]}},
        }
        if mutate:
            mutate(ops)
        circuit = root / 'circuit.json'
        circuit.write_text(json.dumps(template))
        source = {'config': {'model': template['name'], 'arch': template['name'],
                            'num_layers': 1, 'embed_dim': 5, 'num_heads': 1, 'num_kv_heads': 1,
                            'head_dim': 5, 'intermediate_size': 5, 'context_length': 2,
                            'max_seq_len': 2, 'vocab_size': 1, 'epsilon': 1e-5},
                  'entries': bundle['entries'], 'quant_summary': {}, 'template': template}
        layout, call_ir, _library, _loaded, fn = compile_native_graph(root, source, circuit)
        self.assertEqual(call_ir['errors'], [])
        size = layout['memory']['arena']['total_size']
        raw = (ctypes.c_uint8 * (size + 64))()
        arena = (ctypes.c_uint8 * size).from_buffer(raw, (-ctypes.addressof(raw)) & 63)
        payload = (root / 'weights.fixture').read_bytes()
        planned = {e['name']: e for e in layout['memory']['weights']['entries']}
        for entry in bundle['entries']:
            start = planned[entry['name']]['abs_offset']
            arena[start:start + entry['size']] = payload[entry['file_offset']:entry['file_offset'] + entry['size']]
        buffers = {e['name']: e for e in layout['memory']['activations']['buffers']}
        def tensor(name, shape):
            return np.ndarray(shape, dtype='f', buffer=arena, offset=buffers[name]['abs_offset'])
        # Keep the loaded library alive for the ctypes function.
        return arena, tensor, fn, _loaded, planned

    def test_strided_composition_repeated_requests_and_rejection(self):
        arena, tensor, fn, _loaded, weights = self.compile()
        left = tensor('left', (2, 5)); right = tensor('right', (2, 5))
        normalized = tensor('normalized', (2, 7)); activated = tensor('activated', (2, 8))
        for base in (np.arange(10, dtype='f').reshape(2, 5), np.zeros((2, 5), 'f')):
            left[:] = base; right[:] = -0.25
            normalized.fill(-777.); activated.fill(-888.)
            self.assertEqual(fn(arena, len(arena)), 0)
            summed = (base - np.float32(.25)).astype('d')
            expected = ((summed - summed.mean(axis=1, keepdims=True)) /
                        np.sqrt(summed.var(axis=1, keepdims=True) + 1e-5)).astype('f')
            np.testing.assert_allclose(normalized[:, :5], expected, rtol=0, atol=1e-6)
            expected_gelu = .5 * expected * (1 + np.tanh(np.sqrt(2/np.pi) * (expected + .044715 * expected**3)))
            np.testing.assert_allclose(activated[:, :5], expected_gelu, rtol=0, atol=1e-6)
            self.assertTrue((normalized[:, 5:] == -777.).all())
            self.assertTrue((activated[:, 5:] == -888.).all())
        normalized.fill(-777.); activated.fill(-888.)
        gamma = np.ndarray((5,), dtype='f', buffer=arena, offset=weights['fixture.gamma']['abs_offset'])
        gamma[-1] = np.nan
        self.assertNotEqual(fn(arena, len(arena)), 0)
        self.assertTrue((normalized == -777.).all()); self.assertTrue((activated == -888.).all())
        self.assertEqual(fn(arena, len(arena)-1), -2)

    def test_generated_undersized_workspace_stops_consumer(self):
        def mutate(ops):
            ops[1]['params']['Q'] = 13
            ops[1]['params']['call_constants']['norm_scratch_elements'] = 13
        arena, tensor, fn, _loaded, _weights = self.compile(mutate)
        tensor('left', (2, 5))[:] = 1.
        tensor('right', (2, 5))[:] = 2.
        normalized = tensor('normalized', (2, 7)); normalized.fill(-777.)
        activated = tensor('activated', (2, 8)); activated.fill(-888.)
        self.assertEqual(fn(arena, len(arena)), -2)
        self.assertTrue((normalized == -777.).all()); self.assertTrue((activated == -888.).all())
        misaligned = ctypes.cast(ctypes.addressof(arena) + 1, ctypes.POINTER(ctypes.c_uint8))
        self.assertEqual(fn(misaligned, len(arena)), -2)
        self.assertTrue((normalized == -777.).all()); self.assertTrue((activated == -888.).all())

    def test_planner_rejects_false_capacity_and_shape_claims(self):
        for name, value in (('norm_output_elements', 100), ('norm_channels', 6),
                            ('norm_scratch_elements', 13), ('norm_output_stride', 4)):
            def mutate(ops, name=name, value=value):
                ops[1]['params']['call_constants'][name] = value
            with self.subTest(argument=name), self.assertRaises(RuntimeError):
                self.compile(mutate)


if __name__ == '__main__':
    unittest.main()
