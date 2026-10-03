"""Model-neutral dilated Conv1D -> Snake through normal v8 codegen."""

import ctypes
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.v8_checked_graph_test_support import compile_native_graph


class DilatedConvGeneratedTest(unittest.TestCase):
    def compile(self, wrong_weight_capacity=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        weight = np.arange(12, dtype=np.float32).reshape(2, 2, 3) / 20
        bias = np.array([.25, -.125], np.float32)
        alpha = np.array([.75, 1.25], np.float32)
        entries = []
        offset = 0
        for name, array in (('stage.weight', weight), ('stage.bias', bias),
                            ('stage.alpha', alpha)):
            data = array.tobytes()
            entries.append({'name': name, 'dtype': 'fp32',
                'shape': list(array.shape), 'file_offset': offset,
                'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
            offset += len(data)
        conv = {
            'id': 'conv', 'op': 'audio_conv1d_dilated_checked',
            'kernel': 'audio_conv1d_dilated_checked_channel_major_f32',
            'returns_status': True,
            'weight_refs': {'weight': 'stage.weight', 'bias': 'stage.bias'},
            'params': {'I': 2, 'O': 2, 'T': 5, 'K': 3, 'U': 5, 'Q': 10,
                'call_constants': {
                    'conv_input_elements': 14, 'conv_input_stride': 7,
                    'conv_weight_elements': 13 if wrong_weight_capacity else 12,
                    'conv_bias_elements': 2, 'conv_output_elements': 14,
                    'conv_output_stride': 7, 'conv_scratch_elements': 10,
                    'conv_input_channels': 2, 'conv_output_channels': 2,
                    'conv_input_frames': 5, 'conv_kernel_size': 3,
                    'conv_stride': 1, 'conv_padding': 2,
                    'conv_dilation': 2, 'conv_output_frames': 5}},
            'graph_slots': {'inputs': {'input': 'external:signal'},
                            'outputs': {'output': 'convolved'}}}
        snake = {
            'id': 'snake', 'op': 'audio_snake_strided_checked',
            'kernel': 'audio_snake_strided_f32_checked',
            'returns_status': True,
            'weight_refs': {'alpha': 'stage.alpha'},
            'params': {'C': 2, 'T': 5, 'call_constants': {
                'snake_input_elements': 14, 'snake_input_stride': 7,
                'snake_alpha_elements': 2, 'snake_output_elements': 14,
                'snake_output_stride': 7, 'snake_channels': 2,
                'snake_frames': 5}},
            'graph_slots': {'inputs': {'input': 'convolved'},
                            'outputs': {'output': 'activated'}}}
        names = ('signal', 'convolved', 'activated')
        circuit = {
            'version': 3, 'name': 'synthetic_dilated_conv_snake',
            'family': 'bounded_graph', 'checked_native_entry': True,
            'contract': {'runtime_invariants': {'inference_only': True}},
            'activation_buffers': {name: {'shape': [2, 7]} for name in names},
            'activation_bindings': {name: name for name in names},
            'native_entry': {'function': 'ck_synthetic_dilated_conv_snake',
                'params': [{'c_type': 'uint8_t *', 'name': 'arena'},
                           {'c_type': 'size_t', 'name': 'arena_bytes'}],
                'arena': {'pointer': 'arena', 'bytes': 'arena_bytes'}},
            'sequence': ['component'], 'block_types': {'component': {
                'sequence': ['header', 'body', 'footer'],
                'header': [conv], 'body': {'type': 'dense', 'ops': [snake]},
                'footer': []}}}
        path = root / 'circuit.json'
        path.write_text(json.dumps(circuit))
        source = {'config': {'model': circuit['name'], 'arch': circuit['name'],
            'num_layers': 1, 'embed_dim': 2, 'num_heads': 1,
            'num_kv_heads': 1, 'head_dim': 2, 'intermediate_size': 2,
            'context_length': 5, 'max_seq_len': 5, 'vocab_size': 1},
            'entries': entries, 'quant_summary': {}, 'template': circuit}
        return compile_native_graph(root, source, path), weight, bias, alpha

    def test_generated_composition_and_failure_propagation(self):
        (layout, calls, _library, _loaded, function), weight, bias, alpha = self.compile()
        self.assertEqual(calls['errors'], [])
        self.assertEqual([item['function'] for item in calls['operations']],
            ['audio_conv1d_dilated_checked_channel_major_f32',
             'audio_snake_strided_f32_checked'])
        size = layout['memory']['arena']['total_size']
        raw = (ctypes.c_uint8 * (size + 63))()
        arena = (ctypes.c_uint8 * size).from_buffer(
            raw, (-ctypes.addressof(raw)) & 63)
        weights = {item['name']: item for item in
                   layout['memory']['weights']['entries']}
        for name, array in (('stage.weight', weight), ('stage.bias', bias),
                            ('stage.alpha', alpha)):
            payload = array.tobytes()
            location = weights[name]['abs_offset']
            arena[location:location + len(payload)] = payload
        buffers = {item['name']: item for item in
                   layout['memory']['activations']['buffers']}
        def view(name):
            return np.ndarray((2, 7), np.float32, buffer=arena,
                              offset=buffers[name]['abs_offset'])
        signal, convolved, activated = map(view,
                                            ('signal', 'convolved', 'activated'))
        for values in ([[1, 2, 3, 4, 5], [0, -1, 2, -3, 4]],
                       [[-.5, 0, .5, 1, -1], [2, 1, 0, -1, -2]]):
            signal[:] = np.nan
            signal[:, :5] = values
            convolved[:] = -77.
            activated[:] = -91.
            self.assertEqual(function(arena, len(arena)), 0)
            expected = np.empty((2, 5), np.float32)
            for oc in range(2):
                for frame in range(5):
                    expected[oc, frame] = bias[oc] + sum(
                        float(signal[ic, frame + 2 * tap - 2]) *
                        float(weight[oc, ic, tap])
                        for ic in range(2) for tap in range(3)
                        if 0 <= frame + 2 * tap - 2 < 5)
            a = alpha[:, None]
            expected = expected + np.sin(a * expected) ** 2 / a
            np.testing.assert_allclose(activated[:, :5], expected,
                                       rtol=0, atol=1e-6)
            self.assertTrue(np.all(convolved[:, 5:] == -77.))
            self.assertTrue(np.all(activated[:, 5:] == -91.))
        signal[1, 4] = np.nan
        convolved[:] = -77.
        activated[:] = -91.
        self.assertNotEqual(function(arena, len(arena)), 0)
        self.assertTrue(np.all(convolved == -77.))
        self.assertTrue(np.all(activated == -91.))

    def test_wrong_weight_geometry_rejected_before_codegen(self):
        with self.assertRaisesRegex(RuntimeError, 'HARD CALL CONSTANT FAULT'):
            self.compile(wrong_weight_capacity=True)


if __name__ == '__main__':
    unittest.main()
