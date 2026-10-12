"""Explicit pool binding through normal checked circuit lowering and codegen."""
import ctypes
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.v8_checked_graph_test_support import compile_native_graph


class ParallelConvGeneratedTest(unittest.TestCase):
    def test_circuit_selected_pool_and_failure_propagation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weight = np.arange(8 * 3 * 3, dtype=np.float32).reshape(8, 3, 3) / 300
            bias = np.arange(8, dtype=np.float32) / 20
            entries = []
            file_offset = 0
            for name, array in [('stage.weight', weight), ('stage.bias', bias)]:
                data = array.tobytes()
                entries.append({'name': name, 'dtype': 'fp32',
                    'shape': list(array.shape), 'file_offset': file_offset,
                    'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
                file_offset += len(data)
            op = {'id': 'conv', 'op': 'audio_conv1d_checked',
                'kernel': 'audio_conv1d_checked_channel_major_f32_avx2_parallel',
                'returns_status': True,
                'weight_refs': {'weight': 'stage.weight', 'bias': 'stage.bias'},
                'params': {'I': 3, 'O': 8, 'T': 24, 'K': 3, 'U': 24, 'Q': 192,
                    'conv_worker_pool': 'worker_pool', 'conv_worker_count': 4,
                    'call_constants': {
                        'conv_input_elements': 72, 'conv_input_stride': 24,
                        'conv_weight_elements': 72, 'conv_bias_elements': 8,
                        'conv_output_elements': 192, 'conv_output_stride': 24,
                        'conv_scratch_elements': 192, 'conv_input_channels': 3,
                        'conv_output_channels': 8, 'conv_input_frames': 24,
                        'conv_kernel_size': 3, 'conv_stride': 1,
                        'conv_padding': 1, 'conv_output_frames': 24}},
                'graph_slots': {'inputs': {'input': 'external:signal'},
                                'outputs': {'output': 'convolved'}}}
            circuit = {'version': 3, 'name': 'synthetic_parallel_conv',
                'family': 'bounded_graph', 'checked_native_entry': True,
                'contract': {'runtime_invariants': {'inference_only': True}},
                'activation_buffers': {'signal': {'shape': [3, 24]},
                                       'convolved': {'shape': [8, 24]}},
                'activation_bindings': {'signal': 'signal', 'convolved': 'convolved'},
                'native_entry': {'function': 'ck_synthetic_parallel_conv',
                    'params': [{'c_type': 'uint8_t *', 'name': 'arena'},
                               {'c_type': 'size_t', 'name': 'arena_bytes'},
                               {'c_type': 'void *', 'name': 'worker_pool'}],
                    'arena': {'pointer': 'arena', 'bytes': 'arena_bytes'}},
                'sequence': ['component'], 'block_types': {'component': {
                    'sequence': ['header', 'body', 'footer'],
                    'header': [op], 'body': {'type': 'dense', 'ops': []},
                    'footer': []}}}
            path = root / 'circuit.json'
            path.write_text(json.dumps(circuit))
            source = {'config': {'model': circuit['name'], 'arch': circuit['name'],
                'num_layers': 1, 'embed_dim': 3, 'num_heads': 1, 'num_kv_heads': 1,
                'head_dim': 3, 'intermediate_size': 3, 'context_length': 24,
                'max_seq_len': 24, 'vocab_size': 1},
                'entries': entries, 'quant_summary': {}, 'template': circuit}
            layout, calls, _library, loaded, function = compile_native_graph(
                root, source, path,
                extra_exports=('ck_threadpool_create', 'ck_threadpool_destroy'))
            self.assertEqual(calls['errors'], [])
            self.assertEqual(calls['operations'][0]['function'],
                'audio_conv1d_checked_channel_major_f32_avx2_parallel')
            function.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
                                 ctypes.c_void_p]
            loaded.ck_threadpool_create.argtypes = [ctypes.c_int]
            loaded.ck_threadpool_create.restype = ctypes.c_void_p
            loaded.ck_threadpool_destroy.argtypes = [ctypes.c_void_p]
            pool = loaded.ck_threadpool_create(4)
            self.assertTrue(pool)
            try:
                size = layout['memory']['arena']['total_size']
                raw = (ctypes.c_uint8 * (size + 63))()
                arena = (ctypes.c_uint8 * size).from_buffer(raw,
                    (-ctypes.addressof(raw)) & 63)
                weights = {x['name']: x for x in layout['memory']['weights']['entries']}
                for name, array in [('stage.weight', weight), ('stage.bias', bias)]:
                    start = weights[name]['abs_offset']
                    arena[start:start + array.nbytes] = array.tobytes()
                buffers = {x['name']: x for x in layout['memory']['activations']['buffers']}
                def view(name, shape):
                    return np.ndarray(shape, np.float32, buffer=arena,
                        offset=buffers[name]['abs_offset'])
                signal = view('signal', (3, 24))
                convolved = view('convolved', (8, 24))
                signal[:] = np.arange(72, dtype=np.float32).reshape(3, 24) / 50
                convolved[:] = -77
                status = function(arena, len(arena), pool)
                if status == -5:
                    self.skipTest('AVX2/FMA unavailable')
                self.assertEqual(status, 0)
                expected = np.zeros_like(convolved)
                for oc in range(8):
                    for t in range(24):
                        value = bias[oc]
                        for ic in range(3):
                            for tap in range(3):
                                pos = t + tap - 1
                                if 0 <= pos < 24:
                                    value = np.float32(value + np.float32(signal[ic, pos] * weight[oc, ic, tap]))
                        expected[oc, t] = value
                np.testing.assert_allclose(convolved, expected, rtol=0, atol=1e-6)
                signal[0, 0] = np.nan
                convolved[:] = -77
                self.assertEqual(function(arena, len(arena), pool), -4)
                self.assertTrue(np.all(convolved == -77))
            finally:
                loaded.ck_threadpool_destroy(pool)


if __name__ == '__main__':
    unittest.main()
