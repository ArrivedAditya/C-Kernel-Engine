"""Model-neutral harmonic source through normal v8 lowering and generated C."""

import ctypes
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.v8_checked_graph_test_support import compile_native_graph


class HarmonicSourceGeneratedTest(unittest.TestCase):
    def compile_source(self, root, frames, upsample, harmonics,
                       call_constant_overrides=None):
        samples = frames * upsample
        values = samples * harmonics
        scratch = frames * harmonics + values
        op = {
            'id': 'source', 'op': 'audio_harmonic_source_checked',
            'kernel': 'audio_harmonic_source_checked_f32',
            'returns_status': True,
            'params': {'F': frames, 'S': samples, 'H': harmonics,
                'Q': scratch, 'upsample': upsample,
                'harmonics': harmonics,
                'sample_rate': 24000., 'voiced_threshold': 10.,
                'sine_amp': .1, 'noise_std': .003,
                'call_constants': {
                    'source_f0_capacity': frames,
                    'source_gaussian_capacity': values,
                    'source_output_capacity': values,
                    'source_scratch_capacity': scratch,
                    'source_frames': frames}},
            'graph_slots': {'inputs': {'f0': 'external:f0',
                'gaussian': 'external:gaussian'},
                'outputs': {'output': 'harmonics'}}}
        if call_constant_overrides:
            op['params']['call_constants'].update(call_constant_overrides)
        buffers = {'f0': {'shape': [frames]},
                   'gaussian': {'shape': [samples, harmonics]},
                   'harmonics': {'shape': [samples, harmonics]}}
        circuit = {'version': 3, 'name': 'synthetic_harmonic_source',
            'family': 'bounded_graph', 'checked_native_entry': True,
            'contract': {'runtime_invariants': {'inference_only': True}},
            'activation_buffers': buffers,
            'activation_bindings': {name: name for name in buffers},
            'native_entry': {'function': 'ck_synthetic_harmonic_source',
                'params': [{'c_type': 'uint8_t *', 'name': 'arena'},
                           {'c_type': 'size_t', 'name': 'arena_bytes'}],
                'arena': {'pointer': 'arena', 'bytes': 'arena_bytes'}},
            'sequence': ['component'], 'block_types': {'component': {
                'sequence': ['header', 'body', 'footer'],
                'header': [], 'body': {'type': 'dense', 'ops': [op]},
                'footer': []}}}
        path = root / 'circuit.json'
        path.write_text(json.dumps(circuit))
        source = {'config': {'model': circuit['name'], 'arch': circuit['name'],
            'num_layers': 1, 'embed_dim': 2, 'num_heads': 1,
            'num_kv_heads': 1, 'head_dim': 2, 'intermediate_size': 2,
            'context_length': samples, 'max_seq_len': samples,
            'vocab_size': 1},
            'entries': [], 'quant_summary': {}, 'template': circuit}
        layout, calls, _library, _loaded, fn = compile_native_graph(
            root, source, path)
        self.assertEqual(calls['errors'], [])
        self.assertEqual(calls['operations'][-1]['function'],
            'audio_harmonic_source_checked_f32')
        return layout, fn

    def test_generated_source_and_failure_propagation(self):
        with tempfile.TemporaryDirectory() as temporary:
            layout, fn = self.compile_source(Path(temporary), 3, 4, 2)
            size = layout['memory']['arena']['total_size']
            raw = (ctypes.c_uint8 * (size + 63))()
            arena = (ctypes.c_uint8 * size).from_buffer(
                raw, (-ctypes.addressof(raw)) & 63)
            locations = {item['name']: item for item in
                layout['memory']['activations']['buffers']}
            def view(name, shape):
                return np.ndarray(shape, np.float32, buffer=arena,
                    offset=locations[name]['abs_offset'])
            f0 = view('f0', (3,))
            gaussian = view('gaussian', (12, 2))
            output = view('harmonics', (12, 2))
            f0[:] = [100., 0., 220.]
            gaussian[:] = 0.
            output[:] = -91.
            self.assertEqual(fn(arena, len(arena)), 0)
            self.assertTrue(np.isfinite(output).all())
            self.assertTrue(np.all(output[4:8] == 0.))
            first = output.copy()
            self.assertEqual(fn(arena, len(arena)), 0)
            np.testing.assert_array_equal(output, first)
            gaussian[0, 0] = np.nan
            output[:] = -91.
            self.assertNotEqual(fn(arena, len(arena)), 0)
            self.assertTrue(np.all(output == -91.))
            gaussian[0, 0] = 0.
            self.assertEqual(fn(arena, len(arena)), 0)
            np.testing.assert_array_equal(output, first)

    def test_reject_claimed_capacity_beyond_planned_storage(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex((RuntimeError, ValueError),
                                        'capacity|span|shape|storage'):
                self.compile_source(Path(temporary), 3, 4, 2,
                                    {'source_output_capacity': 25})

    def test_generated_source_matches_pinned_model_capture(self):
        fixture = Path(__file__).parent / 'fixtures/tts/kokoro_source_pinned.npz'
        metadata = json.loads(fixture.with_suffix('.json').read_text())
        self.assertEqual(hashlib.sha256(fixture.read_bytes()).hexdigest(),
                         metadata['fixture_sha256'])
        with np.load(fixture) as archive:
            f0_reference = archive['f0'].reshape(-1)
            gaussian_reference = archive['gaussian'].reshape(-1)
            expected = archive['sine_waves'].reshape(-1)
        with tempfile.TemporaryDirectory() as temporary:
            layout, fn = self.compile_source(Path(temporary), 206, 300, 9)
            size = layout['memory']['arena']['total_size']
            raw = (ctypes.c_uint8 * (size + 63))()
            arena = (ctypes.c_uint8 * size).from_buffer(
                raw, (-ctypes.addressof(raw)) & 63)
            locations = {item['name']: item for item in
                layout['memory']['activations']['buffers']}
            def view(name, shape):
                return np.ndarray(shape, np.float32, buffer=arena,
                    offset=locations[name]['abs_offset'])
            f0 = view('f0', (206,))
            gaussian = view('gaussian', (61800, 9))
            output = view('harmonics', (61800, 9))
            f0[:] = f0_reference
            gaussian.reshape(-1)[:] = gaussian_reference
            output.fill(-91.)
            self.assertEqual(fn(arena, len(arena)), 0)
            self.assertTrue(np.isfinite(output).all())
            error = np.abs(output.reshape(-1) - expected)
            point = int(np.argmax(error))
            self.assertLessEqual(float(error[point]), 5e-4,
                (point, float(output.reshape(-1)[point]), float(expected[point])))
            print('CKE_NUMERICAL_CASE ' + json.dumps({
                'case_id': 'kokoro.harmonic-source.generated-vs-pinned-model',
                'name': 'generated harmonic source versus direct full-model hook',
                'provider': 'audio_harmonic_source_checked_f32',
                'dtype': 'fp32', 'direction': 'inference',
                'oracle': 'pinned-full-kmodel-pytorch28',
                'backend_version': metadata['environment']['torch'],
                'status': 'pass', 'max_diff': float(error[point]),
                'worst_index': point, 'tolerance': 5e-4,
                'configuration': '206 F0 frames, 300x, 9 harmonics, captured Gaussian',
                'reproduction_command': 'python3 -m unittest ' + self.id()}))


if __name__ == '__main__':
    unittest.main()
