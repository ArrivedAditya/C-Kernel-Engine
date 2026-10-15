"""One generated encoder and duration predictor with checked token extents."""
import ctypes
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from tests.v8_checked_graph_test_support import compile_native_graph
from tests.v8_kokoro_duration_fixture_support import prepare_duration_fixture

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'version/v8/tts'))
from build_kokoro_duration_circuit import build_circuit


class KokoroRuntimeDurationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        circuit = root / 'circuit.json'
        circuit.write_text(json.dumps(build_circuit(64, runtime_tokens=True)))
        fixture = prepare_duration_fixture(root, circuit)
        fixture.source['config']['activation_buffer_dtypes'].update({
            'request_length': 'i32', 'valid_length': 'i32',
            'valid_residual_elements': 'i32'})
        cls.layout, cls.calls, cls.library, cls.loaded, cls.fn = (
            compile_native_graph(root, fixture.source, circuit))
        cls.fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
                           ctypes.POINTER(ctypes.c_int32),
                           ctypes.POINTER(ctypes.c_int32),
                           ctypes.POINTER(ctypes.c_int32)]
        cls.buffers = {entry['name']: entry for entry in
                       cls.layout['memory']['activations']['buffers']}
        cls.weights = {entry['name']: entry for entry in
                       cls.layout['memory']['weights']['entries']}
        cls.fixture = fixture
        cls.root = root
        cls.captures = {36: fixture.encoder['word_ids']}
        for count in (2, 8, 64):
            name = ('kokoro_encoder_pinned.npz' if count == 36 else
                    f'kokoro_encoder_{count}_pinned.npz')
            cls.captures[count] = dict(np.load(ROOT / 'tests/fixtures/tts' / name))['word_ids']

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def view(self, arena, name, shape, dtype=np.float32):
        return np.ndarray(shape, dtype, buffer=arena,
                          offset=self.buffers[name]['abs_offset'])

    def arena(self):
        size = self.layout['memory']['arena']['total_size']
        raw = (ctypes.c_uint8 * (size + 63))()
        arena = (ctypes.c_uint8 * size).from_buffer(
            raw, (-ctypes.addressof(raw)) & 63)
        for planned in self.weights.values():
            entry = self.fixture.entries[planned['name']]
            start = entry['file_offset']
            arena[planned['abs_offset']:planned['abs_offset'] + entry['size']] = (
                self.fixture.bump[start:start + entry['size']])
        self.view(arena, 'type_ids', (64,), np.int32)[:] = 0
        self.view(arena, 'predictor_style', (128,))[:] = (
            self.fixture.duration['predictor_style'])
        return arena

    def execute(self, arena, count):
        ids = self.view(arena, 'word_ids', (64,), np.int32)
        ids[:] = 178  # Padding is invalid and must never be consumed.
        ids[:count] = self.captures[count]
        self.view(arena, 'request_length', (1,), np.int32)[0] = count
        logits = self.view(arena, 'duration_logits', (64, 50))
        durations = self.view(arena, 'runtime_values', (64,), np.int32)
        logits[:] = -777
        durations[:] = -999
        results = [ctypes.c_int32(-1) for _ in range(3)]
        status = self.fn(arena, len(arena), *(ctypes.byref(x) for x in results))
        self.assertEqual(status, 0)
        self.assertEqual(results[0].value, count)
        self.assertEqual(results[1].value, count * 768)
        self.assertEqual(int(durations[:count].sum(dtype=np.int64)), results[2].value)
        self.assertGreater(results[2].value, 0)
        self.assertLessEqual(results[2].value, 64 * 50)
        self.assertTrue(np.isfinite(logits[:count]).all())
        self.assertTrue(np.all(logits[count:] == -777))
        self.assertTrue(np.all(durations[count:] == -999))
        return logits[:count].copy(), durations[:count].copy(), results[2].value

    def test_one_entry_changes_length_and_matches_pinned_duration(self):
        self.assertEqual(self.calls['errors'], [])
        self.assertEqual(len(self.calls['operations']), 162)
        arena = self.arena()
        first_short = None
        for count in (2, 8, 36, 64, 2):
            with self.subTest(count=count):
                result = self.execute(arena, count)
                if count == 36:
                    expected = self.fixture.duration
                    error = np.abs(result[0].astype(np.float64) -
                                   expected['duration_logits'].astype(np.float64))
                    self.assertLessEqual(float(error.max()), 5e-5)
                    np.testing.assert_array_equal(result[1], expected['durations'])
                    self.assertEqual(result[2], 103)
                    print('CKE_NUMERICAL_CASE ' + json.dumps({
                        'case_id': 'kokoro.runtime-duration.36.pytorch28',
                        'name': 'Kokoro runtime duration logits and frame extent',
                        'provider': 'generated_kokoro_runtime_duration_predictor',
                        'dtype': 'fp32', 'direction': 'inference',
                        'oracle': 'pinned independent PyTorch 2.8 duration capture',
                        'status': 'pass', 'gate': 'composition', 'blocking': True,
                        'max_diff': float(error.max()), 'tolerance': 5e-5,
                        'configuration': 'capacity=64, valid=36, af_heart pinned style',
                        'reproduction_command': 'python3 -m unittest '
                            'tests.test_v8_kokoro_runtime_duration',
                    }, sort_keys=True))
                elif count == 2 and first_short is not None:
                    for actual, previous in zip(result, first_short):
                        np.testing.assert_array_equal(actual, previous)
                elif count == 2:
                    first_short = result

    def test_failure_does_not_publish_lengths_and_recovers(self):
        arena = self.arena()
        baseline = self.execute(arena, 8)
        for count in (0, -1, 65, 2147483647):
            with self.subTest(count=count):
                self.view(arena, 'request_length', (1,), np.int32)[0] = count
                logits = self.view(arena, 'duration_logits', (64, 50))
                before = logits.copy()
                results = [ctypes.c_int32(-1) for _ in range(3)]
                self.assertNotEqual(self.fn(arena, len(arena),
                    *(ctypes.byref(x) for x in results)), 0)
                self.assertEqual([x.value for x in results], [-1, -1, -1])
                np.testing.assert_array_equal(logits, before)
        style = self.view(arena, 'predictor_style', (128,))
        original_style = float(style[0])
        self.view(arena, 'request_length', (1,), np.int32)[0] = 8
        style[0] = np.nan
        logits = self.view(arena, 'duration_logits', (64, 50))
        before = logits.copy()
        results = [ctypes.c_int32(-1) for _ in range(3)]
        self.assertNotEqual(self.fn(arena, len(arena),
            *(ctypes.byref(x) for x in results)), 0)
        self.assertEqual([x.value for x in results], [-1, -1, -1])
        np.testing.assert_array_equal(logits, before)
        style[0] = original_style
        recovered = self.execute(arena, 8)
        for actual, previous in zip(recovered, baseline):
            np.testing.assert_array_equal(actual, previous)

    def test_declared_capacity_must_match_planned_storage(self):
        from tests.v8_checked_graph_test_support import build_ir_v8
        registry = build_ir_v8.load_kernel_registry()
        source = self.fixture.source
        circuit = self.root / 'circuit.json'
        ir1 = build_ir_v8.build_ir1_direct(source, circuit, mode='prefill')
        original = build_ir_v8.generate_ir_lower_1(
            ir1, registry, source, 'prefill')
        layout = build_ir_v8.generate_memory_layout(
            original, source, registry, mode='prefill', context_len=64)
        for name, excess in (('logit_elements', 3201),
                             ('duration_capacity', 65)):
            with self.subTest(name=name):
                lower1 = copy.deepcopy(original)
                lower1[-1]['params']['call_constants'][name] = excess
                with self.assertRaisesRegex(RuntimeError,
                                            'capacity exceeds planned storage'):
                    lower2 = build_ir_v8.generate_ir_lower_2(
                        lower1, layout, source, registry, mode='prefill')
                    build_ir_v8.generate_ir_lower_3(lower2, mode='prefill')


if __name__ == '__main__':
    unittest.main()
