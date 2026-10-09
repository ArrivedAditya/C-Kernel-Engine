"""Complete generated Kokoro F0/noise curves versus pinned model hooks."""
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from tests.v8_checked_graph_test_support import compile_native_graph
from tests.v8_kokoro_duration_fixture_support import (
    populated_duration_arena, prepare_duration_fixture)
from tests import test_v8_kokoro_generated_albert_layer as xray_support

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'version/v8/tts'))
import build_kokoro_prosody_complete_circuit as author


def verified_fixture(name):
    path = ROOT / f'tests/fixtures/tts/kokoro_{name}_pinned.npz'
    meta = json.loads(path.with_suffix('.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta['fixture_sha256']:
        raise RuntimeError(f'fixture mismatch: {name}')
    with np.load(path) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    for key, value in arrays.items():
        if ('array_sha256' in meta and
                hashlib.sha256(value.tobytes()).hexdigest() !=
                meta['array_sha256'][key]):
            raise RuntimeError(f'fixture tensor mismatch: {name}.{key}')
    return arrays, meta


def complete_prosody_weights():
    """Return the pinned effective tensors shared by connected prosody tests."""
    fixtures = {}
    for key in ('prosody_first_block', 'prosody_norm', 'prosody_shared',
                'text_encoder', 'text_embedding',
                'prosody_second_block_weights'):
        fixtures[key], _ = verified_fixture(key)
    direct, direct_meta = verified_fixture('prosody_branch_model')
    weights = {'acoustic_text_encoder.embedding.weight':
               fixtures['text_embedding']['table'].copy()}
    text = fixtures['text_encoder']
    for index in range(3):
        for kind in ('weight', 'bias'):
            weights[f'acoustic_text_encoder.conv{index}.{kind}'] = \
                text[f'conv{index}_{kind}']
        weights[f'acoustic_text_encoder.norm{index}.weight'] = \
            text[f'norm{index}_gamma']
        weights[f'acoustic_text_encoder.norm{index}.bias'] = \
            text[f'norm{index}_beta']
    for kind in ('weight_ih', 'weight_hh', 'bias_ih', 'bias_hh'):
        weights[f'acoustic_text_encoder.lstm.{kind}'] = text[f'lstm_{kind}']
        weights[f'duration_prosody.shared_scan.{kind}'] = \
            fixtures['prosody_shared'][kind]
    for branch in ('F0', 'N'):
        prefix = f'duration_prosody.{branch}.0'
        for norm_name in ('norm1', 'norm2'):
            for kind in ('fc_weight', 'fc_bias', 'norm_weight', 'norm_bias'):
                canonical = f'{prefix}.{norm_name}.{kind.replace("_", ".")}'
                source = ('prosody_norm' if norm_name == 'norm1' else
                          'prosody_first_block')
                source_name = (f'{branch}_{kind}' if norm_name == 'norm1' else
                               f'{branch}_norm2_{kind}')
                weights[canonical] = fixtures[source][source_name]
        for index in (1, 2):
            for kind in ('weight', 'bias'):
                weights[f'{prefix}.conv{index}.{kind}'] = \
                    fixtures['prosody_first_block'][f'{branch}_conv{index}_{kind}']
    weights.update(fixtures['prosody_second_block_weights'])
    return weights, direct, direct_meta


class KokoroGeneratedProsodyCompleteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        weights, cls.direct, cls.direct_meta = complete_prosody_weights()
        cls.projection_weights = {
            branch: (weights[f'duration_prosody.{branch}_proj.weight'].copy(),
                     weights[f'duration_prosody.{branch}_proj.bias'].copy())
            for branch in ('F0', 'N')}
        fixture = prepare_duration_fixture(cls.root, author.OUTPUT, weights)
        for name, value in vars(fixture).items():
            setattr(cls, name, value)
        cls.layout, cls.calls, cls.library, cls.loaded, cls.fn = \
            compile_native_graph(cls.root, cls.source, author.OUTPUT)
        cls.fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
                           ctypes.POINTER(ctypes.c_int32),
                           ctypes.POINTER(ctypes.c_int32)]
        cls.buffers = {item['name']: item for item in
                       cls.layout['memory']['activations']['buffers']}
        cls.weight_layout = {item['name']: item for item in
                             cls.layout['memory']['weights']['entries']}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def view(self, arena, name, dtype=np.float32):
        item = self.buffers[name]
        return np.ndarray((item['size'] // np.dtype(dtype).itemsize,), dtype,
                          buffer=arena, offset=item['abs_offset'])

    def arena(self):
        arena = populated_duration_arena(self.layout, self.entries, self.bump,
                                         self.encoder, self.duration)
        for branch in ('f0', 'n'):
            for name in ('block1_pool', 'block1_conv0',
                         'block1_shortcut_upsample', 'block1_shortcut_conv',
                         'block1_output', 'block2_output', 'output'):
                self.view(arena, f'{branch}_{name}')[:] = -91.
        return arena

    def test_second_blocks_match_direct_full_model_hooks(self):
        self.assertEqual(self.calls['errors'], [])
        arena = self.arena()
        frames = ctypes.c_int32(-999)
        doubled = ctypes.c_int32(-999)
        self.assertEqual(self.fn(arena, len(arena), ctypes.byref(frames),
                                 ctypes.byref(doubled)), 0)
        self.assertEqual((frames.value, doubled.value), (103, 206))
        worst = (0., None)
        f0_stage_errors = []
        for branch, stem in (('F0', 'f0'), ('N', 'n')):
            first_block = self.view(arena, f'{stem}_block0_output').reshape(512, 128)
            first_expected = self.direct[f'predictor_{branch}_0']
            first_error = np.abs(first_block[:, :103] - first_expected)
            if branch == 'F0':
                f0_stage_errors.append(('block0', first_error,
                                        first_block[:, :103], first_expected,
                                        1e-4))
            self.assertTrue(np.isfinite(first_block[:, :103]).all())
            self.assertLessEqual(float(np.max(first_error)), 1e-4)
            self.assertTrue(np.all(first_block[:, 103:] == -777.))
            actual = self.view(arena, f'{stem}_block1_output').reshape(256, 256)
            expected = self.direct[f'predictor_{branch}_1']
            self.assertTrue(np.isfinite(actual[:, :206]).all())
            error = np.abs(actual[:, :206] - expected)
            if branch == 'F0':
                f0_stage_errors.append(('block1', error,
                                        actual[:, :206], expected, 1e-4))
            point = np.unravel_index(np.argmax(error), error.shape)
            if float(error[point]) > worst[0]:
                worst = (float(error[point]),
                         (branch, *(int(x) for x in point)))
            self.assertLessEqual(float(error[point]), 1e-4,
                                 (branch, point, float(error[point])))
            self.assertTrue(np.all(actual[:, 206:] == -91.))
            third = self.view(arena, f'{stem}_block2_output').reshape(256, 256)
            third_expected = self.direct[f'predictor_{branch}_2']
            third_error = np.abs(third[:, :206] - third_expected)
            if branch == 'F0':
                f0_stage_errors.append(('block2', third_error,
                                        third[:, :206], third_expected, 1e-4))
            third_point = np.unravel_index(np.argmax(third_error),
                                           third_error.shape)
            self.assertTrue(np.isfinite(third[:, :206]).all())
            self.assertLessEqual(float(third_error[third_point]), 1e-4,
                (branch, 'third block', third_point,
                 float(third_error[third_point])))
            self.assertTrue(np.all(third[:, 206:] == -91.))
            final = self.view(arena, f'{stem}_output').reshape(1, 256)
            final_expected = self.direct[f'predictor_{branch}_proj']
            self.assertTrue(np.isfinite(final[:, :206]).all())
            final_error = np.abs(final[:, :206] - final_expected)
            if branch == 'F0':
                f0_stage_errors.append(('projection', final_error,
                                        final[:, :206], final_expected, 5e-4))
            final_point = np.unravel_index(np.argmax(final_error),
                                           final_error.shape)
            projection_weight, projection_bias = self.projection_weights[branch]
            fp64_actual = (projection_weight.reshape(256).astype(np.float64)
                           @ third[:, :206].astype(np.float64) +
                           float(projection_bias[0]))
            fp64_expected = (projection_weight.reshape(256).astype(np.float64)
                             @ third_expected.astype(np.float64) +
                             float(projection_bias[0]))
            # F0's 256-channel projection amplifies the upstream block error.
            # Keep the three contributions separate from a final comparison:
            # generated block drift, native reduction, and oracle reduction.
            self.assertLessEqual(float(np.max(np.abs(
                fp64_actual - fp64_expected))), 4e-4)
            self.assertLessEqual(float(np.max(np.abs(
                final[0, :206] - fp64_actual))), 2e-4)
            self.assertLessEqual(float(np.max(np.abs(
                final_expected[0] - fp64_expected))), 2e-4)
            if float(final_error[final_point]) > worst[0]:
                worst = (float(final_error[final_point]),
                         (branch, 'projection',
                          *(int(x) for x in final_point)))
            self.assertLessEqual(float(final_error[final_point]),
                                 5e-4 if branch == 'F0' else 1e-4,
                (branch, 'projection', final_point,
                 float(final_error[final_point])))
            self.assertTrue(np.all(final[:, 206:] == -91.))
        for stage, error, actual, reference, limit in f0_stage_errors:
            point = np.unravel_index(np.argmax(error), error.shape)
            print('CKE_NUMERICAL_CASE ' + json.dumps({
                'case_id': f'kokoro.complete-prosody.F0-{stage}.drift-v1',
                'name': f'Kokoro generated F0 {stage} full-model drift',
                'provider': 'generated_prosody_complete',
                'dtype': 'fp32', 'direction': 'inference',
                'oracle': 'pinned-full-kmodel-pytorch28',
                'backend_version': self.direct_meta['environment']['torch'],
                'status': 'pass' if float(error[point]) <= limit else 'fail',
                'gate': 'diagnostic', 'blocking': False,
                'reason': 'coarse full-model checkpoint; first divergent '
                          'operation still requires identical-input replay',
                'max_diff': float(error[point]),
                'worst_index': list(map(int, point)),
                'actual': float(actual[point]),
                'reference': float(reference[point]),
                'rmse': float(np.sqrt(np.mean(error.astype(np.float64) ** 2))),
                'tolerance': limit,
                'configuration': '36 phonemes; A=103; 2A=206; af_heart',
                'reproduction_command': 'python3 -m unittest '
                    'tests.test_v8_kokoro_generated_prosody_complete.'
                    'KokoroGeneratedProsodyCompleteTest.'
                    'test_second_blocks_match_direct_full_model_hooks'}))
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'kokoro.complete-prosody.generated-v1',
            'name': 'generated complete F0/noise outputs versus direct pinned KModel hooks',
            'provider': 'generated_prosody_complete',
            'dtype': 'fp32', 'direction': 'inference',
            'oracle': 'pinned-full-kmodel-pytorch28',
            'backend_version': self.direct_meta['environment']['torch'],
            'status': 'pass', 'max_diff': worst[0], 'worst_index': worst[1],
            'tolerance': 5e-4,
            'checkpoint_tolerances': {'first_block': 1e-4,
                                      'second_block': 1e-4,
                                      'third_block': 1e-4,
                                      'F0_final': 5e-4,
                                      'N_final': 1e-4},
            'configuration': '36 phonemes; A=103 of 128; 2A=206 of 256; af_heart',
            'reproduction_command': 'python3 -m unittest ' + self.id()}))

    def test_repeated_lengths_and_rejection_preserve_outputs(self):
        arena = self.arena()
        first = ctypes.c_int32(-999); second = ctypes.c_int32(-999)
        self.assertEqual(self.fn(arena, len(arena), ctypes.byref(first),
                                 ctypes.byref(second)), 0)
        reference = self.view(arena, 'f0_block1_output').copy()
        self.view(arena, 'f0_block1_output')[:] = -91.
        self.assertEqual(self.fn(arena, len(arena), ctypes.byref(first),
                                 ctypes.byref(second)), 0)
        np.testing.assert_array_equal(
            self.view(arena, 'f0_block1_output').reshape(256, 256)[:, :206],
            reference.reshape(256, 256)[:, :206])
        weight = self.weight_layout['duration_prosody.duration_head.weight']
        bias = self.weight_layout['duration_prosody.duration_head.bias']
        np.ndarray((weight['size']//4,), np.float32, buffer=arena,
                   offset=weight['abs_offset'])[:] = 0.
        biases = np.ndarray((bias['size']//4,), np.float32, buffer=arena,
                            offset=bias['abs_offset'])
        for bias_value, expected in ((-3.2, 72), (-10., 36)):
            biases[:] = bias_value
            for branch in ('f0', 'n'):
                self.view(arena, f'{branch}_output')[:] = -91.
            self.assertEqual(self.fn(arena, len(arena), ctypes.byref(first),
                                     ctypes.byref(second)), 0)
            self.assertEqual((first.value, second.value),
                             (expected, 2 * expected))
            for branch in ('f0', 'n'):
                output = self.view(arena, f'{branch}_output').reshape(1, 256)
                self.assertTrue(np.isfinite(output[:, :2 * expected]).all())
                self.assertTrue(np.all(output[:, 2 * expected:] == -91.))
        # A corrupt upstream duration fails before 2A becomes valid or a
        # downstream output is published.
        self.view(arena, 'f0_block1_output')[:] = -91.
        biases[:] = 1.
        first.value = second.value = -999
        self.assertNotEqual(self.fn(arena, len(arena), ctypes.byref(first),
                                    ctypes.byref(second)), 0)
        self.assertEqual((first.value, second.value), (-999, -999))
        self.assertTrue(np.all(self.view(arena, 'f0_block1_output') == -91.))

    def test_wrong_final_branch_weight_is_detected(self):
        arena = self.arena()
        source = self.weight_layout['duration_prosody.N_proj.weight']
        target = self.weight_layout['duration_prosody.F0_proj.weight']
        self.assertEqual(source['size'], target['size'])
        arena[target['abs_offset']:target['abs_offset'] + target['size']] = \
            arena[source['abs_offset']:source['abs_offset'] + source['size']]
        first = ctypes.c_int32(-999); second = ctypes.c_int32(-999)
        self.assertEqual(self.fn(arena, len(arena), ctypes.byref(first),
                                 ctypes.byref(second)), 0)
        actual = self.view(arena, 'f0_output').reshape(1, 256)[:, :206]
        self.assertGreater(float(np.max(np.abs(actual -
            self.direct['predictor_F0_proj']))), 1e-3)

    def test_xray_final_curves_resolve_selected_provider(self):
        arena = self.arena()
        first = ctypes.c_int32(-999); second = ctypes.c_int32(-999)
        self.assertEqual(self.fn(arena, len(arena), ctypes.byref(first),
                                 ctypes.byref(second)), 0)
        tensor_report = {'torch': {'tensors': {}}, 'comparisons': {}}
        points = {}
        for branch, stem in (('F0', 'f0'), ('N', 'n')):
            name = f'{stem}_output'
            point = next(item for operation in self.calls['operations']
                for item in operation.get('semantic_checkpoints', [])
                if item['tensor'] == name)
            self.assertEqual(point['resolved_contract_id'],
                             'audio_conv1d_checked_scalar_fma_fp32')
            selector = name if point['layer'] < 0 else \
                f'{name}@{point["layer"]}'
            native = self.root / f'{name}-native.f32'
            oracle = self.root / f'{name}-pytorch.f32'
            self.view(arena, name).tofile(native)
            self.direct[f'predictor_{branch}_proj'].tofile(oracle)
            tensor_report['torch']['tensors'][selector] = {
                'path': str(oracle), 'shape': [1, 206]}
            tensor_report['comparisons'][selector] = {
                'ck_path': str(native), 'shape': [1, 206],
                'physical_shape': [1, 256], 'capacity_shape': [1, 256],
                'valid_shape': [1, 206], 'physical_strides': [256, 1]}
            points[name] = point
        builder = xray_support.xray_builder
        runtime = builder.capture_runtime_library_identity(
            self.loaded, 'ck_kokoro_prosody_complete_bounded')
        subject = builder.build_manifest(backend='ck', call_ir=self.calls,
            tensor_report=tensor_report, model='kokoro_prosody_complete_bounded',
            source='generated_native', phase='prefill',
            loaded_library=self.library, runtime_library=runtime)
        oracle = builder.build_manifest(backend='pytorch', call_ir=self.calls,
            tensor_report=tensor_report, model='kokoro_prosody_complete_bounded',
            source='pinned_full_kmodel', phase='prefill')
        self.assertEqual(subject['run']['artifact_library']['sha256'],
                         subject['run']['runtime_library']['sha256'])
        for name, point in points.items():
            ceiling = 5e-4 if name == 'f0_output' else 1e-4
            profile = {'schema': 'cke.parity_profile', 'schema_version': 1,
                'name': 'kokoro_complete_prosody', 'backend': 'pytorch',
                'contract_schema_version': 1,
                'required_match_fields': ['checkpoint_id', 'producer',
                    'logical_layout', 'axis_names', 'resolved_contract_id',
                    'kernel_id', 'function'],
                'observed_storage': {'default': 'fp32', 'checkpoints': {}},
                'dtype_thresholds': {'fp32': {'cosine_min': .99999,
                    'rmse_max': ceiling, 'relative_rmse_max': ceiling,
                    'max_abs_max': ceiling, 'finite_required': True}},
                'checkpoint_order': [point['id']],
                'interval_expansions': {}, 'backend_mappings': {}}
            one = lambda manifest: {**manifest, 'checkpoints': [entry for entry in
                manifest['checkpoints'] if entry['checkpoint_id'] == point['id']]}
            report = xray_support.xray.compare_manifests(
                one(subject), one(oracle), profile)
            self.assertEqual(report['status'], 'pass', (name, report))


if __name__ == '__main__':
    unittest.main()
