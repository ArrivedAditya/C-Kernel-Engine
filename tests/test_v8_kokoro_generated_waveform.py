"""Complete generated Kokoro waveform execution and diagnostic parity."""

import ctypes
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import wave

import numpy as np

from tests.v8_checked_graph_test_support import compile_native_graph
from tests.v8_kokoro_duration_fixture_support import (
    populated_duration_arena, prepare_duration_fixture)
from tests.test_v8_kokoro_generator_stage0_join import (
    join_weights_and_references)
from tests.test_v8_kokoro_generator_stage0_pool import stage0_pool_weights
from tests import test_v8_kokoro_source_residual_first_pair as fixture_support


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'version/v8/tts'))
import build_kokoro_waveform_circuit as waveform_author
import build_kokoro_waveform_oracle_circuit as direct_author


class KokoroGeneratedWaveformTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        cls.refs = join_weights_and_references()
        first, first_meta = fixture_support.load_fixture(
            'kokoro_generator_main_resblock0_pinned')
        tail, tail_meta = fixture_support.load_fixture(
            'kokoro_generator_stage0_pool_pinned')
        ingress, ingress_meta = fixture_support.load_fixture(
            'kokoro_generator_stage1_ingress_pinned')
        source, source_meta = fixture_support.load_fixture(
            'kokoro_generator_stage1_source_conv_pinned')
        join, join_meta = fixture_support.load_fixture(
            'kokoro_generator_stage1_join_pinned')
        pool, pool_meta = fixture_support.load_fixture(
            'kokoro_generator_stage1_pool_pinned')
        cls.pool = pool
        cls.spectral, spectral_meta = fixture_support.load_fixture(
            'kokoro_spectral_head_pinned')
        cls.spectral_meta = spectral_meta
        for field in ('model_pin', 'code_pin', 'asset_sha256'):
            if any(item[field] != first_meta[field] for item in (
                    tail_meta, ingress_meta, source_meta, join_meta,
                    pool_meta, spectral_meta, cls.refs.join_meta)):
                raise RuntimeError(f'waveform fixture {field} differs')
        previous = ROOT / 'tests/fixtures/tts/kokoro_generator_stage1_pool_pinned.npz'
        if hashlib.sha256(previous.read_bytes()).hexdigest() != \
                spectral_meta['previous_fixture_sha256']:
            raise RuntimeError('waveform fixture upstream hash differs')
        weights = stage0_pool_weights(cls.refs, first, tail)
        weights['waveform_decoder.generator.ups.1.weight'] = ingress['weight']
        weights['waveform_decoder.generator.ups.1.bias'] = ingress['bias']
        weights['waveform_decoder.generator.noise_convs.1.weight'] = source['weight']
        weights['waveform_decoder.generator.noise_convs.1.bias'] = source['bias']
        weights.update({key: value for data in (join, pool)
                        for key, value in data.items()
                        if key.startswith('waveform_decoder.')})
        weights['waveform_decoder.generator.conv_post.weight'] = \
            cls.spectral['weight']
        weights['waveform_decoder.generator.conv_post.bias'] = \
            cls.spectral['bias']
        fixture = prepare_duration_fixture(root, waveform_author.OUTPUT, weights)
        cls.encoder, cls.duration = fixture.encoder, fixture.duration
        cls.entries, cls.bump = fixture.entries, fixture.bump
        native = root / 'waveform'
        native.mkdir()
        cls.native_dir = native
        try:
            cls.layout, cls.calls, _, _, cls.fn = compile_native_graph(
                native, fixture.source, waveform_author.OUTPUT)
        except subprocess.CalledProcessError as exc:
            calls = native / 'call.json'
            errors = json.loads(calls.read_text()).get('errors') if calls.exists() else None
            raise RuntimeError(f'{exc.stderr or exc.stdout}\ncall errors: {errors}') from exc
        cls.fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t] + [
            ctypes.POINTER(ctypes.c_int32)] * 7
        diagnostic = copy.deepcopy(fixture.source)
        diagnostic['template'] = direct_author.build_circuit()
        diagnostic['config']['model'] = diagnostic['template']['name']
        diagnostic['config']['arch'] = diagnostic['template']['name']
        diagnostic['config']['activation_buffer_dtypes'].update({
            name: 'i32' for name in (
                'head_frame_count', 'head_sample_count',
                'head_frame_extent_value', 'head_sample_extent_value')})
        referenced = {name
            for block in diagnostic['template']['block_types'].values()
            for op in block['header'] + block['body']['ops'] + block['footer']
            for name in op.get('weight_refs', {}).values()}
        diagnostic['entries'] = [
            item for item in diagnostic['entries']
            if item['name'] in referenced]
        direct = root / 'direct'
        direct.mkdir()
        cls.direct_layout, cls.direct_calls, _, _, cls.direct_fn = \
            compile_native_graph(direct, diagnostic, direct_author.OUTPUT)
        cls.direct_fn.argtypes = [
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @staticmethod
    def view(layout, arena, name, dtype=np.float32):
        item = next(item for item in layout['memory']['activations']['buffers']
                    if item['name'] == name)
        return np.ndarray((item['size'] // np.dtype(dtype).itemsize,), dtype,
                          buffer=arena, offset=item['abs_offset'])

    @classmethod
    def arena(cls):
        arena = populated_duration_arena(
            cls.layout, cls.entries, cls.bump, cls.encoder, cls.duration)
        cls.view(cls.layout, arena, 'decoder_style')[:] = \
            cls.refs.encode['decoder_style'].ravel()
        cls.view(cls.layout, arena, 'source_gaussian').reshape(
            76800, 9)[:61800] = cls.refs.source['gaussian'].reshape(61800, 9)
        cls.view(cls.layout, arena, 'source_stft_window')[:] = \
            cls.refs.source['window']
        taps = np.arange(20, dtype=np.float64)
        angles = -2 * np.pi * np.arange(11)[:, None] * taps / 20
        cls.view(cls.layout, arena, 'source_stft_cos').reshape(11, 20)[:] = \
            np.cos(angles)
        cls.view(cls.layout, arena, 'source_stft_sin').reshape(11, 20)[:] = \
            np.sin(angles)
        cls.view(cls.layout, arena, 'generator_waveform')[:] = -91.
        return arena

    @classmethod
    def direct_arena(cls):
        layout = cls.direct_layout
        size = layout['memory']['arena']['total_size']
        raw = (ctypes.c_uint8 * (size + 63))()
        arena = (ctypes.c_uint8 * size).from_buffer(
            raw, (-ctypes.addressof(raw)) & 63)
        for item in layout['memory']['weights']['entries']:
            entry = cls.entries[item['name']]
            start = entry['file_offset']
            arena[item['abs_offset']:item['abs_offset'] + entry['size']] = \
                cls.bump[start:start + entry['size']]
        features = cls.view(layout, arena,
            'generator_stage1_final_activation').reshape(128, 15361)
        features[:] = -91.
        features[:, :12361] = cls.pool['final_activation'][0]
        cls.view(layout, arena, 'head_frame_count', np.int32)[0] = 12361
        cls.view(layout, arena, 'head_sample_count', np.int32)[0] = 61800
        for name in ('generator_conv_post', 'generator_magnitude',
                     'generator_phase', 'generator_waveform'):
            cls.view(layout, arena, name)[:] = -91.
        return arena

    def test_direct_generated_tail_matches_full_model(self):
        self.assertEqual(json.loads(direct_author.OUTPUT.read_text()),
                         direct_author.build_circuit())
        self.assertEqual(self.direct_calls['errors'], [])
        arena = self.direct_arena()
        frames = ctypes.c_int32(-999)
        samples = ctypes.c_int32(-999)
        status = self.direct_fn(arena, len(arena),
            ctypes.byref(frames), ctypes.byref(samples))
        if status:
            print('DIRECT_WAVEFORM_DIAGNOSTIC', {
                'status': status,
                'conv_first': float(self.view(self.direct_layout, arena,
                    'generator_conv_post')[0]),
                'magnitude_first': float(self.view(self.direct_layout, arena,
                    'generator_magnitude')[0]),
                'phase_first': float(self.view(self.direct_layout, arena,
                    'generator_phase')[0]),
                'waveform_first': float(self.view(self.direct_layout, arena,
                    'generator_waveform')[0]),
                'frame_input': int(self.view(self.direct_layout, arena,
                    'head_frame_count', np.int32)[0]),
                'sample_input': int(self.view(self.direct_layout, arena,
                    'head_sample_count', np.int32)[0]),
            })
        self.assertEqual(status, 0)
        self.assertEqual((frames.value, samples.value), (12361, 61800))
        layout = self.direct_layout
        cases = (
            ('conv_post', self.view(layout, arena,
                'generator_conv_post').reshape(22, 15361)[:, :12361],
             self.spectral['conv_post'][0], 5e-5),
            ('magnitude', self.view(layout, arena,
                'generator_magnitude').reshape(15361, 11)[:12361],
             self.spectral['magnitude'][0].T, 5e-5),
            ('phase', self.view(layout, arena,
                'generator_phase').reshape(15361, 11)[:12361],
             self.spectral['phase'][0].T, 1e-5),
            ('waveform', self.view(layout, arena,
                'generator_waveform')[:61800],
             self.spectral['waveform'][0, 0], 1e-5),
        )
        for name, actual, reference, limit in cases:
            self.assertTrue(np.isfinite(actual).all(), name)
            error = np.abs(actual - reference)
            point = np.unravel_index(np.argmax(error), error.shape)
            print('CKE_NUMERICAL_CASE ' + json.dumps({
                'case_id': f'kokoro.waveform-tail.{name}.direct-v1',
                'name': f'Kokoro generated waveform tail {name} direct parity',
                'provider': 'generated_kokoro_waveform_tail',
                'dtype': 'fp32', 'direction': 'inference',
                'oracle': 'pinned-PyTorch-2.8-full-model-hook',
                'status': 'pass' if float(error[point]) <= limit else 'fail',
                'max_diff': float(error[point]), 'tolerance': limit,
                'worst_index': list(map(int, point)),
                'actual': float(actual[point]),
                'reference': float(reference[point]),
                'configuration': 'oracle-fed final generator features, 12361 frames',
                'reproduction_command': 'python3 -m unittest '
                    'tests.test_v8_kokoro_generated_waveform.'
                    'KokoroGeneratedWaveformTest.test_direct_generated_tail_matches_full_model'}))
            self.assertLessEqual(float(error[point]), limit, name)
        self.assertTrue(np.all(self.view(layout, arena,
            'generator_conv_post').reshape(22, 15361)[:, 12361:] == -91.))
        for name in ('generator_magnitude', 'generator_phase'):
            self.assertTrue(np.all(self.view(layout, arena,
                name).reshape(15361, 11)[12361:] == -91.))
        self.assertTrue(np.all(self.view(layout, arena,
            'generator_waveform')[61800:] == -91.))

    def test_connected_waveform_is_finite_and_reports_numerical_status(self):
        self.assertEqual(json.loads(waveform_author.OUTPUT.read_text()),
                         waveform_author.build_circuit())
        self.assertEqual(self.calls['errors'], [])
        arena = self.arena()
        lengths = [ctypes.c_int32(-999) for _ in range(7)]
        self.assertEqual(self.fn(arena, len(arena), *(
            ctypes.byref(value) for value in lengths)), 0)
        self.assertEqual(tuple(value.value for value in lengths),
                         (103, 206, 2060, 61800, 12361, 12360, 12361))
        output = self.view(self.layout, arena, 'generator_waveform')
        self.assertTrue(np.isfinite(output[:61800]).all())
        self.assertTrue(np.all(output[61800:] == -91.))
        reference = self.spectral['waveform'][0, 0]
        error = np.abs(output[:61800] - reference)
        point = int(np.argmax(error))
        limit = 2e-4
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'kokoro.waveform.connected-diagnostic-v1',
            'name': 'Kokoro complete generated waveform diagnostic',
            'provider': 'generated_kokoro_waveform', 'dtype': 'fp32',
            'direction': 'inference',
            'oracle': 'pinned-PyTorch-2.8-full-model-waveform',
            'status': 'fail' if float(error[point]) > limit else 'pass',
            'gate': 'diagnostic', 'blocking': False,
            'reason': 'known connected F0 and source/STFT mismatch',
            'max_diff': float(error[point]), 'tolerance': limit,
            'worst_index': point, 'actual': float(output[point]),
            'reference': float(reference[point]),
            'configuration': '36 phonemes, af_heart, captured Gaussian, 61800 samples',
            'reproduction_command': 'python3 -m unittest '
                'tests.test_v8_kokoro_generated_waveform'}))

    def _run_standalone_native_wav_outside_checkout(self):
        retained = os.environ.get('CKE_TTS_RETAIN_DIR')
        root = Path(retained) if retained else Path(self.temp.name) / 'standalone'
        root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.native_dir / 'generated.so', root / 'generated.so')
        shutil.copy2(self.native_dir / 'build-identity.json',
                     root / 'build-identity.json')
        source = ROOT / 'version/v8/tts/native_kokoro_wav_host.c'
        shutil.copy2(source, root / 'host.c')
        binary = root / 'native-kokoro-wav'
        subprocess.run(['cc', '-std=c11', '-O2', '-Wall', '-Wextra',
            '-Werror', '-pedantic', str(root / 'host.c'), '-L', str(root),
            '-l:generated.so', '-Wl,-rpath,$ORIGIN', '-lm', '-o', str(binary)],
            check=True)
        arena = self.arena()
        output_item = next(item for item in
            self.layout['memory']['activations']['buffers']
            if item['name'] == 'generator_waveform')
        package = root / 'prepared-arena.cktts'
        page = 65536
        segments = 0
        with package.open('wb') as stream:
            stream.write(b'CKTTS001')
            stream.write(struct.pack('<QQQI', len(arena),
                output_item['abs_offset'], output_item['size'] // 4, 0))
            for offset in range(0, len(arena), page):
                chunk = bytes(arena[offset:offset + page])
                if any(chunk):
                    stream.write(struct.pack('<QQ', offset, len(chunk)))
                    stream.write(chunk)
                    segments += 1
            stream.seek(32)
            stream.write(struct.pack('<I', segments))
        wav_path = root / 'kokoro-generated-diagnostic.wav'
        subprocess.run([str(binary), str(package), str(wav_path)],
                       cwd=root, check=True, capture_output=True, text=True)
        with wave.open(str(wav_path), 'rb') as stream:
            self.assertEqual(stream.getnchannels(), 1)
            self.assertEqual(stream.getsampwidth(), 2)
            self.assertEqual(stream.getframerate(), 24000)
            self.assertEqual(stream.getnframes(), 61800)
            pcm = stream.readframes(61800)
        self.assertTrue(any(pcm))
        (root / 'reproduction.json').write_text(json.dumps({
            'scope': 'experimental fixed-phoneme diagnostic WAV',
            'model_pin': self.spectral_meta['model_pin'],
            'code_pin': self.spectral_meta['code_pin'],
            'library_sha256': hashlib.sha256(
                (root / 'generated.so').read_bytes()).hexdigest(),
            'package_sha256': hashlib.sha256(package.read_bytes()).hexdigest(),
            'wav_sha256': hashlib.sha256(wav_path.read_bytes()).hexdigest(),
            'sample_rate': 24000, 'channels': 1, 'samples': 61800,
            'segments': segments,
            'command': './native-kokoro-wav prepared-arena.cktts '
                'kokoro-generated-diagnostic.wav',
            'numerical_status': 'FAIL, connected diagnostic',
            'captured_gaussian': True,
        }, indent=2) + '\n')
        invalid_package = root / 'invalid.cktts'
        with package.open('rb') as stream:
            invalid_package.write_bytes(stream.read(20))
        failed = subprocess.run([str(binary), str(invalid_package),
            str(root / 'invalid.wav')], cwd=root)
        self.assertNotEqual(failed.returncode, 0)
        self.assertFalse((root / 'invalid.wav').exists())

if os.environ.get('CKE_TTS_RUN_STANDALONE'):
    KokoroGeneratedWaveformTest.test_standalone_native_wav_outside_checkout = (
        KokoroGeneratedWaveformTest._run_standalone_native_wav_outside_checkout)


if __name__ == '__main__':
    unittest.main()
