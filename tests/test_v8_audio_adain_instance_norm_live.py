"""Separate live PyTorch lane for the channelwise AdaIN provider."""
import json
import unittest

import numpy as np

from tests import test_v8_audio_adain_instance_norm_oracle as support


class AdaINInstanceNormLiveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        support.AdaINInstanceNormOracleTest.setUpClass()
        cls.fixture = support.AdaINInstanceNormOracleTest(
            'test_committed_independent_pytorch_fixture')

    @classmethod
    def tearDownClass(cls):
        support.AdaINInstanceNormOracleTest.tearDownClass()

    def test_live_pytorch_and_fixture_reproduction(self):
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as exc:
            self.skipTest(f'live PyTorch dependency unavailable: {exc}')
        fixture = self.fixture
        worst_fixture, worst_native = 0., 0.
        for index, (_channels, frames) in enumerate(fixture.meta['shapes']):
            values, source, output, stride = fixture.case(index)
            with torch.no_grad():
                normalized = F.instance_norm(
                    torch.from_numpy(values['input'])[None],
                    weight=torch.from_numpy(values['norm_weight']),
                    bias=torch.from_numpy(values['norm_bias']),
                    use_input_stats=True, eps=fixture.meta['epsilon'])[0]
                channels = source.shape[0]
                live = ((1 + torch.from_numpy(
                    values['style_affine'][:channels, None])) * normalized
                    + torch.from_numpy(values['style_affine'][channels:, None]))
            live_np = live.numpy()
            self.assertEqual(fixture.functions[0](*fixture.args(
                values, source, output, stride, frames,
                fixture.meta['epsilon'])), 0)
            worst_fixture = max(worst_fixture, float(np.max(np.abs(
                values['output'] - live_np))))
            worst_native = max(worst_native, float(np.max(np.abs(
                output[:, :frames] - live_np))))
        native_pass = worst_native <= 1e-5
        same_reference_version = torch.__version__ == fixture.meta['backend_version']
        fixture_tolerance = 1e-6 if same_reference_version else None
        fixture_pass = (worst_fixture <= fixture_tolerance
                        if fixture_tolerance is not None else None)
        for case_id, name, direction, error, tolerance in (
            ('tts.adain-instance-norm.native-vs-live',
             'channelwise AdaIN native versus live PyTorch',
             'inference', worst_native, 1e-5),
            ('tts.adain-instance-norm.fixture-vs-live',
             'channelwise AdaIN committed versus live PyTorch oracle',
             'oracle-reproduction', worst_fixture, fixture_tolerance)):
            status = ('pass' if native_pass else 'fail') if direction == 'inference' else (
                'pass' if fixture_pass else 'fail' if fixture_pass is False else 'not_tested')
            print('CKE_NUMERICAL_CASE ' + json.dumps({
                'case_id': case_id, 'name': name,
                'provider': 'audio_adain_instance_norm_f32', 'dtype': 'fp32',
                'direction': direction, 'oracle': 'live-pytorch',
                'backend_version': torch.__version__, 'status': status,
                'max_diff': error, 'tolerance': tolerance,
                'reason': None if fixture_tolerance is not None or direction == 'inference'
                else 'reference version differs from committed fixture',
                'configuration': str(fixture.meta['shapes']),
                'reproduction_command': 'python3 -m unittest ' + self.id(),
            }))
        self.assertTrue(native_pass, f'native versus live PyTorch: {worst_native}')
        if same_reference_version:
            self.assertTrue(fixture_pass, f'fixture versus live PyTorch: {worst_fixture}')


if __name__ == '__main__':
    unittest.main()
