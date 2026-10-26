"""Asset-free admission and generated-map contract for opt-in shared FarSkip."""
import json
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / 'version/v8/kernel_maps/farskip_swiglu_shared_combine_pytorch_onednn_3_12_bf16_storage.json'

class InstellaOneDNNSharedContractTests(unittest.TestCase):
    def test_map_declares_grouped_rows_and_bounded_staging(self):
        data = json.loads(MAP.read_text())
        self.assertEqual(data['op'], 'farskip_routed_shared_combine')
        self.assertEqual(data['selection']['status'], 'candidate')
        self.assertEqual(data['scratch'][0]['size_bytes'], 'R * (16 * I + 8 * H) + 8 * I')
        self.assertEqual(data['scratch'][0]['alignment'], 64)
        self.assertEqual(data['constraints']['external_runtime'], 'onednn_3_12_openmp')
        self.assertIn('-DUSE_ONEDNN', data['impl']['variants'][0]['compile_flags'])
        self.assertEqual(data['call_abi']['params'][-2]['source'], 'scratch:workspace')
        self.assertEqual(data['call_abi']['params'][-1]['source'], 'scratch_size:workspace')

    def test_planner_uses_shared_width_when_routed_width_differs(self):
        data = json.loads(MAP.read_text())
        sys.path.insert(0, str(ROOT / 'version/v8/scripts'))
        try:
            from build_ir_v8 import _bind_farskip_shared_width, _kernel_scratch_size_bytes
        finally:
            sys.path.pop(0)
        for name in ('moe_shared_intermediate_size',
                     'moe_shared_expert_intermediate_size'):
            with self.subTest(config_key=name):
                config = {'embed_dim': 2048, 'moe_intermediate_size': 1408,
                          name: 2816}
                params = {'_m': 3}
                _bind_farskip_shared_width(params, config)
                self.assertEqual(params['I'], 2816)
                actual = _kernel_scratch_size_bytes(data['scratch'][0], params, config)
                self.assertEqual(actual, 3 * (16 * 2816 + 8 * 2048) + 8 * 2816)
                self.assertGreater(actual, 3 * (16 * 1408 + 8 * 2048) + 8 * 1408)
        with self.assertRaisesRegex(RuntimeError, 'explicit positive shared'):
            _bind_farskip_shared_width({'_m': 3},
                                       {'embed_dim': 2048, 'moe_intermediate_size': 1408})
        with self.assertRaisesRegex(RuntimeError, 'conflicting shared expert'):
            _bind_farskip_shared_width({'_m': 3}, {
                'embed_dim': 2048,
                'moe_shared_intermediate_size': 2816,
                'moe_shared_expert_intermediate_size': 1408,
            })

    def test_emitted_call_and_scratch_share_one_width_for_both_config_names(self):
        sys.path.insert(0, str(ROOT / 'version/v8/scripts'))
        try:
            from build_ir_v8 import (
                _bind_farskip_shared_width, _kernel_scratch_size_bytes,
                generate_ir_lower_3,
            )
        finally:
            sys.path.pop(0)
        kernel_map = json.loads(MAP.read_text())
        observed = json.loads((ROOT / 'tests/fixtures/instella_onednn_expert/shared_lowered_op.json').read_text())
        for name in ('moe_shared_intermediate_size',
                     'moe_shared_expert_intermediate_size'):
            with self.subTest(config_key=name):
                config = {'embed_dim': 2048, 'moe_intermediate_size': 1408,
                          'n_shared_experts': 2, name: 2816}
                op = copy.deepcopy(observed)
                op['params'].update({'_m': 3, 'R': 3, 'I': 1408,
                                     'seq_len': 3})
                _bind_farskip_shared_width(op['params'], config)
                expected_bytes = _kernel_scratch_size_bytes(
                    kernel_map['scratch'][0], op['params'], config,
                )
                op['scratch'][0]['size'] = expected_bytes
                call = generate_ir_lower_3(
                    {'config': config, 'operations': [op]}, 'prefill',
                )['operations'][0]
                self.assertEqual(call['errors'], [])
                args = {arg['name']: arg['expr'] for arg in call['args']}
                self.assertEqual(args['intermediate_dim'], '2816')
                self.assertEqual(args['workspace_bytes'], str(expected_bytes))
                self.assertEqual(expected_bytes,
                                 3 * (16 * 2816 + 8 * 2048) + 8 * 2816)

    def test_compiled_group_and_rejection_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / 'probe'
            result = subprocess.run([
                'cc','-O2','-ffunction-sections','-fdata-sections',
                '-I',str(ROOT/'include'),'-Wl,--gc-sections',
                str(ROOT/'tests/fixtures/instella_onednn_expert/shared_contract_probe.c'),
                str(ROOT/'src/kernels/axpy_kernels.c'),'-o',str(binary),'-lm',
            ],capture_output=True,text=True,check=False)
            self.assertEqual(result.returncode,0,result.stderr)
            run = subprocess.run([str(binary)],capture_output=True,text=True,check=False)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            self.assertIn('PASS:',run.stdout)

if __name__ == '__main__':
    unittest.main()
