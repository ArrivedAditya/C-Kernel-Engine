"""Source-path interventions retain all evidence when the oracle drifts."""

import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from tests.diagnose_kokoro_source_waveform import baseline_reproduction


ROOT = Path(__file__).resolve().parents[1]
REPRODUCTION = (
    "CKE_TTS_RUN_SOURCE_IMPACT_FULL=1 "
    "CKE_TTS_SOURCE_IMPACT_MODEL_DIR=<pinned-model> "
    "CKE_TTS_SOURCE_IMPACT_CAPTURE_DIR=<pinned-capture> "
    "python3 -m unittest tests.test_v8_kokoro_source_impact_diagnostic"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def emit(name, status, reason="", **details):
    report = {
        "case_id": f"kokoro.source-impact.{name}-v1",
        "name": f"Kokoro source impact {name}",
        "provider": "pinned_full_model_diagnostic",
        "dtype": "fp32", "direction": "inference",
        "oracle": "pinned PyTorch full model and retained capture",
        "gate": "diagnostic", "blocking": False,
        "status": status, "reproduction_command": REPRODUCTION,
    }
    if reason:
        report["reason"] = reason
    report.update(details)
    print("CKE_NUMERICAL_CASE " + json.dumps(report, sort_keys=True))


class KokoroSourceImpactDiagnosticTest(unittest.TestCase):
    def test_baseline_reports_every_checkpoint_before_failure(self):
        reference = {name: np.array([0., 1.], np.float32)
                     for name in ('f0', 'source', 'first_source_conv', 'waveform')}
        actual = {name: value.copy() for name, value in reference.items()}
        actual['source'][1] = 2.
        actual['waveform'][0] = -1.
        report = baseline_reproduction(reference, actual)
        self.assertEqual(report['status'], 'fail')
        self.assertEqual(set(report['checkpoints']), set(reference))
        self.assertEqual(report['checkpoints']['f0']['max_abs'], 0.)
        self.assertEqual(report['checkpoints']['source']['max_abs'], 1.)
        self.assertEqual(report['checkpoints']['waveform']['max_abs'], 1.)
        self.assertEqual(baseline_reproduction(reference, reference)['status'],
                         'pass')

    def test_asset_coverage_is_explicit(self):
        requested = bool(os.environ.get('CKE_TTS_RUN_SOURCE_IMPACT_FULL'))
        required = ('CKE_TTS_SOURCE_IMPACT_MODEL_DIR',
                    'CKE_TTS_SOURCE_IMPACT_CAPTURE_DIR')
        missing = [name for name in required if not os.environ.get(name)]
        if requested and not missing:
            return
        reason = ('source-impact asset run not requested' if not requested else
                  'missing assets: ' + ', '.join(missing))
        emit('live-oracle-reproduction', 'not_tested', reason)

    @unittest.skipUnless(os.environ.get('CKE_TTS_RUN_SOURCE_IMPACT_FULL'),
                         'source-impact asset run not requested')
    def test_live_interventions_complete_even_if_oracle_drifts(self):
        required = ('CKE_TTS_SOURCE_IMPACT_MODEL_DIR',
                    'CKE_TTS_SOURCE_IMPACT_CAPTURE_DIR')
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            self.skipTest('missing assets: ' + ', '.join(missing))
        command = [sys.executable, '-m', 'tests.diagnose_kokoro_source_waveform',
                   '--model-dir', os.environ[required[0]],
                   '--capture-dir', os.environ[required[1]]]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True,
                                   text=True, timeout=240, check=False)
        records = []
        for line in completed.stdout.splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and 'case' in item:
                records.append(item)
        by_case = {item['case']: item for item in records}
        required_cases = {
            'generated_f0_error_decomposition',
            'pinned_full_model_reproduction',
            'generated_f0_into_pinned_full_model',
            'cke_source_arithmetic_with_identical_generated_f0',
            'generated_f0_phase_sensitivity',
            'pinned_source_cke_stft',
            'generated_source_torch_stft',
            'generated_source_cke_stft',
            'generated_source_stft_incremental',
        }
        self.assertTrue(required_cases <= by_case.keys(),
                        (sorted(required_cases - by_case.keys()), completed.stderr))
        baseline = by_case['pinned_full_model_reproduction']
        self.assertEqual(completed.returncode,
                         1 if baseline['status'] == 'fail' else 0,
                         completed.stderr)
        emit('live-oracle-reproduction', baseline['status'],
             'retained PyTorch baseline differs on this dispatch'
             if baseline['status'] == 'fail' else '',
             max_diff=baseline['checkpoints']['waveform']['max_abs'],
             tolerance=0,
             checkpoints=baseline['checkpoints'],
             interventions={name: by_case[name] for name in
                            sorted(required_cases -
                                   {'pinned_full_model_reproduction'})},
             model_config_sha256=sha256_file(
                 Path(os.environ[required[0]]) / 'config.json'),
             model_weights_sha256=sha256_file(
                 Path(os.environ[required[0]]) / 'kokoro-v1_0.pth'),
             voice_sha256=sha256_file(
                 Path(os.environ[required[0]]) / 'voices' / 'af_heart.pt'),
             capture_manifest_sha256=hashlib.sha256(
                 (Path(os.environ[required[1]]) / 'manifest.json').read_bytes()
             ).hexdigest(),
             sample_count=baseline['sample_count'])


if __name__ == '__main__':
    unittest.main()
