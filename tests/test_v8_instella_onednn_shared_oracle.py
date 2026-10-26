"""Optional layer-3 identical-input replay of Instella's shared FarSkip."""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/instella_onednn_expert'
LAYER = 3


def _child(provider: Path, bundle: Path, xray: Path, onednn: Path) -> None:
    import numpy as np

    def capture(side: str, name: str) -> np.ndarray:
        return np.fromfile(
            xray / side / f'layer_{LAYER:03d}_{name}_full.f32',
            dtype=np.float32,
        ).copy()

    hidden = capture('ck', 'ffn_norm')
    routed = capture('ck', 'moe_routed_output')
    residual = capture('ck', 'after_attn')
    assert all(np.array_equal(actual, capture('pytorch', name)) for actual, name in (
        (hidden, 'ffn_norm'), (routed, 'moe_routed_output'),
        (residual, 'after_attn'),
    )), 'shared-branch inputs differ before replay'
    assert hidden.size == routed.size == residual.size == 3 * 2048
    expected_main = capture('pytorch', 'layer_out')
    expected_free = capture('pytorch', 'routed_free_out')
    old_main = capture('ck', 'layer_out')
    old_free = capture('ck', 'routed_free_out')

    manifest = json.loads((bundle / 'weights_manifest.json').read_text())
    entries = {item['name']: item for item in manifest['entries']}
    weights = {}
    for kind in ('gate', 'up', 'down'):
        entry = entries[f'layer.{LAYER}.moe_shared_{kind}']
        assert entry['dtype'] == 'bf16'
        weights[kind] = np.memmap(
            bundle / 'weights.bump', dtype=np.uint16, mode='r',
            offset=entry['file_offset'], shape=tuple(entry['shape']),
        )
    library = ctypes.CDLL(str(provider))
    assert str(onednn.resolve()) in Path('/proc/self/maps').read_text(), \
        'pinned oneDNN backing is not loaded'
    new = library.ck_test_real_shared_farskip
    new.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 3 + [
        ctypes.c_void_p, ctypes.c_size_t,
    ]
    new.restype = ctypes.c_int
    old = library.ck_test_old_shared_farskip
    old.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 3
    old.restype = None
    rows, h, i = 3, 2048, 2816
    workspace_bytes = rows * (16 * i + 8 * h) + 8 * i
    raw = np.empty(workspace_bytes + 64, dtype=np.uint8)
    scratch = (raw.ctypes.data + 63) & ~63
    main = np.full(rows * h, -999.0, dtype=np.float32)
    free = np.full(rows * h, -999.0, dtype=np.float32)
    pointers = tuple(value.ctypes.data for value in (
        hidden, routed, residual, weights['gate'], weights['up'],
        weights['down'], main, free,
    ))
    minimum = rows * (12 * i + 4 * h + 4 * max(h, i)) + 8 * i
    assert new(*pointers, rows, h, i, scratch, minimum - 1) == -1
    assert np.all(main == -999.0) and np.all(free == -999.0), \
        'failed admission mutated output'
    assert new(*pointers, rows, h, i, scratch, workspace_bytes) == 0
    prior_main = np.empty_like(main)
    prior_free = np.empty_like(free)
    old_pointers = pointers[:6] + (prior_main.ctypes.data, prior_free.ctypes.data)
    old(*old_pointers, rows, h, i)
    counts = {
        'inputs_exact': int(sum(np.count_nonzero(actual == capture('pytorch', name))
                                for actual, name in ((hidden, 'ffn_norm'),
                                                     (routed, 'moe_routed_output'),
                                                     (residual, 'after_attn')))),
        'new_main_exact': int(np.count_nonzero(main == expected_main)),
        'new_free_exact': int(np.count_nonzero(free == expected_free)),
        'old_main_vs_capture_exact': int(np.count_nonzero(prior_main == old_main)),
        'old_free_vs_capture_exact': int(np.count_nonzero(prior_free == old_free)),
        'old_free_vs_publisher_exact': int(np.count_nonzero(prior_free == expected_free)),
        'elements_per_output': rows * h,
    }
    passed = (
        counts['inputs_exact'] == 3 * rows * h
        and counts['new_main_exact'] == rows * h
        and counts['new_free_exact'] == rows * h
        and counts['old_main_vs_capture_exact'] == rows * h
        and counts['old_free_vs_capture_exact'] == rows * h
        and counts['old_free_vs_publisher_exact'] < rows * h
    )
    print('INSTELLA_SHARED_ORACLE ' + json.dumps({
        'status': 'PASS' if passed else 'FAIL', 'counts': counts,
        'layer': LAYER, 'tokens': [1, 5, 7],
        'onednn_sha256': hashlib.sha256(onednn.read_bytes()).hexdigest(),
        'provider_source_sha256': os.environ['CK_COMPILED_PROVIDER_SOURCE_SHA256'],
        'captured_generated_source_sha256': hashlib.sha256(
            (bundle / 'model_v8.c').read_bytes()).hexdigest(),
        'max_cpu_isa': os.environ['DNNL_MAX_CPU_ISA'],
        'omp_threads': os.environ['OMP_NUM_THREADS'],
    }, sort_keys=True))
    assert passed, counts


class InstellaOneDNNSharedOracleTests(unittest.TestCase):
    def test_captured_three_row_shared_boundary(self):
        bundle = Path(os.getenv('CK_INSTELLA_ONEDNN_BUNDLE', ''))
        xray = Path(os.getenv('CK_INSTELLA_ONEDNN_SHARED_XRAY', ''))
        onednn = Path(os.getenv('CK_ONEDNN_LIBRARY', ''))
        sleef = Path(os.getenv('CK_SLEEF_LIBRARY', ''))
        includes = os.getenv('CK_ONEDNN_INCLUDE', '')
        missing = [name for name, path in (
            ('CK_INSTELLA_ONEDNN_BUNDLE', bundle / 'weights.bump'),
            ('CK_INSTELLA_ONEDNN_SHARED_XRAY', xray / 'ck/layer_003_ffn_norm_full.f32'),
            ('CK_ONEDNN_LIBRARY', onednn),
            ('CK_SLEEF_LIBRARY', sleef),
        ) if not path.is_file()]
        if not includes:
            missing.append('CK_ONEDNN_INCLUDE')
        if not Path('/proc/cpuinfo').is_file() or \
                'avx512_bf16' not in Path('/proc/cpuinfo').read_text():
            missing.append('AVX512_BF16 CPU')
        if missing:
            reason = 'pinned shared replay unavailable: ' + ', '.join(missing)
            if os.getenv('CK_REQUIRE_ONEDNN_SHARED_ORACLE') == '1':
                self.fail(reason)
            self.skipTest(reason)
        include_flags = [
            item if item.startswith('-I') else '-I' + item
            for item in shlex.split(includes)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            provider = Path(temporary) / 'libshared_farskip.so'
            command = [
                'cc', '-O2', '-shared', '-fPIC', '-fopenmp', '-DUSE_ONEDNN',
                '-fvisibility=hidden', '-ffunction-sections', '-fdata-sections',
                '-Wl,--gc-sections', '-I' + str(ROOT / 'include'), *include_flags,
                str(ROOT / 'src/kernels/axpy_kernels.c'),
                str(ROOT / 'src/kernels/gemm_kernels_bf16.c'),
                str(ROOT / 'src/kernels/swiglu_kernels.c'),
                str(FIXTURE / 'real_shared_entry.c'),
                '-L' + str(onednn.parent), '-ldnnl',
                '-Wl,-rpath,' + str(onednn.parent),
                '-lm', '-ldl', '-lpthread', '-o', str(provider),
            ]
            built = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            env = os.environ.copy()
            env['OMP_NUM_THREADS'] = '8'
            env['DNNL_MAX_CPU_ISA'] = 'AVX512_CORE_BF16'
            env['CK_COMPILED_PROVIDER_SOURCE_SHA256'] = hashlib.sha256(
                (ROOT / 'src/kernels/axpy_kernels.c').read_bytes()
            ).hexdigest()
            env['LD_LIBRARY_PATH'] = str(onednn.parent) + os.pathsep + env.get('LD_LIBRARY_PATH', '')
            result = subprocess.run(
                [sys.executable, __file__, '--child', str(provider), str(bundle),
                 str(xray), str(onednn)], env=env, capture_output=True,
                text=True, timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('"status": "PASS"', result.stdout)
            print(result.stdout.strip())

if __name__ == '__main__':
    if len(sys.argv) == 6 and sys.argv[1] == '--child':
        _child(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]))
    else:
        unittest.main()
