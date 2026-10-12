import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

os.sched_setaffinity(0, {0})
base = Path('/data/cke/tts')
variants = {
    'hardware_fma': base / 'kokoro-post-fma-profile-20261012',
    'avx2_fma': base / 'kokoro-post-avx2-profile-20261012',
}
environment = os.environ.copy()
environment.update(CK_NUM_THREADS='1', OMP_NUM_THREADS='1')
record = {
    'host': os.uname().nodename,
    'affinity': sorted(os.sched_getaffinity(0)),
    'load_before': os.getloadavg(),
    'runs': [],
}
for repetition in range(-1, 5):
    order = ('hardware_fma', 'avx2_fma') if repetition % 2 == 0 else ('avx2_fma', 'hardware_fma')
    for variant in order:
        directory = variants[variant]
        output = directory / 'review-series.wav'
        started = time.perf_counter()
        result = subprocess.run(
            ['./native-kokoro-wav', 'prepared-native-noise.cktts', output.name, '7'],
            cwd=directory, env=environment, capture_output=True, text=True)
        elapsed = time.perf_counter() - started
        entry = {
            'repetition': repetition,
            'variant': variant,
            'seconds': round(elapsed, 6),
            'exit_code': result.returncode,
            'stdout': result.stdout.strip(),
            'stderr': result.stderr.strip(),
            'load_after': os.getloadavg(),
            'wav_sha256': hashlib.sha256(output.read_bytes()).hexdigest() if result.returncode == 0 else None,
        }
        record['runs'].append(entry)
        print(json.dumps(entry), flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)
record['load_after'] = os.getloadavg()
(base / 'kokoro-post-avx2-profile-20261012' / 'timing-series.json').write_text(json.dumps(record, indent=2) + '\n')
