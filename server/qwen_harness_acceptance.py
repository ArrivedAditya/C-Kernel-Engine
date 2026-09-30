"""Bounded, local Qwen Code serving acceptance with session-bound evidence.

This runner exercises the existing Qwen Code client against an already-running
OpenAI-compatible endpoint. It never starts a model or executes server code.
The disposable task is intentionally small: read, edit, validate, summarize.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import uuid
from urllib.request import urlopen


SOURCE_HTML = ('<!doctype html>\n<html lang="en">\n'
               '<head><meta charset="utf-8"><title>Serving probe</title></head>\n'
               '<body><main><h1>Before</h1><p>Local harness fixture.</p></main></body>\n'
               '</html>\n')
EXPECTED_HTML = SOURCE_HTML.replace('<h1>Before</h1>', '<h1>CKE tool round trip</h1>')
TEST_SCRIPT = ('from pathlib import Path\n'
               'text = Path("index.html").read_text(encoding="utf-8")\n'
               'assert "<h1>CKE tool round trip</h1>" in text\n'
               'print("PASS: expected heading present")\n')
TASK_PROMPT = ('Read index.html. Change only its h1 text from Before to CKE tool round trip '
               'using edit. Run python3 test_html.py with run_shell_command. Summarize the '
               'test result. Do not edit test_html.py or files outside this directory.')
EXPECTED_TOOLS = ('read_file', 'edit', 'run_shell_command')


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f'{path.name}:{number}: invalid JSON: {exc}') from exc
        if not isinstance(row, dict):
            raise ValueError(f'{path.name}:{number}: expected object')
        rows.append(row)
    return rows


def _session_file(qwen_bin: str, cwd: Path, session_id: str) -> Path:
    result = subprocess.run(
        [qwen_bin, 'sessions', 'list', '--json', '--limit', '100'],
        cwd=cwd, capture_output=True, text=True, timeout=30, check=True,
    )
    for line in result.stdout.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get('sessionId') == session_id:
            path = Path(row.get('filePath', ''))
            if path.is_file():
                return path
    raise ValueError(f'Qwen Code recording for session {session_id} is missing')


def _valid_tool_phases(order: list[tuple[str, str]], calls: list[dict]) -> bool:
    """Allow parallel reads, but require each phase to finish before the next."""
    by_id = {call['id']: call['name'] for call in calls}
    outstanding: set[str] = set()
    phase = 0
    phases = ('read_file', 'edit', 'run_shell_command')
    for event, call_id in order:
        if event == 'call':
            name = by_id.get(call_id)
            if name not in phases:
                return False
            next_phase = phases.index(name)
            if next_phase < phase or next_phase > phase + 1:
                return False
            if next_phase > phase:
                if outstanding:
                    return False
                phase = next_phase
            elif phase != 0 and outstanding:
                return False
            if call_id in outstanding:
                return False
            outstanding.add(call_id)
        elif event == 'result':
            if call_id not in outstanding:
                return False
            outstanding.remove(call_id)
        else:
            return False
    return phase == 2 and not outstanding


def _tool_arguments(value: object) -> dict | None:
    """Return the Qwen tool argument object, regardless of JSON key order."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def evaluate(events: list[dict], recording: list[dict], *, session_id: str,
             html: str, exit_code: int, expected_model: str,
             fixture_dir: Path | None = None) -> dict:
    """Compare Qwen's machine stream and retained conversation, fail closed."""
    errors: list[str] = []
    init = next((row for row in events if row.get('type') == 'system'
                 and row.get('subtype') == 'init'), None)
    if not init or init.get('session_id') != session_id:
        errors.append('stream_session_identity_missing_or_mismatched')
    if not init or init.get('model') != expected_model:
        errors.append('stream_model_identity_missing_or_mismatched')
    if any(row.get('session_id') not in (None, session_id) for row in events):
        errors.append('mixed_stream_sessions')
    if any(row.get('sessionId') not in (None, session_id) for row in recording):
        errors.append('mixed_recorded_sessions')

    calls: list[dict] = []
    results: dict[str, dict] = {}
    stream_order: list[tuple[str, str]] = []
    last_tool_result_index = -1
    tool_activity_after_terminal = False
    terminal_indices = [i for i, row in enumerate(events) if row.get('type') == 'result']
    first_terminal_index = terminal_indices[0] if terminal_indices else len(events)
    for event_index, row in enumerate(events):
        message = row.get('message') or {}
        if row.get('type') == 'assistant':
            for part in message.get('content', []):
                if part.get('type') == 'tool_use':
                    arguments = _tool_arguments(part.get('input'))
                    if arguments is None:
                        errors.append('invalid_stream_tool_arguments')
                    calls.append({'id': str(part.get('id', '')),
                                  'name': str(part.get('name', '')),
                                  'input': arguments or {}})
                    stream_order.append(('call', calls[-1]['id']))
                    tool_activity_after_terminal |= event_index > first_terminal_index
        elif row.get('type') == 'user':
            for part in message.get('content', []):
                if part.get('type') == 'tool_result':
                    call_id = str(part.get('tool_use_id', ''))
                    if call_id in results:
                        errors.append('duplicate_tool_result')
                    results[call_id] = part
                    stream_order.append(('result', call_id))
                    last_tool_result_index = event_index
                    tool_activity_after_terminal |= event_index > first_terminal_index
    names = tuple(call['name'] for call in calls)
    valid_order = (len(names) in (3, 4) and names[-2:] == EXPECTED_TOOLS[-2:]
                   and all(name == 'read_file' for name in names[:-2]))
    if not valid_order:
        errors.append('wrong_tool_sequence')
    if len({call['id'] for call in calls}) != len(calls) or any(not call['id'] for call in calls):
        errors.append('invalid_call_ids')
    if set(results) != {call['id'] for call in calls}:
        errors.append('missing_or_unmatched_tool_result')
    if not _valid_tool_phases(stream_order, calls):
        errors.append('tool_result_order_mismatch')
    if any(result.get('is_error') for result in results.values()):
        errors.append('tool_error')
    recorded_calls: list[tuple[str, str, dict | None]] = []
    recorded_results: dict[str, str] = {}
    recorded_order: list[tuple[str, str]] = []
    for row in recording:
        if row.get('type') == 'assistant':
            for part in (row.get('message') or {}).get('parts') or []:
                function = part.get('functionCall') if isinstance(part, dict) else None
                if isinstance(function, dict):
                    arguments = _tool_arguments(function.get('args'))
                    if arguments is None:
                        errors.append('invalid_recorded_tool_arguments')
                    recorded_calls.append((str(function.get('id', '')),
                                           str(function.get('name', '')), arguments))
                    recorded_order.append(('call', recorded_calls[-1][0]))
        elif row.get('type') == 'tool_result':
            result = row.get('toolCallResult') or {}
            if result.get('callId'):
                recorded_results[str(result['callId'])] = str(result.get('status', ''))
                recorded_order.append(('result', str(result['callId'])))
    if [(call_id, name) for call_id, name, _ in recorded_calls] != [
            (call['id'], call['name']) for call in calls]:
        errors.append('stream_recording_tool_calls_mismatch')
    if [args for _, _, args in recorded_calls] != [call['input'] for call in calls]:
        errors.append('stream_recording_tool_arguments_mismatch')
    if set(recorded_results) != set(results):
        errors.append('stream_recording_tool_results_mismatch')
    if not _valid_tool_phases(recorded_order, calls):
        errors.append('recorded_tool_result_order_mismatch')
    if any(status != 'success' for status in recorded_results.values()):
        errors.append('recorded_tool_error')
    if valid_order and 'PASS: expected heading present' not in str(
            results.get(calls[-1]['id'], {}).get('content', '')):
        errors.append('validator_result_missing')
    if fixture_dir is not None:
        fixture_dir = fixture_dir.resolve()
        read_index = False
        for call in calls:
            supplied = call['input']
            if call['name'] in ('read_file', 'edit'):
                path = Path(str(supplied.get('file_path', '')))
                if not path.is_absolute():
                    path = fixture_dir / path
                allowed = ({fixture_dir / 'index.html', fixture_dir / 'test_html.py'}
                           if call['name'] == 'read_file' else {fixture_dir / 'index.html'})
                if path.resolve() not in allowed:
                    errors.append('unexpected_tool_path')
                if call['name'] == 'read_file' and path.resolve() == fixture_dir / 'index.html':
                    read_index = True
            elif call['name'] == 'run_shell_command':
                if supplied.get('command') != 'python3 test_html.py':
                    errors.append('unexpected_validation_command')
                directory = supplied.get('directory')
                if directory and Path(str(directory)).resolve() != fixture_dir:
                    errors.append('unexpected_validation_directory')
        if not read_index:
            errors.append('index_read_missing')
    if html != EXPECTED_HTML:
        errors.append('artifact_mismatch')
    if exit_code != 0:
        errors.append('qwen_nonzero_exit')

    terminal = [row for row in events if row.get('type') == 'result']
    stream_final = terminal[-1] if terminal else None
    if not stream_final or stream_final.get('subtype') != 'success' or stream_final.get('is_error'):
        errors.append('stream_final_missing_or_failed')
    if len(terminal_indices) != 1 or (terminal_indices and terminal_indices[0] <= last_tool_result_index):
        errors.append('stream_terminal_order_mismatch')
    if tool_activity_after_terminal:
        errors.append('tool_activity_after_terminal')
    stream_text = str((stream_final or {}).get('result') or '').strip()
    if not stream_text:
        errors.append('stream_summary_missing')
    elif 'test_html.py' not in stream_text or 'PASS' not in stream_text:
        errors.append('summary_missing_test_outcome')

    last_tool_index = max((i for i, row in enumerate(recording)
                           if row.get('type') == 'tool_result'), default=-1)
    recorded_texts = []
    for row in recording[last_tool_index + 1:]:
        if row.get('type') != 'assistant' or row.get('provenance') != 'assistant_output':
            continue
        parts = (row.get('message') or {}).get('parts') or []
        recorded_texts.extend(str(part['text']).strip() for part in parts
                              if isinstance(part, dict) and part.get('text')
                              and not part.get('thought'))
    recorded_final = '\n'.join(recorded_texts).strip()
    if not recorded_final:
        errors.append('recorded_summary_missing')
    if stream_text and recorded_final and stream_text != recorded_final:
        errors.append('stream_recording_summary_mismatch')

    return {
        'status': 'pass' if not errors else 'fail',
        'errors': errors,
        'tool_sequence': [call['name'] for call in calls],
        'tool_call_ids': [call['id'] for call in calls],
        'stream_summary_present': bool(stream_text),
        'recorded_summary_present': bool(recorded_final),
        'turn_usage': [row['message'].get('usage') for row in events
                       if row.get('type') == 'assistant'
                       and isinstance(row.get('message'), dict)
                       and isinstance(row['message'].get('usage'), dict)],
        'client_usage': (stream_final or {}).get('usage'),
        'client_turns': (stream_final or {}).get('num_turns'),
        'artifact_sha256': hashlib.sha256(html.encode('utf-8')).hexdigest(),
    }


def run(args: argparse.Namespace) -> dict:
    if args.timeout <= 0 or args.max_output_tokens <= 0:
        raise ValueError('timeout and max-output-tokens must be positive')
    version = subprocess.run([args.qwen_bin, '--version'], capture_output=True,
                             text=True, timeout=10, check=True).stdout.strip()
    if version != args.expected_qwen_version:
        raise ValueError(f'Qwen Code version {version!r} differs from pinned '
                         f'{args.expected_qwen_version!r}')
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()):
        raise ValueError('run directory must be empty; existing artifacts are immutable')
    (run_dir / 'index.html').write_text(SOURCE_HTML, encoding='utf-8')
    (run_dir / 'test_html.py').write_text(TEST_SCRIPT, encoding='utf-8')
    session_id = str(uuid.uuid4())
    started = time.time()
    endpoint = args.endpoint.rstrip('/')
    with urlopen(endpoint + '/models', timeout=10) as response:
        model_list = json.load(response)
    advertised = next((row for row in model_list.get('data', [])
                       if row.get('id') == args.model), None)
    if advertised is None:
        raise ValueError(f'model {args.model!r} is not advertised by endpoint')
    cmd = [args.qwen_bin, '--bare', '--auth-type', 'openai-responses',
           '--model', args.model, '--session-id', session_id,
           '--chat-recording', '--allowed-tools', 'read_file', 'edit',
           'run_shell_command', '--max-tool-calls', '6',
           '--output-format', 'stream-json', '--prompt', TASK_PROMPT]
    env = os.environ.copy()
    env.update(OPENAI_BASE_URL=endpoint, OPENAI_MODEL=args.model,
               OPENAI_API_KEY=args.api_key,
               QWEN_CODE_MAX_OUTPUT_TOKENS=str(args.max_output_tokens))
    if args.settings:
        env['QWEN_CODE_SYSTEM_SETTINGS_PATH'] = str(Path(args.settings).resolve())
    with (run_dir / 'events.jsonl').open('wb') as output, (run_dir / 'stderr.log').open('wb') as errors:
        process = subprocess.Popen(cmd, cwd=run_dir, env=env, stdout=output,
                                   stderr=errors, start_new_session=True)
        try:
            exit_code = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            exit_code = 124
    try:
        session_path = _session_file(args.qwen_bin, run_dir, session_id)
        shutil.copyfile(session_path, run_dir / 'session.jsonl')
        recording = _read_jsonl(run_dir / 'session.jsonl')
    except (ValueError, subprocess.SubprocessError, OSError) as exc:
        recording = []
        recording_error = str(exc)
    else:
        recording_error = None
    try:
        events = _read_jsonl(run_dir / 'events.jsonl')
        event_error = None
    except (ValueError, OSError) as exc:
        events, event_error = [], str(exc)
    verdict = evaluate(events, recording, session_id=session_id,
                       html=(run_dir / 'index.html').read_text(encoding='utf-8'),
                       exit_code=exit_code, expected_model=args.model,
                       fixture_dir=run_dir)
    if recording_error:
        verdict['errors'].append('recording_unavailable: ' + recording_error)
    if event_error:
        verdict['errors'].append('event_stream_unavailable: ' + event_error)
    if verdict['errors']:
        verdict['status'] = 'fail'
    report = {
        'schema': 'cke.serving_harness_acceptance.v1',
        'run_id': run_dir.name,
        'session_id': session_id,
        'model': args.model,
        'endpoint': endpoint,
        'started_at_unix': started,
        'elapsed_seconds': time.time() - started,
        'qwen_code_version': version,
        'advertised_model': advertised,
        'exit_code': exit_code,
        'runner_sha256': _sha256(Path(__file__)),
        'fixture_sha256': _sha256(run_dir / 'test_html.py'),
        'events_sha256': _sha256(run_dir / 'events.jsonl'),
        'session_sha256': _sha256(run_dir / 'session.jsonl')
            if (run_dir / 'session.jsonl').is_file() else None,
        'result': verdict,
        'identity': {'level': 'endpoint_model_id_only',
                     'note': 'Loaded model/library/template hashes require a separate attestation.'},
        'certification': {'status': 'incomplete',
                          'reason': 'loaded_artifact_identity_unverified'},
    }
    (run_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', required=True, help='Base URL ending in /v1')
    parser.add_argument('--model', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--qwen-bin', default='qwen')
    parser.add_argument('--expected-qwen-version', default='0.24.6')
    parser.add_argument('--settings')
    parser.add_argument('--api-key', default='cke-local-only')
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--max-output-tokens', type=int, default=768)
    args = parser.parse_args()
    try:
        report = run(args)
    except Exception as exc:
        print(f'acceptance setup failed: {exc}', file=sys.stderr)
        return 2
    print(json.dumps({'report': str(Path(args.run_dir).resolve() / 'report.json'),
                      'status': report['result']['status'],
                      'errors': report['result']['errors']}))
    return 0 if report['result']['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
