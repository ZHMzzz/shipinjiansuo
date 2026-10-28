"""Offline comparison only. Send synthetic queries, never videos or indexes.

Credential is read from DEEPSEEK_API_KEY or an echo-disabled TTY prompt;
it is never saved in the results. No automatic request retries.
"""
import getpass
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'text-query-parser'))
from run import fingerprint, listener, validate

DIR = Path(__file__).resolve().parent
ROOT = DIR.parent.parent

def main():
    output = DIR / 'parser-results.json'
    previous = json.loads(output.read_text()) if output.exists() else None
    design = json.loads((DIR / 'design.json').read_text())
    settings = {
        'endpoint': 'https://api.deepseek.com/chat/completions',
        'model': 'deepseek-flash', 'thinking': {'type': 'disabled'},
        'temperature': 0, 'response_format': {'type': 'json_object'},
        'max_tokens': 640, 'stream': False,
        'expected_calls': len(design['cases']) + 1, 'automatic_retries': 0,
        'note': 'Frozen retrieval dataset and same parser prompt as previous text benchmark. JSON mode is an API-side format constraint absent from local baseline. No videos, indexes, or gold criteria sent.',
    }
    settings_path = DIR / 'api-settings.json'
    settings_path.write_text(json.dumps(settings, ensure_ascii=False, indent=2) + '\n')
    key = os.environ.get('DEEPSEEK_API_KEY') or getpass.getpass('DeepSeek API key (hidden): ')
    if not key.strip():
        raise SystemExit('No API key supplied')
    project = (ROOT / 'projects/demo').resolve()
    report = {'settings': settings, 'started_utc': datetime.now(timezone.utc).isoformat(),
              'design_sha256': hashlib.sha256((DIR / 'design.json').read_bytes()).hexdigest(),
              'index_before': fingerprint(project), 'server_pid_before': listener(), 'results': []}
    if previous:
        assert previous['design_sha256'] == report['design_sha256']
        assert previous['index_before'] == report['index_before']
        report = previous
        report.setdefault('runner_notes', []).append('Resumed after offline runner missing kind field; completed requests preserved, no retry of successful calls.')
    schedule = [('warmup', {'id': 'warmup', 'query': design['warmup'], 'kind': 'warmup'})]
    schedule += [('main', c) for c in design['cases']]
    by_id = {c['id']: c for c in design['cases']}
    for repeat in range(1, 3):
        schedule += [(f'repeat{repeat}', by_id[i]) for i in design['repeats']]
    completed = {(r['phase'], r['id']) for r in report['results']}
    schedule = [(phase, case) for phase, case in schedule if (phase, case['id']) not in completed]
    base = {k: settings[k] for k in ['model', 'thinking', 'temperature', 'response_format', 'max_tokens', 'stream']}
    with httpx.Client(timeout=httpx.Timeout(45, connect=15), headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}, follow_redirects=False) as client:
        for phase, case in schedule:
            payload = dict(base, messages=[{'role': 'system', 'content': design['system_prompt']}, {'role': 'user', 'content': case['query']}])
            row = {'phase': phase, 'id': case['id'], 'kind': case.get('kind', case.get('cohort')), 'query': case['query'],
                   'started_utc': datetime.now(timezone.utc).isoformat(), 'schema_valid': False}
            t = time.perf_counter()
            fatal = False
            try:
                response = client.post(settings['endpoint'], json=payload)
                row['request_seconds'] = time.perf_counter() - t
                row['http_status'] = response.status_code
                if response.status_code != 200:
                    row['error'] = f'HTTP {response.status_code}'
                    fatal = response.status_code in (400, 401, 402, 403, 404)
                else:
                    body = response.json()
                    choice = body['choices'][0]
                    raw = choice['message'].get('content') or ''
                    # Model responses must not accidentally persist a credential.
                    raw = raw.replace(key, '[REDACTED]')
                    parsed, valid = validate(raw)
                    row.update(raw=raw, parsed=parsed, schema_valid=valid, finish_reason=choice['finish_reason'],
                               response_model=body.get('model'), usage=body.get('usage'),
                               system_fingerprint=body.get('system_fingerprint'), reasoning_returned=bool(choice['message'].get('reasoning_content')))
            except (httpx.HTTPError, ValueError, KeyError, IndexError) as exc:
                # Do not save exception strings, which may contain request details.
                row['error'] = type(exc).__name__
            row['total_seconds'] = time.perf_counter() - t
            report['results'].append(row)
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
            print(phase, case['id'], 'schema', row['schema_valid'], 'seconds', round(row['total_seconds'], 3), 'status', row.get('http_status', row.get('error')), flush=True)
            if fatal:
                break
    report['index_after'] = fingerprint(project)
    report['pointer_unchanged'] = project == (ROOT / 'projects/demo').resolve()
    report['server_pid_after'] = listener()
    report['completed_utc'] = datetime.now(timezone.utc).isoformat()
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    assert report['index_before'] == report['index_after'] and report['pointer_unchanged']

if __name__ == '__main__':
    main()
