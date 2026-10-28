"""Read-only fingerprint of production assets before/after the PPE experiment."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def capture():
    pointer = ROOT / 'projects/demo'
    project = pointer.resolve()
    files = list((project / 'action-index').glob('*'))
    files += list((project / 'store').rglob('*.faiss'))
    files += list((project / 'metadata').glob('*.db'))
    for folder in ['scripts', 'ui', 'wise/src']:
        files += [p for p in (ROOT / folder).rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts and p.suffix in {'.py', '.js', '.html', '.css'}]
    files += [ROOT / name for name in ['wise_config.yaml', 'requirements-macos.lock.txt']]
    state = {'project': str(project), 'pointer': os.readlink(pointer),
             'files': {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(files)) if p.is_file()}}
    for endpoint in ['actions/status', 'jobs']:
        state[endpoint] = json.load(urllib.request.urlopen('http://127.0.0.1:9670/mvp-api/' + endpoint, timeout=30))
    state['server_pids'] = subprocess.check_output(['lsof', '-t', '-iTCP:9670', '-sTCP:LISTEN'], text=True).split()
    return state


if __name__ == '__main__':
    name = sys.argv[1]
    path = OUT / f'production-{name}.json'
    if path.exists():
        raise SystemExit(f'Refusing to overwrite {path}')
    data = capture()
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(json.dumps({'snapshot': name, 'project': data['pointer'], 'hashed_files': len(data['files']),
                      'server_pids': data['server_pids'], 'status': data['actions/status']}, ensure_ascii=False))
