"""Download pinned Q8_0 weights only; verify upstream LFS SHA-256."""
import os
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '30'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

from pathlib import Path
import hashlib
import json
import time
import requests
from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent
REPO = 'mradermacher/Qwen3-VL-Embedding-8B-GGUF'
REVISION = 'ffa49879fdb91ed1a436fbc84f37b123f714bb13'
FILES = {
    'Qwen3-VL-Embedding-8B.Q8_0.gguf': 'c77299abab613f121ff918f17d085704952b21e986c73a71ec6cdc8a6e43e34b',
    'Qwen3-VL-Embedding-8B.mmproj-Q8_0.gguf': '36c44edfbe487f743128e369e53966f2e51e1641e604064f68a443c864b5bb9c',
}

if __name__ == '__main__':
    records = []
    for name, digest in FILES.items():
        print('DOWNLOAD', name, flush=True)
        for attempt in range(8):
            try:
                path = Path(hf_hub_download(REPO, name, revision=REVISION, local_dir=HERE / 'models'))
                break
            except requests.RequestException as exc:
                if attempt == 7:
                    raise
                print('RESUME', attempt + 1, type(exc).__name__, flush=True)
                time.sleep(3)
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(chunk)
        if h.hexdigest() != digest:
            raise ValueError(f'SHA-256 mismatch for {name}')
        records.append({'file': name, 'sha256': digest, 'bytes': path.stat().st_size})
        print('VERIFIED', name, flush=True)
    (HERE / 'model-provenance.json').write_text(json.dumps({
        'repo': REPO, 'revision': REVISION, 'files': records}, indent=2))
