"""Download the two pinned public checkpoints and verify their exact content."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request

HERE = Path(__file__).resolve().parent
MODELS = [
    ('yolo11-ppe', 'melihuzunoglu/ppe-detection', 'aed7807c358836eef5a6f67af27dd970ade6fb18',
     'a00b6fce124e63c5d23f44792593983b70e646008b58bc54cf5b0a1c87ba2119'),
    ('yolov8-ppe', 'killuminati1/construction-ppe-yolov8', '06ce70134a78f31881e2f9a223719d60a0a9a6a9',
     '31ef3ca04a17cf545f3fcfc64c4af8993a41d52ccc460e82aff01d5354603533'),
]


if __name__ == '__main__':
    (HERE / 'models').mkdir(exist_ok=True)
    records = []
    for name, repo, revision, expected in MODELS:
        path = HERE / 'models' / f'{name}.pt'
        url = f'https://huggingface.co/{repo}/resolve/{revision}/best.pt'
        started = time.perf_counter()
        cached = path.exists()
        if not cached:
            temporary = path.with_suffix('.part')
            urllib.request.urlretrieve(url, temporary)
            assert hashlib.sha256(temporary.read_bytes()).hexdigest() == expected
            temporary.replace(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == expected, path
        records.append({'id': name, 'repo': repo, 'revision': revision, 'filename': 'best.pt',
                        'url': url, 'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest,
                        'cached': cached, 'download_or_check_seconds': time.perf_counter() - started})
        print(name, path.stat().st_size, digest)
    if not (HERE / 'models.json').exists():
        (HERE / 'models.json').write_text(json.dumps(records, indent=2))
