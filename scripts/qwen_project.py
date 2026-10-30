"""Retire SigLIP on a copy; retain Qwen, InsightFace and WISE OWLv2."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import uuid

try:
    from .action_search import read_snapshot, SETTINGS
    from .ingestion import prepare_versions
    from .release_retention import publish_pointer, prune_releases
except ImportError:
    from action_search import read_snapshot, SETTINGS
    from ingestion import prepare_versions
    from release_retention import publish_pointer, prune_releases

FACE = 'deepinsight/insightface/buffalo_l/_unknown'
OWL = 'transformers/owlv2/google/owlv2-base-patch16-ensemble'
RETIRED = ('mlfoundations/open_clip/ViT-B-16-SigLIP2-256/webli',)


def retained_fingerprint(project):
    files = list((project/'action-index').rglob('*')) + list((project/'store'/FACE).rglob('*'))
    files += list((project/'store'/SETTINGS['model']).rglob('*')) + [project/'qwen-encoding.json']
    files.append(project/'thumbs.db')
    fingerprints = {str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in files if p.is_file()}
    with sqlite3.connect(project/'metadata/internal.db') as db:
        for table, where in [('media', ''), ('source_collections', ''),
                             ('vectors', ' WHERE feature_extractor_id=?'),
                             ('vector_metadata_insightface', '')]:
            rows = db.execute(f'SELECT * FROM {table}{where} ORDER BY 1', (FACE,) if where else ()).fetchall()
            fingerprints[table] = hashlib.sha256(repr(rows).encode()).hexdigest()
        rows = db.execute('SELECT * FROM vectors WHERE feature_extractor_id=? ORDER BY id', (SETTINGS['model'],)).fetchall()
        fingerprints['qwen_vectors'] = hashlib.sha256(repr(rows).encode()).hexdigest()
    return fingerprints


def validate_project(project, require_owl=False):
    with sqlite3.connect(project/'metadata/internal.db') as db:
        models = {r[0] for r in db.execute('SELECT DISTINCT feature_extractor_id FROM vectors')}
    if models - {FACE, OWL, SETTINGS['model']} or any((project/'store'/model).exists() for model in RETIRED):
        raise ValueError('WISE 索引仍包含已停用模型，请先移除 SigLIP，保留 Qwen、OWLv2 与人脸索引')
    if require_owl and not (project/'store'/OWL/'index/video-IndexFlatIP.faiss').exists():
        raise ValueError('缺少 OWLv2 索引，请先运行 scripts/restore_owl_project.py 构建并发布')


def prepare_qwen_project(root):
    current = prepare_versions(root)
    if not any((current/'store'/model).exists() for model in RETIRED):
        validate_project(current)
        return current
    (root/'logs').mkdir(exist_ok=True)
    with (root/'logs/ingestion.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        read_snapshot(current, complete=True)
        before = retained_fingerprint(current)
        release = root/'projects/releases'/('qwen-'+uuid.uuid4().hex)
        stage = release/'demo'
        published = False
        try:
            shutil.copytree(current, stage)
            with sqlite3.connect(stage/'metadata/internal.db') as db:
                db.execute('PRAGMA foreign_keys=ON')
                db.executemany('DELETE FROM vectors WHERE feature_extractor_id=?', [(m,) for m in RETIRED])
            for model in RETIRED:
                shutil.rmtree(stage/'store'/model, ignore_errors=True)
            validate_project(stage)
            read_snapshot(stage, complete=True)
            if retained_fingerprint(stage) != before:
                raise RuntimeError('迁移改变了应保留的数据，未发布')
            (stage/'qwen-migration.json').write_text(json.dumps({
                'source':str(current), 'retired':RETIRED, 'retained':before}, ensure_ascii=False, indent=2))
            pointer = root/'projects/demo'
            if pointer.resolve() != current:
                raise RuntimeError('视频库版本已变化，未发布迁移')
            publish_pointer(root, stage)
            published = True
            prune_releases(root)
            return stage
        finally:
            if not published and release.exists(): shutil.rmtree(release)


if __name__ == '__main__':
    print(prepare_qwen_project(Path(__file__).resolve().parents[1]))
