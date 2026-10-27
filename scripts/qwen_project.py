"""Retire legacy encoders on a copy; preserve Qwen and face indexes byte for byte."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import uuid

try:
    from .action_search import read_snapshot
    from .ingestion import prepare_versions
except ImportError:
    from action_search import read_snapshot
    from ingestion import prepare_versions

FACE = 'deepinsight/insightface/buffalo_l/_unknown'
RETIRED = ('mlfoundations/open_clip/ViT-B-16-SigLIP2-256/webli',
           'transformers/owlv2/google/owlv2-base-patch16-ensemble')


def retained_fingerprint(project):
    files = list((project/'action-index').rglob('*')) + list((project/'store'/FACE).rglob('*'))
    files.append(project/'thumbs.db')
    fingerprints = {str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in files if p.is_file()}
    with sqlite3.connect(project/'metadata/internal.db') as db:
        for table, where in [('media', ''), ('source_collections', ''),
                             ('vectors', ' WHERE feature_extractor_id=?'),
                             ('vector_metadata_insightface', '')]:
            rows = db.execute(f'SELECT * FROM {table}{where} ORDER BY 1', (FACE,) if where else ()).fetchall()
            fingerprints[table] = hashlib.sha256(repr(rows).encode()).hexdigest()
    return fingerprints


def validate_face_project(project):
    with sqlite3.connect(project/'metadata/internal.db') as db:
        models = {r[0] for r in db.execute('SELECT DISTINCT feature_extractor_id FROM vectors')}
    if models - {FACE} or any((project/'store'/model).exists() for model in RETIRED):
        raise ValueError('WISE 索引仍包含已停用模型，请先迁移至 Qwen 片段与人脸索引')


def prepare_qwen_project(root):
    current = prepare_versions(root)
    if not any((current/'store'/model).exists() for model in RETIRED):
        validate_face_project(current)
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
                db.execute('DROP TABLE IF EXISTS vector_metadata_owlv2')
            for model in RETIRED:
                shutil.rmtree(stage/'store'/model, ignore_errors=True)
            validate_face_project(stage)
            read_snapshot(stage, complete=True)
            if retained_fingerprint(stage) != before:
                raise RuntimeError('迁移改变了应保留的数据，未发布')
            (stage/'qwen-migration.json').write_text(json.dumps({
                'source':str(current), 'retired':RETIRED, 'retained':before}, ensure_ascii=False, indent=2))
            pointer = root/'projects/demo'
            if pointer.resolve() != current:
                raise RuntimeError('视频库版本已变化，未发布迁移')
            temporary = root/'projects/demo.next'
            temporary.unlink(missing_ok=True)
            temporary.symlink_to(os.path.relpath(stage, pointer.parent))
            os.replace(temporary, pointer)
            published = True
            return stage
        finally:
            if not published and release.exists(): shutil.rmtree(release)


if __name__ == '__main__':
    print(prepare_qwen_project(Path(__file__).resolve().parents[1]))
