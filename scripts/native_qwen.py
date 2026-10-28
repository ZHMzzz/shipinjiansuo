"""Expose existing Qwen segment vectors to WISE; never re-encode video here."""
import fcntl
import os
import shutil
import sqlite3
import uuid

import faiss
import numpy as np

if __package__:
    from .action_search import SETTINGS, read_snapshot
    from .qwen_project import prepare_qwen_project, validate_project
else:
    from action_search import SETTINGS, read_snapshot
    from qwen_project import prepare_qwen_project, validate_project

QWEN = SETTINGS['model']
STORE_FILES = ('features/video-000000.faiss', 'index/video-IndexFlatIP.faiss')


def segment_key(segment):
    return (segment['media_id'], segment['ts'], segment['te'])


def validate_native_qwen(project):
    meta, matrix, _ = read_snapshot(project, complete=True)
    with sqlite3.connect(project/'metadata/internal.db') as db:
        rows = db.execute('SELECT id,media_id,timestamp,end_timestamp,modality FROM vectors '
                          'WHERE feature_extractor_id=? ORDER BY media_id,timestamp', (QWEN,)).fetchall()
    if [r[1:4] for r in rows] != [segment_key(s) for s in meta['segments']] or any(r[4] != 'VIDEO' for r in rows):
        raise ValueError('WISE Qwen 元数据与片段索引不一致')
    ids = np.asarray([r[0] for r in rows], dtype='int64')
    for relative in STORE_FILES:
        index = faiss.read_index(str(project/'store'/QWEN/relative))
        if index.d != matrix.shape[1] or not np.array_equal(faiss.vector_to_array(index.id_map), ids):
            raise ValueError('WISE Qwen 向量 ID 与元数据不一致')
        if len(ids) and not np.array_equal(index.reconstruct_batch(ids), matrix):
            raise ValueError('WISE Qwen 向量与已有片段向量不一致')


def sync_native_qwen(project):
    """Write only to an unpublished stage. Retain IDs for unchanged segments."""
    meta, matrix, _ = read_snapshot(project, complete=True)
    wanted = {segment_key(s) for s in meta['segments']}
    with sqlite3.connect(project/'metadata/internal.db') as db:
        db.execute('PRAGMA foreign_keys=ON')
        rows = db.execute('SELECT id,media_id,timestamp,end_timestamp FROM vectors '
                          'WHERE feature_extractor_id=?', (QWEN,)).fetchall()
        existing = {}
        for vid, mid, start, end in rows:
            key = (mid, start, end)
            if key not in wanted or key in existing:
                db.execute('DELETE FROM vectors WHERE id=?', (vid,))
            else:
                existing[key] = vid
        ids = []
        for segment in meta['segments']:
            key = segment_key(segment)
            vid = existing.get(key)
            if vid is None:
                vid = db.execute('INSERT INTO vectors(modality,feature_extractor_id,media_id,timestamp,end_timestamp) '
                                 'VALUES(?,?,?,?,?)', ('VIDEO', QWEN, *key)).lastrowid
            ids.append(vid)
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(matrix.shape[1]))
    if ids:
        index.add_with_ids(np.ascontiguousarray(matrix, dtype='float32'), np.asarray(ids, dtype='int64'))
    folder = project/'store'/QWEN
    # This model's feature store is rebuilt from the complete existing snapshot.
    shutil.rmtree(folder, ignore_errors=True)
    for relative in STORE_FILES:
        path = folder/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(path))
    validate_native_qwen(project)


def prepare_native_qwen_project(root):
    current = prepare_qwen_project(root)
    try:
        read_snapshot(current, complete=True)
    except (ValueError, FileNotFoundError):
        # Keep the existing first-start/backfill workflow available. Until its
        # complete stage is published, pure text queries report not ready.
        if not (current/'store'/QWEN).exists():
            return current
        raise
    try:
        validate_native_qwen(current)
        return current
    except (ValueError, RuntimeError, FileNotFoundError):
        pass
    (root/'logs').mkdir(exist_ok=True)
    with (root/'logs/ingestion.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        release = root/'projects/releases'/('native-qwen-'+uuid.uuid4().hex)
        stage = release/'demo'
        published = False
        try:
            shutil.copytree(current, stage)
            sync_native_qwen(stage)
            validate_project(stage, require_owl=True)
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
            if not published and release.exists():
                shutil.rmtree(release)
