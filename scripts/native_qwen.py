"""Canonical WISE Qwen storage; legacy snapshots are migration inputs only."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time
import uuid

import faiss
import numpy as np

if __package__:
    from .action_search import SETTINGS, media_catalog, plan, read_legacy_snapshot
    from .release_retention import publish_pointer, prune_releases
else:
    from action_search import SETTINGS, media_catalog, plan, read_legacy_snapshot
    from release_retention import publish_pointer, prune_releases

QWEN = SETTINGS['model']
MANIFEST = 'qwen-encoding.json'
STORE_FILES = ('features/video-000000.faiss', 'index/video-IndexFlatIP.faiss')


def segment_key(segment):
    return (segment['media_id'], segment['ts'], segment['te'])


def media_signature(media):
    return hashlib.sha256(json.dumps(media, sort_keys=True).encode()).hexdigest()


def read_native_snapshot(project, complete=False):
    project = Path(project)
    marker = project/MANIFEST
    if not marker.exists():
        if complete: raise ValueError('Qwen 编码配置记录尚未建立')
        return None
    config = json.loads(marker.read_text())
    if config['settings'] != SETTINGS: raise ValueError('Qwen 编码配置不匹配，需要重新建立')
    with sqlite3.connect(project/'metadata/internal.db') as db:
        rows = db.execute('SELECT id,media_id,timestamp,end_timestamp,modality FROM vectors '
                          'WHERE feature_extractor_id=? ORDER BY media_id,timestamp,id', (QWEN,)).fetchall()
    segments = [dict(media_id=r[1],ts=r[2],te=r[3]) for r in rows]
    ids = np.asarray([r[0] for r in rows], dtype='int64')
    catalog = media_catalog(project)
    indexed = {s['media_id'] for s in segments}
    media = [m for m in catalog if m['id'] in indexed]
    if any(r[4] != 'VIDEO' for r in rows) or segments != plan(media):
        raise ValueError('WISE Qwen 片段时间覆盖或媒体不一致')
    if any(config['media_signatures'].get(str(m['id'])) != media_signature(m) for m in media):
        raise ValueError('Qwen 索引媒体与视频库不一致')
    if complete and media != catalog: raise ValueError('尚有视频缺少动作索引，未发布')
    folder = project/'store'/QWEN
    index = faiss.read_index(str(folder/STORE_FILES[1]))
    actual = faiss.vector_to_array(index.id_map)
    if index.d != config['dimension'] or len(actual) != len(ids) or set(actual) != set(ids):
        raise ValueError('WISE Qwen 向量 ID 与元数据不一致')
    matrix = index.reconstruct_batch(ids) if len(ids) else np.empty((0,index.d), dtype='float32')
    if not np.isfinite(matrix).all(): raise ValueError('WISE Qwen 包含无效向量')
    positions = {int(vid):i for i,vid in enumerate(ids)}
    featured = []
    for path in sorted((folder/'features').glob('video-*.faiss')):
        feature = faiss.read_index(str(path));fids = faiss.vector_to_array(feature.id_map)
        if feature.d != index.d or any(int(i) not in positions for i in fids):
            raise ValueError('WISE Qwen 特征与索引不一致')
        if len(fids) and not np.array_equal(feature.reconstruct_batch(fids), matrix[[positions[int(i)] for i in fids]]):
            raise ValueError('WISE Qwen 特征与索引向量不一致')
        featured.extend(int(i) for i in fids)
    if len(featured) != len(ids) or set(featured) != set(ids):
        raise ValueError('WISE Qwen 特征 ID 与元数据不一致')
    # Compatibility view for in-memory person/scene scoring; never persisted.
    positional = faiss.IndexFlatIP(index.d);positional.add(matrix)
    return {'settings':SETTINGS,'media':media,'segments':segments,'built_at':config['built_at']}, matrix, positional


def validate_native_qwen(project):
    read_native_snapshot(project, complete=True)


def write_native_qwen(project, catalog, segments, matrix):
    """Write to an unpublished stage, retaining IDs of unchanged windows."""
    project = Path(project)
    order = sorted(range(len(segments)), key=lambda i: segment_key(segments[i]))
    segments = [segments[i] for i in order]
    matrix = np.asarray(matrix, dtype='float32')
    if matrix.size == 0 and not segments and matrix.ndim == 1:
        marker=project/MANIFEST
        dimension=json.loads(marker.read_text())['dimension'] if marker.exists() else 2048
        matrix=np.empty((0,dimension),dtype='float32')
    if matrix.ndim != 2 or len(matrix) != len(segments) or not np.isfinite(matrix).all():
        raise ValueError('Qwen 向量与片段记录不一致')
    matrix = np.ascontiguousarray(matrix[order])
    if catalog != media_catalog(project) or segments != plan(catalog):
        raise ValueError('Qwen 片段时间覆盖或媒体不完整，未发布')
    wanted = {segment_key(s) for s in segments}
    with sqlite3.connect(project/'metadata/internal.db') as db:
        db.execute('PRAGMA foreign_keys=ON')
        rows = db.execute('SELECT id,media_id,timestamp,end_timestamp FROM vectors WHERE feature_extractor_id=?', (QWEN,)).fetchall()
        existing = {}
        for vid,mid,start,end in rows:
            key = (mid,start,end)
            if key not in wanted or key in existing: db.execute('DELETE FROM vectors WHERE id=?',(vid,))
            else: existing[key] = vid
        ids = []
        for segment in segments:
            key = segment_key(segment);vid = existing.get(key)
            if vid is None:
                vid = db.execute('INSERT INTO vectors(modality,feature_extractor_id,media_id,timestamp,end_timestamp) VALUES(?,?,?,?,?)', ('VIDEO',QWEN,*key)).lastrowid
            ids.append(vid)
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(matrix.shape[1]))
    if ids: index.add_with_ids(matrix, np.asarray(ids,dtype='int64'))
    folder = project/'store'/QWEN
    shutil.rmtree(folder, ignore_errors=True)
    for relative in STORE_FILES:
        path = folder/relative;path.parent.mkdir(parents=True,exist_ok=True)
        faiss.write_index(index,str(path))
    marker = {'settings':SETTINGS,'dimension':matrix.shape[1],'built_at':time.time(),
              'media_signatures':{str(m['id']):media_signature(m) for m in catalog}}
    (project/MANIFEST).write_text(json.dumps(marker,ensure_ascii=False,indent=2))
    validate_native_qwen(project)
    shutil.rmtree(project/'action-index',ignore_errors=True)


def sync_native_qwen(project):
    """One-time legacy migration, kept for old maintenance scripts."""
    project = Path(project)
    if (project/MANIFEST).exists():
        validate_native_qwen(project)
        shutil.rmtree(project/'action-index',ignore_errors=True)
        return
    meta,matrix,_ = read_legacy_snapshot(project,complete=True)
    write_native_qwen(project,meta['media'],meta['segments'],matrix)


def reset_native_qwen(project):
    """Discard a damaged Qwen store on an unpublished rebuild stage only."""
    project = Path(project)
    with sqlite3.connect(project/'metadata/internal.db') as db:
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('DELETE FROM vectors WHERE feature_extractor_id=?',(QWEN,))
    shutil.rmtree(project/'store'/QWEN,ignore_errors=True)
    shutil.rmtree(project/'action-index',ignore_errors=True)
    (project/MANIFEST).unlink(missing_ok=True)


@contextmanager
def suspend_native_store(project):
    """Hide Qwen from WISE frame-model discovery without discarding vectors."""
    project = Path(project);folder = project/'store'/QWEN;parked = project/'.qwen-store-pending'
    if parked.exists(): raise ValueError('Qwen 暂存目录已存在，拒绝覆盖')
    moved = folder.exists()
    if moved: folder.rename(parked)
    try: yield
    finally:
        if moved:
            if folder.exists(): raise RuntimeError('帧提取意外创建 Qwen 索引，未发布')
            parked.rename(folder)


def prepare_native_qwen_project(root):
    if __package__:
        from .qwen_project import prepare_qwen_project,validate_project
    else:
        from qwen_project import prepare_qwen_project,validate_project
    current = prepare_qwen_project(root)
    if (current/MANIFEST).exists():
        # The normal background backfill path handles incomplete/damaged stores
        # on a copy. Do not silently fall back to an old duplicate snapshot.
        return current
    if not (current/'action-index').exists():
        return current
    try:
        read_legacy_snapshot(current,complete=True)
    except (ValueError,FileNotFoundError):
        if not (current/'store'/QWEN).exists():
            return current  # Legacy partial/first-start backfill remains available.
        raise
    (root/'logs').mkdir(exist_ok=True)
    with (root/'logs/ingestion.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        release = root/'projects/releases'/('native-qwen-'+uuid.uuid4().hex)
        stage = release/'demo';published = False
        try:
            shutil.copytree(current,stage)
            sync_native_qwen(stage)
            validate_project(stage,require_owl=True)
            pointer = root/'projects/demo'
            if pointer.resolve() != current: raise RuntimeError('视频库版本已变化，未发布迁移')
            publish_pointer(root, stage)
            published = True
            prune_releases(root)
            return stage
        finally:
            if not published and release.exists(): shutil.rmtree(release)
