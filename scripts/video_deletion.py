"""Delete a media identity from staged and historical WISE/Qwen snapshots."""
import hashlib
import json
import os
import shutil
import sqlite3
import time
import uuid
from pathlib import Path

import faiss
import numpy as np
from fastapi import HTTPException

if __package__:
    from .action_search import read_snapshot
    from .qwen_project import validate_project
else:
    from action_search import read_snapshot
    from qwen_project import validate_project


def media_identity(row):
    return hashlib.sha256(json.dumps([row['location'],row['path'],
        bytes(row['checksum']).hex() if row['checksum'] else None],ensure_ascii=False).encode()).hexdigest()


def media_revision(row, project):
    # A version-bound token also prevents re-imports/ID reuse from accepting a
    # stale delete confirmation, even when the file bytes are identical.
    return hashlib.sha256((str(project.resolve())+media_identity(row)).encode()).hexdigest()


def read_media(project, media_id):
    with sqlite3.connect(project/'metadata/internal.db') as db:
        db.row_factory=sqlite3.Row
        row=db.execute('SELECT m.*,s.location FROM media m JOIN source_collections s ON s.id=m.source_collection_id WHERE m.id=?',(media_id,)).fetchone()
        return dict(row) if row else None


def write_record(stage, record):
    temporary=stage/'deletion.json.tmp'
    temporary.write_text(json.dumps(record,ensure_ascii=False,indent=2))
    os.replace(temporary,stage/'deletion.json')


def prune_project(project, media_id):
    """Idempotent removal, metadata last so an interrupted cleanup can resume."""
    with sqlite3.connect(project/'metadata/internal.db') as db:
        vector_ids=np.array([r[0] for r in db.execute('SELECT id FROM vectors WHERE media_id=?',(media_id,))],dtype='int64')
        qwen_count=db.execute('SELECT count(*) FROM vectors WHERE media_id=? AND feature_extractor_id=?',(media_id,'hf/Qwen/Qwen3-VL-Embedding/2B')).fetchone()[0]
    removed={'vectors':len(vector_ids),'segments':qwen_count}
    if len(vector_ids):
        for path in (project/'store').rglob('*.faiss'):
            index=faiss.read_index(str(path))
            if not hasattr(index,'id_map'): raise RuntimeError('仅支持当前 WISE IDMap 索引，未删除视频')
            ids=faiss.vector_to_array(index.id_map)
            selected=np.intersect1d(ids,vector_ids)
            if not len(selected): continue
            index.remove_ids(selected)
            if np.isin(faiss.vector_to_array(index.id_map),vector_ids).any():
                raise RuntimeError('删除后仍有残留向量')
            temporary=path.with_suffix('.faiss.tmp');faiss.write_index(index,str(temporary));os.replace(temporary,path)
    folder=project/'action-index'
    if (folder/'metadata.json').exists():
        meta=json.loads((folder/'metadata.json').read_text())
        matrix=np.load(folder/'vectors.npy',allow_pickle=False)
        keep=np.array([s['media_id']!=media_id for s in meta['segments']],dtype=bool)
        if len(keep)!=len(matrix): raise RuntimeError('Qwen 片段与向量不一致')
        removed['segments']=int((~keep).sum())
        # Write the three matching files together at directory granularity. The
        # source stays intact until replaced, including on a zero-video library.
        if removed['segments'] or any(m['id']==media_id for m in meta['media']):
            temp=project/'action-delete-next';shutil.rmtree(temp,ignore_errors=True);temp.mkdir()
            matrix=matrix[keep]
            meta['segments']=[s for s,k in zip(meta['segments'],keep) if k]
            meta['media']=[m for m in meta['media'] if m['id']!=media_id]
            meta['built_at']=time.time()
            index=faiss.IndexFlatIP(matrix.shape[1]);index.add(matrix)
            np.save(temp/'vectors.npy',matrix);faiss.write_index(index,str(temp/'index.faiss'))
            (temp/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False))
            backup=project/'action-delete-previous';shutil.rmtree(backup,ignore_errors=True)
            folder.rename(backup);temp.rename(folder);shutil.rmtree(backup)
    with sqlite3.connect(project/'thumbs.db') as db:
        db.execute('DELETE FROM thumbnails WHERE media_id=?',(media_id,));db.commit();db.execute('VACUUM')
    with sqlite3.connect(project/'metadata/internal.db') as db:
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('DELETE FROM media WHERE id=?',(media_id,))
        db.execute('DELETE FROM source_collections WHERE id NOT IN (SELECT source_collection_id FROM media)')
        db.commit();db.execute('VACUUM')
        if db.execute('PRAGMA foreign_key_check').fetchall(): raise RuntimeError('删除后元数据引用不完整')
    marker=project/'qwen-encoding.json'
    if marker.exists():
        config=json.loads(marker.read_text());config['media_signatures'].pop(str(media_id),None)
        temporary=marker.with_suffix('.json.tmp');temporary.write_text(json.dumps(config,ensure_ascii=False,indent=2));os.replace(temporary,marker)
    return removed


def validate_stores(project):
    with sqlite3.connect(project/'metadata/internal.db') as db:
        expected={}
        for vid,model in db.execute('SELECT id,feature_extractor_id FROM vectors'):
            expected.setdefault(model,set()).add(vid)
    for folder in (project/'store').glob('*/*/*/*'):
        if not (folder/'index').exists(): continue
        model=str(folder.relative_to(project/'store'));want=expected.get(model,set())
        features=[]
        for path in (folder/'features').glob('video-*.faiss'):
            index=faiss.read_index(str(path));features.extend(int(v) for v in faiss.vector_to_array(index.id_map))
        index=faiss.read_index(str(folder/'index/video-IndexFlatIP.faiss'))
        ids=faiss.vector_to_array(index.id_map)
        if set(features)!=want or len(features)!=len(want) or set(ids)!=want or len(ids)!=len(want):
            raise RuntimeError('删除后特征、索引和数据库不一致：'+model)
    validate_project(project,require_owl=True)
    read_snapshot(project,complete=True)


def recover_action_swap(project):
    """Recover a crash between the two Qwen directory renames."""
    folder=project/'action-index';backup=project/'action-delete-previous'
    if not folder.exists() and backup.exists(): backup.rename(folder)
    if folder.exists(): shutil.rmtree(backup,ignore_errors=True)


def cleanup_deleted(manager, stage, record):
    root=manager.root.resolve(); current=(root/'projects/demo').resolve()
    for entry in record['history']:
        path=Path(entry['project']).resolve()
        if path==current or not path.is_relative_to(root/'projects/releases'):
            raise RuntimeError('历史清理路径不合法')
        if not path.exists(): continue
        recover_action_swap(path)
        row=read_media(path,entry['media_id'])
        if row is not None and media_identity(row)!=record['identity']:
            raise RuntimeError('历史媒体身份改变，停止清理')
        prune_project(path,entry['media_id'])
    # Only unlink application-managed copies. Never remove arbitrary imports.
    live_paths=set()
    with sqlite3.connect(current/'metadata/internal.db') as db:
        for location,path in db.execute('SELECT s.location,m.path FROM media m JOIN source_collections s ON s.id=m.source_collection_id'):
            live_paths.add((root/location/path).resolve())
    for value in record['files']:
        path=Path(value).resolve()
        if not any(path.is_relative_to(root/base) for base in ('data/uploads','data/videos')):
            raise RuntimeError('文件不在视频库托管目录内')
        if path not in live_paths: path.unlink(missing_ok=True)
    with manager.connect() as db:
        for jid in record['jobs']:
            db.execute('DELETE FROM jobs WHERE id=?',(jid,))
    record['cleanup_complete']=True;write_record(stage,record)


def pending_cleanup(manager):
    stage=(manager.root/'projects/demo').resolve();path=stage/'deletion.json'
    if path.exists():
        record=json.loads(path.read_text())
        if not record['cleanup_complete']: cleanup_deleted(manager,stage,record)


def delete_video(manager, media_id, revision):
    if not manager.mutation_lock.acquire(blocking=False):
        raise HTTPException(409,'视频正在入库或删除，请处理完成后重试')
    release=None
    try:
        pending_cleanup(manager)
        current=(manager.root/'projects/demo').resolve()
        row=read_media(current,media_id)
        prior=current/'deletion.json'
        if row is None and prior.exists():
            previous=json.loads(prior.read_text())
            if previous['media_id']==media_id and previous['revision']==revision:
                return {'deleted':media_id,'removed':previous['removed'],'cleanup_pending':False}
        if row is None: raise HTTPException(404,'视频已不存在，请刷新列表')
        if media_revision(row,current)!=revision: raise HTTPException(409,'视频记录已变化，请刷新后重新确认')
        history=[]
        for project in (manager.root/'projects/releases').glob('*/demo'):
            if not (project/'metadata/internal.db').exists(): continue
            with sqlite3.connect(project/'metadata/internal.db') as db:
                db.row_factory=sqlite3.Row
                candidates=db.execute('SELECT m.*,s.location FROM media m JOIN source_collections s ON s.id=m.source_collection_id WHERE m.path=? AND s.location=?',(row['path'],row['location'])).fetchall()
                history.extend({'project':str(project.resolve()),'media_id':r['id']} for r in candidates if media_identity(r)==media_identity(row))
        files=[];root=manager.root.resolve();video=(root/row['location']/row['path']).resolve()
        if any(video.is_relative_to(root/base) for base in ('data/uploads','data/videos')): files.append(str(video))
        jobs=[]
        for job in manager.jobs():
            folder=(manager.uploads/job['id']).resolve()
            if not folder.is_relative_to(manager.uploads.resolve()): continue
            if (root/row['location']).resolve()==folder/'prepared':
                jobs.append(job['id']);files.append(str(folder/('source'+job['suffix'])))
        release=manager.root/'projects/releases'/('delete-'+uuid.uuid4().hex);stage=release/'demo'
        shutil.copytree(current,stage)
        recover_action_swap(stage)
        removed=prune_project(stage,media_id);validate_stores(stage)
        record=dict(media_id=media_id,revision=revision,identity=media_identity(row),history=history,files=files,jobs=jobs,
            removed=removed,cleanup_complete=False)
        write_record(stage,record)
        if manager.stopping.is_set(): raise RuntimeError('服务正在停止，未删除视频')
        manager.wait_async(manager.publisher(stage))
        # Online queries now use the complete new snapshot. Cleanup is resumable.
        try:
            cleanup_deleted(manager,stage,record)
        except Exception:
            import logging
            logging.exception('Video removed; durable cleanup will resume')
            return {'deleted':media_id,'removed':removed,'cleanup_pending':True}
        return {'deleted':media_id,'removed':removed,'cleanup_pending':False}
    finally:
        if release and release.exists() and (manager.root/'projects/demo').resolve()!=release/'demo':
            shutil.rmtree(release)
        manager.mutation_lock.release()
