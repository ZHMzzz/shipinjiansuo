import asyncio
import io
import json
import os
import shutil
import sqlite3
from pathlib import Path
from unittest.mock import Mock

import faiss
import numpy as np
import pytest
from fastapi import HTTPException, UploadFile
from wise.wise_project import WiseProject
from scripts.action_search import ActionEngine, SETTINGS, media_catalog, plan, read_snapshot
from scripts.ingestion import IngestionManager
from scripts.qwen_project import FACE, OWL
from scripts.video_deletion import (delete_video, prune_project, read_media,
    media_revision, pending_cleanup, validate_stores, recover_action_swap)


def fixture(root):
    (root/'data').mkdir();(root/'logs').mkdir()
    manager=IngestionManager(root,None)
    job=manager.enqueue(UploadFile(file=io.BytesIO(b'owned test clip'),filename='delete.mp4'))['job']
    folder=manager.uploads/job['id']/'prepared';folder.mkdir();(folder/'delete.mp4').write_bytes(b'delete')
    keep=root/'data/videos';keep.mkdir();(keep/'keep.mp4').write_bytes(b'keep')
    stage=root/'projects/releases/initial/demo';stage.parent.mkdir(parents=True)
    project=WiseProject(stage,create_project=True);project.db_engine.dispose();project.thumbsdb_engine.dispose()
    with sqlite3.connect(stage/'metadata/internal.db') as db:
        db.executemany('INSERT INTO source_collections VALUES(?,?,?)',[(1,str(keep),'DIR'),(2,str(folder),'DIR')])
        db.executemany('INSERT INTO media(id,source_collection_id,path,checksum,media_type,format,width,height,duration) VALUES(?,?,?,?,?,?,?,?,?)',
            [(10,1,'keep.mp4',b'keep','VIDEO','mp4',64,64,2.),(20,2,'delete.mp4',b'delete','VIDEO','mp4',64,64,2.)])
        for model,base in [(FACE,100),(OWL,200)]:
            ids=np.array([base+10,base+20],dtype='int64');vectors=np.array([[1,0,0],[0,1,0]],dtype='float32')
            db.executemany('INSERT INTO vectors(id,modality,feature_extractor_id,media_id,timestamp) VALUES(?,?,?,?,?)',[(int(ids[0]),'VIDEO',model,10,0.),(int(ids[1]),'VIDEO',model,20,0.)])
            db.execute('INSERT INTO featured VALUES(?)',(int(ids[1]),))
            for sub,file in [('features','video-000000.faiss'),('index','video-IndexFlatIP.faiss')]:
                path=stage/'store'/model/sub;path.mkdir(parents=True)
                index=faiss.IndexIDMap2(faiss.IndexFlatIP(3));index.add_with_ids(vectors,ids);faiss.write_index(index,str(path/file))
    with sqlite3.connect(stage/'thumbs.db') as db:
        db.executemany('INSERT INTO thumbnails(media_id,timestamp,content) VALUES(?,?,?)',[(10,0,b'keep'),(20,0,b'delete')])
    # Historical releases must remain deletable after native-only migration.
    catalog=media_catalog(stage);matrix=np.array([[1,0,0],[0,1,0]],dtype='float32')
    folder=stage/'action-index';folder.mkdir()
    index=faiss.IndexFlatIP(3);index.add(matrix)
    np.save(folder/'vectors.npy',matrix);faiss.write_index(index,str(folder/'index.faiss'))
    (folder/'metadata.json').write_text(json.dumps({'settings':SETTINGS,'media':catalog,'segments':plan(catalog)}))
    (root/'projects/demo').symlink_to(stage)
    manager.update(job['id'],status='completed',media_id=20)
    async def publish(new):
        temporary=root/'projects/next';temporary.symlink_to(new);os.replace(temporary,root/'projects/demo')
    manager.publisher=publish;manager.wait_async=asyncio.run
    return manager,stage,job


def test_delete_cleans_all_stores_snapshots_files_jobs_and_preserves_other_vectors(tmp_path):
    manager,old,job=fixture(tmp_path);revision=media_revision(read_media(old,20),old)
    second=tmp_path/'projects/releases/backup/demo';shutil.copytree(old,second)
    result=delete_video(manager,20,revision)
    assert result['removed']=={'vectors':2,'segments':1} and not result['cleanup_pending']
    for project in [old,second,(tmp_path/'projects/demo').resolve()]:
        assert read_media(project,20) is None and read_media(project,10)
        validate_stores(project)
        assert read_snapshot(project,complete=True)[1].tolist()==[[1,0,0]]
        index=faiss.read_index(str(project/'store'/OWL/'features/video-000000.faiss'))
        np.testing.assert_array_equal(index.reconstruct(210),[1,0,0])
        with sqlite3.connect(project/'thumbs.db') as db:
            assert db.execute('SELECT media_id FROM thumbnails').fetchall()==[(10,)]
        with sqlite3.connect(project/'metadata/internal.db') as db:
            assert not db.execute('SELECT * FROM featured').fetchall()
    assert not (manager.uploads/job['id']/'prepared/delete.mp4').exists()
    assert not (manager.uploads/job['id']/'source.mp4').exists()
    assert (tmp_path/'data/videos/keep.mp4').read_bytes()==b'keep'
    assert manager.jobs()==[]
    # Retrying the same request is harmless; re-upload no longer hits dedup.
    assert delete_video(manager,20,revision)['deleted']==20
    assert not manager.enqueue(UploadFile(file=io.BytesIO(b'owned test clip'),filename='again.mp4'))['duplicate']


def test_failed_publication_preserves_active_data_and_files(tmp_path):
    manager,old,job=fixture(tmp_path)
    async def fail(_): raise RuntimeError('injected publish failure')
    manager.publisher=fail
    with pytest.raises(RuntimeError,match='injected'):
        delete_video(manager,20,media_revision(read_media(old,20),old))
    assert (tmp_path/'projects/demo').resolve()==old and read_media(old,20)
    validate_stores(old)
    assert (manager.uploads/job['id']/'source.mp4').exists()
    assert not list((tmp_path/'projects/releases').glob('delete-*'))


def test_stale_identity_and_writer_lock_do_not_mutate(tmp_path):
    manager,old,_=fixture(tmp_path);revision=media_revision(read_media(old,20),old)
    with pytest.raises(HTTPException) as exc: delete_video(manager,20,'0'*64)
    assert exc.value.status_code==409
    with manager.mutation_lock:
        with pytest.raises(HTTPException) as exc: delete_video(manager,20,revision)
        assert exc.value.status_code==409
    assert read_media(old,20)


def test_post_publish_cleanup_failure_is_resumed_from_durable_record(tmp_path,monkeypatch):
    from scripts import video_deletion as module
    manager,old,job=fixture(tmp_path);cleanup=module.cleanup_deleted
    monkeypatch.setattr(module,'cleanup_deleted',Mock(side_effect=OSError('injected cleanup failure')))
    result=delete_video(manager,20,media_revision(read_media(old,20),old))
    assert result['cleanup_pending'] and read_media((tmp_path/'projects/demo').resolve(),20) is None
    assert (manager.uploads/job['id']/'source.mp4').exists()
    monkeypatch.setattr(module,'cleanup_deleted',cleanup)
    pending_cleanup(manager)
    assert read_media(old,20) is None
    assert not (manager.uploads/job['id']/'source.mp4').exists()
    assert json.loads((tmp_path/'projects/demo/deletion.json').read_text())['cleanup_complete']


def test_last_video_can_be_deleted_and_empty_indices_reload(tmp_path):
    from wise.api.services.project import LocalWiseProjectService
    from wise.api.services.search import LocalSearchService
    from wise.config import APIConfig
    manager,old,_=fixture(tmp_path)
    delete_video(manager,20,media_revision(read_media(old,20),old))
    current=(tmp_path/'projects/demo').resolve()
    delete_video(manager,10,media_revision(read_media(current,10),current))
    current=(tmp_path/'projects/demo').resolve();validate_stores(current)
    meta,matrix,index=read_snapshot(current,complete=True)
    assert not meta['segments'] and matrix.shape==(0,3) and index.ntotal==0
    project=WiseProject(current,read_only=True)
    service=LocalWiseProjectService(project,APIConfig(command='serve',project_dir=current))
    assert service.info().num_media_files==0
    LocalSearchService(service,Mock())
    project.db_engine.dispose();project.thumbsdb_engine.dispose()


def test_recover_qwen_directory_swap(tmp_path):
    manager,old,_=fixture(tmp_path)
    (old/'action-index').rename(old/'action-delete-previous')
    recover_action_swap(old)
    assert read_snapshot(old,complete=True)


def test_reused_media_id_rejects_old_confirmation_even_with_copied_delete_record(tmp_path):
    manager,old,_=fixture(tmp_path);revision=media_revision(read_media(old,20),old)
    delete_video(manager,20,revision)
    current=(tmp_path/'projects/demo').resolve()
    with sqlite3.connect(current/'metadata/internal.db') as db:
        db.execute("INSERT INTO media(id,source_collection_id,path,checksum,media_type,format,width,height,duration) VALUES(20,1,'new.mp4',X'1234','VIDEO','mp4',64,64,2)")
    with pytest.raises(HTTPException) as error: delete_video(manager,20,revision)
    assert error.value.status_code==409 and read_media(current,20)['path']=='new.mp4'
