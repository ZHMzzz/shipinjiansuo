import io
import sqlite3
from pathlib import Path

import pytest
from fastapi import HTTPException, UploadFile
from scripts.ingestion import IngestionManager, prepare_versions


def manager(tmp_path):
    (tmp_path/'data').mkdir()
    (tmp_path/'logs').mkdir()
    return IngestionManager(tmp_path,None)


def test_upload_deduplicates_content_and_confines_filename(tmp_path):
    m=manager(tmp_path)
    first=m.enqueue(UploadFile(file=io.BytesIO(b'example video bytes'),filename='../../outside.mp4'))
    second=m.enqueue(UploadFile(file=io.BytesIO(b'example video bytes'),filename='different.mp4'))
    assert first['job']['filename']=='outside.mp4'
    assert second['duplicate'] is True
    assert second['job']['id']==first['job']['id']
    assert len(list(m.uploads.iterdir()))==1
    assert not (tmp_path.parent/'outside.mp4').exists()


def test_invalid_extension_and_empty_upload_leave_no_job(tmp_path):
    m=manager(tmp_path)
    for name,data in [('file.html',b'not video'),('empty.mp4',b'')]:
        with pytest.raises(HTTPException) as e:
            m.enqueue(UploadFile(file=io.BytesIO(data),filename=name))
        assert e.value.status_code==400
    assert m.jobs()==[]
    assert list(m.uploads.iterdir())==[]


def test_retry_is_only_allowed_for_failed_tasks(tmp_path):
    m=manager(tmp_path)
    job=m.enqueue(UploadFile(file=io.BytesIO(b'x'),filename='video.mp4'))['job']
    with pytest.raises(HTTPException) as e: m.retry(job['id'])
    assert e.value.status_code==409
    m.update(job['id'],status='failed',message='decode error')
    m.retry(job['id'])
    assert m.jobs()[0]['status']=='queued'
    assert m.jobs()[0]['progress']==0


def test_project_pointer_migration_preserves_original_and_is_repeatable(tmp_path):
    project=tmp_path/'projects/demo'
    project.mkdir(parents=True)
    (project/'original').write_bytes(b'original-index')
    first=prepare_versions(tmp_path)
    second=prepare_versions(tmp_path)
    assert first==second
    assert project.is_symlink()
    assert (first/'original').read_bytes()==b'original-index'


def test_restart_recovers_published_job_and_marks_unpublished_interrupted(tmp_path):
    m=manager(tmp_path)
    project=tmp_path/'projects/demo/metadata'
    project.mkdir(parents=True)
    a=m.enqueue(UploadFile(file=io.BytesIO(b'a'),filename='a.mp4'))['job']
    b=m.enqueue(UploadFile(file=io.BytesIO(b'b'),filename='b.mp4'))['job']
    m.update(a['id'],status='publishing')
    m.update(b['id'],status='extracting')
    with sqlite3.connect(project/'internal.db') as db:
        db.execute('CREATE TABLE media(id INTEGER,source_collection_id INTEGER)')
        db.execute('CREATE TABLE source_collections(id INTEGER,location TEXT)')
        db.execute('INSERT INTO media VALUES(15,2)')
        db.execute('INSERT INTO source_collections VALUES(2,?)',(str(m.uploads/a['id']/'prepared'),))
    m.recover()
    state={j['id']:j for j in m.jobs()}
    assert state[a['id']]['status']=='completed'
    assert state[a['id']]['media_id']==15
    assert state[b['id']]['status']=='failed'
    assert '重试' in state[b['id']]['message']


def test_batch_publications_prune_history_but_failed_upload_keeps_both_versions(tmp_path, monkeypatch):
    import asyncio
    import concurrent.futures
    import json
    from types import SimpleNamespace
    from test_video_deletion import fixture
    from scripts import ingestion
    from scripts.qwen_project import OWL

    m, original, _ = fixture(tmp_path)
    m.frame_worker = None
    m.build_actions = lambda *args: None
    monkeypatch.setattr(ingestion.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout=json.dumps({'streams':[{'codec_type':'video'}], 'format':{'duration':2}})))

    def run_coroutine(coroutine, loop):
        future = concurrent.futures.Future()
        try: future.set_result(asyncio.run(coroutine))
        except Exception as exc: future.set_exception(exc)
        return future
    monkeypatch.setattr(ingestion.asyncio, 'run_coroutine_threadsafe', run_coroutine)
    def run(command, log, on_line=None):
        if 'create-index' not in command: return
        stage = Path(command[command.index('--project-dir')+1])
        with sqlite3.connect(stage/'metadata/internal.db') as db:
            db.execute('INSERT INTO source_collections(id,location,type) VALUES(?,?,?)', (next_id,str((m.uploads/job['id'])/'prepared'),'DIR'))
            db.execute("INSERT INTO media(id,source_collection_id,path,media_type,format,width,height,duration) VALUES(?,?,'clip.mp4','VIDEO','mp4',64,64,2)", (next_id,next_id))
            db.execute('INSERT INTO vectors(id,modality,feature_extractor_id,media_id,timestamp) VALUES(?,?,?,?,?)', (next_id,'VIDEO',OWL,next_id,0))
        with sqlite3.connect(stage/'thumbs.db') as db:
            db.execute('INSERT INTO thumbnails(media_id,timestamp,content) VALUES(?,0,?)',(next_id,b'thumbnail'))
    m.run = run
    for next_id in (1000,1001,1002):
        job = m.enqueue(UploadFile(file=io.BytesIO(str(next_id).encode()),filename='clip.mp4'))['job']
        with m.mutation_lock: m.process_job(job)
        assert next(j for j in m.jobs() if j['id']==job['id'])['status']=='completed'
        assert len(list((tmp_path/'projects/releases').iterdir()))==2
        assert ((m.uploads/job['id'])/'source.mp4').exists()
    assert not original.parent.exists()
    before = set((tmp_path/'projects/releases').iterdir())
    current = (tmp_path/'projects/demo').resolve()
    async def fail(stage): raise RuntimeError('injected publication failure')
    m.publisher = fail
    next_id = 1003
    job = m.enqueue(UploadFile(file=io.BytesIO(b'fail'),filename='clip.mp4'))['job']
    with m.mutation_lock: m.process_job(job)
    assert next(j for j in m.jobs() if j['id']==job['id'])['status']=='failed'
    assert (tmp_path/'projects/demo').resolve()==current
    assert set((tmp_path/'projects/releases').iterdir())==before


def test_successful_backfill_also_retires_older_release(tmp_path, monkeypatch):
    import shutil
    from test_video_deletion import fixture
    from scripts.release_retention import publish_pointer
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    import action_search
    m, old, _ = fixture(tmp_path)
    previous = tmp_path/'projects/releases/previous/demo'
    shutil.copytree(old,previous); publish_pointer(tmp_path,previous)
    def incomplete(project, complete=False):
        if complete: raise ValueError('missing segment')
        return None
    monkeypatch.setattr(action_search,'read_snapshot',incomplete)
    m.build_actions = lambda *args: None
    with m.mutation_lock: m.backfill_actions()
    assert m.action_state['status']=='ready'
    assert previous.exists() and not old.parent.exists()
    assert len(list((tmp_path/'projects/releases').iterdir()))==2
