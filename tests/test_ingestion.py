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
