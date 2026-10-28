import hashlib
import shutil
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi import HTTPException
from wise.api.common import TextQueryTerm
from wise.api.services.embedding import EmbeddingConfig, EmbeddingService
from wise.api.services.project import LocalWiseProjectService
from wise.api.services.search import LocalSearchService
from wise.api.standalone.search import search_output_to_response
from wise.config import APIConfig
from wise.data_models import MediaType, ModalityType
from wise.wise_project import WiseProject

from scripts.action_search import ActionEngine, media_catalog, plan, read_snapshot
from scripts.library_search import native_text_search, normalize
from scripts.native_qwen import QWEN, prepare_native_qwen_project, sync_native_qwen, validate_native_qwen
from scripts.qwen_project import FACE, OWL
from scripts.video_deletion import delete_video, media_revision, read_media, validate_stores
from test_video_deletion import fixture


def test_existing_native_vectors_migrate_without_encoding_or_id_changes(tmp_path, monkeypatch):
    from scripts.native_qwen import MANIFEST
    _, old, _ = fixture(tmp_path)
    saved = tmp_path/'legacy';shutil.copytree(old/'action-index',saved)
    sync_native_qwen(old)
    shutil.copytree(saved,old/'action-index');(old/MANIFEST).unlink()
    with sqlite3.connect(old/'metadata/internal.db') as db:
        before=db.execute('SELECT * FROM vectors ORDER BY id').fetchall()
    matrix=read_snapshot(old)[1].copy()
    monkeypatch.setattr(ActionEngine,'encode_segment',Mock(side_effect=AssertionError('must reuse')))
    stage=prepare_native_qwen_project(tmp_path)
    with sqlite3.connect(stage/'metadata/internal.db') as db:
        assert db.execute('SELECT * FROM vectors ORDER BY id').fetchall()==before
    np.testing.assert_array_equal(read_snapshot(stage)[1],matrix)
    assert not (stage/'action-index').exists()


def test_upload_hides_and_restores_only_native_store_even_on_failure(tmp_path):
    from scripts.native_qwen import suspend_native_store
    _, stage, _ = fixture(tmp_path);sync_native_qwen(stage)
    matrix=read_snapshot(stage)[1].copy()
    with pytest.raises(RuntimeError,match='frame failure'):
        with suspend_native_store(stage):
            assert not (stage/'store'/QWEN).exists()
            assert (stage/'store'/OWL).exists()
            with sqlite3.connect(stage/'metadata/internal.db') as db:
                assert db.execute('SELECT count(*) FROM vectors WHERE feature_extractor_id=?',(QWEN,)).fetchone()[0]==2
            raise RuntimeError('frame failure')
    np.testing.assert_array_equal(read_snapshot(stage)[1],matrix)
    assert not (stage/'.qwen-store-pending').exists()


def test_corrupt_native_does_not_fall_back_to_stale_legacy(tmp_path):
    import faiss
    _, stage, _ = fixture(tmp_path)
    saved=tmp_path/'legacy';shutil.copytree(stage/'action-index',saved)
    sync_native_qwen(stage);shutil.copytree(saved,stage/'action-index')
    faiss.write_index(faiss.IndexIDMap2(faiss.IndexFlatIP(3)),str(stage/'store'/QWEN/'index/video-IndexFlatIP.faiss'))
    with pytest.raises(ValueError,match='不一致'):read_snapshot(stage)


def test_native_missing_window_and_changed_settings_are_rejected(tmp_path):
    import json
    from scripts.native_qwen import MANIFEST
    _, stage, _ = fixture(tmp_path);sync_native_qwen(stage)
    marker=stage/MANIFEST;data=json.loads(marker.read_text());data['settings']['frames']=8;marker.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='配置'):read_snapshot(stage)
    data['settings']['frames']=16;marker.write_text(json.dumps(data))
    with sqlite3.connect(stage/'metadata/internal.db') as db:
        db.execute('UPDATE media SET duration=12 WHERE id=10')
    with pytest.raises(ValueError,match='覆盖'):read_snapshot(stage)


def test_native_missing_video_is_incremental_not_full_rebuild(tmp_path):
    _, stage, _ = fixture(tmp_path);sync_native_qwen(stage)
    original=read_snapshot(stage)[1].copy()
    with sqlite3.connect(stage/'metadata/internal.db') as db:
        db.execute("INSERT INTO media(id,source_collection_id,path,media_type,format,width,height,duration) VALUES(30,1,'new.mp4','VIDEO','mp4',64,64,2)")
    catalog,segments,vectors,missing=ActionEngine().prepare(stage)
    assert [m['id'] for m in missing]==[30]
    np.testing.assert_array_equal(vectors,original)
    ActionEngine().finish(stage,catalog,segments+plan(missing),vectors+[np.array([0,0,1],dtype='float32')])
    assert not (stage/'action-index').exists()
    validate_native_qwen(stage)


def test_legacy_partial_first_start_retains_vectors_for_backfill(tmp_path):
    _, old, _ = fixture(tmp_path)
    with sqlite3.connect(old/'metadata/internal.db') as db:
        db.execute("INSERT INTO media(id,source_collection_id,path,media_type,format,width,height,duration) VALUES(30,1,'new.mp4','VIDEO','mp4',64,64,2)")
    assert prepare_native_qwen_project(tmp_path)==old
    _,_,vectors,missing=ActionEngine().prepare(old)
    assert len(vectors)==2 and [m['id'] for m in missing]==[30]


def test_person_scene_uses_native_matrix_after_migration(tmp_path):
    from scripts.combined_search import person_scene_search
    from wise.api.services.search.base import SearchOutput
    from types import SimpleNamespace
    _, stage, _ = fixture(tmp_path)
    engine=ActionEngine();engine.query_vector=Mock(return_value=(np.array([[1,0,0]],dtype='float32'),False))
    face=SimpleNamespace(media_id=10,timestamp=0.)
    output=SearchOutput();output.ids=[110];output.distances=[.9];output.metadata=[face];output.ext_metadata=[None]
    search=SimpleNamespace(search_indices={ModalityType.VIDEO:{FACE:SimpleNamespace(index=SimpleNamespace(ntotal=1))}},search_with_feature=Mock(return_value=output))
    before=person_scene_search(np.zeros((1,3)),'写字',.45,face_model=FACE,search=search,actions=engine,project=stage)
    sync_native_qwen(stage);engine.cached_project=None
    after=person_scene_search(np.zeros((1,3)),'写字',.45,face_model=FACE,search=search,actions=engine,project=stage)
    assert before[0].ids==after[0].ids and before[0].distances==after[0].distances
    assert before[1]==after[1]


def retained(project):
    paths = [project/'thumbs.db']
    for model in (FACE, OWL):
        paths.extend((project/'store'/model).rglob('*'))
    return {str(p.relative_to(project)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


def test_migration_reuses_vectors_preserves_other_stores_and_old_snapshot(tmp_path):
    _, old, _ = fixture(tmp_path)
    before = retained(old)
    stage = prepare_native_qwen_project(tmp_path)
    assert stage != old and (tmp_path/'projects/demo').resolve() == stage
    assert before == retained(stage) == retained(old)
    assert (old/'action-index').exists() and not (stage/'action-index').exists()
    assert not (old/'store'/QWEN).exists()
    validate_native_qwen(stage)
    validate_stores(stage)
    assert prepare_native_qwen_project(tmp_path) == stage


def test_failed_migration_does_not_change_active_snapshot(tmp_path, monkeypatch):
    from scripts import native_qwen
    _, old, _ = fixture(tmp_path)
    monkeypatch.setattr(native_qwen,'sync_native_qwen',Mock(side_effect=ValueError('injected')))
    with pytest.raises(ValueError,match='injected'):
        prepare_native_qwen_project(tmp_path)
    assert (tmp_path/'projects/demo').resolve() == old
    assert not list((tmp_path/'projects/releases').glob('native-qwen-*'))


def test_first_start_without_segments_allows_background_backfill(tmp_path):
    _, old, _ = fixture(tmp_path)
    shutil.rmtree(old/'action-index')
    assert prepare_native_qwen_project(tmp_path) == old
    with pytest.raises(HTTPException) as error:
        native_text_search(None,SimpleNamespace(search_indices={}),None,'写字',20)
    assert error.value.status_code == 409


def test_incremental_sync_preserves_ids_and_delete_removes_native_vectors(tmp_path):
    manager, stage, _ = fixture(tmp_path)
    sync_native_qwen(stage)
    def rows():
        with sqlite3.connect(stage/'metadata/internal.db') as db:
            return db.execute('SELECT id,media_id,timestamp FROM vectors WHERE feature_extractor_id=? ORDER BY id',(QWEN,)).fetchall()
    original = rows()
    with sqlite3.connect(stage/'metadata/internal.db') as db:
        db.execute("INSERT INTO media(id,source_collection_id,path,media_type,format,width,height,duration) VALUES(30,1,'added.mp4','VIDEO','mp4',64,64,2)")
    catalog = media_catalog(stage)
    ActionEngine().finish(stage,catalog,plan(catalog),np.eye(3,dtype='float32'))
    sync_native_qwen(stage)
    assert rows()[:2] == original and rows()[-1][1] == 30
    validate_stores(stage)
    result = delete_video(manager,20,media_revision(read_media(stage,20),stage))
    assert result['removed'] == {'vectors':3,'segments':1}
    current = (tmp_path/'projects/demo').resolve()
    validate_native_qwen(current)
    validate_native_qwen(stage)  # Historical cleanup also removes Qwen vectors.
    # Deleting all remaining media leaves a loadable empty native store.
    for mid in (10,30):
        delete_video(manager,mid,media_revision(read_media(current,mid),current))
        current = (tmp_path/'projects/demo').resolve()
        validate_native_qwen(current)


def test_real_wise_ranking_truncation_merging_and_filename_constraint(tmp_path):
    _, stage, _ = fixture(tmp_path)
    with sqlite3.connect(stage/'metadata/internal.db') as db:
        db.execute('UPDATE media SET duration=16 WHERE id=10')
        db.execute("UPDATE media SET size_in_bytes=100,date_modified='2026-10-27 00:00:00',num_frames=100")
    with sqlite3.connect(stage/'thumbs.db') as db:
        db.execute("INSERT INTO thumbnails(media_id,timestamp,content) VALUES(10,4,?)",(b'thumbnail',))
    catalog = media_catalog(stage)
    # Best hit is a later overlapping window; all scores below the old gate.
    scores = np.array([.3,.4,.2,.1],dtype='float32')
    matrix = np.column_stack([scores,np.sqrt(1-scores**2),np.zeros(4)]).astype('float32')
    ActionEngine().finish(stage,catalog,plan(catalog),matrix)
    sync_native_qwen(stage)
    config = APIConfig(command='serve',project_dir=stage)
    extractor = Mock()
    extractor.extract_text_features.return_value = np.array([[1,0,0]],dtype='float32')
    extractor.transform_faiss_distances_hook.side_effect = lambda value:value
    embedding = EmbeddingService({QWEN:extractor})
    project = LocalWiseProjectService(WiseProject(stage,read_only=True),config)
    search = LocalSearchService(project,embedding)
    data, limited = native_text_search(project,search,config,'骑自行车',2)
    terms = [TextQueryTerm(term_id='text',is_negative=False,txt='骑自行车')]
    raw = search.search(terms,EmbeddingConfig(query_prefix=''),modality_type=ModalityType.VIDEO,
        feature_extractor_id=QWEN,start=0,end=2,filter_specs={})
    expected = search_output_to_response(config,project,terms,MediaType.VIDEO,raw,1).model_dump(mode='json',exclude={'query'})
    assert data == expected and limited
    hits = data['video_results']['unmerged_windows']
    assert [h['ts'] for h in hits] == [4,0]
    assert all(h['distance'] < .52 for h in hits)
    cards = [dict(id=m['id'],name=m['path'],duration=m['duration']) for m in catalog]
    result = normalize(data,cards,'text',2,.45)
    assert [x['id'] for x in result['items']] == [10]  # No expansion to fill two videos.
    assert result['items'][0]['start'] == 4 and result['items'][0]['end'] == 12
    filtered, capped = native_text_search(project,search,config,'骑自行车',2,allowed={20})
    assert not capped
    assert [h['media_id'] for h in filtered['video_results']['merged_windows']] == ['20']
    assert native_text_search(project,search,config,'骑自行车',2,allowed=set()) == ({'video_results':None},False)
    extractor.extract_text_features.assert_called_with(['骑自行车'])
    project.wise_project.db_engine.dispose()
    project.wise_project.thumbsdb_engine.dispose()
