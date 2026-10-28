import json
import sqlite3
from unittest.mock import Mock

import numpy as np
import faiss
import pytest
from scripts.action_search import ActionEngine, windows, plan, media_catalog, read_snapshot, select_hits


def project(tmp_path, duration=18):
    path = tmp_path/'demo'; (path/'metadata').mkdir(parents=True)
    with sqlite3.connect(path/'metadata/internal.db') as db:
        db.executescript('''CREATE TABLE source_collections(id INTEGER,location TEXT);
            CREATE TABLE media(id INTEGER,path TEXT,duration REAL,source_collection_id INTEGER,media_type TEXT);
            CREATE TABLE vectors(id INTEGER PRIMARY KEY,modality TEXT,feature_extractor_id TEXT,media_id INTEGER,timestamp REAL,end_timestamp REAL);
            INSERT INTO source_collections VALUES(1,'/example');''')
        db.execute("INSERT INTO media VALUES(7,'a.mp4',?,1,'VIDEO')", (duration,))
    return path


def save(path):
    engine = ActionEngine(); catalog = media_catalog(path); records = plan(catalog)
    matrix = np.eye(len(records), dtype='float32')
    engine.finish(path, catalog, records, matrix)
    return engine, records, matrix


@pytest.mark.parametrize('duration,expected', [(.2, [(0,.2)]), (8, [(0,8)]),
    (8.1, [(0,8),(4,8.1)]), (12, [(0,8),(4,12)]), (18, [(0,8),(4,12),(8,16),(12,18)])])
def test_windows_cover_tail_without_redundant_terminal_clip(duration, expected):
    assert list(windows(duration)) == expected


@pytest.mark.parametrize('duration', [0, -1, float('nan'), float('inf')])
def test_bad_duration_rejected(duration):
    with pytest.raises(ValueError): list(windows(duration))


def test_incremental_index_preserves_vectors_and_blocks_incomplete_publication(tmp_path):
    path = project(tmp_path); engine, records, matrix = save(path)
    with sqlite3.connect(path/'metadata/internal.db') as db:
        db.execute("INSERT INTO media VALUES(19,'new.mp4',3,1,'AV')")
    with pytest.raises(ValueError, match='缺少动作索引'): read_snapshot(path, complete=True)
    catalog, old_records, old_vectors, missing = engine.prepare(path)
    assert [m['id'] for m in missing] == [19]
    np.testing.assert_array_equal(old_vectors, matrix)
    new_record = plan(missing)[0]
    engine.finish(path, catalog, old_records+[new_record], old_vectors+[np.ones(matrix.shape[1], dtype='float32')])
    meta, result, index = read_snapshot(path, complete=True)
    np.testing.assert_array_equal(result[:-1], matrix)
    assert index.ntotal == len(records)+1
    assert meta['segments'][-1] == dict(media_id=19,ts=0.,te=3.)


def test_snapshot_rejects_changed_media_and_misaligned_vectors(tmp_path):
    path=project(tmp_path); save(path)
    from scripts.native_qwen import QWEN
    folder=path/'store'/QWEN
    damaged=faiss.IndexIDMap2(faiss.IndexFlatIP(4))
    faiss.write_index(damaged,str(folder/'features/video-000000.faiss'))
    with pytest.raises(ValueError,match='不一致'): read_snapshot(path)
    save(path)
    with sqlite3.connect(path/'metadata/internal.db') as db:
        db.execute("UPDATE media SET path='other.mp4'")
    with pytest.raises(ValueError,match='媒体'): read_snapshot(path)


def test_arbitrary_query_reaches_encoder_and_returns_ranked_nonzero_timestamps(tmp_path):
    path=project(tmp_path); engine, records, _ = save(path)
    encoder = Mock(); encoder.extract_text_features.return_value = np.array([[0,0,1,0]],dtype='float32')
    engine.model = Mock(return_value=encoder)
    query='一个人从椅子旁边突然向前倒下，这是未预设的描述'
    result=engine.search(path,query,4)
    encoder.extract_text_features.assert_called_once_with([query])
    best=result['video_results']['unmerged_windows'][0]
    assert (best['ts'],best['te'])==(8,16)
    assert best['link']=='/demo/media/7#t=8.0,16.0'
    assert result['indexed_segments']==4
    assert result['score_kind']=='action'


def test_overlapping_duplicates_suppressed_but_separate_events_kept():
    segments=[dict(media_id=1,ts=0,te=8), dict(media_id=1,ts=1,te=9),
              dict(media_id=1,ts=12,te=20), dict(media_id=2,ts=0,te=8)]
    hits=select_hits(segments,[.9,.8,.7,.6],[0,1,2,3],20)
    assert [(h['media_id'],h['ts']) for h in hits]==[(1,0),(1,12),(2,0)]


def test_failed_backfill_preserves_live_version_and_is_retryable(tmp_path, monkeypatch):
    from pathlib import Path
    from scripts.ingestion import IngestionManager
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    (tmp_path/'data').mkdir(); (tmp_path/'logs').mkdir()
    stage=project(tmp_path/'projects')
    (stage/'preserve.txt').write_text('existing video library')
    manager=IngestionManager(tmp_path, Mock(), action_builder=Mock())
    manager.build_actions=Mock(side_effect=RuntimeError('test model failure'))
    manager.backfill_actions()
    assert manager.action_state['status']=='failed'
    assert (stage/'preserve.txt').read_text()=='existing video library'
    assert not (stage/'action-index').exists()
    assert list((tmp_path/'projects/releases').iterdir())==[]
    manager.publisher.assert_not_called()
    manager.request_backfill()
    assert manager.action_state['status']=='pending' and manager.backfill_requested


def test_text_cache_uses_trimmed_query_and_rejects_empty():
    engine=ActionEngine(); model=Mock()
    model.extract_text_features.return_value=np.array([[0.,2.]],dtype='float32')
    engine.model=Mock(return_value=model)
    vector,cached=engine.query_vector('街道')
    assert not cached and np.linalg.norm(vector)==pytest.approx(1)
    model.extract_text_features.assert_called_once_with(['街道'])
    assert engine.query_vector(' 街道 ')[1]
    engine.query_vector('室内')
    assert model.extract_text_features.call_count==2
    with pytest.raises(ValueError): engine.query_vector(' ')


def test_filename_candidate_is_not_lost_beyond_initial_shortlist(tmp_path):
    path=project(tmp_path,duration=500)
    with sqlite3.connect(path/'metadata/internal.db') as db:
        db.execute("INSERT INTO media VALUES(19,'short.mp4',3,1,'VIDEO')")
    engine=ActionEngine(); catalog=media_catalog(path); records=plan(catalog)
    vectors=np.tile([1.,0.],(len(records),1)).astype('float32'); vectors[-1]=[0.,1.]
    engine.finish(path,catalog,records,vectors)
    engine.query_vector=Mock(return_value=(np.array([[1.,0.]],dtype='float32'),False))
    result=engine.search(path,'test',2,media_ids={19})
    assert [h['media_id'] for h in result['video_results']['unmerged_windows']]==[19]
    assert result['score_kind']=='action'


def test_video_grouping_searches_beyond_duplicate_windows_before_limit():
    # A long video occupies the first 150 ranked windows. The next video must
    # still be found, retaining the first video's highest-scoring timestamp.
    engine = ActionEngine()
    segments = [dict(media_id=1, ts=float(600-i*4), te=float(608-i*4)) for i in range(150)]
    segments += [dict(media_id=2, ts=12., te=20.), dict(media_id=3, ts=0., te=8.)]
    index = Mock(ntotal=len(segments))
    index.search.return_value = (np.linspace(.99,.1,len(segments))[None,:], np.arange(len(segments))[None,:])
    engine.get_snapshot = Mock(return_value=({'segments':segments, 'media':[
        dict(id=i,path=f'{i}.mp4',duration=700.) for i in (1,2,3)]}, None, index))
    engine.query_vector = Mock(return_value=(np.ones((1,2),dtype='float32'),False))
    result = engine.search('unused','查询',2,group_by_video=True)
    hits = result['video_results']['unmerged_windows']
    assert [(h['media_id'],h['ts']) for h in hits] == [(1,600.),(2,12.)]
    assert hits[0]['distance'] > hits[1]['distance']
    assert hits[0]['link'] == '/demo/media/1#t=600.0,608.0'
    assert index.search.call_args.args[1] == len(segments)
    filtered = engine.search('unused','查询',2,media_ids={2,3},group_by_video=True)
    assert [h['media_id'] for h in filtered['video_results']['unmerged_windows']] == [2,3]
