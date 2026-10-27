import sqlite3
from types import SimpleNamespace as NS
from unittest.mock import Mock
import pytest
from fastapi import HTTPException
from scripts.library_search import matching_media, object_constraints, normalize, search_library, OWL


def test_filename_matches_display_name_case_insensitively():
    catalog = [dict(id=1,name='张学友.MP4'),dict(id=2,name='other.mp4')]
    assert matching_media(catalog,'张学友.mp4') == {1}
    assert matching_media(catalog,'MP4') == {1,2}
    assert matching_media(catalog,'missing') == set()


def test_object_constraints_enforce_media_and_face_window_before_ranking(tmp_path):
    db_path = tmp_path/'vectors.db'
    with sqlite3.connect(db_path) as db:
        db.execute('CREATE TABLE vectors(id,media_id,timestamp,feature_extractor_id)')
        db.executemany('INSERT INTO vectors VALUES(?,?,?,?)',[(1,1,0,OWL),(2,1,4,OWL),
            (3,2,0,OWL),(4,1,0,'face'),(5,1,8,OWL)])
    assert object_constraints(db_path,{1}).tolist() == [1,2,5]
    # A face exactly at the end is excluded from the preceding object's window.
    assert object_constraints(db_path,{1},{1:[4.,3.9]}).tolist() == [1,2]
    assert object_constraints(db_path,{1},{1:[4.]}).tolist() == [2]
    assert object_constraints(db_path,set()).tolist() == []


def test_mixed_query_is_rejected_before_any_model_or_database_access():
    with pytest.raises(HTTPException) as error:
        search_library(catalog=[],query='街道',person='',filename='',image_bytes=b'image',
            threshold=.45,limit=20,project=None,search=None,embedding=None,actions=None,config=None,people_db=None)
    assert error.value.status_code == 400


def test_empty_filename_candidate_skips_model():
    actions=Mock()
    result=search_library(catalog=[dict(id=1,name='a.mp4')],query='汽车',person='',filename='missing',image_bytes=None,
        threshold=.45,limit=20,project=None,search=None,embedding=None,actions=actions,config=None,people_db=None)
    assert result['items']==[]
    actions.search.assert_not_called()


def test_normalize_keeps_multiple_text_windows_but_deduplicates_objects():
    catalog=[dict(id=1,name='真实名称.mp4',duration=20,url='/demo/media/1')]
    hits=[dict(media_id=1,ts=0,te=4,thumbnail_ts=1,thumbnail='thumbnail/a',distance=.9),
          dict(media_id=1,ts=8,te=12,thumbnail_ts=9,thumbnail='thumbnail/b',distance=.8)]
    data={'video_results':{'unmerged_windows':hits}}
    text=normalize(data,catalog,'text',100,.45)
    assert [h['start'] for h in text['items']]==[0,8]
    objects=normalize(data,catalog,'object',100,.45)
    assert len(objects['items'])==1 and objects['items'][0]['start']==1
    assert objects['items'][0]['name']=='真实名称.mp4'


def test_native_object_routing_does_not_serialize_uploaded_binary(monkeypatch):
    from wise.api.common import SearchResponse
    import scripts.library_search as module
    search=Mock()
    def response(config,project,terms,media_type,result,thumbs):
        assert terms[0].src == b'\xff\xd8\xff'
        return SearchResponse(time=0,query=terms,video_results=None,video_audio_results=None,image_results=None)
    monkeypatch.setattr(module,'search_output_to_response',response)
    result=search_library(catalog=[dict(id=1,name='a.mp4')],query='',person='',filename='',image_bytes=b'\xff\xd8\xff',
        threshold=.45,limit=100,project=None,search=search,embedding=None,actions=None,
        config=NS(text_queries_weight=2,negative_queries_weight=.2),people_db=None)
    assert result['items']==[] and result['mode']=='object'
    assert search.search.call_args.kwargs['feature_extractor_id']==OWL
