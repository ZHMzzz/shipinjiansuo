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


def test_empty_filename_candidate_skips_model():
    actions=Mock()
    result=search_library(catalog=[dict(id=1,name='a.mp4')],query='汽车',person='',filename='missing',image_bytes=None,
        threshold=.45,limit=20,project=None,search=None,embedding=None,actions=actions,config=None,people_db=None)
    assert result['items']==[]
    actions.search.assert_not_called()


def test_normalize_keeps_best_window_per_video():
    catalog=[dict(id=1,name='真实名称.mp4',duration=20,url='/demo/media/1')]
    hits=[dict(media_id=1,ts=0,te=4,thumbnail_ts=1,thumbnail='thumbnail/a',distance=.9),
          dict(media_id=1,ts=8,te=12,thumbnail_ts=9,thumbnail='thumbnail/b',distance=.8)]
    data={'video_results':{'unmerged_windows':hits,'merged_windows':hits}}
    text=normalize(data,catalog,'text',100,.45)
    assert [h['start'] for h in text['items']]==[1]
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


@pytest.mark.parametrize('limit', [1, 100])
@pytest.mark.parametrize('region_count', [130, 501])
def test_object_candidates_do_not_let_duplicate_regions_hide_other_videos(monkeypatch, limit, region_count):
    import scripts.library_search as module
    # The second video's best region ranks 128th, beyond the former 100-region cutoff.
    hits=[dict(media_id=1 if i < 127 else 2,ts=i*.5,te=i*.5+4,
               thumbnail=f'thumb/{i}',distance=1-i*.001) for i in range(region_count)]
    search=NS(search=Mock(side_effect=lambda *args, **kw: hits[:kw['end']]))
    monkeypatch.setattr(module,'search_output_to_response',
        lambda config,project,terms,media_type,result,thumbs:
            NS(model_dump=lambda **kw:{'video_results':{'unmerged_windows':result}}))
    result=search_library(catalog=[dict(id=i,name=f'{i}.mp4',duration=300) for i in (1,2)],
        query='',person='',filename='',image_bytes=b'image',threshold=.45,limit=limit,
        project=None,search=search,embedding=None,actions=None,
        config=NS(text_queries_weight=2,negative_queries_weight=.2,max_search_results=100),people_db=None)
    assert [x['id'] for x in result['items']] == [1,2][:limit]
    assert [x['start'] for x in result['items']] == [0,63.5][:limit]
    assert result['limited'] is (region_count > 500 or limit == 1)
    assert search.search.call_args.kwargs['end'] == 500


def test_text_route_uses_native_candidates_without_score_filter(monkeypatch):
    import scripts.library_search as module
    catalog=[dict(id=i,name=f'{i}.mp4',duration=20) for i in range(1,4)]
    actions=Mock()
    native=Mock(return_value=({'video_results':{'merged_windows':[
        dict(media_id=i,ts=8,te=16,thumbnail=f'thumb/{i}',distance=.4-i*.1)
        for i in (1,1,2)]}},True))
    monkeypatch.setattr(module,'native_segment_search',native)
    project=NS(wise_project=NS(_project_dir='project'))
    result=search_library(catalog=catalog,query='写字',person='',filename='',image_bytes=None,
        threshold=.45,limit=3,project=project,
        search=None,embedding=None,actions=actions,config=None,people_db=None)
    native.assert_called_once_with(project,None,None,'写字',3,allowed=None)
    actions.search.assert_not_called()
    assert [item['id'] for item in result['items']] == [1,2]
    assert result['limited'] is True


def test_mixed_route_passes_decoded_image_and_filename_to_qwen(monkeypatch):
    import io
    from PIL import Image
    import scripts.library_search as module
    raw = io.BytesIO(); Image.new('RGB', (8,8), 'red').save(raw, format='PNG')
    native = Mock(return_value=({'video_results':{'merged_windows':[
        dict(media_id=2,ts=4,te=12,thumbnail='thumb',distance=.3)]}}, False))
    monkeypatch.setattr(module, 'native_segment_search', native)
    actions = Mock()
    result = search_library(catalog=[dict(id=1,name='a.mp4',duration=20),dict(id=2,name='b.mp4',duration=20)],
        query='街道',person='',filename='b.mp4',image_bytes=raw.getvalue(),threshold=.45,limit=20,
        project=None,search=None,embedding=None,actions=actions,config=None,people_db=None)
    assert result['mode'] == 'image_text' and [x['id'] for x in result['items']] == [2]
    assert native.call_args.kwargs['allowed'] == {2}
    assert native.call_args.kwargs['image'].getpixel((0,0)) == (255,0,0)
    assert native.call_args.kwargs['actions'] is actions


@pytest.mark.parametrize('with_image', [False, True])
def test_person_routes_use_all_photos_before_object_time_constraints(tmp_path, monkeypatch, with_image):
    import numpy as np
    import scripts.library_search as module
    from scripts.people_store import initialize_people, add_photos
    from scripts.qwen_project import FACE
    from wise.api.services.search.base import SearchOutput
    from wise.data_models import ModalityType
    people = tmp_path/'people.sqlite'; initialize_people(people)
    vectors = np.eye(2,512,dtype=np.float32)
    add_photos(people,'同一人',[(vectors[i].tobytes(),b'photo',str(i)) for i in range(2)],FACE)
    project_dir=tmp_path/'project'; (project_dir/'metadata').mkdir(parents=True)
    with sqlite3.connect(project_dir/'metadata/internal.db') as db:
        db.execute('CREATE TABLE vectors(id,media_id,timestamp,feature_extractor_id)')
        db.executemany('INSERT INTO vectors VALUES(?,?,?,?)',[(101,1,0,OWL),(102,2,4,OWL),(103,2,0,OWL)])
    faces=[SearchOutput.model_construct(ids=[10,20],distances=scores,
        metadata=[NS(media_id=1,timestamp=1.),NS(media_id=2,timestamp=5.)],ext_metadata=[None,None])
        for scores in ([.8,.2],[.1,.9])]
    search=NS(search_indices={ModalityType.VIDEO:{FACE:NS(index=NS(ntotal=2))}},
              search_with_feature=Mock(side_effect=faces),search=Mock(return_value=SearchOutput()))
    def response(config,project,terms,media_type,result,thumbs):
        hits=[dict(media_id=m.media_id,ts=m.timestamp,te=m.timestamp+4,thumbnail='thumb',distance=s)
              for m,s in zip(result.metadata,result.distances)]
        return NS(model_dump=lambda **kw:{'video_results':{'unmerged_windows':hits}})
    monkeypatch.setattr(module,'search_output_to_response',response)
    result=search_library(catalog=[dict(id=i,name=f'{i}.mp4',duration=12) for i in (1,2)],
        query='',person='同一人',filename='',image_bytes=b'image' if with_image else None,
        threshold=.45,limit=100,project=NS(wise_project=NS(_project_dir=project_dir)),search=search,
        embedding=None,actions=None,config=NS(text_queries_weight=2,negative_queries_weight=.2),people_db=people)
    assert search.search_with_feature.call_count == 2
    if with_image:
        # Second photo supplies the second video's face, only in its containing object window.
        assert search.search.call_args.kwargs['vector_id_constraint'].tolist() == [101,102]
    else:
        assert [item['id'] for item in result['items']] == [2,1]
        assert [item['score'] for item in result['items']] == [.9,.8]
