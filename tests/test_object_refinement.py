from types import SimpleNamespace as NS
from unittest.mock import Mock
import numpy as np
from scripts.object_refinement import ObjectRefiner


def test_frame_comparison_reranks_pixels_and_cache_is_version_bound(monkeypatch,tmp_path):
    image=object();frame1=object();frame2=object()
    read=Mock(side_effect=lambda media,ts:(frame1 if media['id']==1 else frame2,ts))
    monkeypatch.setattr('scripts.object_refinement.read_frame',read)
    def vector(**kwargs):
        value=kwargs['image']
        return np.array([[1.,0.]]) if value is image or value is frame2 else np.array([[0.,1.]]),False
    engine=NS(get_snapshot=Mock(return_value=({'media':[{'id':1},{'id':2}]},None,None)),query_vector=Mock(side_effect=vector))
    result={'video_results':{'videos':{},'unmerged_windows':[
        dict(media_id=1,ts=0,te=8,distance=.8),dict(media_id=2,ts=4,te=12,distance=.6)]}}
    refiner=ObjectRefiner();out=refiner.refine(engine,tmp_path/'v1',result,image,negative='汽车')
    hits=out['video_results']['unmerged_windows'];assert [h['media_id'] for h in hits]==[2,1]
    assert hits[0]['embedding_score']==.6 and hits[0]['original_rank']==2
    assert hits[0]['ts']==4 and hits[0]['te']==12
    assert engine.query_vector.call_args_list[0].kwargs['negative']=='汽车'
    refiner.refine(engine,tmp_path/'v1',result,image)
    assert read.call_count==2
    refiner.refine(engine,tmp_path/'v2',result,image)
    assert read.call_count==4
    assert result['video_results']['unmerged_windows'][0]['media_id']==1
