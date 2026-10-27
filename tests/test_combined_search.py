from types import SimpleNamespace as NS
from unittest.mock import Mock
import numpy as np
from scripts.combined_search import person_scene_search
from wise.api.services.search.base import SearchOutput
from wise.data_models import ModalityType


def face_search(scores, times):
    faces = SearchOutput.model_construct(ids=list(range(10,10+len(scores))), distances=scores,
        metadata=[NS(media_id=mid,timestamp=ts) for mid,ts in times], ext_metadata=[None]*len(scores))
    return NS(search_indices={ModalityType.VIDEO:{'face':NS(index=NS(ntotal=len(scores)))}},
              search_with_feature=Mock(return_value=faces))


def test_scene_ranking_enforces_identity_video_and_containing_window():
    search = face_search([.9,.1,.8,.7,.6], [(1,1.),(1,2.),(2,1.),(1,8.),(3,2.)])
    segments = [dict(media_id=1,ts=0,te=8),dict(media_id=1,ts=8,te=16),
                dict(media_id=2,ts=0,te=8),dict(media_id=99,ts=0,te=8)]
    actions = NS(get_snapshot=Mock(return_value=({'segments':segments},np.array([[.2],[.3],[.7],[1.]]),None)),
                 query_vector=Mock(return_value=(np.ones((1,1)),False)))
    result,evidence = person_scene_search(np.ones((1,4)), '汽车', .45,
        face_model='face',search=search,actions=actions,project='project')
    assert set(result.ids)=={10,12,13} and result.ids[0]==12
    assert evidence['10']['scene_similarity']==.2
    assert evidence['13']['scene_ts']==8  # End is exclusive; next segment contains timestamp 8.
    assert all(v['face_similarity']>=.45 for v in evidence.values())
    actions.query_vector.assert_called_once_with('汽车')
    assert search.search_with_feature.call_count==1


def test_no_identity_match_does_not_load_qwen():
    search = face_search([.1],[(1,0.)]); actions=Mock()
    result,evidence=person_scene_search(np.ones((1,4)), '汽车', .45,
        face_model='face',search=search,actions=actions,project=None)
    assert not result.ids and not evidence
    actions.get_snapshot.assert_not_called()
    actions.query_vector.assert_not_called()
