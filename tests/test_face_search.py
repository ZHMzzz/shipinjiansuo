from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
from wise.api.services.search.base import SearchOutput
from wise.data_models import ModalityType

from scripts.face_search import search_registered_faces
from scripts.combined_search import person_scene_search


def output(ids, scores, times=None):
    return SearchOutput.model_construct(ids=ids, distances=scores,
        metadata=[NS(media_id=i, timestamp=0. if times is None else times[k]) for k,i in enumerate(ids)],
        ext_metadata=[None]*len(ids))


def searcher(outputs, count=4):
    return NS(search_indices={ModalityType.VIDEO:{'face':NS(index=NS(ntotal=count))}},
              search_with_feature=Mock(side_effect=outputs))


def test_union_keeps_old_hits_adds_second_photo_hits_and_takes_max_not_average():
    search = searcher([output([1,2,3,4],[.9,.3,.5,.2]), output([2,3,1,4],[.8,.7,.1,.44])])
    vectors = np.eye(2, dtype=np.float32)
    result = search_registered_faces(vectors, search=search, face_model='face', threshold=.45)
    assert result.ids == [1,2,3]
    assert result.distances == [.9,.8,.7]
    calls = search.search_with_feature.call_args_list
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0].args[0], vectors[:1])
    np.testing.assert_array_equal(calls[1].args[0], vectors[1:])
    assert all(c.kwargs['end'] == 4 for c in calls)


def test_threshold_is_applied_before_rounding_and_limit_after_union():
    search = searcher([output([1,2],[.45,.4499]), output([3,1],[.9,.2])])
    result = search_registered_faces(np.eye(2), search=search, face_model='face', threshold=.45, limit=1)
    assert result.ids == [3]


def test_empty_index_skips_search():
    search = searcher([], count=0)
    assert not search_registered_faces(np.ones((1,2)), search=search, face_model='face').ids
    search.search_with_feature.assert_not_called()


def test_nonfinite_scores_cannot_escape_threshold():
    search = searcher([output([1,2,3],[float('nan'),float('inf'),.7])])
    result = search_registered_faces(np.ones((1,2)), search=search, face_model='face', threshold=.45)
    assert result.ids == [3]


def test_person_scene_uses_second_photo_but_retains_time_and_identity_gates():
    search = searcher([output([1,2,3],[.8,.2,.1],[1.,2.,10.]),
                       output([2,3,1],[.9,.7,.3],[2.,10.,1.])])
    actions = NS(get_snapshot=Mock(return_value=({'segments':[
        dict(media_id=1,ts=0,te=8),dict(media_id=2,ts=0,te=8),dict(media_id=3,ts=0,te=8)]},
        np.ones((3,1)),None)), query_vector=Mock(return_value=(np.ones((1,1)),False)))
    result,evidence = person_scene_search(np.eye(2),'场景',.45,face_model='face',search=search,
                                          actions=actions,project=None)
    assert set(result.ids) == {1,2}
    assert evidence['2']['face_similarity'] == .9
    assert '3' not in evidence  # Face is outside the semantic window.
