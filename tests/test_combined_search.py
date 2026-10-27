from types import SimpleNamespace as NS
from unittest.mock import Mock
import numpy as np

from scripts.combined_search import person_scene_search
from wise.api.services.search.base import SearchOutput
from wise.data_models import ModalityType


def test_scene_ranking_keeps_identity_and_timestamp_constraints():
    metadata = [NS(media_id=1,timestamp=1.), NS(media_id=1,timestamp=2.),
                NS(media_id=2,timestamp=1.)]
    faces = SearchOutput.model_construct(ids=[10,11,12], distances=[.9,.1,.8],
                                         metadata=metadata, ext_metadata=[None]*3)
    scenes = SearchOutput.model_construct(ids=[102,100], distances=[.7,.2],
                                          metadata=[metadata[2],metadata[0]], ext_metadata=[None]*2)
    search = NS(search_indices={ModalityType.VIDEO:{'face':NS(index=NS(ntotal=3))}},
                search_with_feature=Mock(side_effect=[faces,scenes]))
    project = NS(wise_project=NS(get_vector_ids=Mock(return_value=[100,102])))
    embedding = NS(embed=Mock(return_value=np.ones((1,4),dtype=np.float32)))
    result,evidence = person_scene_search(np.ones((1,4)), '汽车', '室内', .45,
        face_model='face',text_model='text',search=search,embedding=embedding,
        project=project,config=NS(text_queries_weight=1,negative_queries_weight=.2))
    project.wise_project.get_vector_ids.assert_called_once_with(
        [1,2],ModalityType.VIDEO,'text',timestamps=[1.,1.])
    assert search.search_with_feature.call_args.kwargs['vector_id_constraint']==[100,102]
    assert set(result.ids)=={10,12} and result.ids[0]==12
    assert all(v['face_similarity']>=.45 for v in evidence.values())
    terms=embedding.embed.call_args.args[2]
    assert terms[1].is_negative and terms[1].txt=='室内'


def test_no_identity_match_does_not_search_scene():
    search = NS(search_indices={ModalityType.VIDEO:{'face':NS(index=NS(ntotal=1))}},
        search_with_feature=Mock(return_value=SearchOutput.model_construct(
            ids=[1],distances=[.1],metadata=[NS(media_id=1,timestamp=0.)],ext_metadata=[None])))
    embedding=NS(embed=Mock())
    result,evidence=person_scene_search(np.ones((1,4)), '汽车', '', .45,
        face_model='face',text_model='text',search=search,embedding=embedding,
        project=None,config=None)
    assert not result.ids and not evidence
    embedding.embed.assert_not_called()
