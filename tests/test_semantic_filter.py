from scripts.semantic_filter import filter_semantic_hits
from scripts.library_search import normalize


def test_no_forced_top_result_when_query_is_weak():
    assert filter_semantic_hits([{'distance':.509},{'distance':.42}]) == []
    assert filter_semantic_hits([]) == []


def test_relative_gap_and_absolute_floor_both_apply():
    hits=[{'distance':s} for s in [.5624,.4809,.4706]]
    assert filter_semantic_hits(hits) == hits[:1]
    hits=[{'distance':s} for s in [.52,.46,.449]]
    assert filter_semantic_hits(hits) == hits[:2]
    assert len(filter_semantic_hits([{'distance':.629},{'distance':.619}])) == 2


def test_invalid_scores_are_excluded_and_order_is_preserved():
    hits=[{'distance':.6},{'distance':float('nan')},{'distance':float('inf')}, {}, {'distance':.65}]
    assert filter_semantic_hits(hits) == [hits[0],hits[4]]


def test_person_scene_uses_semantic_score_not_fused_or_face_score():
    catalog=[dict(id=i,name=f'{i}.mp4',duration=20) for i in (1,2)]
    hits=[dict(media_id=i,ts=0,te=8,thumbnail='thumb',distance=.04,face_similarity=.99,
               scene_similarity=s) for i,s in [(1,.7),(2,.4)]]
    result=normalize({'video_results':{'unmerged_windows':hits}},catalog,'person_scene',100,.45)
    assert [h['id'] for h in result['items']] == [1]
    faces=normalize({'video_results':{'unmerged_windows':[{**h,'distance':.9} for h in hits]}},catalog,'face',100,.45)
    assert len(faces['items']) == 2
    objects=normalize({'video_results':{'unmerged_windows':hits}},catalog,'object',100,.45)
    assert len(objects['items']) == 2
