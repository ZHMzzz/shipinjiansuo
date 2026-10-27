from unittest.mock import Mock

import numpy as np
import pytest

from scripts.action_reranker import reorder_hits, read_frames, VideoReranker
from scripts.action_search import ActionEngine


def test_reorder_preserves_candidate_identity_timestamps_and_separate_scores():
    hits = [{'media_id': 3, 'ts': 16., 'te': 24., 'distance': .8},
            {'media_id': 1, 'ts': 4., 'te': 12., 'distance': .7}]
    ranked = reorder_hits(hits, [.1, .9])
    assert [(h['media_id'], h['ts'], h['te']) for h in ranked] == [(1,4.,12.),(3,16.,24.)]
    assert ranked[0]['original_rank'] == 2
    assert ranked[0]['distance'] == ranked[0]['embedding_score'] == .7
    assert ranked[0]['rerank_score'] == .9
    assert 'original_rank' not in hits[0]
    assert reorder_hits(hits, [.5,.5])[0]['media_id'] == 3


@pytest.mark.parametrize('scores', [[.2], [.2,float('nan')], [.2,float('inf')]])
def test_incomplete_or_nonfinite_reranking_is_rejected(scores):
    with pytest.raises(RuntimeError):
        reorder_hits([{'distance':.9},{'distance':.8}], scores)


def test_decode_failure_is_not_scored_as_empty_document(tmp_path):
    with pytest.raises(RuntimeError, match='无法解码'):
        read_frames({'location':str(tmp_path),'path':'absent.mp4'}, {'ts':0,'te':8})


def test_reranker_passes_only_query_and_pixels_and_preserves_window(monkeypatch):
    import scripts.action_reranker as module
    model = VideoReranker(); model.load = Mock(return_value=0)
    model.score = Mock(side_effect=[.2,.8])
    pixels = object()
    monkeypatch.setattr(module, 'read_frames', Mock(return_value=pixels))
    import psutil
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: Mock(available=10*1024**3))
    hits = [{'media_id':1,'ts':16.,'te':24.,'distance':.9},
            {'media_id':2,'ts':0.,'te':8.,'distance':.7}]
    catalog = [{'id':1,'path':'不应进入提示词的标签.mp4'}, {'id':2,'path':'helmet.mp4'}]
    ranked, timing = model.rank('没有佩戴安全帽', hits, catalog)
    model.score.assert_called_with('没有佩戴安全帽', pixels)
    assert module.read_frames.call_args_list[0].args == (catalog[0],hits[0])
    assert ranked[0]['media_id'] == 2
    assert len(timing['candidates']) == 2


def test_query_cache_avoids_reloading_embedding_and_is_bounded():
    engine = ActionEngine()
    index = Mock(); index.ntotal = 1
    index.search.return_value = (np.array([[.8]]),np.array([[0]]))
    engine.get_snapshot = Mock(return_value=({'media':[{'id':1,'path':'v.mp4','duration':8}],
         'segments':[{'media_id':1,'ts':0,'te':8}]}, None, index))
    encoder = Mock(); encoder.extract_text_features.return_value = np.ones((1,4),dtype='float32')
    engine.model = Mock(return_value=encoder)
    assert not engine.search(None,'同一查询',1)['query_embedding_cached']
    assert engine.search(None,'同一查询',1)['query_embedding_cached']
    assert encoder.extract_text_features.call_count == 1
    for i in range(65): engine.search(None,str(i),1)
    assert len(engine.query_vectors) == 64


@pytest.mark.parametrize('enabled', [False, True])
def test_disabled_does_not_call_model_and_failed_rerank_preserves_original(monkeypatch, enabled):
    import scripts.action_reranker as module
    sampler = Mock(result={'total_seconds':1})
    sampler.__enter__ = Mock(return_value=sampler)
    sampler.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(module, 'ResourceSampler', lambda: sampler)
    hits = [{'media_id':1,'ts':16,'te':24,'distance':.8}]
    actions = Mock(snapshot=[{'media':[]}])
    actions.search.return_value = {'video_results':{'unmerged_windows':hits}}
    ranker = Mock(); ranker.rank.side_effect = RuntimeError('test memory exhaustion')
    result = module.search_with_reranking(actions, ranker, 'project', 'query', enabled=enabled)
    assert result['video_results']['unmerged_windows'] == hits
    assert not result['rerank']['applied']
    if enabled:
        assert '当前显示原始排序' in result['rerank']['error']
        ranker.unload.assert_called_once()
    else:
        ranker.rank.assert_not_called()
        actions.unload.assert_not_called()
