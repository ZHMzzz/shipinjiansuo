"""Exercise the production encoder against frozen experimental Linear vectors."""
import json
from pathlib import Path
import time

import bench


def main():
    import numpy as np
    import torch
    from action_search import ActionEngine

    source = bench.OUT/'qwen-linear-20261028'
    output = bench.OUT/'linear-production-20261028'
    design = json.loads((source/'design.json').read_text())
    trial = json.loads((source/'linear-sample-r1.json').read_text())
    reference = np.load(source/'linear-sample-r1.npy')
    media = {m['id']: m for m in design['media']}
    engine = ActionEngine()
    query = '仓库货架通道里，戴黄色安全帽的人把手抬到嘴边'
    before, _ = engine.query_vector(query)
    patch = engine.extractor._model.model.visual.patch_embed
    assert 'forward' not in patch.__dict__
    records = []
    # Different videos/positions from the original 31-segment sample.
    for index in (0, len(reference)//2, len(reference)-1):
        segment = trial['segments'][index]
        item = media[segment['media_id']]
        assert bench.sha(item['absolute_path']) == item['sha256']
        started = time.perf_counter()
        vector = engine.encode_segment(item, segment)
        torch.mps.synchronize()
        elapsed = time.perf_counter() - started
        assert 'forward' not in patch.__dict__
        np.testing.assert_array_equal(vector, reference[index])
        records.append({'segment': segment, 'seconds': elapsed,
                        'matches_experimental_linear_exactly': True})
    engine.query_vectors.clear()
    after, _ = engine.query_vector(query)
    np.testing.assert_array_equal(before, after)
    result = {'status': 'passed', 'segments': records,
              'query_vector_unchanged_after_video_encoding': True,
              'original_forward_restored': True,
              'production_index_written': False}
    bench.dump(output/'encoder-verification.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
