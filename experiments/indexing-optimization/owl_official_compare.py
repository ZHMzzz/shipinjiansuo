"""Numerical and frozen-reference retrieval regression on 44 sampled frames."""
import json
from pathlib import Path
import sys

import numpy as np
import bench
from owl_official_bench import OUT


def ranking(values, frames, queries):
    vectors = np.concatenate([values[f'vectors_{i}'] for i in range(len(frames))])
    frame_ids = np.concatenate([np.full(len(values[f'vectors_{i}']), i) for i in range(len(frames))])
    result = {}
    for name in queries.files:
        scores = (vectors @ queries[name].reshape(-1)).astype('float64')
        frame_scores = np.full(len(frames), -np.inf)
        np.maximum.at(frame_scores, frame_ids, scores)
        top_frames = np.argsort(-frame_scores, kind='stable')[:5].tolist()
        media_scores = {}
        for score, frame in zip(frame_scores, frames):
            mid = frame['media_id']; media_scores[mid] = max(media_scores.get(mid, -np.inf), float(score))
        top_media = sorted(media_scores, key=lambda mid: (-media_scores[mid], mid))[:5]
        result[name] = {'frames_top5': top_frames, 'videos_top5': top_media,
                        'frame_scores_top5': frame_scores[top_frames].tolist()}
    return result


def compare(base, candidate):
    design = json.loads((OUT/'design.json').read_text())
    assert bench.sha(bench.OUT/'query-baseline.npz') == design['query_sha256']
    queries = np.load(bench.OUT/'query-baseline.npz')
    with np.load(OUT/(base+'.npz')) as a, np.load(OUT/(candidate+'.npz')) as b:
        rows = []; cosines = []
        for i, frame in enumerate(design['frames']):
            s, t = a[f'raw_scores_{i}'], b[f'raw_scores_{i}']
            v, w = a[f'raw_vectors_{i}'], b[f'raw_vectors_{i}']
            boxes, boxes2 = a[f'raw_boxes_{i}'], b[f'raw_boxes_{i}']
            assert np.isfinite(s).all() and np.isfinite(t).all() and np.isfinite(v).all() and np.isfinite(w).all()
            keep = (s >= .02) & (t >= .02)
            x, y = v[keep].astype('float64'), w[keep].astype('float64')
            if len(x):
                cosine = (x*y).sum(-1)/(np.linalg.norm(x, axis=-1)*np.linalg.norm(y, axis=-1))
                cosines.extend(cosine.tolist())
            rows.append({'media_id': frame['media_id'], 'ts': frame['ts'],
                'threshold_crossings': int(((s >= .02) != (t >= .02)).sum()),
                'regions_baseline': int((s >= .02).sum()), 'regions_candidate': int((t >= .02).sum()),
                'scores_max_abs': float(np.abs(s-t).max()),
                'retained_boxes_max_abs': float(np.abs(boxes[keep]-boxes2[keep]).max()) if keep.any() else 0.,
                'raw_exact': all(np.array_equal(a[f'{key}_{i}'], b[f'{key}_{i}']) for key in ('raw_scores','raw_vectors','raw_boxes'))})
        left, right = ranking(a, design['frames'], queries), ranking(b, design['frames'], queries)
    result = {'baseline': base, 'candidate': candidate, 'frames': rows,
        'all_raw_outputs_exact': all(r['raw_exact'] for r in rows),
        'threshold_crossings': sum(r['threshold_crossings'] for r in rows),
        'regions_baseline': sum(r['regions_baseline'] for r in rows),
        'regions_candidate': sum(r['regions_candidate'] for r in rows),
        'retained_aligned_cosine_min': min(cosines), 'retained_aligned_cosine_median': float(np.median(cosines)),
        'changed_video_top5_queries': [q for q in left if left[q]['videos_top5'] != right[q]['videos_top5']],
        'changed_frame_top5_queries': [q for q in left if left[q]['frames_top5'] != right[q]['frames_top5']],
        'queries': {q: {'baseline': left[q], 'candidate': right[q]} for q in left},
        'scope': 'Frozen production-side reference-image query vectors; 44 sampled gallery frames only; not full-library recall or an accuracy benchmark'}
    path = OUT/('compare-'+base.replace('/', '-')+'--'+candidate.replace('/', '-')+'.json')
    bench.dump(path, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('frames','queries')}, ensure_ascii=False))


if __name__ == '__main__':
    compare(sys.argv[1], sys.argv[2])
