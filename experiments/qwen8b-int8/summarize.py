"""Validate the completed runs, audit production hashes and export comparisons."""
from pathlib import Path
import csv
import hashlib
import json
import os
import statistics
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def summary(name):
    d = json.loads((HERE / (name + '.json')).read_text())
    r = json.loads((HERE / (name + '-resources.json')).read_text())
    assert d['status'] == 'completed' and r['exit_code'] == 0
    assert len(d['clips']) == 10 and len(d['queries']) == 36
    videos = np.load(HERE / (name + '-vectors.npy'))
    queries = np.load(HERE / (name + '-query-vectors.npy'))
    assert videos.shape[0] == 10 and queries.shape[0] == 36
    assert videos.shape[1] == queries.shape[1] == (2048 if name == '2b-fp16' else 4096)
    assert np.isfinite(videos).all() and np.isfinite(queries).all()
    assert np.allclose(np.linalg.norm(videos, axis=1), 1, atol=1e-5)
    assert np.allclose(np.linalg.norm(queries, axis=1), 1, atol=1e-5)
    first = [x for x in d['queries'] if x['round'] == 0]
    warm = [x for x in d['queries'] if x['round'] > 0]
    groups = {}
    for language in ['all', 'zh', 'en']:
        rows = [x for x in first if language == 'all' or x['language'] == language]
        groups[language] = {'queries': len(rows), **{
            k: statistics.mean(x['metrics'][k] for x in rows) for k in first[0]['metrics']}}
    stable = all(
        [y['id'] for y in q['ranking']] == [y['id'] for y in first[q['query_index']]['ranking']]
        for q in d['queries'])
    return {'quality': groups, 'load_seconds': d['model_load_seconds'],
            'warm_video_median_seconds': statistics.median(x['encode_seconds'] for x in d['warm_video_repeats']),
            'warm_query_median_seconds': statistics.median(x['encode_seconds'] + x['search_seconds'] for x in warm),
            'warm_query_p95_seconds': float(np.percentile([x['encode_seconds'] + x['search_seconds'] for x in warm], 95)),
            'faiss_median_seconds': statistics.median(x['search_seconds'] for x in warm),
            'peak_metal_bytes': d['final_memory'].get('sampled_metal_peak_bytes'),
            'peak_rss_bytes': r['peak_rss_bytes'],
            'peak_physical_footprint_bytes': r['peak_physical_footprint_bytes'],
            'minimum_system_available_bytes': r['minimum_system_available_bytes'],
            'initial_swap_used_bytes': r['initial_swap_used_bytes'],
            'peak_swap_used_bytes': r['peak_swap_used_bytes'],
            'rankings_identical_across_rounds': stable,
            'repeated_video_min_cosine': min(x['cosine_to_first'] for x in d['warm_video_repeats']),
            'runtime': d['runtime']}, d


if __name__ == '__main__':
    a, da = summary('2b-fp16')
    b, db = summary('8b-q8')
    reference = json.loads((HERE / 'template-reference.json').read_text())
    expected_text = [x for x in reference['tokens'] if x != 151656]
    alignment = []
    for ca, cb in zip(da['clips'], db['clips']):
        assert ca['id'] == cb['id']
        grid = ca['input']['video_grid_thw'][0]
        ok = (cb['input']['text_token_ids'] == expected_text
              and cb['input']['visual_chunk_tokens'] == [grid[1]*grid[2]//4]*grid[0]
              and ca['input']['tokens'] == cb['input']['tokens'])
        assert ok, ca['id']
        alignment.append({'id': ca['id'], 'tokens': ca['input']['tokens'], 'aligned': ok})
    before = json.loads((HERE / 'production-before.json').read_text())
    after = {p: sha256(p) for p in before['files']}
    changed = [p for p, h in after.items() if h != before['files'][p]]
    pointer_same = os.readlink(ROOT / 'projects/demo') == before['pointer']
    audit = {'project_pointer_unchanged': pointer_same, 'checked_file_count': len(after),
             'changed_files': changed, 'sha256': after}
    (HERE / 'production-after.json').write_text(json.dumps(audit, indent=2))
    assert not changed and pointer_same, 'Production data changed; inspect audit before reporting'
    result = {'2b-fp16': a, '8b-q8': b, 'input_alignment': alignment,
              'production_unchanged': True, 'sample': '10 clips; 6 intents in Chinese and English'}
    (HERE / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    with (HERE / 'query-comparison.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['query', 'expected', '2B_top1', '8B_Q8_top1', '2B_hit1', '8B_hit1', '2B_top5', '8B_top5'])
        for qa, qb in zip(da['queries'][:12], db['queries'][:12]):
            assert qa['text'] == qb['text']
            writer.writerow([qa['text'], ', '.join(qa['expected']), qa['ranking'][0]['id'], qb['ranking'][0]['id'],
                qa['metrics']['hit_at_1'], qb['metrics']['hit_at_1'],
                ', '.join(x['id'] for x in qa['ranking'][:5]), ', '.join(x['id'] for x in qb['ranking'][:5])])
    print(json.dumps({k: v for k, v in result.items() if k != 'input_alignment'}, ensure_ascii=False, indent=2))
