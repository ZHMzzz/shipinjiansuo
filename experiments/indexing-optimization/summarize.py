"""Summarize saved measurements without running inference or opening indexes."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def summarize():
    names = ['baseline-r1', 'fast-r1', 'fp16-r1', 'fp16_weights-r1',
             'fast_fp16_weights-r1', 'fast_fp16_coreml-r1']
    rows = []
    for name in names:
        timing = read(ROOT / 'runs' / name / 'timing.json')
        metrics = read(ROOT / 'runs' / name / 'metrics.json')
        if timing['status'] != 'completed':
            raise ValueError(f'Incomplete sample run: {name}')
        rows.append({'name': name, 'owl': timing['owl'], 'face': timing['face'],
                     'wise_seconds': timing['wise_total_seconds'],
                     'extract_seconds': timing['extract_wall_seconds'],
                     'index_seconds': timing['index_wall_seconds'],
                     'vector_counts': timing['vector_counts'],
                     'timers': metrics['timers'], 'resources': metrics['resources']})
    for row in rows:
        row['speedup'] = rows[0]['wise_seconds'] / row['wise_seconds']
        row['time_reduction_fraction'] = 1 - row['wise_seconds'] / rows[0]['wise_seconds']
    result = {'sample_runs': rows, 'sample_scope': '11 videos, 151.3346 seconds, 303 frames; WISE only; one run per configuration'}
    evaluation = read(ROOT / 'eval-fast_fp16_coreml-fixed.json')
    result['fixed_query_quality'] = evaluation['summary']
    result['face_quality'] = evaluation['faces']
    suite_path = ROOT / 'replays/e2e-r1/suite.json'
    if suite_path.exists():
        suite = read(suite_path)
        result['replay_status'] = suite['status']
        result['qwen_load_and_text_warmup_seconds'] = suite.get('qwen_load_and_text_warmup_seconds')
        trials = []
        for path in sorted(suite_path.parent.glob('r*/timing.json')):
            timing = read(path)
            if timing['status'] == 'completed':
                trials.append({'trial': path.parent.name, **timing})
        result['replay_trials'] = trials
        grouped = []
        for media_id in (40, 41):
            pair = {}
            for config in ('baseline', 'optimized'):
                group = [r for r in trials if r['original_media_id'] == media_id and r['config']['name'] == config]
                if not group:
                    continue
                stages = sorted(set.intersection(*(set(r['stages']) for r in group)))
                pair[config] = {
                    'trials': len(group),
                    'raw_seconds': [r['elapsed_wall_seconds'] for r in group],
                    'adjusted_seconds': [r['adjusted_wall_seconds'] for r in group],
                    'mean_raw_seconds': statistics.mean(r['elapsed_wall_seconds'] for r in group),
                    'mean_adjusted_seconds': statistics.mean(r['adjusted_wall_seconds'] for r in group),
                    'mean_stages': {s: statistics.mean(r['stages'][s] for r in group) for s in stages},
                    'qwen_new_vector_hashes': sorted({r['new_qwen_vectors_sha256'] for r in group}),
                }
            row = {'media_id': media_id, 'configurations': pair}
            if set(pair) == {'baseline', 'optimized'}:
                before = pair['baseline']['mean_adjusted_seconds']
                after = pair['optimized']['mean_adjusted_seconds']
                row.update(speedup=before / after, time_reduction_fraction=1 - after / before)
            grouped.append(row)
        result['replay_comparison'] = grouped
    review_path = ROOT / 'integrity-review.json'
    if review_path.exists():
        result['integrity_review'] = read(review_path)
    (ROOT / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'sample_seconds': {r['name']: r['wise_seconds'] for r in rows},
                      'replay_status': result.get('replay_status'),
                      'replay_comparison': result.get('replay_comparison')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    summarize()
