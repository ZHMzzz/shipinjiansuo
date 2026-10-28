import json
import statistics
from pathlib import Path

DIR = Path(__file__).resolve().parent

def percentile(values, q):
    xs = sorted(values)
    position = (len(xs) - 1) * q
    low = int(position)
    high = min(low + 1, len(xs) - 1)
    return xs[low] + (xs[high] - xs[low]) * (position - low)

def main():
    data = json.loads((DIR / 'results.json').read_text())
    design = json.loads((DIR / 'design.json').read_text())
    reviews = {r['id']: r for r in json.loads((DIR / 'review.json').read_text())['cases']}
    rows = [r for r in data['results'] if r['phase'] == 'main']
    assert {r['id'] for r in rows} == {c['id'] for c in design['cases']} == set(reviews)
    summary = {'model_load_seconds': data['model_load_seconds'], 'groups': {}}
    for kind in ['short', 'paragraph', 'all']:
        items = [r for r in rows if kind == 'all' or r['kind'] == kind]
        timings = [r['total_seconds'] for r in items]
        summary['groups'][kind] = {
            'n': len(items), 'schema_valid': sum(r['schema_valid'] for r in items),
            'strict_pass': sum(r['schema_valid'] and reviews[r['id']]['semantic_pass'] for r in items),
            'p50_seconds': statistics.median(timings), 'p95_seconds': percentile(timings, .95),
            'min_seconds': min(timings), 'max_seconds': max(timings),
            'median_output_tokens': statistics.median([r['output_tokens'] for r in items]),
        }
    main_by_id = {r['id']: r for r in rows}
    repeats = [r for r in data['results'] if r['phase'].startswith('repeat')]
    summary['repeat_calls'] = len(repeats)
    summary['repeat_exact_matches'] = sum(r['raw'] == main_by_id[r['id']]['raw'] for r in repeats)
    summary['repeat_p50_seconds'] = statistics.median([r['total_seconds'] for r in repeats])
    summary['warmup_seconds'] = data['results'][0]['total_seconds']
    summary['memory'] = {key: max(r[key] for r in data['results']) for key in ['rss_bytes', 'mps_driver_bytes', 'swap_bytes']}
    summary['index_unchanged'] = data['index_before'] == data['index_after'] and data['pointer_unchanged']
    summary['pid_unchanged'] = data['server_pid_before'] == data['server_pid_after']
    summary['pid'] = data['server_pid_after']
    (DIR / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
