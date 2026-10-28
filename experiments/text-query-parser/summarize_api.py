import json
import statistics
from pathlib import Path
from summarize import percentile

DIR = Path(__file__).resolve().parent

def main():
    data = json.loads((DIR / 'results-deepseek.json').read_text())
    design = json.loads((DIR / 'design.json').read_text())
    review = json.loads((DIR / 'review-deepseek.json').read_text())
    reviews = {r['id']: r for r in review['cases']}
    rows = [r for r in data['results'] if r['phase'] == 'main']
    assert len(rows) == len(design['cases']) == len(reviews) == 24
    summary = {'model_requested': data['settings']['model'], 'groups': {}}
    for kind in ['short', 'paragraph', 'all']:
        items = [r for r in rows if kind == 'all' or r['kind'] == kind]
        timings = [r['total_seconds'] for r in items]
        summary['groups'][kind] = {
            'n': len(items), 'schema_valid': sum(r['schema_valid'] for r in items),
            'strict_pass': sum(r['schema_valid'] and reviews[r['id']]['semantic_pass'] for r in items),
            'p50_seconds': statistics.median(timings), 'p95_seconds': percentile(timings, .95),
            'min_seconds': min(timings), 'max_seconds': max(timings),
        }
    main_by_id = {r['id']: r for r in rows}
    repeats = [r for r in data['results'] if r['phase'].startswith('repeat')]
    summary['repeat_calls'] = len(repeats)
    summary['repeat_exact_matches'] = sum(r['raw'] == main_by_id[r['id']]['raw'] for r in repeats)
    summary['repeat_semantic_equivalent'] = review['repeats']['semantic_equivalent']
    summary['repeat_p50_seconds'] = statistics.median(r['total_seconds'] for r in repeats)
    summary['first_request_seconds'] = data['results'][0]['total_seconds']
    summary['calls'] = len(data['results'])
    summary['errors'] = sum('error' in r for r in data['results'])
    summary['truncated'] = sum(r.get('finish_reason') == 'length' for r in data['results'])
    summary['reasoning_returned'] = sum(r.get('reasoning_returned', False) for r in data['results'])
    usage = {key: sum(r.get('usage', {}).get(key, 0) for r in data['results'])
             for key in ['prompt_tokens', 'completion_tokens', 'prompt_cache_hit_tokens', 'prompt_cache_miss_tokens']}
    summary['usage_all_33_calls'] = usage
    assert usage['prompt_tokens'] == usage['prompt_cache_hit_tokens'] + usage['prompt_cache_miss_tokens']
    # Official CNY / 1M token rates, observed 2026-10-27. Estimate, not invoice.
    estimated = (.02 * usage['prompt_cache_hit_tokens'] + usage['prompt_cache_miss_tokens'] + 4 * usage['completion_tokens']) / 1e6
    summary['estimated_cost_cny_offpeak'] = estimated
    summary['estimated_cost_cny_peak'] = 2 * estimated
    summary['price_source'] = 'https://api-docs.deepseek.com/zh-cn/quick_start/pricing/'
    summary['index_unchanged'] = data['index_before'] == data['index_after'] and data['pointer_unchanged']
    summary['pid_unchanged'] = data['server_pid_before'] == data['server_pid_after']
    summary['pid'] = data['server_pid_after']
    (DIR / 'summary-deepseek.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
