"""Fixed gates, no threshold search and no decisions based on labels."""
import json
import statistics
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

def best_per_video(indices, full_scores, segments):
    seen, output = set(), []
    for i in sorted(indices, key=lambda i: (-full_scores[i], i)):
        mid = segments[i]['media_id']
        if mid not in seen:
            output.append(i)
            seen.add(mid)
    return output

def condition_filter(eligible, groups, texts, minimum=.35, gap=.15):
    """OR between groups, AND across conditions at the SAME segment index."""
    checks = [[(texts[t]['scores'], max(minimum, max(texts[t]['scores'])-gap)) for t in g] for g in groups]
    if not checks or any(not group for group in checks):
        raise ValueError('Empty groups must explicitly fall back, never vacuous truth')
    return [i for i in eligible if any(all(scores[i] >= cut for scores, cut in group) for group in checks)]

def main():
    design = json.loads((HERE/'design.json').read_text())
    raw = json.loads((HERE/'scores.json').read_text())
    plans = {p['id']:p for p in json.loads((HERE/'plans.json').read_text())}
    parser = {r['id']:r for r in json.loads((HERE/'parser-results.json').read_text())['results'] if r['phase']=='main'}
    segments, texts = raw['segments'], raw['texts']
    universe = set(raw['previous_label_universe'])
    rule = design['rule']
    report = {'rule': rule, 'cases': [], 'cohorts': {}, 'new_unlabelled_media': sorted(set(s['media_id'] for s in segments)-universe)}
    for case in design['cases']:
        p = plans[case['id']]
        scores = texts[case['query']]['scores']
        best = max(scores)
        eligible = [i for i, s in enumerate(scores) if best >= rule['baseline_min_best'] and s >= max(rule['baseline_min_result'], best-rule['baseline_gap'])]
        baseline = best_per_video(eligible, scores, segments)
        bset = {segments[i]['media_id'] for i in baseline}
        positive = set(case['positive_media'])
        negative = (bset & universe) if case.get('expected_empty') else set(case['negative_media']) & bset
        row = {'id':case['id'], 'query':case['query'], 'cohort':case['cohort'], 'baseline_ids':sorted(bset),
               'baseline_positive_ids':sorted(positive & bset), 'baseline_false_positive_ids':sorted(negative),
               'baseline_unlabelled_ids':sorted(bset-positive-negative), 'baseline_missed_known_positive_ids':sorted(positive-bset),
               'fallback_reasons':p['fallback_reasons'], 'parser_seconds':parser[case['id']]['total_seconds'], 'variants':{}}
        variants = {'auto': p['groups_normalized'], 'raw': p['groups_raw'], 'manual': [case['conditions']] if case['conditions'] else None}
        for name, groups in variants.items():
            fallback = groups is None or (name != 'manual' and bool(p['fallback_reasons']))
            started = time.perf_counter()
            keep = eligible if fallback else condition_filter(eligible, groups, texts, rule['condition_min'], rule['condition_gap'])
            chosen = best_per_video(keep, scores, segments)
            elapsed = time.perf_counter()-started
            after = {segments[i]['media_id'] for i in chosen}
            assert after <= bset
            def describe(i):
                return dict(segments[i], index=i, full_score=scores[i], groups=[] if fallback else [
                    [{'text':t, 'score':texts[t]['scores'][i], 'cutoff':max(rule['condition_min'],max(texts[t]['scores'])-rule['condition_gap'])} for t in g] for g in groups])
            row['variants'][name] = {'after_ids':sorted(after), 'positive_deleted_ids':sorted((positive & bset)-after),
                'false_removed_ids':sorted(negative-after), 'false_remaining_ids':sorted(negative & after),
                'unlabelled_after_ids':sorted(after-positive-negative), 'unlabelled_removed_ids':sorted((bset-positive-negative)-after),
                'selected_windows':[describe(i) for i in chosen], 'baseline_best_windows':[describe(i) for i in baseline],
                'filter_seconds':elapsed, 'fallback':fallback, 'groups':groups}
        report['cases'].append(row)
    for cohort in ['original26','paragraph6','logic3','all']:
        rows = [r for r in report['cases'] if cohort=='all' or r['cohort']==cohort]
        stats = {'queries':len(rows),'baseline_candidates':sum(len(r['baseline_ids']) for r in rows),
                 'baseline_positive':sum(len(r['baseline_positive_ids']) for r in rows),
                 'baseline_false_positive':sum(len(r['baseline_false_positive_ids']) for r in rows),
                 'baseline_unlabelled':sum(len(r['baseline_unlabelled_ids']) for r in rows),
                 'baseline_missed_known_positive':sum(len(r['baseline_missed_known_positive_ids']) for r in rows),
                 'fallback_queries':sum(bool(r['fallback_reasons']) for r in rows),'variants':{}}
        for variant in ['auto','raw','manual']:
            stats['variants'][variant] = {name:sum(len(r['variants'][variant][field]) for r in rows) for name,field in [
                ('after_candidates','after_ids'), ('positive_deleted','positive_deleted_ids'), ('false_removed','false_removed_ids'),
                ('false_remaining','false_remaining_ids'), ('unlabelled_remaining','unlabelled_after_ids'), ('unlabelled_removed','unlabelled_removed_ids')]}
        report['cohorts'][cohort] = stats
    timing = raw['fresh_query_timings']
    combined = [dict(t, parser_seconds=parser[t['id']]['total_seconds'], composed_seconds=parser[t['id']]['total_seconds']+t['query_seconds']+t['conditions_seconds']) for t in timing]
    report['timing'] = {'parser_median_seconds':statistics.median(r['total_seconds'] for r in parser.values()),
                        'model_load_seconds':raw['model_load_seconds'], 'fresh_examples':combined,
                        'conditions_median_seconds':statistics.median(t['conditions_seconds'] for t in timing),
                        'composed_median_seconds':statistics.median(t['composed_seconds'] for t in combined),
                        'note':'Separate API call and fresh local encoding durations added; not measured integrated endpoint latency.'}
    report['index_unchanged'] = raw['index_before']==raw['index_after'] and raw['pointer_unchanged']
    report['pid_unchanged'] = raw['server_pid_before']==raw['server_pid_after']
    (HERE/'evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report['cohorts'],ensure_ascii=False,indent=2))
    for row in report['cases']:
        a=row['variants']['auto']
        print(row['id'], row['baseline_ids'], '->',a['after_ids'], 'lost',a['positive_deleted_ids'],'FP_removed',a['false_removed_ids'],'fallback',row['fallback_reasons'])

if __name__ == '__main__':
    main()
