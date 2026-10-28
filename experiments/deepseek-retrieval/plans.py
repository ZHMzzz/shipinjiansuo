"""Deterministic compiler for this offline experiment; no per-case edits."""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

def normalize(text):
    text = re.sub(r'人物[A-Z0-9]+', '有人', text)
    return re.sub(r'工人[A-Z0-9]+', '工人', text)

def compile_plan(result):
    plan = {'id': result['id'], 'query': result['query'], 'groups_raw': [], 'groups_normalized': [], 'fallback_reasons': [], 'unverified_relations': []}
    if not result.get('schema_valid') or not result.get('parsed'):
        plan['fallback_reasons'] = ['invalid_parser_output']
        return plan
    parsed = result['parsed']
    if parsed['uncertain']:
        plan['fallback_reasons'].append('uncertainty')
    if not parsed['groups']:
        plan['fallback_reasons'].append('no_groups')
    for group in parsed['groups']:
        if group['must_not']:
            plan['fallback_reasons'].append('negative_constraints')
        if not group['must']:
            plan['fallback_reasons'].append('empty_mandatory_group')
        mandatory = '；'.join(group['must'])
        relations = '；'.join(group['relations'])
        if re.search(r'未|没有|没戴|不戴|不是|不在|不佩戴|无人', mandatory):
            plan['fallback_reasons'].append('negative_fact_in_must')
        if re.search(r'先.*(?:再|后|接着)|随后|之后|接着', mandatory + relations):
            plan['fallback_reasons'].append('temporal_order')
        subjects = set(re.findall(r'人物[A-Z0-9]+', mandatory + relations))
        if len(subjects) > 1 or re.search(r'至少|至多|只有|两名|两人|两个人|多人|不同的人|不同人物|另一个', mandatory + relations):
            plan['fallback_reasons'].append('count_or_multiple_people')
        raw = list(dict.fromkeys(group['must']))
        plan['groups_raw'].append(raw)
        plan['groups_normalized'].append(list(dict.fromkeys(normalize(s) for s in raw)))
        plan['unverified_relations'].extend(group['relations'])
    plan['fallback_reasons'] = sorted(set(plan['fallback_reasons']))
    plan['status'] = 'baseline_fallback' if plan['fallback_reasons'] else 'approximate_condition_gate'
    return plan

def main():
    target = HERE / 'plans.json'
    if target.exists():
        raise SystemExit('Preserve pre-scoring plans')
    design = json.loads((HERE / 'design.json').read_text())
    results = json.loads((HERE / 'parser-results.json').read_text())
    rows = [r for r in results['results'] if r['phase'] == 'main']
    assert {r['id'] for r in rows} == {c['id'] for c in design['cases']}
    plans = [compile_plan(r) for r in rows]
    target.write_text(json.dumps(plans, ensure_ascii=False, indent=2)+'\n')
    for p in plans:
        print(p['id'], p.get('status'), p['fallback_reasons'], p['groups_normalized'])

if __name__ == '__main__':
    main()
