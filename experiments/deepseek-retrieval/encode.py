"""Read existing vectors, encode actual parser output, never publish indexes."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from scripts.action_search import ActionEngine, read_snapshot, SETTINGS
sys.path.insert(0, str(ROOT / 'experiments/text-query-parser'))
from run import fingerprint, listener

def main():
    target = HERE / 'scores.json'
    if target.exists():
        raise SystemExit('Preserve scores')
    design = json.loads((HERE / 'design.json').read_text())
    plans = json.loads((HERE / 'plans.json').read_text())
    old = json.loads((ROOT / 'experiments/compound-filter/scores-extended.json').read_text())
    project = (ROOT / 'projects/demo').resolve()
    before = fingerprint(project)
    meta, matrix, _ = read_snapshot(project, complete=True)
    can_reuse = old['index_after'] == before and old['segments'] == meta['segments']
    current_media = {m['id']:m for m in meta['media']}
    for m in old['media']:
        assert all(current_media[m['id']][k] == m[k] for k in ['path','duration'])
    texts = [c['query'] for c in design['cases']]
    for p in plans:
        if not p['fallback_reasons']:
            texts.extend(t for groups in (p['groups_raw'], p['groups_normalized']) for g in groups for t in g)
    texts.extend(t for c in design['cases'] for t in c['conditions'])
    texts = list(dict.fromkeys(texts))
    records = {t: dict(old['texts'][t], source='previous_identical_index') for t in texts if can_reuse and t in old['texts']}
    engine = ActionEngine()
    report = {'settings': SETTINGS, 'design_sha256': hashlib.sha256((HERE/'design.json').read_bytes()).hexdigest(),
              'plans_sha256': hashlib.sha256((HERE/'plans.json').read_bytes()).hexdigest(),
              'project': str(project), 'index_before': before, 'server_pid_before': listener(),
              'previous_label_universe': [m['id'] for m in old['media']], 'previous_scores_reusable': can_reuse,
              'segments': meta['segments'], 'media': [{k:v for k,v in m.items() if k!='location'} for m in meta['media']], 'texts': records}
    engine.query_vector('视频中的场景和动作')
    report['model_load_seconds'] = engine.last_load_seconds
    new = [t for t in texts if t not in records]
    for n, text in enumerate(new, 1):
        t = time.perf_counter()
        vector, cached = engine.query_vector(text)
        records[text] = {'seconds': time.perf_counter()-t, 'cached': cached, 'source': 'fresh', 'scores': (matrix @ vector[0]).tolist()}
        (HERE/'progress.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print(f'{n}/{len(new)} {records[text]["seconds"]:.3f}s {text}', flush=True)
    # Fresh uncached local encoding costs for representative full queries + conditions.
    selected = ['outdoor_run','helmet_bike','write','smoke','paragraph_fire','logic_or']
    by_id = {p['id']:p for p in plans}
    timings = []
    for cid in selected:
        p = by_id[cid]
        engine.query_vectors.clear()
        query_start = time.perf_counter()
        engine.query_vector(p['query'])
        full_seconds = time.perf_counter()-query_start
        conditions = list(dict.fromkeys(t for g in p['groups_normalized'] for t in g)) if not p['fallback_reasons'] else []
        condition_start = time.perf_counter()
        for text in conditions:
            engine.query_vector(text)
        timings.append({'id':cid, 'query_seconds':full_seconds,'conditions_seconds':time.perf_counter()-condition_start,'condition_count':len(conditions)})
    report['fresh_query_timings'] = timings
    report['index_after'] = fingerprint(project)
    report['pointer_unchanged'] = project == (ROOT/'projects/demo').resolve()
    report['server_pid_after'] = listener()
    assert before == report['index_after'] and report['pointer_unchanged']
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    (HERE/'progress.json').unlink(missing_ok=True)
    print('FINISHED', len(meta['media']), 'videos', len(meta['segments']), 'segments', len(new), 'new texts', flush=True)

if __name__ == '__main__':
    main()
