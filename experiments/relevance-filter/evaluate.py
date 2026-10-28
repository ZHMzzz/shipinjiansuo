"""Replay saved pre-filter results without model inference or API changes."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.semantic_filter import filter_semantic_hits,MIN_BEST_SCORE,MIN_RESULT_SCORE,MAX_SCORE_GAP
folder=Path(__file__).resolve().parent
rows=json.loads((folder/'baseline.json').read_text())
report={'settings':dict(min_best=MIN_BEST_SCORE,min_result=MIN_RESULT_SCORE,max_gap=MAX_SCORE_GAP),'cases':[]}
for row in rows:
    kept=filter_semantic_hits(row['items'],score_key='score')
    ids={h['id'] for h in kept}; targets=set(row['known_positive_media'])
    report['cases'].append(dict(split=row['split'],query=row['query'],before=len(row['items']),after=len(kept),retained_media=sorted(ids),
        known_positive_media=sorted(targets),missed_positive_media=sorted(targets-ids),expected_empty=row['expected_empty'],
        passed=not ids if row['expected_empty'] else targets<=ids))
for split in ('calibration','holdout'):
    cases=[c for c in report['cases'] if c['split']==split]
    report[split]=dict(cases=len(cases),passed=sum(c['passed'] for c in cases),before=sum(c['before'] for c in cases),after=sum(c['after'] for c in cases))
(folder/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='cases'},ensure_ascii=False))
