"""Replay same-window gates and select a rule without using holdout labels."""
import json,sys,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.semantic_filter import MIN_BEST_SCORE,MIN_RESULT_SCORE,MAX_SCORE_GAP
HERE=Path(__file__).resolve().parent

def evaluate(design,raw):
    segments=raw['segments'];texts=raw['texts']; mids=[s['media_id'] for s in segments]
    def ranked(indices,scores):
        seen=set();out=[]
        for i in sorted(indices,key=lambda i:(-scores[i],i)):
            if mids[i] not in seen:out.append(i);seen.add(mids[i])
        return out
    bases={}
    for c in design['cases']:
        scores=texts[c['query']]['scores'];best=max(scores)
        eligible=[i for i,s in enumerate(scores) if best>=MIN_BEST_SCORE and s>=max(MIN_RESULT_SCORE,best-MAX_SCORE_GAP)]
        bases[c['id']]=(eligible,ranked(eligible,scores))
    def run_rule(minimum,gap):
        cases=[]
        for c in design['cases']:
            scores=texts[c['query']]['scores'];eligible,base=bases[c['id']]
            conditions=[texts[t]['scores'] for t in c['conditions']]
            cuts=[max(minimum,max(ss)-gap) if gap is not None else minimum for ss in conditions]
            def passes(i):return all(ss[i]>=cut for ss,cut in zip(conditions,cuts))
            chosen=ranked([i for i in eligible if passes(i)],scores)
            best_only=[i for i in base if passes(i)]
            bset={mids[i] for i in base};aset={mids[i] for i in chosen}
            pos=set(c['positive_media']);neg=bset if c.get('expected_empty') else set(c['negative_media'])&bset
            cases.append(dict(id=c['id'],split=c['split'],query=c['query'],conditions=c['conditions'],cutoffs=cuts,
                baseline_ids=sorted(bset),after_ids=sorted(aset),best_only_ids=sorted({mids[i] for i in best_only}),
                known_positive_ids=sorted(pos),baseline_positive_ids=sorted(pos&bset),newly_deleted_positive_ids=sorted((pos&bset)-aset),
                baseline_false_positive_ids=sorted(neg),removed_false_positive_ids=sorted(neg-aset),remaining_false_positive_ids=sorted(neg&aset),
                baseline_windows=[dict(index=i,**segments[i],score=scores[i],condition_scores=[ss[i] for ss in conditions],passes=passes(i)) for i in base],
                selected_windows=[dict(index=i,**segments[i],score=scores[i],condition_scores=[ss[i] for ss in conditions]) for i in chosen]))
        stats={}
        for split in ['calibration','holdout','supplemental']:
            subset=[c for c in cases if c['split']==split]
            stats[split]={k:sum(len(c[field]) for c in subset) for k,field in [('before','baseline_ids'),('after','after_ids'),('positive_before','baseline_positive_ids'),('positive_deleted','newly_deleted_positive_ids'),('false_before','baseline_false_positive_ids'),('false_removed','removed_false_positive_ids')]}
        return {'minimum':minimum,'gap':gap,**stats,'cases':cases}
    rules=[run_rule(m,g) for m in design['rules']['minimum_condition_scores'] for g in design['rules']['maximum_condition_gaps']]
    safe=[r for r in rules if r['calibration']['positive_deleted']==0]
    chosen=min(safe,key=lambda r:(-r['calibration']['false_removed'],r['minimum'],-(r['gap'] if r['gap'] is not None else 10))) if safe else None
    report={'selection':design['selection'],'selected':chosen,'grid':[{k:v for k,v in r.items() if k!='cases'} for r in rules]}
    seconds=[v['seconds'] for v in texts.values()]
    report['encoding']={'unique_texts':len(seconds),'median_seconds':statistics.median(seconds),'min_seconds':min(seconds),'max_seconds':max(seconds),'model_load_seconds':raw['model_load_seconds']}
    report['index_unchanged']=raw['index_before']==raw['index_after'] and raw['pointer_unchanged']
    return report

if __name__=='__main__':
    result=evaluate(json.loads((HERE/'design.json').read_text()),json.loads((HERE/('scores-extended.json' if (HERE/'scores-extended.json').exists() else 'scores.json')).read_text()))
    (HERE/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    selected=result['selected']
    print(json.dumps({k:v for k,v in result.items() if k not in ('grid','selected')},ensure_ascii=False))
    if selected:
        print('SELECTED', {k:v for k,v in selected.items() if k!='cases'})
        for c in selected['cases']:print(c['id'],c['baseline_ids'],'->',c['after_ids'],'lost:',c['newly_deleted_positive_ids'])
    else:print('No rule preserves known calibration positives.')
