import json,statistics,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
source=sys.argv[1] if len(sys.argv)>1 else 'results.json'
r=json.loads((HERE/source).read_text()); main=[dict(x) for x in r['results'] if x['phase']=='main']
design=json.loads((HERE/r.get('design_file','design.json')).read_text());labels={c['id']:c['expected'] for c in design['cases']}
for x in main:x['expected']=labels[x['id']]
summary={'model_load_seconds':r['model_load_seconds'],'warmup_seconds':r['warmup']['total_seconds'],'completed_cases':len(main),'groups':{},'queries':{},'repeated_candidate_batches':{}}
for group in sorted({x['group'] for x in main}):
 rows=[x for x in main if x['group']==group];labelled=[x for x in rows if x['expected'] is not None]
 summary['groups'][group]={'cases':len(rows),'labelled':len(labelled),'correct':sum(x['expected']==x['match'] for x in labelled),'true_positive':sum(x['expected']=='yes' and x['match']=='yes' for x in labelled),'positives':sum(x['expected']=='yes' for x in labelled),'false_positive':sum(x['expected']=='no' and x['match']=='yes' for x in labelled),'negatives':sum(x['expected']=='no' for x in labelled),'unknown_or_invalid':sum(x['match'] in ('unknown','invalid') for x in rows),'total_seconds':sum(x['total_seconds'] for x in rows),'median_seconds':statistics.median(x['total_seconds'] for x in rows)}
for q in dict.fromkeys(x['query'] for x in main):
 rows=[x for x in main if x['query']==q]
 summary['queries'][q]=[{'id':x['id'],'expected':x['expected'],'match':x['match'],'seconds':x['total_seconds']} for x in rows]
for phase in ['main','repeat1','repeat2']:
 rows=[x for x in r['results'] if x['phase']==phase and x['group']=='helmet_candidates']
 summary['repeated_candidate_batches'][phase]={'count':len(rows),'first5_seconds':sum(x['total_seconds'] for x in rows[:5]),'all_seconds':sum(x['total_seconds'] for x in rows),'decisions':[x['match'] for x in rows]}
values=r.get('resource_samples',[])
if values:summary['resources']={'max_rss_GiB':max(x['rss'] for x in values)/1024**3,'max_mps_driver_GiB':max(x['mps_driver'] for x in values)/1024**3,'min_system_available_GiB':min(x['available'] for x in values)/1024**3,'swap_GiB_range':[min(x['swap'] for x in values)/1024**3,max(x['swap'] for x in values)/1024**3]}
summary['index_unchanged']=r.get('index_before')==r.get('index_after') and r.get('pointer_unchanged')
(HERE/('summary.json' if source=='results.json' else Path(source).stem+'-summary.json')).write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k!='queries'},ensure_ascii=False,indent=2))
