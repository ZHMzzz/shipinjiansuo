"""Aggregate thread factorial without treating CPU blocking time as GPU queue time."""
import contextlib
import io
import json
from pathlib import Path
import statistics as stats
import summarize_concurrency as old

OUT=Path(__file__).resolve().parent/'thread-contention-20261028-rerun'


def numbers(values):
    if not values:return None
    xs=sorted(values)
    return {'mean':stats.mean(xs),'min':min(xs),'max':max(xs),'p95':xs[min(len(xs)-1,int(len(xs)*.95))]}


def host_metrics(rows):
    if len(rows)<2:return None
    fields={'swap_in':('Swapins',),'swap_out':('Swapouts',),'page_in':('Pageins',),
            'page_out':('Pageouts',),'compression':('Compressions',),'decompression':('Decompressions',)}
    result={}
    for key,(field,) in fields.items():
        rates=[]
        for a,b in zip(rows,rows[1:]):
            delta=(b['vm_pages'][field]-a['vm_pages'][field])*b['page_size']/1024**2
            rates.append(delta/(b['at']-a['at']))
        result[key]={'total_mib':(rows[-1]['vm_pages'][field]-rows[0]['vm_pages'][field])*rows[0]['page_size']/1024**2,
                     'peak_mib_per_second':max(rates)}
    gpu=[r['gpu']['Device Utilization %'] for r in rows]
    result.update(gpu_percent=numbers(gpu),gpu_samples_ge90_percent=100*sum(x>=90 for x in gpu)/len(gpu),
                  cpu_percent=numbers([r['cpu_percent'] for r in rows]),sample_count=len(rows),
                  sample_cost_seconds=numbers([r['sample_cost_seconds'] for r in rows]))
    return result


def summarize():
    old.OUT=OUT
    with contextlib.redirect_stdout(io.StringIO()):old.summarize()
    summary=json.loads((OUT/'summary.json').read_text())
    raw=json.loads((OUT/'result.json').read_text())
    all_events=[]
    for path in (OUT/'events').glob('*.jsonl'):
        for line in path.read_text().splitlines():
            try:all_events.append(json.loads(line))
            except json.JSONDecodeError:pass
    trials=[]
    baseline_retrieval=raw['trials'][0].get('retrieval') if raw['trials'] else None
    for row in raw['trials']:
        if row['status']!='completed':continue
        result=next(x for x in summary['trials'] if x['name']==row['name'])
        result['threads']=row['threads']
        result['new_vector_counts']=row['new_vector_counts']
        result['qwen_max_abs_diff']=row.get('qwen_max_abs_diff',0)
        if baseline_retrieval:
            current=row['retrieval'];reference=baseline_retrieval
            def keys(items,fields):return [tuple(x[k] for k in fields) for x in items]
            result['quality']={
                'text_top10_identity_same_queries':sum(keys(a,('media_id','ts','te'))==keys(b,('media_id','ts','te')) for a,b in zip(current['qwen_top10'],reference['qwen_top10'])),
                'image_top5_video_order_same_queries':sum(keys(current['owl'][k]['videos_top5'],('media_id',))==keys(reference['owl'][k]['videos_top5'],('media_id',)) for k in current['owl']),
                'image_top10_media_time_order_same_queries':sum(keys(current['owl'][k]['regions_top10'],('media_id','time'))==keys(reference['owl'][k]['regions_top10'],('media_id','time')) for k in current['owl']),
                'face_candidate_order_same_queries':sum(keys(a,('media_id','time'))==keys(b,('media_id','time')) for a,b in zip(current['registered_faces'],reference['registered_faces']))}
        start,end=row['resources']['start'],row['resources']['end']
        samples=[]
        for line in (OUT/row['name']/'host.jsonl').read_text().splitlines():
            v=json.loads(line)
            if 'error' not in v and start<=v['at']<=end:samples.append(v)
        result['host']=host_metrics(samples)
        result['background']=host_metrics(row['background_samples'])
        result['waits']={};result['thread_proof']=[]
        worker=row['worker']['pid']
        parent=next(e['pid'] for e in all_events if e['kind']=='thread_setting' and e.get('trial')=='startup')
        for event in all_events:
            if event['kind']=='thread_setting' and event.get('trial')==row['name']:
                result['thread_proof'].append({k:event[k] for k in ('pid','requested','actual')})
            if not (start<=event['start'] and event['end']<=end) or event['kind']=='thread_setting':continue
            assert event.get('actual_threads',row['threads'])==row['threads'],event
            role='qwen_parent' if event['pid']==parent else 'frame_worker' if event['pid']==worker else 'owl_backend'
            key=role+':'+event['kind'];entry=result['waits'].setdefault(key,{'count':0,'total_seconds':0,'max_seconds':0})
            duration=event['end']-event['start'];entry['count']+=1;entry['total_seconds']+=duration;entry['max_seconds']=max(entry['max_seconds'],duration)
        proof={(p['pid'],p['requested'],p['actual']) for p in result['thread_proof']}
        result['thread_proof']=[dict(zip(('pid','requested','actual'),x)) for x in sorted(proof)]
        assert len(proof)==3 and all(p[1:]==(row['threads'],row['threads']) for p in proof),proof
        trials.append(result)
    groups={}
    for threads in (1,2,4):
        for mode in ('serial','full','cpu'):
            rows=[x for x in trials if x['threads']==threads and x['mode']==mode and 'warmup' not in x['name']]
            if not rows:continue
            group={'n':len(rows),'seconds':[r['total_seconds'] for r in rows]}
            for key in ('total_seconds','frame_task_seconds','qwen_task_seconds','rss_peak_gib','mps_driver_sum_peak_gib'):
                group[key]=numbers([r[key] for r in rows])
            group['gpu_mean_percent']=stats.mean(r['host']['gpu_percent']['mean'] for r in rows)
            group['background_gpu_mean_percent']=stats.mean(r['background']['gpu_percent']['mean'] for r in rows)
            group['gpu_ge90_sample_percent']=stats.mean(r['host']['gpu_samples_ge90_percent'] for r in rows)
            for key in ('swap_in','swap_out','compression','decompression'):
                group[key+'_mib_mean']=stats.mean(r['host'][key]['total_mib'] for r in rows)
            group['waits']={}
            for key in sorted({k for r in rows for k in r['waits']}):
                group['waits'][key]=stats.mean(r['waits'].get(key,{}).get('total_seconds',0) for r in rows)
            group['quality_by_trial']=[r.get('quality') for r in rows]
            group['qwen_max_abs_diff']=max(r['qwen_max_abs_diff'] for r in rows)
            group['all_outputs_exact']=all(r['outputs_exact'] for r in rows)
            group['all_retrieval_exact']=all(r['retrieval_exact'] for r in rows)
            groups[f'{mode}-t{threads}']=group
    baseline=groups.get('serial-t1',{}).get('total_seconds',{}).get('mean')
    for key,g in groups.items():
        g['reduction_vs_serial_t1_percent']=100*(1-g['total_seconds']['mean']/baseline) if baseline else None
    summary.update(groups=groups,trials=trials)
    summary['same_thread_consistency']={}
    for n in (1,2,4):
        rows=[r for r in raw['trials'] if r['status']=='completed' and r['threads']==n and 'warmup' not in r['name']]
        if not rows:continue
        ref=rows[0]
        summary['same_thread_consistency'][str(n)]={'n':len(rows),
            'frame_signatures_identical':all(r['frame_signatures']==ref['frame_signatures'] for r in rows),
            'qwen_vectors_identical':all(r['new_qwen_sha256']==ref['new_qwen_sha256'] for r in rows),
            'retrieval_identical':all(r['retrieval']==ref['retrieval'] for r in rows)}
    summary['image_order_changes']=[]
    for row in raw['trials']:
        if row['status']!='completed':continue
        for key,base in baseline_retrieval['owl'].items():
            a=[x['media_id'] for x in base['videos_top5']]
            b=[x['media_id'] for x in row['retrieval']['owl'][key]['videos_top5']]
            if a!=b:summary['image_order_changes'].append({'trial':row['name'],'query':key,'baseline':a,'current':b})
    summary['method_notes']=[
        'Three scheduling modes x torch CPU intra-op threads 1/2/4; 3 warm repetitions per cell with rotated/reversed order (recorded); one initial serial warmup excluded. This is not a fully counterbalanced positional design.',
        'Same original process_job and 35.44s media; actual thread count recorded at each child request. OMP_NUM_THREADS and OPENBLAS_NUM_THREADS environment remain 1; no production edits.',
        'IORegistry Device Utilization % is whole-device observational sampling; includes unrelated applications, no per-process attribution, no claimed exclusive access.',
        'vm_stat counters are whole-system interval deltas. Swap rates do not isolate this job. Sample interval is approximately 0.5s plus measurement cost.',
        'Existing Tensor.cpu on MPS and torch.mps.synchronize wrapped for CPU-visible blocking duration; includes outstanding compute, copies, scheduling; not pure GPU queue wait. No extra synchronizations added.',
        'Full-mode join and bounded producer/consumer queue waits include application-level waiting. Different wait categories/processes overlap; do not add to obtain total delay.',
        'Metal exact queue/compute trace unavailable: xctrace requires full Xcode; powermetrics requires admin password. No privilege changes attempted.',
        'Background host samples before each trial; no unrelated apps closed or production service stopped. Formal resident models remain loaded while the isolated experiment owns another set, so memory pressure is not identical to normal production reuse.',
        'Fixed-query consistency regression, not independent retrieval accuracy; model resident timings exclude first load and HTTP transfer/queue wait.'
    ]
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps({'status':summary['status'],'groups':{k:{'n':v['n'],'seconds':v['seconds'],'gpu':v['gpu_mean_percent'],'swap_in_mib':v['swap_in_mib_mean'],'swap_out_mib':v['swap_out_mib_mean'],'outputs_exact':v['all_outputs_exact'],'queries_exact':v['all_retrieval_exact']} for k,v in groups.items()}}))


if __name__=='__main__':summarize()
