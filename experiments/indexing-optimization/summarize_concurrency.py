"""Aggregate balanced warm runs and time-aligned sampled allocator metrics."""
import bisect
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
OUT=HERE/'qwen-frame-concurrency-20261028'


def summarize():
    result=json.loads((OUT/'result.json').read_text())
    traces={}
    for path in (OUT/'mps').glob('*.jsonl'):
        rows=[]
        for line in path.read_text().splitlines():
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:pass
        if rows:traces[int(path.stem)]=([r['at'] for r in rows],rows)
    trials=[]
    for row in result['trials']:
        if row['status']!='completed':continue
        memory=row['resources'];start,end=memory['start'],memory['end']
        rss_rows=memory['samples'];rss_times=[x['at'] for x in rss_rows]
        moments=sorted({x['at'] for _,rows in traces.values() for x in rows if start<=x['at']<=end})
        driver=0;allocated=0;max_age=0
        for at in moments:
            sample=rss_rows[max(0,bisect.bisect_right(rss_times,at)-1)]
            d=a=0
            for pid,(times,rows) in traces.items():
                if pid not in sample['pids']:continue
                index=bisect.bisect_right(times,at)-1
                if index<0:continue
                value=rows[index];d+=value['driver'];a+=value['allocated'];max_age=max(max_age,at-value['at'])
            driver=max(driver,d);allocated=max(allocated,a)
        events={e['name']:e['seconds'] for e in row['schedule_events']}
        overlap=max(0,min(events['frame_end'],events['qwen_gpu_end'])-max(events['frame_start'],events['qwen_gpu_start']))
        trials.append({'name':row['name'],'mode':row['mode'],'total_seconds':row['process_job_seconds'],
            'frame_task_seconds':row['frame_worker_seconds'],
            'qwen_task_seconds':events['qwen_gpu_end']-events['qwen_gpu_start'],
            'frame_qwen_task_overlap_seconds':overlap,
            'rss_peak_gib':memory['rss_peak_bytes']/1024**3,
            'uss_peak_gib':memory['uss_peak_bytes']/1024**3 if memory['uss_peak_bytes'] is not None else None,
            'mps_driver_sum_peak_gib':driver/1024**3,'mps_tensor_sum_peak_gib':allocated/1024**3,
            'mps_max_sample_age_seconds':max_age,
            'system_available_min_gib':memory['system_available_min_bytes']/1024**3,
            'system_swap_out_delta_mib':memory['swap_out_delta_bytes']/1024**2,
            'system_swap_used_before_gib':memory['swap_used_before_bytes']/1024**3,
            'outputs_exact':row['frame_outputs_exact_vs_serial'] and row['qwen_vectors_exact_vs_serial'],
            'retrieval_exact':row['retrieval_exact_vs_serial'],'stages':row['stages']})
    groups={}
    for mode in ('serial','full','cpu'):
        rows=[x for x in trials if x['mode']==mode and 'warmup' not in x['name']]
        if not rows:continue
        groups[mode]={'n':len(rows),'seconds':[x['total_seconds'] for x in rows],
            'mean_seconds':statistics.mean(x['total_seconds'] for x in rows),
            'median_seconds':statistics.median(x['total_seconds'] for x in rows),
            'all_outputs_exact':all(x['outputs_exact'] for x in rows),
            'all_retrieval_exact':all(x['retrieval_exact'] for x in rows)}
        for key in ['rss_peak_gib','mps_driver_sum_peak_gib','mps_tensor_sum_peak_gib','frame_task_seconds','qwen_task_seconds','frame_qwen_task_overlap_seconds']:
            values=[x[key] for x in rows];groups[mode][key]={'mean':statistics.mean(values),'min':min(values),'max':max(values)}
    if 'serial' in groups:
        base=groups['serial']['mean_seconds']
        for value in groups.values():value['mean_time_reduction_percent']=100*(1-value['mean_seconds']/base)
    final={'status':result['status'],'source_video_seconds':35.44,
        'scope':'Actual process_job on isolated clone of current 44-video library; each trial adds one duplicate content video. Qwen and frame models resident after serial warmup. No HTTP transfer, queue wait or online query contention.',
        'order':[x['name'] for x in trials],'groups':groups,'trials':trials,
        'retrieval_scope':{'text_queries':73,'text_result':'FAISS Top10 segment records and scores',
            'reference_images':9,'image_result':'Top10 regions and Top5 distinct videos from top100 regions, including times/boxes/scores',
            'registered_faces':5,'face_result':'All candidates at threshold 0.45, times and scores'},
        'memory_notes':['RSS is sum over benchmark parent and descendants, including shared-page double counting; live server excluded.',
            'Cross-process USS is denied by macOS and is unavailable; not reported as zero.',
            'MPS counters are sampled in Qwen, frame and OWLv2 model processes at 100ms; sums aligned by timestamps. Tensor allocation and driver reserved memory overlap and must not be added to RSS or each other.',
            'MPS allocator metrics do not include all CoreML allocations. System available memory and swap include unrelated processes; existing swap does not imply every byte was created by this test.',
            'Peaks are sampled, not exact hardware peaks.'],
        'method_notes':['Full mode starts original Qwen encoding with frame extraction, joins both before index construction; native writes remain serial.',
            'CPU mode copies original decode/processor steps into a CPU producer, queue capacity 2 (plus in-flight preparation/consumption); Qwen forward stays after frame extraction and index creation.',
            'Schedule timestamps delimit tasks, including CPU work/waits; they are not Metal kernel traces.',
            'Fixed query regression measures consistency, not an independent accuracy or recall benchmark.',
            'Aborted attempt1 stopped at original disk safety threshold; its partial data is excluded. Only regenerated experimental buffers/indexes were removed; production remained unchanged.'],
        'production_unchanged':result.get('production_unchanged')}
    (OUT/'summary.json').write_text(json.dumps(final,ensure_ascii=False,indent=2))
    print(json.dumps({'status':final['status'],'groups':groups},ensure_ascii=False,indent=2))


if __name__=='__main__':summarize()
