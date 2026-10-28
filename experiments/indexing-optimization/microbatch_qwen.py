"""Independent Qwen segment microbatch experiment, unchanged production decode/settings."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace
import bench

OUT=Path(os.environ.get('MICROBATCH_OUT', bench.OUT/'microbatch-20261028')).resolve()

def prepare_inputs(extractor, frames_list):
    import torch
    from torchvision.transforms import functional as TF
    from qwen_vl_utils import process_vision_info
    processor=extractor._processor
    conversations=[]
    for frames in frames_list:
        frames=frames.cpu()
        if frames.dtype!=torch.uint8:frames=(frames*255).clamp(0,255).to(torch.uint8)
        conversations.append([{'role':'user','content':[{'type':'video','video':[TF.to_pil_image(f) for f in frames],'total_pixels':extractor.max_total_pixels}]}])
    texts=[processor.apply_chat_template(m,tokenize=False,add_generation_prompt=True) for m in conversations]
    images,video_inputs,kwargs=process_vision_info(conversations,image_patch_size=16,return_video_metadata=True,return_video_kwargs=True)
    videos,metadata=map(list,zip(*video_inputs))
    return processor(text=texts,images=images,videos=videos,video_metadata=metadata,return_tensors='pt',padding=True,do_resize=False,**kwargs)

def worker(batch,repeat):
    import numpy as np
    import torch
    import psutil
    from action_search import ActionEngine,windows,SETTINGS
    from resource_metrics import ResourceSampler
    torch.set_num_threads(1)
    path=OUT/f'qwen-b{batch}-r{repeat}.json'
    report={'batch':batch,'repeat':repeat,'settings':SETTINGS,'status':'running','groups':[]}
    design=json.loads((bench.OUT/'design.json').read_text())
    media=[m for m in design['media'] if m['id'] in design['sample_media_ids']]
    class Capture:
        def extract_video_segment_features(self,frames):
            self.frames=frames
            return SimpleNamespace(vectors=np.zeros((1,2048),dtype='float32'))
    capture=Capture();decoder=ActionEngine();decoder.model=lambda:capture
    def decode(m,seg):
        decoder.encode_segment(m,seg)
        return capture.frames
    groups=[];segments=[]
    for m in media:
        items=[dict(media_id=m['id'],ts=a,te=b) for a,b in windows(m['duration'])]
        for i in range(0,len(items),batch):groups.append((m,items[i:i+batch]))
        segments.extend(items)
    report['segments']=segments
    engine=ActionEngine();t=time.perf_counter();ext=engine.model();report['load_seconds']=time.perf_counter()-t
    assert str(next(ext._model.parameters()).dtype)=='torch.float16'
    report['dtype']=str(next(ext._model.parameters()).dtype)
    @torch.inference_mode()
    def encode(frames):
        if batch==1:return ext.extract_video_segment_features(frames[0]).vectors
        inputs={k:v.to(ext.DEVICE) for k,v in prepare_inputs(ext,frames).items()}
        outputs=ext._model(**inputs)
        x=ext._last_token_pool(outputs.last_hidden_state,inputs['attention_mask']).float()
        return (x/x.norm(dim=-1,keepdim=True)).cpu().numpy()
    # Verify candidate preprocessing against the actual single-segment method,
    # including variable-length right padding. No forward execution in this audit.
    import wise.feature.qwen3vl as module
    src=textwrap.dedent(inspect.getsource(type(ext).extract_video_segment_features))
    assert src.count('outputs = model(**inputs)')==1
    src=src.replace('outputs = model(**inputs)','return inputs')
    ns={};exec(compile(src,'<actual-single-preprocessor>','exec'),vars(module),ns)
    actual_single=ns['extract_video_segment_features']
    if batch==2 and repeat==1:
        validations=[]
        checks=list(groups)
        # Explicit cross-video shape case is validation only, not throughput timing.
        mixed=[decode(media[0],{'ts':0,'te':min(8,media[0]['duration'])}),decode(media[-2],{'ts':0,'te':min(8,media[-2]['duration'])})]
        for gi in range(len(checks)+1):
            frames=mixed if gi==len(checks) else [decode(checks[gi][0],s) for s in checks[gi][1]]
            combined=prepare_inputs(ext,frames);singles=[{k:v.cpu() for k,v in actual_single(ext,f).items()} for f in frames]
            assert set(combined)==set(singles[0]),(combined.keys(),singles[0].keys())
            lengths=[];offset=0
            for i,single in enumerate(singles):
                length=int(single['attention_mask'].sum());lengths.append(length)
                assert torch.equal(combined['input_ids'][i,:length],single['input_ids'][0])
                assert combined['attention_mask'][i,:length].all() and not combined['attention_mask'][i,length:].any()
                grid=single['video_grid_thw'];assert torch.equal(combined['video_grid_thw'][i:i+1],grid)
                n=len(single['pixel_values_videos'])
                assert torch.equal(combined['pixel_values_videos'][offset:offset+n],single['pixel_values_videos'])
                offset+=n
            assert offset==len(combined['pixel_values_videos'])
            if gi==len(checks):assert len(set(lengths))>1,'Mixed-shape case must exercise unequal token lengths'
            validations.append({'case':gi,'mixed_shapes':gi==len(checks),'lengths':lengths,'padded_length':int(combined['input_ids'].shape[1]),'exact_input_match':True})
        report['preprocessing_validation']=validations
        # Compare actual embeddings for the mixed-shape case outside timed rounds.
        with torch.inference_mode():
            single_vectors=np.concatenate([ext.extract_video_segment_features(f).vectors for f in mixed])
            mixed_vectors=encode(mixed)
        report['mixed_shape_cosines']=(single_vectors*mixed_vectors).sum(axis=1).tolist()
        del mixed,combined,singles,frames
    warm=[decode(groups[0][0],s) for s in groups[0][1]]
    t=time.perf_counter();encode(warm);torch.mps.synchronize();report['warmup_seconds']=time.perf_counter()-t
    del warm
    vectors=[];hashes=[]
    with ResourceSampler() as resources:
        for gi,(m,segs) in enumerate(groups):
            assert psutil.virtual_memory().available>2*1024**3,'Available RAM below 2GiB; stop own experiment'
            t=time.perf_counter();frames=[decode(m,s) for s in segs];decode_seconds=time.perf_counter()-t
            hashes.extend(hashlib.sha256(f.numpy().tobytes()).hexdigest() for f in frames)
            torch.mps.synchronize();t=time.perf_counter();value=encode(frames);torch.mps.synchronize();seconds=time.perf_counter()-t
            assert value.shape[0]==len(segs) and np.isfinite(value).all()
            vectors.extend(value)
            report['groups'].append({'media_id':m['id'],'count':len(segs),'decode_seconds':decode_seconds,'encode_seconds':seconds,'raw_shapes':[list(f.shape) for f in frames]})
            bench.dump(path,report)
            print(f'qwen b{batch} r{repeat} group {gi+1}/{len(groups)}: {seconds:.3f}s',flush=True)
            del frames,value
    report['resources']=resources.result
    report['input_hashes']=hashes
    report['encode_seconds']=sum(g['encode_seconds'] for g in report['groups'])
    report['decode_seconds']=sum(g['decode_seconds'] for g in report['groups'])
    report['decode_and_encode_seconds']=report['encode_seconds']+report['decode_seconds']
    report['status']='completed';np.save(path.with_suffix('.npy'),np.stack(vectors));bench.dump(path,report)

def suite():
    OUT.mkdir(parents=True,exist_ok=True)
    if any(OUT.glob('qwen-b*.json')):raise FileExistsError('Use a fresh MICROBATCH_OUT; existing trials are immutable')
    before=bench.fingerprint();bench.dump(OUT/'production-before-qwen.json',before)
    from action_search import read_snapshot
    meta,vectors,_=read_snapshot(Path(before['project']),complete=True)
    bench.dump(OUT/'qwen-background.json',meta)
    import numpy as np
    np.save(OUT/'qwen-background.npy',vectors)
    env={**os.environ,'HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'}
    for batch,rep in [(1,1),(2,1),(2,2),(1,2)]:
        path=OUT/f'qwen-b{batch}-r{rep}.json';assert not path.exists()
        with path.with_suffix('.log').open('w') as log:
            result=subprocess.run([sys.executable,'-u',__file__,'worker','--batch',str(batch),'--repeat',str(rep)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
        print(path.name,'exit',result.returncode,flush=True)
        if result.returncode:raise RuntimeError(str(path.with_suffix('.log')))
    after=bench.fingerprint();bench.dump(OUT/'production-after-qwen.json',after);assert before==after

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['suite','worker']);p.add_argument('--batch',type=int,choices=[1,2]);p.add_argument('--repeat',type=int);a=p.parse_args()
    suite() if a.mode=='suite' else worker(a.batch,a.repeat)
