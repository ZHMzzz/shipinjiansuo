"""Local offline visual verification. Does not import the production server."""
import os
os.environ.setdefault('HF_HUB_OFFLINE','1')
import json,time,math,hashlib,threading,platform,sys
from pathlib import Path
import numpy as np
import cv2,psutil,torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration,AutoProcessor
from qwen_vl_utils import process_vision_info
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent

def fingerprint(project):
 return {str(p.relative_to(project)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((project/'action-index').iterdir()) if p.is_file()}

def frames(case,count):
 cap=cv2.VideoCapture(str(ROOT/case['path']))
 try:
  fps=cap.get(cv2.CAP_PROP_FPS);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  assert fps>0 and n>0
  first=min(n-1,max(0,math.ceil(case['start']*fps)));last=min(n-1,max(first,math.ceil(case['end']*fps)-1))
  images=[];times=[]
  for i in np.linspace(first,last,count).astype(int):
   cap.set(cv2.CAP_PROP_POS_FRAMES,int(i));ok,frame=cap.read()
   if not ok:raise RuntimeError('Video frame decode failed')
   images.append(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)));times.append(float(i/fps))
  return images,times
 finally:cap.release()

class Sampler:
 def __init__(self):self.stop=threading.Event();self.values=[];self.thread=threading.Thread(target=self.run,daemon=True)
 def run(self):
  process=psutil.Process()
  while not self.stop.is_set():
   self.values.append({'time':time.time(),'rss':process.memory_info().rss,'available':psutil.virtual_memory().available,'swap':psutil.swap_memory().used,'mps_driver':torch.mps.driver_allocated_memory()})
   self.stop.wait(.5)
 def start(self):self.thread.start()
 def close(self):self.stop.set();self.thread.join()

def main():
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--design',default='design.json');parser.add_argument('--output',default='results.json');parser.add_argument('--no-repeat',action='store_true');args=parser.parse_args()
 design=json.loads((HERE/args.design).read_text());info=json.loads((HERE/'model.json').read_text())
 output=HERE/args.output
 if output.exists():raise SystemExit('Preserve results.json; use a separate experiment for changes.')
 project=(ROOT/'projects/demo').resolve();before=fingerprint(project)
 torch.set_num_threads(1);torch.manual_seed(0);torch.mps.set_per_process_memory_fraction(.55)
 monitor=Sampler();monitor.start();started=time.perf_counter()
 model=Qwen3VLForConditionalGeneration.from_pretrained(info['local_path'],dtype=torch.float16,device_map='mps',attn_implementation='sdpa',local_files_only=True).eval()
 processor=AutoProcessor.from_pretrained(info['local_path'],local_files_only=True)
 torch.mps.synchronize();load_seconds=time.perf_counter()-started
 report={'design_file':args.design,'settings':{k:v for k,v in design.items() if k!='cases'},'revision':info['revision'],'model_load_seconds':load_seconds,'environment':{'platform':platform.platform(),'torch':torch.__version__,'ram':psutil.virtual_memory().total},'project':str(project),'index_before':before,'results':[]}
 def save():output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
 def infer(case,phase):
  t=time.perf_counter();times=[]
  if case['type']=='video':
   count=case.get('frames',design['frames']);imgs,times=frames(case,count)
   visual={'type':'video','video':imgs,'sample_fps':count/(case['end']-case['start']),'min_pixels':4096,'max_pixels':case.get('max_pixels_per_frame',design['max_pixels_per_frame']),'total_pixels':case.get('total_pixels',design['total_pixels'])}
  else:visual={'type':'image','image':Image.open(ROOT/case['path']).convert('RGB'),'min_pixels':4096,'max_pixels':design['image_max_pixels']}
  messages=[{'role':'system','content':case.get('system',design['system'])},{'role':'user','content':[visual,{'type':'text','text':case.get('prompt','请核验：'+case['query'])}]}]
  text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
  images,videos,kwargs=process_vision_info(messages,image_patch_size=16,return_video_kwargs=True,return_video_metadata=True)
  if videos is not None:videos,metadata=map(list,zip(*videos))
  else:metadata=None
  inputs=processor(text=[text],images=images,videos=videos,video_metadata=metadata,return_tensors='pt',do_resize=False,**kwargs).to('mps')
  torch.mps.synchronize();prepared=time.perf_counter()
  with torch.inference_mode():generated=model.generate(**inputs,max_new_tokens=design['max_new_tokens'],do_sample=False)
  torch.mps.synchronize();ended=time.perf_counter()
  tokens=generated[:,inputs['input_ids'].shape[1]:]
  answer=processor.batch_decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
  try:
   obj=json.loads(answer.strip().removeprefix('```json').removesuffix('```').strip())
   if case.get('mode')=='helmet_observation':
    heads=obj.get('heads');scene=obj.get('scene')
    if not isinstance(heads,list) or not all(h in ('helmet','bare','unclear') for h in heads) or scene not in ('work','other','unclear'):match='invalid'
    elif scene=='other':match='no'
    elif scene=='unclear' or not heads:match='unknown'
    elif 'bare' in heads:match='yes'
    elif 'unclear' in heads:match='unknown'
    else:match='no'
   else:match=obj['match'] if obj.get('match') in ('yes','no','unknown') else 'invalid'
  except (ValueError,KeyError,TypeError):obj={};match='invalid'
  result={'id':case['id'],'phase':phase,'group':case['group'],'query':case['query'],'expected':case['expected'],'match':match,'answer':answer,'sample_times':times,'input_tokens':int(inputs['input_ids'].shape[1]),'output_tokens':int(tokens.shape[1]),'preprocess_seconds':prepared-t,'inference_seconds':ended-prepared,'total_seconds':ended-t,'rss_after':psutil.Process().memory_info().rss,'mps_driver_after':torch.mps.driver_allocated_memory()}
  if 'pixel_values_videos' in inputs:result['video_grid_thw']=inputs['video_grid_thw'].tolist()
  return result
 try:
  print('MODEL_LOADED',load_seconds,flush=True)
  report['warmup']=infer(design['cases'][0],'warmup');save()
  todo=[(c,'main') for c in design['cases']]
  repeat=[c for c in design['cases'] if c['group']=='helmet_candidates'][:5]
  if not args.no_repeat:todo += [(c,f'repeat{r}') for r in (1,2) for c in repeat]
  for n,(case,phase) in enumerate(todo):
   result=infer(case,phase);report['results'].append(result);save()
   print(n+1,len(todo),case['id'],phase,result['match'],round(result['total_seconds'],2),result['answer'],flush=True)
 finally:
  monitor.close();report['resource_samples']=monitor.values
  report['index_after']=fingerprint(project);report['pointer_unchanged']=(ROOT/'projects/demo').resolve()==project
  save()
 assert report['index_after']==before and report['pointer_unchanged']
 print('DONE',flush=True)
if __name__=='__main__':main()
