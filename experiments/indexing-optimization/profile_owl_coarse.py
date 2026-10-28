"""Coarse synchronized OWLv2 scopes, preprocessed resident inputs, same work."""
import torch
import numpy as np
from profile_microbatch import build,inputs,Timers,OUT
import bench
import wise.feature.transformers_owlv2 as o

torch.set_num_threads(1)
ext,net,_,_,_=build('owl');data=inputs('owl')
pixels=[ext.processor(images=p,return_tensors='pt')['pixel_values'].to('mps') for p in data]
rows=[];active={'timers':None}
modules=[(net.owlv2.vision_model,'vision_model'),(net.class_head,'class_head'),(net.objectness_head,'objectness_head'),(net.box_head,'box_head')]
originals=[]
for module,name in modules:
 original=module.forward
 def wrap(*a,_original=original,_name=name,**kw):
  t=active['timers']
  if t is None:return _original(*a,**kw)
  with t.measure(_name):return _original(*a,**kw)
 module.forward=wrap;originals.append((module,original))
try:
 with torch.inference_mode():
  for b in [1,2]:
   for p in pixels:
    for part in p.split(b):o.get_object_features(net,part)
  torch.mps.synchronize()
  for measured in [False,True]:
   for turn,b in enumerate([1,2,2,1,2,1,1,2]):
    t=Timers();active['timers']=t if measured else None
    with t.measure('total'):
     for p in pixels:
      for part in p.split(b):o.get_object_features(net,part)
    rows.append({'module_sync_enabled':measured,'turn':turn,'batch':b,'logical_images':4,'timers':dict(t.rows)})
    print(measured,b,dict(t.rows),flush=True)
finally:
 for m,f in originals:m.forward=f
bench.dump(OUT/'owl-coarse.json',{'rows':rows,'scope':'model only; identical preprocessed resident MPS input tensors; excludes preprocessing/output conversion'})
