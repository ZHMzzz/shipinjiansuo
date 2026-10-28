"""Local FP32-accumulation candidate check, unchanged FP16 model weights."""
import json
import torch
import torch.nn.functional as F
import bench
from qwen_linear import OUT
from profile_microbatch import inputs
from microbatch_qwen import prepare_inputs
from action_search import ActionEngine
from profile_kernels import measure

torch.set_num_threads(1);data=inputs('qwen')[0];ext=ActionEngine().model();patch=ext._model.model.visual.patch_embed
prepared=prepare_inputs(ext,data);x=prepared['pixel_values_videos'].to(device='mps',dtype=torch.float16)
w=patch.proj.weight.reshape(patch.embed_dim,-1);b=patch.proj.bias
with torch.inference_mode():
 original=patch(x).float();fp16=F.linear(x,w,b).float();w32=w.float();b32=b.float()
 def fp32():return F.linear(x.float(),w32,b32).to(torch.float16)
 full=fp32().float();t,_=measure(fp32,5)
 result={'fp32_projection_seconds':t,'items':2}
 for name,value in [('linear_fp16',fp16),('linear_fp32',full)]:
  d=(original-value).abs();result[name]={'max_abs':float(d.max()),'mean_abs':float(d.mean()),'relative_l2':float(torch.linalg.vector_norm(original-value)/torch.linalg.vector_norm(original)),'exact_fraction':float((d==0).float().mean())}
bench.dump(OUT/'precision-probe.json',result);print(json.dumps(result,indent=2))
