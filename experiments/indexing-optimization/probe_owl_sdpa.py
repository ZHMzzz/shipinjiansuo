"""Small real-frame SDPA feasibility probe. Not a retrieval-quality approval."""
import json
from pathlib import Path
import statistics
import time
from types import MethodType
import bench


def main():
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    from frame_acceleration import enable
    from action_search import media_catalog
    from wise.feature.feature_extractor_factory import FeatureExtractorFactory
    from wise.feature.transformers_owlv2 import get_object_features
    torch.set_num_threads(1);enable()
    before=bench.fingerprint()
    media=next(m for m in media_catalog(Path(before['project'])) if m['id']==40)
    source=Path(media['location'])/media['path'];source=source if source.is_absolute() else bench.ROOT/source
    cap=cv2.VideoCapture(str(source));frames=[]
    for sec in (0,5,15,30):
        cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000);ok,bgr=cap.read();assert ok
        frames.append(torch.from_numpy(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).permute(2,0,1).unsqueeze(0))
    cap.release()
    ext=FeatureExtractorFactory(bench.OWL,{'default':{'device':'mps','compile':False,'warmup':True}})
    net=ext.model.model
    layers=net.owlv2.vision_model.encoder.layers
    originals=[layer.self_attn.forward for layer in layers]
    def replacement(attn,original):
        def forward(self,hidden_states,attention_mask=None,causal_attention_mask=None,output_attentions=False):
            if attention_mask is not None or causal_attention_mask is not None or output_attentions or self.training:
                return original(hidden_states,attention_mask,causal_attention_mask,output_attentions)
            b,n,d=hidden_states.shape
            q=self._shape(self.q_proj(hidden_states)*self.scale,n,b)
            k=self._shape(self.k_proj(hidden_states),n,b);v=self._shape(self.v_proj(hidden_states),n,b)
            out=F.scaled_dot_product_attention(q,k,v,dropout_p=0.0,scale=1.0)
            return self.out_proj(out.transpose(1,2).reshape(b,n,d)),None
        return MethodType(forward,attn)
    candidates=[replacement(layer.self_attn,old) for layer,old in zip(layers,originals)]
    def select(variant):
        for layer,method in zip(layers,originals if variant=='baseline' else candidates):layer.self_attn.forward=method
    rows=[];samples={}
    try:
        with torch.inference_mode():
            for variant in ('baseline','sdpa'):
                select(variant)
                for x in frames:ext.extract_image_features(x)
            for turn,variant in enumerate(['baseline','sdpa','sdpa','baseline']*2):
                select(variant);torch.mps.synchronize();start=time.perf_counter()
                result=[ext.extract_image_features(x) for x in frames];torch.mps.synchronize()
                rows.append({'turn':turn,'variant':variant,'four_frames_seconds':time.perf_counter()-start,
                             'regions':[len(x[0].vectors) for x in result]})
            # Raw patch alignment avoids confusing sorted/thresholded region IDs.
            for variant in ('baseline','sdpa'):
                select(variant);values=[]
                for x in frames:
                    pixels=ext.processor(images=x,return_tensors='pt')['pixel_values'].to('mps')
                    values.append(tuple(v.float().cpu().numpy() for v in get_object_features(net,pixels)))
                samples[variant]=values
        means={v:statistics.mean(r['four_frames_seconds'] for r in rows if r['variant']==v) for v in ('baseline','sdpa')}
        diffs=[]
        for base,new in zip(samples['baseline'],samples['sdpa']):
            scores,emb,boxes=base;s2,e2,b2=new
            a=emb.reshape(-1,emb.shape[-1]).astype('float64');b=e2.reshape(-1,e2.shape[-1]).astype('float64')
            cosine=(a*b).sum(-1)/(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1))
            diffs.append({'max_objectness_difference':float(np.abs(scores-s2).max()),
                'objectness_threshold_crossings':int(((scores>=.02)!=(s2>=.02)).sum()),
                'aligned_patch_cosine_min':float(cosine.min()),'aligned_patch_cosine_median':float(np.median(cosine)),
                'max_box_component_difference':float(np.abs(boxes-b2).max())})
        report={'status':'completed','frames_seconds':[0,5,15,30],'rows':rows,'mean_seconds':means,
                'time_reduction_percent':100*(1-means['sdpa']/means['baseline']),'differences':diffs,
                'scope':'Resident full OWLv2 image feature extraction including preprocessing and postprocessing; 4 frames, ABBA twice. No startup, database, full indexing or retrieval-quality evaluation.',
                'production_unchanged':before==bench.fingerprint()}
        assert report['production_unchanged']
        bench.dump(bench.OUT/'frame-breakdown-20261028/sdpa-probe.json',report)
        print(json.dumps(report,ensure_ascii=False),flush=True)
    finally:select('baseline')

if __name__=='__main__':main()
