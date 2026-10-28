"""Small serial screening for actual FP16 weights versus MPS autocast.

Uses the real WISE extractor. This is not an end-to-end indexing benchmark.
"""
import argparse
import gc
import json
from pathlib import Path
import statistics
import time

from bench import ROOT, OUT, OWL, dump
from owl_variants import install_variant, describe_processor


def main():
    import torch
    from PIL import Image
    from resource_metrics import ResourceSampler
    from wise.feature.transformers_owlv2 import TransformersOWLv2FeatureExtractor as Ext
    parser=argparse.ArgumentParser()
    parser.add_argument('--variants',nargs='+',default=['baseline','fp16','fp16_weights'])
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    paths=[ROOT/'experiments/owl-qwen-comparison'/f'source-{i}.png' for i in [1,13,21]]
    images=[Image.open(p).convert('RGB') for p in paths]
    report={'scope':'three images, three measured warm repeats each; no decode, database or indexing',
            'images':[str(p) for p in paths],'runs':[]}
    for name in args.variants:
        install_variant(name)
        started=time.perf_counter()
        ext=Ext(OWL,device='mps',warmup=False,config=Ext.Config(device='mps',compile=False,warmup=False))
        model=ext.model.model
        tensors=[ext.preprocess_image([im]) for im in images]
        description=describe_processor(ext)
        observed={}
        hook=model.owlv2.vision_model.encoder.layers[0].self_attn.q_proj.register_forward_hook(
            lambda module, inputs, output: observed.update(projection_output_dtype=str(output.dtype)))
        ext.extract_image_features(tensors[0]);torch.mps.synchronize()
        hook.remove()
        row={'name':name,'load_and_warmup_seconds':time.perf_counter()-started,
             'parameter_dtype':str(next(model.parameters()).dtype),**observed,'processor':description,'samples':[]}
        with ResourceSampler() as resources:
            for repeat in range(3):
                for i,tensor in enumerate(tensors):
                    torch.mps.synchronize();started=time.perf_counter()
                    features=ext.extract_image_features(tensor)
                    torch.mps.synchronize()
                    row['samples'].append({'image':i,'repeat':repeat,'seconds':time.perf_counter()-started,
                                           'regions':len(features[0].vectors)})
        row['resources']=resources.result
        row['median_seconds']=statistics.median(s['seconds'] for s in row['samples'])
        report['runs'].append(row);dump(args.output,report)
        print(name,round(row['median_seconds'],4),row['parameter_dtype'],flush=True)
        # WISE caches models by wrapper instance, so dropping ext alone does
        # not release model weights between variants in this isolated process.
        ext.model._build_torch_model.cache_clear()
        del model,ext,tensors,features
        gc.collect();torch.mps.empty_cache()
    install_variant('baseline')


if __name__=='__main__':main()
