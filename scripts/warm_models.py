import gc
import logging
import time
import torch
from wise.feature.feature_extractor_factory import FeatureExtractorFactory

logging.basicConfig(level=logging.INFO)
models = [
    'mlfoundations/open_clip/ViT-B-16-SigLIP2-256/webli',
    'transformers/owlv2/google/owlv2-base-patch16-ensemble',
    'deepinsight/insightface/buffalo_l/_unknown',
]
for model_id in models:
    t = time.monotonic()
    print('START', model_id, flush=True)
    extractor = FeatureExtractorFactory(model_id, {'default': {
        'device': 'cpu' if 'insightface' in model_id else 'mps',
        'compile': False, 'warmup': True,
    }})
    print('READY', model_id, round(time.monotonic()-t, 2), flush=True)
    del extractor
    gc.collect()
    torch.mps.empty_cache()
