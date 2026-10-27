"""Download/warm only the encoders used by the live application."""
import logging
import gc
import torch
from wise.feature.feature_extractor_factory import FeatureExtractorFactory
from action_search import ActionEngine

logging.basicConfig(level=logging.INFO)
face = FeatureExtractorFactory('deepinsight/insightface/buffalo_l/_unknown',
    {'default': {'device':'cpu', 'compile':False, 'warmup':True}})
print('READY InsightFace', flush=True)
owl = FeatureExtractorFactory('transformers/owlv2/google/owlv2-base-patch16-ensemble',
    {'default': {'device':'mps', 'compile':False, 'warmup':True}})
print('READY OWLv2', flush=True)
del owl
gc.collect(); torch.mps.empty_cache()
engine = ActionEngine()
engine.query_vector('一个人正在走路')
print('READY Qwen3-VL-Embedding 2B FP16', flush=True)
