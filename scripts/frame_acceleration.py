"""Apple Silicon frame-indexing acceleration, installed only by extract_frames.

Importing this module changes nothing. Never call enable() in the query server:
the class adapters intentionally live only for one extraction child process.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import logging
from pathlib import Path
import tempfile

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT/'data/model-cache/frame-acceleration'
MODELS = Path.home()/'.insightface/models/buffalo_l'
SHAPES = {'det_10g.onnx': (1,3,640,640), 'w600k_r50.onnx': (1,3,112,112),
          'genderage.onnx': (1,3,96,96)}
_enabled = False


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def weights_hash(model):
    digest = hashlib.sha256()
    for tensor in model.graph.initializer:
        digest.update(tensor.SerializeToString(deterministic=True))
    return digest.hexdigest()


@contextmanager
def cache_lock(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def static_model(source, cache_dir=CACHE):
    """Create a verified fixed-shape copy; never write the downloaded model."""
    import onnx
    source = Path(source).resolve()
    shape = SHAPES[source.name]
    source_hash = file_hash(source)
    key = hashlib.sha256((source_hash+repr(shape)+'fixed-input-v1').encode()).hexdigest()
    folder = Path(cache_dir)/'onnx'/key
    target, manifest = folder/source.name, folder/'manifest.json'
    with cache_lock(folder):
        if target.exists() and manifest.exists():
            record = json.loads(manifest.read_text())
            if record['source_sha256'] != source_hash or record['model_sha256'] != file_hash(target):
                raise RuntimeError(f'CoreML 模型缓存校验失败，请移除后重试：{folder}')
            return target
        model = onnx.load(source)
        original_weights = weights_hash(model)
        if len(model.graph.input)!=1 or len(model.graph.input[0].type.tensor_type.shape.dim)!=len(shape):
            raise ValueError(f'Unsupported InsightFace input shape: {source}')
        for dim, size in zip(model.graph.input[0].type.tensor_type.shape.dim,shape):
            dim.ClearField('dim_param'); dim.dim_value=size
        model = onnx.shape_inference.infer_shapes(model, strict_mode=False)
        properties = {item.key:item.value for item in model.metadata_props}
        properties['CACHE_KEY']=key[:63]
        onnx.helper.set_model_props(model,properties)
        onnx.checker.check_model(model)
        if weights_hash(model)!=original_weights:
            raise RuntimeError('Shape specialization changed InsightFace weights')
        with tempfile.NamedTemporaryFile(dir=folder, suffix='.onnx', delete=False) as tmp:
            temporary=Path(tmp.name)
        try:
            onnx.save(model,temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        record={'source_sha256':source_hash, 'model_sha256':file_hash(target),
                'initializer_sha256':original_weights, 'shape':shape}
        temporary=manifest.with_suffix('.tmp')
        temporary.write_text(json.dumps(record,indent=2)+'\n')
        temporary.replace(manifest)
    return target


def install_coreml(cache_dir=CACHE, source_dir=MODELS):
    import cv2  # Must precede ORT/InsightFace imports on supported platforms.
    import onnxruntime as ort
    from insightface.model_zoo import model_zoo
    if 'CoreMLExecutionProvider' not in ort.get_available_providers():
        raise RuntimeError('入库加速需要 CoreMLExecutionProvider；可用 VIDEO_FRAME_ACCELERATION=0 回退')
    original=model_zoo.PickableInferenceSession
    source_dir=Path(source_dir).resolve()
    cache_dir=Path(cache_dir).resolve()

    class IndexingSession(original):
        def __init__(self, model_path, **kwargs):
            source=Path(model_path).resolve()
            if source.parent!=source_dir:
                super().__init__(model_path,**kwargs)
                return
            if source.name not in SHAPES:
                # InsightFace initializes two landmarks which WISE discards.
                kwargs.update(providers=['CPUExecutionProvider'],provider_options=[{}])
                super().__init__(model_path,**kwargs)
                return
            prepared=static_model(source,cache_dir)
            compiled=cache_dir/'compiled'
            options={'ModelFormat':'MLProgram','MLComputeUnits':'ALL',
                     'RequireStaticInputShapes':'1','EnableOnSubgraphs':'0',
                     'AllowLowPrecisionAccumulationOnGPU':'0',
                     'ModelCacheDirectory':str(compiled)}
            kwargs.update(providers=['CoreMLExecutionProvider','CPUExecutionProvider'],
                          provider_options=[options,{}])
            session_options=kwargs.get('sess_options') or ort.SessionOptions()
            session_options.log_severity_level=3
            kwargs['sess_options']=session_options
            with cache_lock(compiled):
                super().__init__(str(prepared),**kwargs)
            if 'CoreMLExecutionProvider' not in self.get_providers():
                raise RuntimeError(f'CoreML 初始化失败，未发布入库结果：{source.name}')
            logger.info('Frame acceleration: %s providers=%s',source.name,self.get_providers())

    model_zoo.PickableInferenceSession=IndexingSession


def install_owl():
    import torch
    from transformers import Owlv2ForObjectDetection, Owlv2Processor, Owlv2ImageProcessorFast
    from wise.feature.transformers_owlv2 import TransformersOWLv2FeatureExtractor, TransformersOWLv2Model
    original_preprocess=Owlv2ImageProcessorFast.preprocess

    def preprocess(processor,*args,**kwargs):
        if str(kwargs.get('device','cpu')) not in ('cpu','None'):
            raise ValueError('Indexing fast processor must run on CPU')
        kwargs['device']='cpu'
        return original_preprocess(processor,*args,**kwargs)

    def processor(extractor):
        cached=extractor.__dict__.get('_frame_index_processor')
        if cached is None:
            cached=Owlv2Processor.from_pretrained(extractor.model_name,use_fast=True,local_files_only=True)
            if not isinstance(cached.image_processor,Owlv2ImageProcessorFast):
                raise RuntimeError('OWLv2 did not load the required fast image processor')
            extractor.__dict__['_frame_index_processor']=cached
            logger.info('Frame acceleration: OWLv2 processor=%s device=cpu',type(cached.image_processor).__name__)
        return cached

    def load_model(model):
        if torch.device(model.DEVICE).type!='mps':
            raise RuntimeError('Frame FP16 acceleration requires MPS')
        kwargs=dict(model.model_kwargs)
        if kwargs.get('quantization_config') is not None:
            raise ValueError('Frame FP16 acceleration cannot be combined with quantization')
        kwargs.pop('dtype',None);kwargs['torch_dtype']=torch.float16
        loaded=Owlv2ForObjectDetection.from_pretrained(model.model_id,**kwargs)
        if {p.dtype for p in loaded.parameters() if p.is_floating_point()}!={torch.float16}:
            raise RuntimeError('OWLv2 weights are not all float16')
        logger.info('Frame acceleration: OWLv2 weights=float16 device=mps')
        return loaded

    Owlv2ImageProcessorFast.preprocess=preprocess
    TransformersOWLv2FeatureExtractor.processor=property(processor)
    TransformersOWLv2Model._load_torch_model=load_model


def enable():
    global _enabled
    if _enabled:
        return
    import torch
    if not torch.backends.mps.is_available():
        raise RuntimeError('入库加速需要 Apple MPS；可用 VIDEO_FRAME_ACCELERATION=0 回退')
    install_coreml()
    install_owl()
    _enabled=True
    logger.info('Frame acceleration enabled: fast CPU preprocessing + MPS FP16 weights + CoreML; extraction process only')
