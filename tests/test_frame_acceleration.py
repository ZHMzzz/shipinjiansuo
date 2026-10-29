import importlib
import json

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper


def test_importing_entrypoint_does_not_patch_query_models():
    from wise.feature.transformers_owlv2 import TransformersOWLv2FeatureExtractor, TransformersOWLv2Model
    from insightface.model_zoo import model_zoo
    from transformers import Owlv2ImageProcessorFast
    before=(TransformersOWLv2FeatureExtractor.__dict__['processor'],
            TransformersOWLv2Model._load_torch_model,
            model_zoo.PickableInferenceSession,Owlv2ImageProcessorFast.preprocess)
    importlib.import_module('scripts.frame_acceleration')
    importlib.import_module('scripts.extract_frames')
    importlib.import_module('scripts.frame_worker')
    assert before==(TransformersOWLv2FeatureExtractor.__dict__['processor'],
                    TransformersOWLv2Model._load_torch_model,
                    model_zoo.PickableInferenceSession,Owlv2ImageProcessorFast.preprocess)


def test_rollback_skips_adapters_but_runs_the_same_wise_command(monkeypatch):
    from scripts import extract_frames, frame_acceleration
    from wise._main import extract_features
    calls=[]
    monkeypatch.setenv('VIDEO_FRAME_ACCELERATION','0')
    monkeypatch.setattr(frame_acceleration,'enable',lambda:pytest.fail('rollback must not install adapters'))
    monkeypatch.setattr(extract_features,'main',lambda argv:calls.append(argv))
    extract_frames.main(['incoming','--project-dir','staging','--num-workers','0'])
    assert calls==[['extract-features','incoming','--project-dir','staging','--num-workers','0']]


def test_acceleration_failure_does_not_silently_run_baseline(monkeypatch):
    from scripts import extract_frames, frame_acceleration
    from wise._main import extract_features
    monkeypatch.setenv('VIDEO_FRAME_ACCELERATION','1')
    def fail():raise RuntimeError('CoreML unavailable')
    monkeypatch.setattr(frame_acceleration,'enable',fail)
    monkeypatch.setattr(extract_features,'main',lambda argv:pytest.fail('must not write any index after failed setup'))
    with pytest.raises(RuntimeError,match='CoreML unavailable'):
        extract_frames.main(['incoming','--project-dir','staging'])


def test_missing_coreml_does_not_change_session_factory(tmp_path,monkeypatch):
    import onnxruntime as ort
    from insightface.model_zoo import model_zoo
    from scripts.frame_acceleration import install_coreml
    previous=model_zoo.PickableInferenceSession
    monkeypatch.setattr(ort,'get_available_providers',lambda:['CPUExecutionProvider'])
    with pytest.raises(RuntimeError,match='CoreMLExecutionProvider'):
        install_coreml(tmp_path)
    assert model_zoo.PickableInferenceSession is previous
    assert not list(tmp_path.iterdir())


def test_static_model_keeps_source_and_weights_and_detects_tampering(tmp_path):
    from scripts.frame_acceleration import static_model, file_hash, weights_hash
    source=tmp_path/'det_10g.onnx'
    shape=['batch',3,'height','width']
    graph=helper.make_graph([helper.make_node('Mul',['input','weight'],['output'])],'test',
        [helper.make_tensor_value_info('input',TensorProto.FLOAT,shape)],
        [helper.make_tensor_value_info('output',TensorProto.FLOAT,shape)],
        [numpy_helper.from_array(np.array([2.0],dtype=np.float32),'weight')])
    onnx.save(helper.make_model(graph),source)
    original=file_hash(source); original_weights=weights_hash(onnx.load(source))
    cached=static_model(source,tmp_path/'cache')
    model=onnx.load(cached)
    assert cached!=source and file_hash(source)==original
    assert weights_hash(model)==original_weights
    assert [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]==[1,3,640,640]
    assert static_model(source,tmp_path/'cache')==cached
    record=json.loads((cached.parent/'manifest.json').read_text())
    assert record['source_sha256']==original and record['initializer_sha256']==original_weights
    cached.write_bytes(b'corrupt cached model')
    with pytest.raises(RuntimeError,match='缓存校验失败'):
        static_model(source,tmp_path/'cache')
    assert file_hash(source)==original
