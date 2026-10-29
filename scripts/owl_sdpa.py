"""Indexing-only OWLv2 SDPA bridge. Importing this module has no side effects.

The legacy decoder and query server keep their original Torch environment.
One serial frame worker owns one lazy model subprocess and reusable mmap buffers.
"""
import atexit
import json
import logging
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'data/model-cache/owl-sdpa'
MODEL = 'google/owlv2-base-patch16-ensemble'
VERSIONS = {'torch': '2.14.1', 'torchvision': '0.29.1', 'transformers': '5.21.0'}
SHAPES = {'pixels': (1, 3, 960, 960), 'scores': (1, 3600),
          'embeddings': (1, 3600, 513), 'boxes': (1, 3600, 4)}
logger = logging.getLogger(__name__)
_installed = False
_bridge = None


def runtime_python():
    # Do not resolve the executable symlink: Python needs the venv's bin path.
    path = Path(os.environ.get('VIDEO_OWL_SDPA_PYTHON', str(CACHE / 'runtime/bin/python'))).expanduser().absolute()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError('OWLv2 SDPA 运行时未就绪：执行 ./scripts/setup_owl_sdpa.sh；'
                           '或设置 VIDEO_OWL_SDPA=0 回退。路径：' + str(path))
    return path


class Bridge:
    def __init__(self, python=None, cache_dir=CACHE, timeout=180):
        import numpy as np
        self.process = None
        self.result_fd = None
        self.temp = None
        self.arrays = {}
        self.buffer = b''
        self.timeout = timeout
        self.calls = 0
        self.wall = 0.
        executable = python or runtime_python()
        try:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            self.temp = tempfile.TemporaryDirectory(prefix='shared-', dir=cache_dir)
            self.arrays = {key: np.memmap(Path(self.temp.name) / (key + '.bin'),
                           mode='w+', dtype='float32', shape=shape) for key, shape in SHAPES.items()}
            self.result_fd, write_fd = os.pipe()
            env = {k: v for k, v in os.environ.items() if k not in ('PYTHONHOME', 'PYTHONPATH')}
            env.update(OWL_SDPA_RESULT_FD=str(write_fd), HF_HUB_OFFLINE='1',
                       PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1')
            try:
                self.process = subprocess.Popen([str(executable), '-u', str(Path(__file__).resolve()),
                    '--backend', self.temp.name], cwd=ROOT, stdin=subprocess.PIPE,
                    text=True, pass_fds=(write_fd,), env=env)
            finally:
                os.close(write_fd)
            self.ready = self.receive('ready')
            logger.info('OWL_SDPA_BACKEND %s', json.dumps(self.ready))
        except BaseException:
            self.close()
            raise

    def receive(self, expected):
        deadline = time.monotonic() + self.timeout
        while b'\n' not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.result_fd], [], [], remaining)[0]:
                raise TimeoutError('OWLv2 SDPA 推理进程超时，当前入库任务不会发布')
            block = os.read(self.result_fd, 65536)
            if not block:
                raise RuntimeError('OWLv2 SDPA 推理进程意外退出')
            self.buffer += block
        line, self.buffer = self.buffer.split(b'\n', 1)
        value = json.loads(line)
        if value.get('event') != expected:
            raise RuntimeError('OWLv2 SDPA 失败：' + str(value.get('error', value)))
        return value

    def request(self, command):
        self.process.stdin.write(json.dumps(command) + '\n')
        self.process.stdin.flush()
        return self.receive('done')

    def images(self, images):
        import numpy as np
        import torch
        if not isinstance(images, torch.Tensor) or tuple(images.shape) != SHAPES['pixels']:
            raise ValueError('OWLv2 SDPA 入库要求 batch=1、960×960 输入')
        started = time.perf_counter()
        np.copyto(self.arrays['pixels'], images.detach().cpu().numpy())
        self.request({'op': 'image'})
        # Preserve existing MPS sorting and CPU thresholding; finish copies
        # before allowing the backend to overwrite shared arrays on the next call.
        result = {key: torch.from_numpy(value).to('mps')
                  for key, value in self.arrays.items() if key != 'pixels'}
        torch.mps.synchronize()
        self.calls += 1
        self.wall += time.perf_counter() - started
        return result

    def text(self, inputs):
        import torch
        if set(inputs) != {'input_ids', 'attention_mask'}:
            raise ValueError('OWLv2 文本预热需要 input_ids 和 attention_mask')
        result = self.request({'op': 'text', 'inputs': {k: v.cpu().tolist() for k, v in inputs.items()}})
        return torch.tensor(result['vector'], dtype=torch.float32, device='mps')

    def close(self):
        process, self.process = self.process, None
        try:
            if process is not None:
                try:
                    process.stdin.close()
                except (OSError, ValueError):
                    pass
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        finally:
            if self.result_fd is not None:
                os.close(self.result_fd)
                self.result_fd = None
            for array in self.arrays.values():
                array._mmap.close()
            self.arrays.clear()
            if self.temp is not None:
                self.temp.cleanup()
                self.temp = None


def close():
    global _bridge
    bridge, _bridge = _bridge, None
    if bridge is not None:
        logger.info('OWL_SDPA_BRIDGE images=%s wall_with_communication=%.3f', bridge.calls, bridge.wall)
        bridge.close()


def install():
    global _installed
    if _installed:
        return
    runtime_python()  # Fail before patching/writing when setup is missing.
    from wise.feature.transformers_owlv2 import TransformersOWLv2Model

    def get_bridge(model):
        global _bridge
        if model.model_id != MODEL or str(model.DEVICE) != 'mps':
            raise ValueError('OWLv2 SDPA 仅支持已验证的 base-patch16-ensemble / MPS 配置')
        if model.model_kwargs.get('quantization_config') is not None:
            raise ValueError('OWLv2 SDPA 不支持量化配置')
        if _bridge is None:
            _bridge = Bridge()
        return _bridge

    TransformersOWLv2Model.get_image_features = lambda model, **kw: get_bridge(model).images(kw.get('images'))
    TransformersOWLv2Model.get_text_features = lambda model, **kw: get_bridge(model).text(kw)
    # Both CLI extraction and the resident worker must release the model child
    # when the server terminates them. Worker failures also exit through atexit.
    def terminate(*_):
        raise SystemExit('帧提取进程停止')
    signal.signal(signal.SIGTERM, terminate)
    atexit.register(close)
    _installed = True
    logger.info('OWLv2 indexing backend=sdpa runtime=%s', runtime_python())


def check_runtime():
    from importlib.metadata import version
    actual = {name: version(name) for name in VERSIONS}
    if actual != VERSIONS:
        raise RuntimeError(f'OWLv2 SDPA 依赖版本不符：{actual}，要求 {VERSIONS}')
    import torch
    if not torch.backends.mps.is_available():
        raise RuntimeError('OWLv2 SDPA 需要 Apple MPS')
    sys.path.insert(0, str(ROOT / 'wise/src'))
    from wise.feature.transformers_owlv2 import get_object_features, get_text_embeddings
    return actual


def backend(folder):
    # Report even dependency/model-loading failures through the result channel.
    with os.fdopen(int(os.environ['OWL_SDPA_RESULT_FD']), 'w', buffering=1) as pipe:
        def send(value):
            pipe.write(json.dumps(value) + '\n')
        arrays = {}
        try:
            versions = check_runtime()
            import numpy as np
            import torch
            from transformers import Owlv2ForObjectDetection
            from wise.feature.transformers_owlv2 import get_object_features, get_text_embeddings
            torch.set_num_threads(1)
            arrays = {key: np.memmap(Path(folder) / (key + '.bin'), mode='r+', dtype='float32', shape=shape)
                      for key, shape in SHAPES.items()}
            model = Owlv2ForObjectDetection.from_pretrained(MODEL, local_files_only=True,
                        torch_dtype=torch.float16, attn_implementation='sdpa').eval().to('mps')
            if model.config._attn_implementation != 'sdpa' or {p.dtype for p in model.parameters()} != {torch.float16}:
                raise RuntimeError('OWLv2 未启用要求的 SDPA / FP16 权重')
            send({'event': 'ready', **versions, 'attention': 'sdpa', 'dtype': 'float16',
                  'model': MODEL, 'pid': os.getpid()})
            with torch.inference_mode():
                for line in sys.stdin:
                    command = json.loads(line)
                    if command['op'] == 'image':
                        result = get_object_features(model, torch.from_numpy(arrays['pixels']).to('mps'))
                        for key, tensor in zip(('scores', 'embeddings', 'boxes'), result):
                            np.copyto(arrays[key], tensor.float().cpu().numpy())
                        send({'event': 'done'})
                    elif command['op'] == 'text':
                        inputs = {k: torch.tensor(v, device='mps') for k, v in command['inputs'].items()}
                        vector = get_text_embeddings(model, **inputs)
                        send({'event': 'done', 'vector': vector.float().cpu().tolist()})
                    else:
                        raise ValueError('Unknown OWLv2 SDPA operation')
        except BaseException as exc:
            send({'event': 'failed', 'error': repr(exc)})
            raise
        finally:
            for array in arrays.values():
                array._mmap.close()


if __name__ == '__main__':
    if sys.argv[1:] == ['--check']:
        print(json.dumps(check_runtime()))
    elif len(sys.argv) == 3 and sys.argv[1] == '--backend':
        backend(sys.argv[2])
    else:
        raise SystemExit('Usage: owl_sdpa.py --check | --backend SHARED_DIRECTORY')
