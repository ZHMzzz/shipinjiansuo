"""Experiment-only OWLv2 RPC: old decoder/processor, new official SDPA model."""
import codecs
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import time

import bench
from frame_worker import FrameWorker

RUNTIME = bench.ROOT/'data/model-cache/owl-optimization-runtime/torch214/bin/python'
SHAPES = {'pixels': (1, 3, 960, 960), 'scores': (1, 3600),
          'embeddings': (1, 3600, 513), 'boxes': (1, 3600, 4)}


class SDPAWorker(FrameWorker):
    def _start(self, log):
        read_fd, write_fd = os.pipe()
        self.result_fd = read_fd
        try:
            self.process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'front'],
                cwd=self.root, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                pass_fds=(write_fd,), env={**os.environ, 'HF_HUB_OFFLINE': '1',
                    'PYTHONUNBUFFERED': '1', 'VIDEO_FRAME_RESULT_FD': str(write_fd)})
        finally:
            os.close(write_fd)
        self._result_buffer = b''; self._log_buffer = ''
        self._decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        assert self._receive(self.startup_seconds, log)['event'] == 'ready'


class Bridge:
    def __init__(self):
        import numpy as np
        root = bench.ROOT/'data/model-cache/owl-optimization-runtime'
        self.temp = tempfile.TemporaryDirectory(prefix='shared-', dir=root)
        self.arrays = {key: np.memmap(Path(self.temp.name)/(key+'.bin'), mode='w+', dtype='float32', shape=shape)
                       for key, shape in SHAPES.items()}
        rfd, wfd = os.pipe()
        self.pipe = os.fdopen(rfd)
        try:
            self.process = subprocess.Popen([str(RUNTIME), '-u', str(Path(__file__).resolve()), 'backend', self.temp.name],
                cwd=bench.ROOT, stdin=subprocess.PIPE, text=True, pass_fds=(wfd,),
                env={**os.environ, 'OWL_RPC_FD': str(wfd), 'HF_HUB_OFFLINE': '1'})
        finally:
            os.close(wfd)
        self.calls = 0; self.wall = 0.
        try:
            ready = self.receive()
            assert ready['event'] == 'ready', ready
            print('OWL_SDPA_BACKEND', json.dumps(ready), flush=True)
        except BaseException:
            self.close(); raise

    def receive(self):
        if not select.select([self.pipe], [], [], 180)[0]:
            raise TimeoutError('OWLv2 SDPA backend timed out')
        line = self.pipe.readline()
        if not line:
            raise RuntimeError('OWLv2 SDPA backend exited')
        value = json.loads(line)
        if value['event'] == 'failed':
            raise RuntimeError(value['error'])
        return value

    def request(self, value):
        self.process.stdin.write(json.dumps(value)+'\n'); self.process.stdin.flush()
        return self.receive()

    def images(self, images):
        import numpy as np
        import torch
        assert tuple(images.shape) == SHAPES['pixels']
        started = time.perf_counter()
        np.copyto(self.arrays['pixels'], images.detach().cpu().numpy())
        assert self.request({'op': 'image'})['event'] == 'done'
        # Preserve the existing MPS sort and CPU threshold path in WISE.
        result = {key: torch.from_numpy(value).to('mps') for key, value in self.arrays.items() if key != 'pixels'}
        torch.mps.synchronize()
        self.calls += 1; self.wall += time.perf_counter()-started
        return result

    def text(self, inputs):
        import torch
        result = self.request({'op': 'text', 'inputs': {key: value.cpu().tolist() for key, value in inputs.items()}})
        return torch.tensor(result['vector'], dtype=torch.float32, device='mps')

    def close(self):
        process = getattr(self, 'process', None)
        if process:
            try:
                process.stdin.close()
            except OSError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
        self.pipe.close()
        for array in self.arrays.values():
            array._mmap.close()
        self.arrays.clear(); self.temp.cleanup()


def front():
    from frame_acceleration import enable
    enable()
    from wise.feature.transformers_owlv2 import TransformersOWLv2Model
    bridge = None
    def get_bridge():
        nonlocal bridge
        if bridge is None: bridge = Bridge()
        return bridge
    TransformersOWLv2Model.get_image_features = lambda self, **kw: get_bridge().images(kw['images'])
    TransformersOWLv2Model.get_text_features = lambda self, **kw: get_bridge().text(kw)
    def stop(*_): raise SystemExit('Experiment worker stopped')
    signal.signal(signal.SIGTERM, stop)
    try:
        from frame_worker import worker_main
        return worker_main()
    finally:
        if bridge:
            print('OWL_SDPA_BRIDGE', json.dumps({'image_calls': bridge.calls, 'wall_seconds_including_communication': bridge.wall}), flush=True)
            bridge.close()


def backend(folder):
    import numpy as np
    import torch
    import transformers
    from transformers import Owlv2ForObjectDetection
    from wise.feature.transformers_owlv2 import get_object_features, get_text_embeddings
    torch.set_num_threads(1)
    arrays = {key: np.memmap(Path(folder)/(key+'.bin'), mode='r+', dtype='float32', shape=shape)
              for key, shape in SHAPES.items()}
    with os.fdopen(int(os.environ['OWL_RPC_FD']), 'w', buffering=1) as pipe:
        def send(value): pipe.write(json.dumps(value)+'\n')
        try:
            model = Owlv2ForObjectDetection.from_pretrained('google/owlv2-base-patch16-ensemble',
                local_files_only=True, torch_dtype=torch.float16, attn_implementation='sdpa').eval().to('mps')
            send({'event': 'ready', 'torch': torch.__version__, 'transformers': transformers.__version__,
                  'attention': model.config._attn_implementation, 'pid': os.getpid()})
            with torch.inference_mode():
                for line in sys.stdin:
                    command = json.loads(line)
                    if command['op'] == 'image':
                        result = get_object_features(model, torch.from_numpy(arrays['pixels']).to('mps'))
                        for key, tensor in zip(('scores', 'embeddings', 'boxes'), result):
                            np.copyto(arrays[key], tensor.float().cpu().numpy())
                        send({'event': 'done'})
                    else:
                        inputs = {key: torch.tensor(value, device='mps') for key, value in command['inputs'].items()}
                        vector = get_text_embeddings(model, **inputs)
                        send({'event': 'done', 'vector': vector.float().cpu().tolist()})
        except BaseException as exc:
            send({'event': 'failed', 'error': repr(exc)}); raise
        finally:
            for array in arrays.values(): array._mmap.close()


if __name__ == '__main__':
    raise SystemExit(front() if sys.argv[1] == 'front' else backend(sys.argv[2]))
