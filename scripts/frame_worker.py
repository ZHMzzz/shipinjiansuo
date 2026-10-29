"""Serial extraction subprocess: reuse models, recreate all project state per job."""
import codecs
import gc
import json
import logging
import math
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time


def positive_seconds(name, default):
    value = float(os.environ.get(name, default))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a positive finite number')
    return value


class FrameWorker:
    """Owned by one ingestion thread; stop_event also interrupts blocked reads."""

    def __init__(self, root, stop_event):
        self.root = Path(root)
        self.stop_event = stop_event
        self.idle_seconds = positive_seconds('VIDEO_FRAME_WORKER_IDLE_SECONDS', 600)
        self.timeout_seconds = positive_seconds('VIDEO_FRAME_WORKER_TIMEOUT_SECONDS', 1800)
        self.startup_seconds = positive_seconds('VIDEO_FRAME_WORKER_STARTUP_SECONDS', 120)
        self.process = None
        self.result_fd = None
        self.last_used = 0

    def close(self):
        process, self.process = self.process, None
        try:
            if process is not None:
                try:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                finally:
                    for stream in (process.stdin, process.stdout):
                        try:
                            stream.close()
                        except OSError:
                            # A child dying during stdin.flush can leave a
                            # buffered write; close must still release all FDs.
                            pass
        finally:
            if self.result_fd is not None:
                os.close(self.result_fd)
                self.result_fd = None

    def close_if_idle(self):
        if self.process is not None and (self.process.poll() is not None or
                time.monotonic() - self.last_used >= self.idle_seconds):
            self.close()

    def _start(self, log):
        read_fd, write_fd = os.pipe()
        self.result_fd = read_fd
        try:
            self.process = subprocess.Popen(
                [sys.executable, '-u', str(Path(__file__).resolve()), '--worker'],
                cwd=self.root, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, pass_fds=(write_fd,),
                env={**os.environ, 'HF_HUB_OFFLINE': '1', 'PYTHONUNBUFFERED': '1',
                     'VIDEO_FRAME_RESULT_FD': str(write_fd)})
        finally:
            os.close(write_fd)
        self._result_buffer = b''
        self._log_buffer = ''
        self._decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        result = self._receive(self.startup_seconds, log)
        if result.get('event') != 'ready':
            raise RuntimeError(f'入库进程启动失败：{result}')
        logging.info('Frame worker ready: pid=%s', self.process.pid)

    def _receive(self, timeout, log, on_line=None):
        deadline = time.monotonic() + timeout
        tail = ''
        with selectors.DefaultSelector() as selector:
            selector.register(self.result_fd, selectors.EVENT_READ, 'result')
            selector.register(self.process.stdout, selectors.EVENT_READ, 'log')
            while True:
                if self.stop_event.is_set():
                    raise RuntimeError('服务停止，帧特征提取已中断')
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f'帧特征提取超过 {timeout:g} 秒，进程将重建')
                events = selector.select(min(remaining, .2))
                # Drain logs before accepting completion so progress stays ordered.
                events.sort(key=lambda event: event[0].data == 'result')
                for key, _ in events:
                    chunk = os.read(key.fd, 65536)
                    if key.data == 'log':
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        text = self._decoder.decode(chunk)
                        log.write(text)
                        log.flush()
                        tail = (tail + text)[-1200:]
                        self._log_buffer += text.replace('\r', '\n')
                        while '\n' in self._log_buffer:
                            line, self._log_buffer = self._log_buffer.split('\n', 1)
                            if on_line:
                                on_line(line)
                        self._log_buffer = self._log_buffer[-4096:]
                    else:
                        if not chunk:
                            raise RuntimeError('入库进程意外退出：' + tail)
                        self._result_buffer += chunk
                        if b'\n' in self._result_buffer:
                            line, self._result_buffer = self._result_buffer.split(b'\n', 1)
                            return json.loads(line)

    def extract(self, args, log, on_line=None):
        """No automatic retry: a partially written staging project is discarded."""
        if self.stop_event.is_set():
            raise RuntimeError('服务停止，帧特征提取已中断')
        self.close_if_idle()
        try:
            if self.process is None:
                self._start(log)
            self.process.stdin.write((json.dumps({'args': list(map(str, args))}) + '\n').encode())
            self.process.stdin.flush()
            result = self._receive(self.timeout_seconds, log, on_line)
            if result.get('event') != 'completed':
                raise RuntimeError('帧特征提取失败：' + result.get('error', str(result)))
            self.last_used = time.monotonic()
            log.write(f'\nFrame worker pid={self.process.pid} models={result["models"]} '
                      f'created={result["created"]} reused={result["reused"]}\n')
            log.flush()
            return result
        except BaseException:
            self.close()
            raise


def worker_main():
    # Imports and class adapters remain private to the indexing subprocess.
    from extract_frames import main as extract, configure
    configure([])
    import wise._main.extract_features as module
    original = module.FeatureExtractorFactory
    cache = {}
    counts = {'created': 0, 'reused': 0}

    def factory(feature_id, config):
        key = json.dumps(config, sort_keys=True)
        previous = cache.get(feature_id)
        if previous is not None and previous[0] == key:
            counts['reused'] += 1
            return previous[1]
        # A changed configuration replaces the previous instance for this model.
        cache.pop(feature_id, None)
        model = original(feature_id, config)
        cache[feature_id] = (key, model)
        counts['created'] += 1
        return model

    module.FeatureExtractorFactory = factory
    with os.fdopen(int(os.environ['VIDEO_FRAME_RESULT_FD']), 'w', buffering=1) as pipe:
        def send(value):
            sys.stdout.flush()
            sys.stderr.flush()
            pipe.write(json.dumps(value) + '\n')
        send({'event': 'ready'})
        for line in sys.stdin:
            try:
                request = json.loads(line)
                extract(request['args'])
                import torch
                if torch.backends.mps.is_available():
                    torch.mps.synchronize()
                gc.collect()
                send({'event': 'completed', 'models': len(cache), **counts})
            except BaseException as exc:
                logging.exception('Frame worker failed; process will exit')
                send({'event': 'failed', 'error': repr(exc)[:1500]})
                return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(worker_main())
