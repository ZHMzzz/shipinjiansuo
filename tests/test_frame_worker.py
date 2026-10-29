"""Exercise the real subprocess protocol without loading accelerator models."""
import io
import subprocess
import threading
import time

import pytest

from scripts.frame_worker import FrameWorker


@pytest.fixture
def worker(tmp_path, monkeypatch):
    child = tmp_path/'fake_worker.py'
    child.write_text('''
import json, os, sys, time
pipe = os.fdopen(int(os.environ['VIDEO_FRAME_RESULT_FD']), 'w', buffering=1)
def send(value): pipe.write(json.dumps(value)+'\\n')
if os.environ.get('TEST_SLOW_START'): time.sleep(20)
send({'event':'ready'})
count = 0
for line in sys.stdin:
    args = json.loads(line)['args']
    if args[0] == 'crash': os._exit(9)
    if args[0] == 'hang': time.sleep(20)
    if args[0] == 'fail':
        send({'event':'failed','error':'invalid input'})
        break
    # More than pipe capacity: a parent waiting only for the result would hang.
    sys.stdout.write('x'*100000+'\\nFeature extraction: 7it\\r')
    sys.stdout.flush()
    count += 1
    send({'event':'completed','models':2,'created':2,'reused':2*(count-1)})
''')
    original = subprocess.Popen
    def spawn(command, **kwargs):
        return original([command[0], '-u', str(child)], **kwargs)
    monkeypatch.setattr('scripts.frame_worker.subprocess.Popen', spawn)
    value = FrameWorker(tmp_path, threading.Event())
    value.timeout_seconds = 3
    value.startup_seconds = 3
    yield value
    value.close()


def test_reuses_process_and_drains_large_logs_and_progress(worker):
    lines = []
    log = io.StringIO()
    worker.extract(['ok'], log, lines.append)
    process = worker.process
    result = worker.extract(['ok'], log, lines.append)
    assert worker.process is process and process.poll() is None
    assert result['reused'] == 2
    assert 'Feature extraction: 7it' in lines
    assert len(log.getvalue()) > 200000


@pytest.mark.parametrize('mode, error', [('fail', RuntimeError), ('crash', RuntimeError), ('hang', TimeoutError)])
def test_failure_or_timeout_closes_process_and_next_task_restarts(worker, mode, error):
    worker.extract(['ok'], io.StringIO())
    previous = worker.process
    worker.timeout_seconds = .15
    with pytest.raises(error):
        worker.extract([mode], io.StringIO())
    assert previous.poll() is not None and worker.process is None
    worker.timeout_seconds = 3
    worker.extract(['ok'], io.StringIO())
    assert worker.process.pid != previous.pid


def test_idle_expiry_reaps_worker_and_next_task_restarts(worker):
    worker.extract(['ok'], io.StringIO())
    previous = worker.process
    worker.last_used = time.monotonic() - worker.idle_seconds - 1
    worker.close_if_idle()
    assert previous.poll() is not None and worker.process is None
    worker.extract(['ok'], io.StringIO())
    assert worker.process.pid != previous.pid


def test_stop_event_interrupts_running_task(worker):
    worker.extract(['ok'], io.StringIO())
    previous = worker.process
    timer = threading.Timer(.1, worker.stop_event.set)
    timer.start()
    try:
        with pytest.raises(RuntimeError, match='服务停止'):
            worker.extract(['hang'], io.StringIO())
    finally:
        timer.join()
    assert previous.poll() is not None and worker.process is None


def test_startup_timeout_reaps_child(worker, monkeypatch):
    monkeypatch.setenv('TEST_SLOW_START', '1')
    worker.startup_seconds = .1
    with pytest.raises(TimeoutError):
        worker.extract(['ok'], io.StringIO())
    assert worker.process is None and worker.result_fd is None


def test_ingestion_worker_rollback_switch(tmp_path, monkeypatch):
    from scripts.ingestion import IngestionManager
    monkeypatch.setenv('VIDEO_FRAME_WORKER_REUSE', '0')
    manager = IngestionManager(tmp_path, None)
    assert manager.frame_worker is None
