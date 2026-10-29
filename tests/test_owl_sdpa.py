"""Protocol/error-path coverage without loading model weights."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import owl_sdpa


@pytest.fixture
def fake_backend(tmp_path, monkeypatch):
    child = tmp_path / 'fake_backend.py'
    child.write_text('''
import json,os,sys,time
pipe=os.fdopen(int(os.environ['OWL_SDPA_RESULT_FD']),'w',buffering=1)
def send(value):pipe.write(json.dumps(value)+'\\n')
if os.environ.get('FAKE_START_FAIL'):
    send({'event':'failed','error':'bad runtime'})
    sys.exit(1)
if os.environ.get('FAKE_START_HANG'):time.sleep(20)
send({'event':'ready','pid':os.getpid()})
for line in sys.stdin:
    op=json.loads(line)['op']
    if op=='crash':os._exit(9)
    if op=='hang':time.sleep(20)
    if op=='fail':send({'event':'failed','error':'model failed'})
    else:send({'event':'done','value':op})
''')
    original = subprocess.Popen
    processes = []
    def spawn(command, **kwargs):
        p = original([sys.executable, '-u', str(child)], **kwargs)
        processes.append(p)
        return p
    monkeypatch.setattr(owl_sdpa.subprocess, 'Popen', spawn)
    yield tmp_path, processes
    for p in processes:
        if p.poll() is None:
            p.kill()
        p.wait()


def test_protocol_reuses_child_and_releases_mmaps(fake_backend):
    root, processes = fake_backend
    bridge = owl_sdpa.Bridge(python=sys.executable, cache_dir=root)
    folder = Path(bridge.temp.name)
    assert bridge.request({'op':'one'})['value'] == 'one'
    assert bridge.request({'op':'two'})['value'] == 'two'
    assert len(processes) == 1
    bridge.close()
    bridge.close()
    assert processes[0].poll() is not None
    assert not folder.exists() and bridge.result_fd is None


@pytest.mark.parametrize('op,error', [('crash',RuntimeError),('hang',TimeoutError),('fail',RuntimeError)])
def test_errors_propagate_and_cleanup_is_bounded(fake_backend, op, error):
    root, processes = fake_backend
    bridge = owl_sdpa.Bridge(python=sys.executable, cache_dir=root)
    bridge.timeout = .1
    try:
        with pytest.raises(error):
            bridge.request({'op':op})
    finally:
        bridge.close()
    assert processes[0].poll() is not None
    assert not list(root.glob('shared-*'))


@pytest.mark.parametrize('flag,error', [('FAKE_START_FAIL',RuntimeError),('FAKE_START_HANG',TimeoutError)])
def test_failed_start_cleans_child_and_buffers(fake_backend, monkeypatch, flag, error):
    root, processes = fake_backend
    monkeypatch.setenv(flag, '1')
    with pytest.raises(error):
        owl_sdpa.Bridge(python=sys.executable, cache_dir=root, timeout=.2)
    assert processes[0].poll() is not None
    assert not list(root.glob('shared-*'))


def test_missing_runtime_does_not_patch_model_methods(tmp_path, monkeypatch):
    from wise.feature.transformers_owlv2 import TransformersOWLv2Model
    monkeypatch.setenv('VIDEO_OWL_SDPA_PYTHON', str(tmp_path/'missing'))
    before = TransformersOWLv2Model.get_image_features
    with pytest.raises(RuntimeError, match='setup_owl_sdpa'):
        owl_sdpa.install()
    assert TransformersOWLv2Model.get_image_features is before


def test_sdpa_rollback_keeps_existing_frame_acceleration(monkeypatch):
    from scripts import frame_acceleration
    calls = []
    monkeypatch.setattr(frame_acceleration, '_enabled', False)
    monkeypatch.setenv('VIDEO_OWL_SDPA','0')
    monkeypatch.setattr(frame_acceleration,'install_coreml',lambda:calls.append('face'))
    monkeypatch.setattr(frame_acceleration,'install_owl',lambda:calls.append('owl'))
    monkeypatch.setattr(owl_sdpa,'install',lambda:pytest.fail('rollback must skip SDPA'))
    frame_acceleration.enable()
    assert calls == ['face','owl']


def test_invalid_backend_setting_fails_before_patching(monkeypatch):
    from scripts import frame_acceleration
    monkeypatch.setattr(frame_acceleration,'_enabled',False)
    monkeypatch.setenv('VIDEO_OWL_SDPA','invalid')
    monkeypatch.setattr(frame_acceleration,'install_coreml',lambda:pytest.fail('must validate first'))
    with pytest.raises(ValueError,match='VIDEO_OWL_SDPA'):
        frame_acceleration.enable()
