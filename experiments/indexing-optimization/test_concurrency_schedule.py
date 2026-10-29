import threading
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import concurrency_support as c

class Engine:
    def __init__(self):self.entered=threading.Event();self.release=threading.Event()
    def encode_segment(self,media,segment):
        self.entered.set();assert self.release.wait(2);return 'vector'


def test_full_starts_before_frame_completion(tmp_path):
    (tmp_path/'x.mp4').touch();engine=Engine();s=c.Schedule('full',engine,{})
    try:
        s.start(tmp_path,1)
        assert engine.entered.wait(1)
        assert not any(x['name']=='frame_end' for x in s.events)
        engine.release.set();s.frame_done()
        assert s.values_for(s.media)==['vector']
    finally:s.close()


def test_cpu_prepares_early_but_forward_waits_for_frame_and_merge(tmp_path,monkeypatch):
    (tmp_path/'x.mp4').touch();prepared=threading.Event();calls=[]
    def prep(*args):prepared.set();return {'cpu':True},'hash'
    monkeypatch.setattr(c,'prepare',prep)
    monkeypatch.setattr(c,'forward',lambda engine,inputs:calls.append(inputs) or 'vector')
    s=c.Schedule('cpu',object(),{})
    try:
        s.start(tmp_path,1);assert prepared.wait(1);assert not calls
        s.frame_done();assert not calls
        assert s.values_for(s.media)==['vector'];assert calls==[{'cpu':True}]
        assert s.row['cpu_queue_peak_items']<=2
    finally:s.close()


def test_cpu_preparation_error_propagates_and_cleans_up(tmp_path,monkeypatch):
    (tmp_path/'x.mp4').touch()
    def fail(*args):raise ValueError('broken decode')
    monkeypatch.setattr(c,'prepare',fail)
    s=c.Schedule('cpu',object(),{})
    try:
        s.start(tmp_path,1);s.frame_done()
        with pytest.raises(ValueError,match='broken decode'):s.values_for(s.media)
    finally:s.close()
