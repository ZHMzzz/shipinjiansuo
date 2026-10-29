import threading
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import thread_concurrency_support as c

@pytest.fixture(autouse=True)
def isolated_telemetry(tmp_path, monkeypatch):
    import thread_telemetry as t
    monkeypatch.setattr(t, 'emit', lambda *a, **k: None)

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


def test_host_rates_are_interval_deltas():
    from summarize_thread_contention import host_metrics
    def row(at,count):
        return dict(at=at, vm_pages={k:count for k in ('Swapins','Swapouts','Pageins','Pageouts','Compressions','Decompressions')},page_size=16384,gpu={'Device Utilization %':50},cpu_percent=20,sample_cost_seconds=.01)
    result=host_metrics([row(10,100),row(10.5,132)])
    assert result['swap_in']['total_mib']==.5
    assert result['swap_out']['peak_mib_per_second']==1
    assert result['gpu_percent']['mean']==50


def test_threads_applied_inside_existing_worker(tmp_path,monkeypatch):
    import json
    import torch
    import thread_telemetry as t
    from concurrent.futures import ThreadPoolExecutor
    control=tmp_path/'control.json'
    monkeypatch.setattr(t,'CONTROL',control)
    monkeypatch.setattr(t,'_real_set_threads',torch.set_num_threads)
    with ThreadPoolExecutor(max_workers=1) as executor:
        for n in (1,2,4):
            control.write_text(json.dumps({'threads':n,'trial':'test'}))
            def work():
                t.apply_threads()
                return torch.get_num_threads()
            assert executor.submit(work).result()==n
