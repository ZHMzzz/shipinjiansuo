"""Read-only host sampling and experiment-process-only instrumentation.

Wait durations are CPU-visible blocking call intervals, NOT Metal queue latency.
No new device synchronizations are introduced.
"""
import json
import os
from pathlib import Path
import plistlib
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
OUT = HERE/'thread-contention-20261028-rerun'
CONTROL = OUT/'control.json'
_local = threading.local()
_lock = threading.Lock()
_writer = None
_real_set_threads = None


def emit(kind, start, **extra):
    global _writer
    with _lock:
        if _writer is None:
            (OUT/'events').mkdir(parents=True, exist_ok=True)
            _writer = (OUT/'events'/f'{os.getpid()}.jsonl').open('a', buffering=1)
        _writer.write(json.dumps(dict(kind=kind, start=start, end=time.monotonic(), pid=os.getpid(), **extra))+'\n')


def apply_threads():
    import torch
    cfg = json.loads(CONTROL.read_text())
    desired = cfg['threads']
    assert desired in (1,2,4)
    _real_set_threads(desired)
    emit('thread_setting',time.monotonic(),requested=desired,actual=torch.get_num_threads(),trial=cfg['trial'])


def install_hooks():
    global _real_set_threads
    import torch
    if _real_set_threads is not None:return
    _real_set_threads = torch.set_num_threads
    def setter(_): apply_threads()
    torch.set_num_threads = setter
    apply_threads()
    real_sync = torch.mps.synchronize
    def sync(*a,**kw):
        start=time.monotonic()
        try:return real_sync(*a,**kw)
        finally:emit('mps_synchronize',start,actual_threads=torch.get_num_threads(),thread_id=threading.get_ident())
    torch.mps.synchronize=sync
    real_cpu=torch.Tensor.cpu
    def cpu(self,*a,**kw):
        if self.device.type!='mps':return real_cpu(self,*a,**kw)
        start=time.monotonic()
        try:return real_cpu(self,*a,**kw)
        finally:emit('mps_to_cpu',start,actual_threads=torch.get_num_threads(),thread_id=threading.get_ident())
    torch.Tensor.cpu=cpu


def wrap_input():
    class Input:
        def __init__(self,stream):self.stream=stream
        def __iter__(self):return self
        def __next__(self):
            line=next(self.stream)
            apply_threads()
            return line
        def __getattr__(self,key):return getattr(self.stream,key)
    sys.stdin=Input(sys.stdin)


class TimedQueue:
    @staticmethod
    def make(*args,**kwargs):
        import queue
        class Q(queue.Queue):
            def put(self,*a,**kw):
                start=time.monotonic()
                try:return super().put(*a,**kw)
                finally:emit('producer_queue_put',start)
            def get(self,*a,**kw):
                start=time.monotonic()
                try:return super().get(*a,**kw)
                finally:emit('consumer_queue_get',start)
        return Q(*args,**kwargs)


def host_snapshot():
    import psutil
    start=time.monotonic()
    data=plistlib.loads(subprocess.check_output(['/usr/sbin/ioreg','-a','-r','-c','AGXAccelerator','-d','1'],timeout=3))
    perf=data[0]['PerformanceStatistics']
    vm=subprocess.check_output(['/usr/bin/vm_stat'],text=True,timeout=3)
    pages={}
    for line in vm.splitlines()[1:]:
        if ':' in line:
            k,v=line.split(':',1)
            try:pages[k]=int(v.strip().rstrip('.'))
            except ValueError:pass
    import re
    page_size=int(re.search(r'page size of (\d+) bytes',vm).group(1))
    swap=psutil.swap_memory();mem=psutil.virtual_memory()
    return dict(at=time.monotonic(),sample_cost_seconds=time.monotonic()-start,
        gpu=perf,vm_pages=pages,page_size=page_size,swap_in=swap.sin,swap_out=swap.sout,
        swap_used=swap.used,available=mem.available,cpu_percent=psutil.cpu_percent(),load=os.getloadavg())


class HostSampler:
    def __init__(self,path):self.path=Path(path);self.stop=threading.Event();self.rows=[]
    def __enter__(self):
        def run():
            with self.path.open('w',buffering=1) as f:
                while not self.stop.is_set():
                    try:r=host_snapshot();self.rows.append(r);f.write(json.dumps(r)+'\n')
                    except Exception as e:f.write(json.dumps({'at':time.monotonic(),'error':repr(e)})+'\n')
                    self.stop.wait(.5)
        self.thread=threading.Thread(target=run,daemon=True);self.thread.start();return self
    def __exit__(self,*args):self.stop.set();self.thread.join(timeout=5)


def quiet_sample(seconds=8):
    """Record background; never kill/reconfigure unrelated programs."""
    rows=[];end=time.monotonic()+seconds
    while time.monotonic()<end:
        rows.append(host_snapshot());time.sleep(.5)
    return rows
