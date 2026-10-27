"""Resource sampling for offline benchmark scripts."""
import threading
import time

class ResourceSampler:
    """Sample the whole server process; overlapping RSS/MPS figures must not be added."""
    def snapshot(self):
        import psutil
        import torch
        return {'rss_bytes': psutil.Process().memory_info().rss,
                'mps_allocated_bytes': torch.mps.current_allocated_memory(),
                'mps_driver_bytes': torch.mps.driver_allocated_memory(),
                'system_available_bytes': psutil.virtual_memory().available,
                'system_swap_used_bytes': psutil.swap_memory().used}

    def __enter__(self):
        self.started = time.perf_counter()
        self.before = self.snapshot()
        self.peaks = dict(self.before)
        self.min_available = self.before['system_available_bytes']
        self.stop = threading.Event()
        self.worker = threading.Thread(target=self.monitor, daemon=True)
        self.worker.start()
        return self

    def sample(self):
        current = self.snapshot()
        for key, value in current.items(): self.peaks[key] = max(self.peaks[key], value)
        self.min_available = min(self.min_available, current['system_available_bytes'])

    def monitor(self):
        while not self.stop.wait(.05): self.sample()

    def __exit__(self, *args):
        self.stop.set(); self.worker.join()
        self.sample()
        self.result = {'scope': 'current_process', 'sample_interval_seconds': .05,
                       'total_seconds': time.perf_counter()-self.started,
                       'before': self.before, 'after': self.snapshot(),
                       'sampled_peak': self.peaks, 'min_system_available_bytes': self.min_available}
