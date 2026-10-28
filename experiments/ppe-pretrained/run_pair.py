"""Serialize model processes and wait for unrelated experiments to stop."""
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil

HERE = Path(__file__).resolve().parent


def others():
    found = []
    for p in psutil.process_iter(['pid', 'cmdline']):
        scripts = [x for x in (p.info['cmdline'] or []) if 'experiments/' in x and x.endswith('.py')]
        if scripts and not any('ppe-pretrained' in x for x in scripts):
            found.append({'pid': p.pid, 'scripts': scripts})
    return found


for model_id in ['yolo11-ppe', 'yolov8-ppe']:
    waited = 0
    while True:
        active = others()
        if not active:
            time.sleep(3)
            if not others():
                break
        if waited % 60 == 0:
            print(json.dumps({'waiting_for_other_experiments': active, 'seconds_waited': waited}), flush=True)
        time.sleep(10)
        waited += 10
    print(json.dumps({'starting': model_id, 'seconds_waited': waited}), flush=True)
    subprocess.run([sys.executable, str(HERE / 'benchmark.py'), model_id], check=True)
