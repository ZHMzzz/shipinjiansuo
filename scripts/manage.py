"""Start a detached loopback server or stop only this project's recorded process."""
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parents[1]
pidfile = root / 'logs/server.pid'
server = root / 'scripts/server.py'

def running_pid():
    if not pidfile.exists():
        return None
    pid = int(pidfile.read_text())
    command = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True).stdout
    return pid if str(server) in command else None

if sys.argv[1] == 'stop':
    pid = running_pid()
    if pid:
        os.kill(pid, signal.SIGTERM)
        print('正在停止服务…', flush=True)
        for _ in range(200):
            if running_pid() is None:
                break
            time.sleep(0.1)
        else:
            raise SystemExit('服务仍在退出中，保留 PID 文件；稍后重试。')
        print('服务已停止。')
    else:
        print('本项目服务未运行。')
    pidfile.unlink(missing_ok=True)
else:
    pid = running_pid()
    if pid:
        print(f'服务已启动，PID {pid}：http://127.0.0.1:9670/mvp/')
        sys.exit()
    try:
        urllib.request.urlopen('http://127.0.0.1:9670/', timeout=1)
    except Exception:
        pass
    else:
        raise SystemExit('9670 端口已有服务，请检查后再启动。')
    (root / 'logs').mkdir(exist_ok=True)
    with open(root / 'logs/server.log', 'ab', buffering=0) as log:
        proc = subprocess.Popen([sys.executable, '-u', str(server)], cwd=root,
                                stdout=log, stderr=log, start_new_session=True)
    pidfile.write_text(str(proc.pid))
    print(f'启动中，PID {proc.pid}。模型加载完成后打开 http://127.0.0.1:9670/mvp/')
    print('日志：logs/server.log')
