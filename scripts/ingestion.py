"""Persistent, serial video ingestion. Work on a copy, publish only validated indexes."""
import asyncio
import fcntl
import hashlib
import json
import logging
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi import HTTPException

MAX_UPLOAD = 1024 * 1024 * 1024
EXTENSIONS = {'.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v', '.mpeg', '.mpg', '.ts'}
ACTIVE = ('validating', 'transcoding', 'extracting', 'indexing', 'action_indexing', 'publishing')


def prepare_versions(root):
    """Give the live project an atomic pointer; all readers bind a fixed version."""
    current = root / 'projects/demo'
    initial = root / 'projects/releases/initial/demo'
    if current.is_symlink():
        return current.resolve(strict=True)
    initial.parent.mkdir(parents=True, exist_ok=True)
    if current.exists():
        if initial.exists():
            raise RuntimeError('Initial project backup already exists; refusing to overwrite it')
        current.rename(initial)
    if not initial.exists():
        raise RuntimeError('No WISE project found')
    current.symlink_to(os.path.relpath(initial, current.parent))
    return initial


class IngestionManager:
    def __init__(self, root, publisher, action_builder=None):
        self.root = Path(root)
        self.publisher = publisher
        self.action_builder = action_builder
        self.backfill_requested = action_builder is not None
        self.action_state = {'status': 'pending', 'progress': 0, 'message': '正在检查已有视频的 Qwen 片段索引'}
        self.dbpath = self.root / 'data/ingestion.sqlite'
        self.uploads = self.root / 'data/uploads'
        self.uploads.mkdir(parents=True, exist_ok=True)
        self.stopping = threading.Event()
        self.wakeup = threading.Event()
        self.process = None
        reuse = os.environ.get('VIDEO_FRAME_WORKER_REUSE', '1')
        if reuse not in ('0', '1'):
            raise ValueError('VIDEO_FRAME_WORKER_REUSE must be 0 or 1')
        try:
            from .frame_worker import FrameWorker
        except ImportError:
            from frame_worker import FrameWorker
        self.frame_worker = FrameWorker(self.root, self.stopping) if reuse == '1' else None
        self.mutation_lock = threading.Lock()
        self.loop = None
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('''CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, filename TEXT NOT NULL, suffix TEXT NOT NULL,
                sha256 TEXT UNIQUE NOT NULL, bytes INTEGER NOT NULL, status TEXT NOT NULL,
                stage TEXT NOT NULL, progress REAL NOT NULL, message TEXT NOT NULL,
                created REAL NOT NULL, updated REAL NOT NULL, duration REAL,
                media_id INTEGER, attempts INTEGER NOT NULL DEFAULT 0)''')

    def connect(self):
        db = sqlite3.connect(self.dbpath, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def jobs(self):
        with self.connect() as db:
            return [dict(r) for r in db.execute('SELECT * FROM jobs ORDER BY created DESC')]

    def update(self, job_id, **values):
        values['updated'] = time.time()
        with self.connect() as db:
            db.execute('UPDATE jobs SET '+','.join(f'{k}=?' for k in values)+' WHERE id=?', [*values.values(), job_id])

    def enqueue(self, file):
        filename = Path((file.filename or 'video').replace('\\', '/')).name[:180]
        suffix = Path(filename).suffix.lower()
        if suffix not in EXTENSIONS:
            raise HTTPException(400, '请选择 MP4、MOV、MKV、AVI、WebM 等视频文件')
        if shutil.disk_usage(self.root).free < MAX_UPLOAD * 3:
            raise HTTPException(507, '磁盘可用空间不足 3GB，请先释放空间')
        job_id = uuid.uuid4().hex
        folder = self.uploads / job_id
        folder.mkdir()
        size = 0
        digest = hashlib.sha256()
        try:
            with (folder / ('source'+suffix)).open('wb') as out:
                while chunk := file.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD:
                        raise HTTPException(413, 'MVP 单个视频最大 1GB')
                    digest.update(chunk)
                    out.write(chunk)
            if size == 0:
                raise HTTPException(400, '视频文件为空')
            now = time.time()
            with self.connect() as db:
                try:
                    db.execute('INSERT INTO jobs(id,filename,suffix,sha256,bytes,status,stage,progress,message,created,updated) VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                        (job_id,filename,suffix,digest.hexdigest(),size,'queued','等待处理',0,'视频已上传，等待后台处理',now,now))
                except sqlite3.IntegrityError:
                    row = db.execute('SELECT * FROM jobs WHERE sha256=?',(digest.hexdigest(),)).fetchone()
                    if row is None: raise
                    shutil.rmtree(folder)
                    return {'job':dict(row),'duplicate':True}
            self.wakeup.set()
            return {'job':next(j for j in self.jobs() if j['id']==job_id),'duplicate':False}
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise

    def retry(self, job_id):
        with self.connect() as db:
            row = db.execute('SELECT * FROM jobs WHERE id=?',(job_id,)).fetchone()
            if not row: raise HTTPException(404, '任务不存在')
            if row['status'] != 'failed': raise HTTPException(409, '只有失败的任务可以重试')
            db.execute("UPDATE jobs SET status='queued',stage='等待重试',progress=0,message='已重新加入处理队列',updated=? WHERE id=?",(time.time(),job_id))
        self.wakeup.set()
        return {'queued':job_id}

    def recover(self):
        """Reconcile an interrupted publication before marking other active work failed."""
        for job in self.jobs():
            if job['status'] not in ACTIVE: continue
            current = self.root/'projects/demo/metadata/internal.db'
            with sqlite3.connect(current) as db:
                row = db.execute('SELECT media.id FROM media JOIN source_collections s ON s.id=media.source_collection_id WHERE s.location=?',
                                 (str(self.uploads/job['id']/'prepared'),)).fetchone()
            if row:
                self.update(job['id'],status='completed',stage='已入库',progress=100,media_id=row[0],message='视频已加入检索库')
            else:
                self.update(job['id'],status='failed',stage='处理已中断',message='服务曾停止，视频未发布；可点击重试')

    def start(self, loop):
        self.lockfile = (self.root/'logs/ingestion.lock').open('a')
        fcntl.flock(self.lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.loop = loop
        self.recover()
        logging.info('Frame worker reuse=%s idle_seconds=%s timeout_seconds=%s owl_sdpa=%s',
                     self.frame_worker is not None,
                     self.frame_worker.idle_seconds if self.frame_worker else None,
                     self.frame_worker.timeout_seconds if self.frame_worker else None,
                     os.environ.get('VIDEO_FRAME_ACCELERATION', '1') == '1' and
                     os.environ.get('VIDEO_OWL_SDPA', '1') == '1')
        self.thread = threading.Thread(target=self.worker, name='video-ingestion', daemon=True)
        self.thread.start()

    def stop(self):
        self.stopping.set()
        self.wakeup.set()
        if self.process and self.process.poll() is None:
            self.process.terminate()
        if hasattr(self,'thread'):
            self.thread.join(timeout=15)
        if hasattr(self,'lockfile'): self.lockfile.close()

    def worker(self):
        try:
            self._worker_loop()
        finally:
            if self.frame_worker is not None:
                self.frame_worker.close()

    def _worker_loop(self):
        try:
            from .video_deletion import pending_cleanup
        except ImportError:
            from video_deletion import pending_cleanup
        while not self.stopping.is_set():
            if self.frame_worker is not None:
                self.frame_worker.close_if_idle()
            row = None
            with self.mutation_lock:
                try:
                    pending_cleanup(self)
                except Exception:
                    logging.exception('视频清理尚未完成，暂缓发布新版本')
                    self.stopping.wait(2)
                    continue
                if self.backfill_requested:
                    self.backfill_requested = False
                    self.backfill_actions()
                    continue
                with self.connect() as db:
                    row = db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
                    if row:
                        db.execute("UPDATE jobs SET status='validating',stage='检查视频',progress=2,attempts=attempts+1,updated=? WHERE id=?",(time.time(),row['id']))
                if row:
                    self.process_job(dict(row))
            if not row:
                self.wakeup.wait(1)
                self.wakeup.clear()

    def request_backfill(self):
        if self.action_state['status'] in ('pending', 'building'):
            return
        self.action_state = {'status': 'pending', 'progress': 0, 'message': '等待补建 Qwen 片段索引'}
        self.backfill_requested = True
        self.wakeup.set()

    def wait_async(self, coroutine):
        future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        while True:
            try: return future.result(timeout=.5)
            except TimeoutError:
                # The coroutine checks stopping between windows. Do not cancel a
                # to_thread write then remove its directory while it still runs.
                continue

    def build_actions(self, stage, jid=None):
        if self.action_builder is None: return
        def progress(done, total, message):
            percent = 100*done/max(1,total)
            self.action_state = {'status': 'building', 'progress': percent,
                'completed_segments': done, 'total_segments': total, 'message': message}
            if jid:
                self.update(jid, status='action_indexing', stage='建立Qwen 片段索引',
                    progress=85+10*percent/100, message=message)
        self.wait_async(self.action_builder(stage, progress))

    def backfill_actions(self):
        release = self.root/'projects/releases'/uuid.uuid4().hex
        stage = release/'demo'
        try:
            from action_search import read_snapshot
            current = (self.root/'projects/demo').resolve()
            if read_snapshot(current, complete=True):
                self.action_state = {'status': 'ready', 'progress': 100, 'message': '全部已入库视频支持 Qwen 片段检索'}
                return
        except Exception:
            # Missing, outdated or damaged action indexes are rebuilt on a copy.
            pass
        try:
            self.action_state = {'status': 'building', 'progress': 0, 'message': '准备补建已有视频的 Qwen 片段索引'}
            shutil.copytree((self.root/'projects/demo').resolve(), stage)
            try:
                read_snapshot(stage)
            except Exception:
                from native_qwen import reset_native_qwen
                reset_native_qwen(stage)
            self.build_actions(stage)
            self.wait_async(self.publisher(stage))
            self.action_state = {'status': 'ready', 'progress': 100, 'message': '全部已入库视频支持 Qwen 片段检索'}
        except Exception as exc:
            self.action_state = {'status': 'failed', 'progress': 0, 'message': str(exc)[:1500]}
        finally:
            if release.exists() and (self.root/'projects/demo').resolve() != stage.resolve():
                shutil.rmtree(release)

    def run(self, command, log, on_line=None):
        if self.stopping.is_set(): raise RuntimeError('服务停止，任务已中断')
        self.process = subprocess.Popen(command,cwd=self.root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            text=True,bufsize=1,env={**os.environ,'HF_HUB_OFFLINE':'1','PYTHONUNBUFFERED':'1'})
        tail=[]
        try:
            for line in self.process.stdout:
                log.write(line)
                log.flush()
                tail=(tail+[line])[-10:]
                if on_line: on_line(line)
                if self.stopping.is_set(): self.process.terminate()
            code=self.process.wait()
            if code: raise RuntimeError(f'处理程序退出（{code}）：'+''.join(tail)[-1200:])
        finally:
            if self.process.poll() is None:
                self.process.terminate()
                try: self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process=None

    def process_job(self, job):
        jid=job['id']
        folder=self.uploads/jid
        release=self.root/'projects/releases'/uuid.uuid4().hex
        stage=release/'demo'
        published=False
        try:
            probe=subprocess.run([str(self.root/'.media-runtime/bin/ffprobe'),'-v','error','-show_format','-show_streams','-of','json',str(folder/('source'+job['suffix']))],capture_output=True,text=True,timeout=30)
            if probe.returncode: raise RuntimeError('无法解码此文件，请确认它是有效视频')
            meta=json.loads(probe.stdout)
            if not any(s.get('codec_type')=='video' for s in meta.get('streams',[])):
                raise RuntimeError('文件中没有视频画面')
            duration=float(meta.get('format',{}).get('duration',0))
            if not math.isfinite(duration) or duration <= 0: raise RuntimeError('无法确定有效视频时长')
            prepared=folder/'prepared'
            prepared.mkdir(exist_ok=True)
            stem=re.sub(r'[^\w\-\u4e00-\u9fff]+','_',Path(job['filename']).stem)[:90] or 'video'
            video=prepared/(stem+'.mp4')
            self.update(jid,status='transcoding',stage='转码与抽帧准备',progress=5,duration=duration,message='转换为浏览器可播放的 H.264 视频')
            with (self.root/'logs'/f'upload-{jid}.log').open('a',buffering=1) as log:
                def transcode_progress(line):
                    if line.startswith('out_time_us='):
                        try: fraction=min(1,max(0,float(line.split('=')[1])/1e6/duration))
                        except ValueError: return
                        self.update(jid,progress=5+10*fraction)
                self.run([str(self.root/'.media-runtime/bin/ffmpeg'),'-y','-v','error','-nostdin','-i',str(folder/('source'+job['suffix'])),
                    '-map','0:v:0','-map','0:a:0?','-vf',"scale=640:640:force_original_aspect_ratio=decrease:force_divisible_by=2",'-c:v','libx264','-crf','23','-preset','fast','-pix_fmt','yuv420p','-c:a','aac','-movflags','+faststart','-progress','pipe:1',str(video)],log,transcode_progress)
                if shutil.disk_usage(self.root).free < 2*1024**3: raise RuntimeError('磁盘空间不足，无法创建新索引版本')
                self.update(jid,status='extracting',stage='提取物体与人脸特征',progress=15,message='通过 WISE 提取 OWLv2 物体与人脸特征，随后建立 Qwen 片段索引')
                shutil.copytree((self.root/'projects/demo').resolve(),stage)
                try:
                    from .qwen_project import validate_project, FACE, OWL
                except ImportError:
                    from qwen_project import validate_project, FACE, OWL
                validate_project(stage, require_owl=True)
                # Keep the sole Qwen store while WISE discovers frame models.
                try:
                    from .native_qwen import suspend_native_store
                except ImportError:
                    from native_qwen import suspend_native_store
                with suspend_native_store(stage):
                    expected=max(1,math.ceil(duration*2)*2)
                    def extraction_progress(line):
                        match=re.search(r'Feature extraction:\s*(\d+)it',line)
                        if match:
                            done=int(match[1])
                            self.update(jid,progress=min(78,15+63*done/expected),message=f'已完成 {done} 次帧特征计算，预计约 {expected} 次')
                    args=[str(prepared),'--project-dir',str(stage),'--num-workers','0','--yes']
                    if self.frame_worker is not None:
                        self.frame_worker.extract(args,log,extraction_progress)
                    else:
                        self.run([sys.executable,'-u',str(Path(__file__).resolve().with_name('extract_frames.py')),*args],log,extraction_progress)
                    self.update(jid,status='indexing',stage='构建检索索引',progress=80,message='合并新旧物体和人脸特征，构建 WISE 检索索引')
                    self.run([sys.executable,'-u','-m','wise','create-index','--project-dir',str(stage),'--overwrite'],log)
            with sqlite3.connect(stage/'metadata/internal.db') as db:
                row=db.execute('SELECT media.id FROM media JOIN source_collections s ON s.id=media.source_collection_id WHERE s.location=?',(str(prepared),)).fetchone()
                if not row: raise RuntimeError('处理结束但没有产生视频记录，未发布此任务')
                media_id=row[0]
                counts=dict(db.execute('SELECT feature_extractor_id,count(*) FROM vectors WHERE media_id=? GROUP BY feature_extractor_id',(media_id,)))
                if set(counts) - {FACE, OWL}: raise RuntimeError('出现已停用的特征模型，未发布此任务')
            with sqlite3.connect(stage/'thumbs.db') as db:
                if not db.execute('SELECT count(*) FROM thumbnails WHERE media_id=?',(media_id,)).fetchone()[0]:
                    raise RuntimeError('缺少视频缩略图，未发布此任务')
            self.build_actions(stage, jid)
            self.update(jid,status='publishing',stage='加入视频库',progress=97,message='检查索引并更新在线检索服务')
            future=asyncio.run_coroutine_threadsafe(self.publisher(stage),self.loop)
            # Cancellation must cancel the pending publish before staging cleanup.
            deadline=time.monotonic()+120
            while True:
                try:
                    future.result(timeout=0.5)
                    break
                except TimeoutError:
                    if self.stopping.is_set() or time.monotonic()>deadline:
                        future.cancel()
                        raise RuntimeError('发布等待已中断，可重试处理')
            published=True
            if self.action_builder:
                self.action_state = {'status': 'ready', 'progress': 100, 'message': '全部已入库视频支持 Qwen 片段检索'}
            self.update(jid,status='completed',stage='已入库',progress=100,media_id=media_id,message='已加入视频库，可直接播放和检索')
        except Exception as exc:
            # If pointer publication succeeded immediately before an interruption,
            # never delete that active version. Startup recovery reconciles it.
            published=published or (stage.exists() and (self.root/'projects/demo').resolve()==stage.resolve())
            self.update(jid,status='failed',stage='处理失败',message=str(exc)[:1500])
            if self.action_builder and self.action_state['status'] == 'building':
                self.action_state = {'status': 'failed', 'progress': 0, 'message': '新视频未发布，请在失败的上传任务中重试：'+str(exc)[:1000]}
        finally:
            if not published and release.exists(): shutil.rmtree(release)
