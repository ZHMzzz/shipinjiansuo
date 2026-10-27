"""Local research extension. WISE retains decoding, embeddings, FAISS and playback APIs."""
import io
import logging
import sqlite3
import time
import asyncio
import os
import json
os.environ.setdefault('HF_HUB_OFFLINE', '1')
from typing import Literal
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from wise.api import create_app
from wise.api.common import VectorQueryTerm
from wise.api.dependencies import EmbeddingServiceDep, ProjectServiceDep, SearchServiceDep
from wise.api.standalone.search import search_output_to_response
from wise.config import APIConfig
from wise.data_models import MediaType, ModalityType
from wise.api import dependencies
from wise.api.services.project import LocalWiseProjectService
from wise.api.services.search import LocalSearchService
from wise.wise_project import WiseProject
from ingestion import IngestionManager, MAX_UPLOAD
from combined_search import person_scene_search
from query_image import decode_query_image
from qwen_project import prepare_qwen_project, validate_project, FACE
from library_search import search_library
from action_search import ActionEngine, SETTINGS as ACTION_SETTINGS, read_snapshot, windows

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / 'data/people.sqlite'
logging.basicConfig(level=logging.INFO)
config = APIConfig(command='serve', project_dir=prepare_qwen_project(ROOT))
validate_project(config.project_dir, require_owl=True)
app = create_app(config, ROOT / 'wise/frontend/dist')
search_gate = asyncio.Lock()
actions = ActionEngine()


@app.middleware('http')
async def serialize_search_and_publication(request, call_next):
    # Keep dependency resolution and inference consistent during a version swap.
    # Uploads and task polling remain available while a search is running.
    if request.url.path.startswith(('/demo/', '/mvp-api/people', '/mvp-api/search', '/mvp-api/library/search')):
        async with search_gate:
            return await call_next(request)
    return await call_next(request)


async def publish_project(stage):
    global config
    async with search_gate:
        if ingestion.stopping.is_set(): raise RuntimeError('服务停止，索引未发布')
        new_config = config.model_copy(update={'project_dir':stage})
        def load():
            validate_project(stage, require_owl=True)
            read_snapshot(stage, complete=True)
            project = WiseProject(stage, read_only=True)
            service = LocalWiseProjectService(project, new_config)
            search = LocalSearchService(service, dependencies.embedding_service)
            # Verify each loaded index matches database metadata before publishing.
            with sqlite3.connect(stage/'metadata/internal.db') as db:
                for modality, indexes in search.search_indices.items():
                    for model, index in indexes.items():
                        expected = db.execute('SELECT count(*) FROM vectors WHERE feature_extractor_id=? AND modality=?',(model,modality.name)).fetchone()[0]
                        if index.index.ntotal != expected:
                            raise RuntimeError(f'索引向量数量不一致：{model}')
            info=service.info()
            info.max_search_results=new_config.max_search_results
            return service,search,info
        service,search,info = await asyncio.to_thread(load)
        pointer=ROOT/'projects/demo'
        temporary=ROOT/'projects/demo.next'
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(os.path.relpath(stage,pointer.parent))
        os.replace(temporary,pointer)
        old=dependencies.project_service
        config=new_config
        dependencies.config=new_config
        dependencies.project_service=service
        dependencies.project_info=info
        dependencies.search_service=search
        old.wise_project.db_engine.dispose()
        old.wise_project.thumbsdb_engine.dispose()


async def build_action_index(stage, progress):
    catalog, records, vectors, missing = await asyncio.to_thread(actions.prepare, stage)
    total = sum(len(list(windows(float(m['duration'])))) for m in missing)
    done = 0
    progress(0, total, f'准备处理 {len(missing)} 个视频、{total} 个Qwen 片段；首次需要加载 Qwen 模型')
    for media in missing:
        for start, end in windows(float(media['duration'])):
            if ingestion.stopping.is_set(): raise RuntimeError('服务停止，动作索引未发布')
            segment = {'media_id':media['id'], 'ts':start, 'te':end}
            # Yield between windows so online searches can interleave with indexing.
            async with search_gate:
                vector = await asyncio.to_thread(actions.encode_segment, media, segment)
            records.append(segment); vectors.append(vector); done += 1
            progress(done, total, f'Qwen 片段 {done}/{total} · {media["path"]} · {start:.1f}–{end:.1f} 秒')
            await asyncio.sleep(0)
    if ingestion.stopping.is_set(): raise RuntimeError('服务停止，动作索引未发布')
    await asyncio.to_thread(actions.finish, stage, catalog, records, vectors)


ingestion = IngestionManager(ROOT, publish_project, build_action_index)
wise_lifespan = app.router.lifespan_context


@asynccontextmanager
async def lifespan(application):
    async with wise_lifespan(application) as state:
        ingestion.start(asyncio.get_running_loop())
        try:
            yield state
        finally:
            await asyncio.to_thread(ingestion.stop)


app.router.lifespan_context = lifespan


@app.get('/mvp-api/jobs')
def jobs():
    return {'jobs':ingestion.jobs(),'max_upload_bytes':MAX_UPLOAD}


@app.get('/mvp-api/actions/status')
def action_status(project: ProjectServiceDep):
    path = project.wise_project._project_dir/'action-index/metadata.json'
    meta = json.loads(path.read_text()) if path.exists() else {'media':[], 'segments':[]}
    return {**ingestion.action_state, 'indexed_videos':len(meta['media']),
        'indexed_segments':len(meta['segments']), 'model_loaded':actions.extractor is not None,
        'model_load_seconds':actions.last_load_seconds, 'settings':ACTION_SETTINGS}


@app.post('/mvp-api/actions/retry')
def retry_actions():
    ingestion.request_backfill()
    return {'message':'已检查并安排补建任务'}


@app.get('/mvp-api/action-report')
def action_report(run: Literal['qwen-fp16', 'qwen-default-fp32'] = 'qwen-fp16'):
    folder = ROOT/'experiments/action-search'
    path = folder/(run+'.json')
    if not path.exists(): raise HTTPException(404, '此配置尚无实验记录')
    report = json.loads(path.read_text())
    resources = folder/(run+'-resources.json')
    if resources.exists():
        report['resources'] = {k:v for k,v in json.loads(resources.read_text()).items() if k!='samples'}
    return report


if (ROOT/'experiments/action-search/videos').exists():
    app.mount('/action-media', StaticFiles(directory=ROOT/'experiments/action-search/videos'), name='action-media')


@app.post('/mvp-api/uploads',status_code=202)
def upload_video(video: UploadFile = File(...)):
    return ingestion.enqueue(video)


@app.post('/mvp-api/jobs/{job_id}/retry')
def retry_job(job_id: str):
    return ingestion.retry(job_id)


@app.get('/mvp-api/videos')
def list_videos(project: ProjectServiceDep):
    # Read the immutable version bound to this request, not the moving symlink.
    with sqlite3.connect(project.wise_project._project_dir/'metadata/internal.db') as db:
        db.row_factory=sqlite3.Row
        rows=db.execute("SELECT id,path,duration,width,height,size_in_bytes FROM media WHERE media_type IN ('VIDEO','AV') ORDER BY id DESC").fetchall()
        counts={}
        for mid,model,count in db.execute('SELECT media_id,feature_extractor_id,count(*) FROM vectors GROUP BY media_id,feature_extractor_id'):
            counts.setdefault(mid,{})[model]=count
    uploaded={job['media_id']:job for job in ingestion.jobs() if job['status']=='completed'}
    action_meta = project.wise_project._project_dir/'action-index/metadata.json'
    action_counts = {}
    if action_meta.exists():
        for segment in json.loads(action_meta.read_text())['segments']:
            mid = segment['media_id']; action_counts[mid] = action_counts.get(mid, 0)+1
    videos=[]
    for row in rows:
        item=dict(row)
        job=uploaded.get(row['id'])
        item.update(name=job['filename'] if job else row['path'],created=job['created'] if job else None,
                    url=f"/demo/media/{row['id']}",thumbnail=f"/demo/thumbnail?media_id={row['id']}&timestamp=0",
                    features=counts.get(row['id'],{}))
        item['action_segments'] = action_counts.get(row['id'], 0)
        videos.append(item)
    return {'videos':videos,'count':len(videos),'total_duration':sum(v['duration'] or 0 for v in videos)}

with sqlite3.connect(DB) as db:
    db.execute('CREATE TABLE IF NOT EXISTS people (name TEXT PRIMARY KEY, vector BLOB NOT NULL, photo BLOB NOT NULL, model TEXT NOT NULL)')


@app.get('/')
def home():
    return RedirectResponse('/mvp/')


@app.get('/mvp-api/people')
def people():
    with sqlite3.connect(DB) as db:
        return [row[0] for row in db.execute('SELECT name FROM people ORDER BY name')]


@app.get('/mvp-api/people/{name}/photo')
def person_photo(name: str):
    """Read the existing enrollment photo for the maintenance list."""
    with sqlite3.connect(DB) as db:
        row = db.execute('SELECT photo FROM people WHERE name=?', (name,)).fetchone()
    if row is None:
        raise HTTPException(404, '此姓名尚未登记')
    return Response(content=row[0], media_type='image/jpeg', headers={'Cache-Control':'no-store'})


@app.post('/mvp-api/people')
def register(embedding: EmbeddingServiceDep, name: str = Form(...), photo: UploadFile = File(...)):
    name = name.strip()
    if not 1 <= len(name) <= 60:
        raise HTTPException(400, '姓名须为 1～60 个字符')
    raw = photo.file.read(10 * 1024 * 1024 + 1)
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(400, '照片应小于 10MB')
    try:
        image = decode_query_image(raw)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, '无法读取图片')
    extractor = embedding.feature_extractors[FACE]
    features = extractor.extract_image_features(extractor.preprocess_image([image]))[0]
    if len(features.vectors) != 1:
        raise HTTPException(400, f'检测到 {len(features.vectors)} 张人脸，请使用清晰的单人照片')
    vector = np.asarray(features.vectors[0:1], dtype=np.float32)
    output = io.BytesIO()
    image.save(output, 'JPEG', quality=90)
    with sqlite3.connect(DB) as db:
        try:
            db.execute('INSERT INTO people VALUES (?, ?, ?, ?)', (name, vector.tobytes(), output.getvalue(), FACE))
        except sqlite3.IntegrityError:
            raise HTTPException(409, '此姓名已登记，请先删除或使用另一姓名')
    return {'name': name, 'dimensions': vector.shape[1]}


@app.delete('/mvp-api/people/{name}')
def remove_person(name: str):
    with sqlite3.connect(DB) as db:
        db.execute('DELETE FROM people WHERE name=?', (name,))
    return {'deleted': name}


@app.post('/mvp-api/people/{name}/search')
def search_person(name: str, search: SearchServiceDep, project: ProjectServiceDep,
                  embedding: EmbeddingServiceDep, scene: str = Form('', max_length=1000),
                  threshold: float = Form(0.45, ge=0, le=1)):
    started = time.perf_counter()
    with sqlite3.connect(DB) as db:
        row = db.execute('SELECT vector, model FROM people WHERE name=?', (name,)).fetchone()
    if row is None:
        raise HTTPException(404, '未登记此姓名，请先上传照片登记')
    if row[1] != FACE:
        raise HTTPException(409, '人脸模型已改变，请重新登记')
    vector = np.frombuffer(row[0], dtype=np.float32).reshape(1, -1).copy()
    scene = scene.strip()
    if scene:
        try:
            result, evidence = person_scene_search(vector, scene, threshold,
                face_model=FACE, search=search, actions=actions, project=project.wise_project._project_dir)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)[:500]) from exc
        response = search_output_to_response(config, project, [], MediaType.VIDEO, result, 1)
        payload = response.model_dump(mode='json')
        payload.update(score_kind='fusion', person=name, scene=scene,
                       face_threshold=threshold, scene_scope='containing_segment')
        if payload.get('video_results'):
            for hit in payload['video_results']['unmerged_windows']:
                hit.update(evidence.get(str(hit['vector_id']), {}))
        payload['time'] = time.perf_counter()-started
        return payload
    result = search.search_with_feature(vector, modality_type=ModalityType.VIDEO,
        feature_extractor_id=FACE, start=0, end=100, filter_specs={}, vector_id_constraint=None)
    response = search_output_to_response(config, project,
        [VectorQueryTerm(term_id='person', is_negative=False, vector=vector)], MediaType.VIDEO, result, 1)
    response.time = time.perf_counter() - started
    return response


def query_image(photo):
    raw = photo.file.read(10 * 1024 * 1024 + 1)
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(400, '图片应小于 10MB')
    try:
        image = decode_query_image(raw)
        return image
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(400, '无法读取图片')


def segment_search(project, query, limit=20):
    try:
        return actions.search(project.wise_project._project_dir, query, limit)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except RuntimeError as exc:
        logging.exception('Qwen segment search failed')
        raise HTTPException(503, 'Qwen 暂不可用：'+str(exc)[:500]) from exc


@app.post('/mvp-api/search')
def search_segments(project: ProjectServiceDep, query: str = Form('', max_length=1000),
                    photo: UploadFile | None = File(None),
                    limit: int = Form(20, ge=1, le=50), refine: bool = Form(False)):
    query = query.strip()
    if photo is not None: raise HTTPException(400, '图片请使用物体检索，已移除图文联合检索')
    if not query: raise HTTPException(400, '请输入场景或动作描述')
    if refine: raise HTTPException(410, '物体图片已恢复 WISE OWLv2，请刷新页面使用物体图片入口')
    return segment_search(project, query, limit)


@app.post('/mvp-api/library/search')
def library_query(project: ProjectServiceDep, search: SearchServiceDep, embedding: EmbeddingServiceDep,
                  query: str = Form('', max_length=1000), person: str = Form('', max_length=60),
                  filename: str = Form('', max_length=255), photo: UploadFile | None = File(None),
                  threshold: float = Form(.45, ge=0, le=1), limit: int = Form(100, ge=1, le=100)):
    query, person, filename = query.strip(), person.strip(), filename.strip()
    if query and photo is not None: raise HTTPException(400, '自然语言与参考图片请分开查询')
    raw = None
    if photo is not None:
        query_image(photo)  # Validate size and readability before the native WISE image path.
        photo.file.seek(0)
        raw = photo.file.read()
    try:
        return search_library(catalog=list_videos(project)['videos'], query=query,person=person,
            filename=filename,image_bytes=raw,threshold=threshold,limit=limit,project=project,
            search=search,embedding=embedding,actions=actions,config=config,people_db=DB)
    except ValueError as exc:
        raise HTTPException(409,str(exc)) from exc
    except RuntimeError as exc:
        logging.exception('Library search failed')
        raise HTTPException(503,'检索暂不可用：'+str(exc)[:300]) from exc


app.mount('/queries', StaticFiles(directory=ROOT / 'data/queries'), name='queries')
app.mount('/mvp', StaticFiles(directory=ROOT / 'ui', html=True), name='mvp')

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=9670)
