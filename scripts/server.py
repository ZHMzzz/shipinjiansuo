"""Local research extension. WISE retains decoding, embeddings, FAISS and playback APIs."""
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

import uvicorn
from fastapi import File, Form, Header, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from wise.api import create_app
from wise.api.dependencies import EmbeddingServiceDep, ProjectServiceDep, SearchServiceDep
from wise.api.standalone.search import search_output_to_response
from wise.config import APIConfig
from wise.data_models import MediaType, ModalityType
from wise.api import dependencies
from wise.api.services.project import LocalWiseProjectService
from wise.api.services.search import LocalSearchService
from wise.wise_project import WiseProject
from ingestion import IngestionManager, MAX_UPLOAD
from release_retention import publish_pointer
from combined_search import person_scene_search
from people_store import initialize_people, load_person_vectors
from people_api import create_people_router
from face_search import search_registered_faces
from query_image import decode_query_image
from qwen_project import validate_project, FACE
from native_qwen import prepare_native_qwen_project, validate_native_qwen, QWEN
from library_search import search_library, native_segment_search
from video_deletion import delete_video
from library_catalog import list_video_catalog
from original_media import original_response
from action_search import ActionEngine, SETTINGS as ACTION_SETTINGS, VIDEO_PATCH_PROJECTION, read_snapshot, windows

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / 'data/people.sqlite'
logging.basicConfig(level=logging.INFO)
config = APIConfig(command='serve', project_dir=prepare_native_qwen_project(ROOT))
validate_project(config.project_dir, require_owl=True)
app = create_app(config, ROOT / 'wise/frontend/dist')
search_gate = asyncio.Lock()
actions = ActionEngine()


@app.middleware('http')
async def serialize_search_and_publication(request, call_next):
    # Keep dependency resolution and inference consistent during a version swap.
    # Uploads and task polling remain available while a search is running.
    reads_catalog = request.method in ('GET', 'HEAD') and request.url.path.startswith(
        ('/mvp-api/videos', '/mvp-api/actions/status'))
    if reads_catalog or request.url.path.startswith(('/demo/', '/mvp-api/people', '/mvp-api/search', '/mvp-api/library/search')):
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
            validate_native_qwen(stage)
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
        publish_pointer(ROOT, stage)
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
    progress(0, total, f'准备处理 {len(missing)} 个视频、{total} 个Qwen 片段')
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
        # Finish loading and the first text inference before accepting requests
        # or starting ingestion, which shares the same model and MPS device.
        started = time.perf_counter()
        logging.info('正在加载并预热 Qwen 自然语言检索模型…')
        try:
            await asyncio.to_thread(actions.query_vector, '视频中的场景和动作')
            dependencies.embedding_service.feature_extractors[QWEN] = actions.extractor
        except Exception:
            logging.exception('Qwen 启动预热失败，服务未就绪；请处理错误后重新启动')
            raise
        logging.info('Qwen 自然语言检索模型已就绪，加载及预热耗时 %.2f 秒', time.perf_counter() - started)
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
    snapshot = actions.get_snapshot(project.wise_project._project_dir)
    meta = snapshot[0] if snapshot else {'media':[], 'segments':[]}
    return {**ingestion.action_state, 'indexed_videos':len(meta['media']),
        'indexed_segments':len(meta['segments']), 'model_loaded':actions.extractor is not None,
        'model_load_seconds':actions.last_load_seconds, 'settings':ACTION_SETTINGS,
        'video_patch_projection':VIDEO_PATCH_PROJECTION}


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
    return list_video_catalog(project.wise_project._project_dir, ingestion.jobs())


@app.api_route('/mvp-api/videos/{media_id}/original', methods=['GET', 'HEAD'])
def play_original(media_id: int, revision: str, project: ProjectServiceDep):
    return original_response(media_id, revision, project.wise_project._project_dir, ingestion.jobs())

@app.delete('/mvp-api/videos/{media_id}')
async def remove_video(media_id: int, revision: str = Header(..., alias='If-Match', min_length=64, max_length=64)):
    try:
        return await asyncio.to_thread(delete_video, ingestion, media_id, revision)
    except HTTPException:
        raise
    except Exception as exc:
        logging.exception('Video deletion failed')
        raise HTTPException(503, '删除未完成，请重试：'+str(exc)[:200]) from exc


initialize_people(DB)
app.include_router(create_people_router(DB, FACE))


@app.get('/')
def home():
    return RedirectResponse('/mvp/')


@app.post('/mvp-api/people/{name}/search')
def search_person(name: str, search: SearchServiceDep, project: ProjectServiceDep,
                  embedding: EmbeddingServiceDep, scene: str = Form('', max_length=1000),
                  threshold: float = Form(0.45, ge=0, le=1)):
    started = time.perf_counter()
    vector = load_person_vectors(DB, name, FACE)
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
    result = search_registered_faces(vector, search=search, face_model=FACE, threshold=threshold, limit=100)
    response = search_output_to_response(config, project, [], MediaType.VIDEO, result, 1)
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


def segment_search(project, search, query, limit=20, image=None):
    try:
        return native_segment_search(project, search, config, query, limit, image=image, actions=actions)[0]
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except RuntimeError as exc:
        logging.exception('Qwen segment search failed')
        raise HTTPException(503, 'Qwen 暂不可用：'+str(exc)[:500]) from exc


@app.post('/mvp-api/search')
def search_segments(project: ProjectServiceDep, search: SearchServiceDep, query: str = Form('', max_length=1000),
                    photo: UploadFile | None = File(None),
                    limit: int = Form(20, ge=1, le=50), refine: bool = Form(False)):
    query = query.strip()
    if not query: raise HTTPException(400, '请输入场景或动作描述')
    if refine: raise HTTPException(410, '物体图片已恢复 WISE OWLv2，请刷新页面使用物体图片入口')
    image = query_image(photo) if photo is not None else None
    return segment_search(project, search, query, limit, image=image)


@app.post('/mvp-api/library/search')
def library_query(project: ProjectServiceDep, search: SearchServiceDep, embedding: EmbeddingServiceDep,
                  query: str = Form('', max_length=1000), person: str = Form('', max_length=60),
                  filename: str = Form('', max_length=255), photo: UploadFile | None = File(None),
                  threshold: float = Form(.45, ge=0, le=1), limit: int = Form(100, ge=1, le=100)):
    query, person, filename = query.strip(), person.strip(), filename.strip()
    raw = None
    if photo is not None:
        query_image(photo)  # Validate size and readability before either image search path.
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
