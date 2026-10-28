"""Real upload/delete on an isolated project root, never the production pointer."""
import asyncio
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time

import numpy as np
from fastapi import UploadFile

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from scripts.action_search import ActionEngine, read_snapshot, windows, media_catalog
from scripts.ingestion import IngestionManager
from scripts.native_qwen import QWEN, prepare_native_qwen_project, validate_native_qwen
from scripts.video_deletion import delete_video, media_revision, read_media, validate_stores


def qwen_rows(project):
    with sqlite3.connect(project/'metadata/internal.db') as db:
        return db.execute('SELECT id,media_id,timestamp,end_timestamp FROM vectors WHERE feature_extractor_id=? ORDER BY id',(QWEN,)).fetchall()


async def run():
    source=(ROOT/'projects/demo').resolve()
    before=json.loads((HERE/'before.json').read_text());assert str(source)==before['project']
    root=HERE/'work';assert not root.exists();root.mkdir()
    engine=ActionEngine();result={'status':'running'}
    try:
        (root/'logs').mkdir();(root/'data').mkdir();(root/'projects/releases/initial').mkdir(parents=True)
        shutil.copy2(ROOT/'wise_config.yaml',root/'wise_config.yaml')
        (root/'.media-runtime').symlink_to(ROOT/'.media-runtime',target_is_directory=True)
        initial=root/'projects/releases/initial/demo';shutil.copytree(source,initial)
        (root/'projects/demo').symlink_to(initial)
        old_meta,old_matrix,_=read_snapshot(initial,complete=True);old_ids=qwen_rows(initial)
        migrated=prepare_native_qwen_project(root)
        np.testing.assert_array_equal(old_matrix,read_snapshot(migrated,complete=True)[1])
        assert old_ids==qwen_rows(migrated) and not (migrated/'action-index').exists()
        result['migration']={'videos':len(old_meta['media']),'segments':len(old_meta['segments']),'vectors_and_ids_exact':True}
        print('migration verified',flush=True)
        async def publish(stage):
            validate_native_qwen(stage);validate_stores(stage)
            assert not (stage/'action-index').exists()
            pointer=root/'projects/demo';tmp=root/'projects/next';tmp.symlink_to(stage);os.replace(tmp,pointer)
        encoded=[]
        async def build(stage,progress):
            catalog,records,vectors,missing=await asyncio.to_thread(engine.prepare,stage)
            for media in missing:
                for ts,te in windows(float(media['duration'])):
                    seg=dict(media_id=media['id'],ts=ts,te=te)
                    vectors.append(await asyncio.to_thread(engine.encode_segment,media,seg));records.append(seg);encoded.append(seg)
            await asyncio.to_thread(engine.finish,stage,catalog,records,vectors)
        manager=IngestionManager(root,publish,build);manager.loop=asyncio.get_running_loop()
        media=next(m for m in old_meta['media'] if m['id']==18)
        clip=Path(media['location'])/media['path']
        if not clip.is_absolute():clip=ROOT/clip
        digest=hashlib.sha256(clip.read_bytes()).hexdigest()
        job=manager.enqueue(UploadFile(file=io.BytesIO(clip.read_bytes()),filename='统一索引上传验证.mp4'))['job']
        started=time.perf_counter();await asyncio.to_thread(manager.process_job,job)
        job=next(j for j in manager.jobs() if j['id']==job['id'])
        assert job['status']=='completed',job
        uploaded=(root/'projects/demo').resolve();meta,matrix,_=read_snapshot(uploaded,complete=True)
        assert len(meta['media'])==42 and len(meta['segments'])==136
        np.testing.assert_array_equal(matrix[:len(old_matrix)],old_matrix)
        assert [r for r in qwen_rows(uploaded) if r[1]!=job['media_id']]==old_ids
        assert len(encoded)==1 and encoded[0]['media_id']==job['media_id']
        result['upload']={'seconds':time.perf_counter()-started,'new_media_id':job['media_id'],'encoded_segments':encoded,'existing_vectors_and_ids_exact':True,'videos':42,'segments':136}
        print('real upload verified',flush=True)
        deletion=await asyncio.to_thread(delete_video,manager,job['media_id'],media_revision(read_media(uploaded,job['media_id']),uploaded))
        final=(root/'projects/demo').resolve();meta,matrix,_=read_snapshot(final,complete=True)
        np.testing.assert_array_equal(matrix,old_matrix);assert qwen_rows(final)==old_ids
        assert len(meta['media'])==41 and not deletion['cleanup_pending']
        assert not (manager.uploads/job['id']/'source.mp4').exists()
        assert clip.exists() and hashlib.sha256(clip.read_bytes()).hexdigest()==digest
        result['delete']={'response':deletion,'baseline_vectors_and_ids_restored':True,'original_source_untouched':True}
        await asyncio.to_thread(manager.backfill_actions)
        assert manager.action_state['status']=='ready' and len(encoded)==1
        result['backfill']={'complete_store_skips_encoding':True}
        result['status']='completed';print('delete and backfill verified',flush=True)
    except BaseException as exc:
        result['status']='failed';result['error']=repr(exc)
        raise
    finally:
        (HERE/'integration.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        engine.unload()
        assert root.parent==HERE and root.name=='work'
        shutil.rmtree(root,ignore_errors=True)


if __name__=='__main__':asyncio.run(run())
