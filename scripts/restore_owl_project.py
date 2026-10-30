"""Restore OWLv2 through WISE extraction, merge and index APIs on a staged copy."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import uuid

try:
    from .qwen_project import retained_fingerprint, FACE
    from .release_retention import publish_pointer, prune_releases
except ImportError:
    from qwen_project import retained_fingerprint, FACE
    from release_retention import publish_pointer, prune_releases

OWL = 'transformers/owlv2/google/owlv2-base-patch16-ensemble'
ROOT = Path(__file__).resolve().parents[1]


def catalog(project):
    with sqlite3.connect(project/'metadata/internal.db') as db:
        return {r[0]:r[1:] for r in db.execute('SELECT id,source_collection_id,path,checksum,size_in_bytes,duration FROM media ORDER BY id')}


def metadata_copy(source, dest):
    dest.mkdir(parents=True)
    shutil.copytree(source/'metadata',dest/'metadata')
    shutil.copy2(source/'thumbs.db',dest/'thumbs.db')


def merge(target, source):
    from wise.wise_project import WiseProject
    a,b = WiseProject(target),WiseProject(source)
    try: a.merge(b,dry_run=False)
    finally:
        for p in (a,b):
            p.db_engine.dispose();p.thumbsdb_engine.dispose()


def validate_owl(project):
    import faiss
    import numpy as np
    from wise.data_models import ModalityType
    from wise.feature.store import FaissStore
    with sqlite3.connect(project/'metadata/internal.db') as db:
        rows=db.execute('SELECT id FROM vectors WHERE feature_extractor_id=? ORDER BY id',(OWL,)).fetchall()
        ids={r[0] for r in rows}
        metadata={r[0] for r in db.execute('SELECT vector_id FROM vector_metadata_owlv2')}
    if not ids or ids!=metadata: raise ValueError('OWLv2 区域向量与元数据不一致')
    store=FaissStore(ModalityType.VIDEO,project/'store'/OWL/'features')
    feature_ids=set()
    for batch_ids, vectors in store.iter_batch(4096):
        if not np.isfinite(vectors).all(): raise ValueError('OWLv2 特征非有限数值')
        feature_ids.update(int(i) for i in batch_ids)
    index=faiss.read_index(str(project/'store'/OWL/'index/video-IndexFlatIP.faiss'))
    if feature_ids!=ids or set(int(i) for i in faiss.vector_to_array(index.id_map))!=ids or index.ntotal!=len(ids):
        raise ValueError('OWLv2 数据库、特征、索引 ID 不一致')
    return len(ids)


def build(root, old=None):
    current=(root/'projects/demo').resolve(strict=True)
    if (current/'store'/OWL/'index/video-IndexFlatIP.faiss').exists():
        validate_owl(current)
        return current
    if old is None and (current/'qwen-migration.json').exists():
        candidate=Path(json.loads((current/'qwen-migration.json').read_text())['source'])
        if (candidate/'store'/OWL/'index/video-IndexFlatIP.faiss').exists(): old=candidate
    before=retained_fingerprint(current)
    release=root/'projects/releases'/('owl-'+uuid.uuid4().hex)
    stage=release/'demo';work=release/'work'
    shutil.copytree(current,stage);work.mkdir()
    all_media=catalog(current); reused=[]
    try:
        if old:
            old=Path(old).resolve(strict=True);old_media=catalog(old)
            if not old_media or any(all_media.get(mid)!=row for mid,row in old_media.items()):
                raise ValueError('旧项目媒体与当前版本不一致，不能复用其 OWLv2 特征')
            validate_owl(old)
            source=work/'old-owl';metadata_copy(old,source)
            shutil.copytree(old/'store'/OWL,source/'store'/OWL)
            merge(stage,source)
            reused=list(old_media)
            if retained_fingerprint(stage)!=before: raise ValueError('合并改变了保留数据')
        missing=sorted(set(all_media)-set(reused))
        if missing:
            source=work/'new-owl';metadata_copy(current,source)
            with sqlite3.connect(source/'metadata/internal.db') as db:
                db.execute('PRAGMA foreign_keys=ON')
                db.execute('DELETE FROM vectors')
                db.execute('DELETE FROM media WHERE id NOT IN ('+','.join('?' for _ in missing)+')',missing)
            # Existing metadata + no stores selects WISE add-feature-extractor mode.
            command=[sys.executable,'-u','-m','wise','extract-features','--project-dir',str(source),
                '--video-feature-id',OWL,'--no-thumbnails','--num-workers','0','--yes']
            subprocess.run(command,cwd=root,check=True,env={**os.environ,'HF_HUB_OFFLINE':'1'})
            # Do not merge copied thumbnails for media outside this source catalog.
            with sqlite3.connect(source/'thumbs.db') as db:
                db.execute('DELETE FROM thumbnails')
            merge(stage,source)
        subprocess.run([sys.executable,'-u','-m','wise','create-index','--project-dir',str(stage),'--overwrite'],cwd=root,check=True)
        count=validate_owl(stage)
        if retained_fingerprint(stage)!=before: raise ValueError('恢复改变了应保留的数据，未发布')
        (stage/'owl-restoration.json').write_text(json.dumps({'base':str(current),'old':str(old) if old else None,
            'reused_media':reused,'extracted_media':missing,'owl_vectors':count,'retained':before},ensure_ascii=False,indent=2))
        shutil.rmtree(work)
        return stage
    except BaseException:
        shutil.rmtree(release,ignore_errors=True)
        raise


def publish(root, stage):
    stage=Path(stage).resolve(strict=True)
    with (root/'logs/ingestion.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        current=(root/'projects/demo').resolve(strict=True)
        if current==stage: return
        record=json.loads((stage/'owl-restoration.json').read_text())
        if str(current)!=record['base']: raise RuntimeError('视频库已更新，请基于最新版本重新恢复')
        if retained_fingerprint(stage)!=record['retained']: raise RuntimeError('恢复版本校验失败')
        validate_owl(stage)
        publish_pointer(root, stage)
        prune_releases(root)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-project',type=Path)
    parser.add_argument('--publish',type=Path)
    args=parser.parse_args()
    if args.publish: publish(ROOT,args.publish)
    else: print('STAGED_PROJECT='+str(build(ROOT,args.old_project)),flush=True)
