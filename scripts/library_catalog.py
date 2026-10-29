"""Read metadata tied to an immutable published project; overlay live job names."""
import json
from functools import lru_cache
import sqlite3
from pathlib import Path

if __package__:
    from .video_deletion import media_revision
    from .original_media import ROOT, upload_for_media, original_path, original_metadata
else:
    from video_deletion import media_revision
    from original_media import ROOT, upload_for_media, original_path, original_metadata


@lru_cache(maxsize=4)
def read_catalog_snapshot(project):
    with sqlite3.connect(project/'metadata/internal.db') as db:
        db.row_factory=sqlite3.Row
        rows=[dict(r) for r in db.execute("SELECT m.id,m.path,m.checksum,s.location,m.duration,m.width,m.height,m.size_in_bytes FROM media m JOIN source_collections s ON s.id=m.source_collection_id WHERE media_type IN ('VIDEO','AV') ORDER BY m.id DESC")]
        counts={}
        for mid,model,count in db.execute('SELECT media_id,feature_extractor_id,count(*) FROM vectors GROUP BY media_id,feature_extractor_id'):
            counts.setdefault(mid,{})[model]=count
    action_counts={}
    metadata=project/'action-index/metadata.json'
    if (project/'qwen-encoding.json').exists():
        action_counts={mid:models.get('hf/Qwen/Qwen3-VL-Embedding/2B',0) for mid,models in counts.items()}
    elif metadata.exists():
        for segment in json.loads(metadata.read_text())['segments']:
            mid=segment['media_id'];action_counts[mid]=action_counts.get(mid,0)+1
    for item in rows:
        item['revision']=media_revision(item,project)
        item.pop('checksum')
        item.update(url=f"/demo/media/{item['id']}",thumbnail=f"/demo/thumbnail?media_id={item['id']}&timestamp=0",
            features=counts.get(item['id'],{}),action_segments=action_counts.get(item['id'],0))
    return rows


def list_video_catalog(project, jobs, root=ROOT):
    # Resolve before the cache boundary: uploads/deletions publish a new path.
    project=Path(project).resolve()
    videos=[]
    for row in read_catalog_snapshot(project):
        job=upload_for_media(row,jobs,root)
        item={**row,'features':dict(row['features']),
            'name':job['filename'] if job else row['path'], 'created':job['created'] if job else None}
        item.pop('location')
        source=original_path(job,root)
        if source is not None:
            try:
                item.update(original_metadata(source,root))
                item['fallback_url']=row['url']
                item['url']=f"/mvp-api/videos/{row['id']}/original?revision={row['revision']}"
            except OSError:
                pass  # Original disappeared; the indexed copy remains playable.
        videos.append(item)
    return {'videos':videos,'count':len(videos),'total_duration':sum(v['duration'] or 0 for v in videos)}
