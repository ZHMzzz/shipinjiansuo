"""Read metadata tied to an immutable published project; overlay live job names."""
import json
from functools import lru_cache
import sqlite3
from pathlib import Path

if __package__:
    from .video_deletion import media_revision
else:
    from video_deletion import media_revision


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
    if metadata.exists():
        for segment in json.loads(metadata.read_text())['segments']:
            mid=segment['media_id'];action_counts[mid]=action_counts.get(mid,0)+1
    for item in rows:
        item['revision']=media_revision(item,project)
        item.pop('checksum');item.pop('location')
        item.update(url=f"/demo/media/{item['id']}",thumbnail=f"/demo/thumbnail?media_id={item['id']}&timestamp=0",
            features=counts.get(item['id'],{}),action_segments=action_counts.get(item['id'],0))
    return rows


def list_video_catalog(project, jobs):
    # Resolve before the cache boundary: uploads/deletions publish a new path.
    project=Path(project).resolve()
    uploaded={job['media_id']:job for job in jobs if job['status']=='completed'}
    videos=[]
    for row in read_catalog_snapshot(project):
        job=uploaded.get(row['id'])
        videos.append({**row,'features':dict(row['features']),
            'name':job['filename'] if job else row['path'], 'created':job['created'] if job else None})
    return {'videos':videos,'count':len(videos),'total_duration':sum(v['duration'] or 0 for v in videos)}
