"""Resolve uploaded originals for display without changing indexed media paths."""
from functools import lru_cache
import json
import mimetypes
from pathlib import Path
import subprocess

from fastapi import HTTPException
from fastapi.responses import FileResponse

if __package__:
    from .video_deletion import read_media, media_revision
else:
    from video_deletion import read_media, media_revision

ROOT = Path(__file__).resolve().parents[1]


def upload_for_media(row, jobs, root=ROOT):
    """Match the source collection as well as ID, which can be reused."""
    uploads = (Path(root)/'data/uploads').resolve()
    for job in jobs:
        if job['status'] != 'completed' or job['media_id'] != row['id']:
            continue
        folder = (uploads/job['id']).resolve()
        if folder.parent != uploads:
            continue
        if (Path(root)/row['location']).resolve() == folder/'prepared':
            return job
    return None


def original_path(job, root=ROOT):
    if job is None:
        return None
    uploads = (Path(root)/'data/uploads').resolve()
    folder = (uploads/job['id']).resolve()
    source = (folder/('source'+job['suffix'])).resolve()
    if folder.parent == uploads and source.parent == folder and source.is_file():
        return source
    return None


@lru_cache(maxsize=512)
def _dimensions(path, probe, size, mtime_ns, ctime_ns):
    # The stat signature invalidates cached metadata when a file is replaced.
    try:
        result = subprocess.run([probe, '-v', 'error', '-select_streams', 'v:0',
            '-show_streams', '-of', 'json', path], capture_output=True, text=True,
            check=True, timeout=10)
        stream = json.loads(result.stdout)['streams'][0]
        width, height = int(stream['width']), int(stream['height'])
        rotation = float(stream.get('tags', {}).get('rotate', 0))
        for side in stream.get('side_data_list', []):
            if 'rotation' in side:
                rotation = float(side['rotation'])
        if round(rotation) % 180 == 90:
            width, height = height, width
        return (width, height) if width > 0 and height > 0 else (None, None)
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError, TypeError):
        return None, None


def original_metadata(path, root=ROOT):
    stat = path.stat()
    width, height = _dimensions(str(path), str(Path(root)/'.media-runtime/bin/ffprobe'),
        stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    return {'width': width, 'height': height, 'size_in_bytes': stat.st_size}


def original_response(media_id, revision, project, jobs, root=ROOT):
    project = Path(project).resolve()
    row = read_media(project, media_id)
    if row is None:
        raise HTTPException(404, '视频不存在')
    if revision != media_revision(row, project):
        raise HTTPException(409, '视频库已更新，请刷新后重试')
    source = original_path(upload_for_media(row, jobs, root), root)
    if source is None:
        raise HTTPException(404, '原视频不存在，请使用兼容播放')
    return FileResponse(source, media_type=mimetypes.guess_type(source.name)[0] or 'application/octet-stream',
        headers={'Cache-Control': 'no-cache'})
