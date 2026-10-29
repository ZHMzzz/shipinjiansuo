import json
import shutil
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from scripts.library_catalog import list_video_catalog, read_catalog_snapshot
from scripts.original_media import original_response, original_metadata, original_path
from scripts.video_deletion import read_media
from test_video_deletion import fixture


def probe_result(width=1920, height=1080, rotation=0):
    return SimpleNamespace(stdout=json.dumps({'streams': [{'width': width, 'height': height,
        'side_data_list': [{'rotation': rotation}]}]}))


def test_original_display_keeps_index_paths_thumbnails_and_duration(tmp_path):
    manager, project, job = fixture(tmp_path)
    indexed = read_media(project, 20)
    with patch('scripts.original_media.subprocess.run', return_value=probe_result()) as probe:
        first = list_video_catalog(project, manager.jobs(), tmp_path)['videos'][0]
        second = list_video_catalog(project, manager.jobs(), tmp_path)['videos'][0]
    assert probe.call_count == 1 and first == second
    assert (first['width'], first['height'], first['size_in_bytes']) == (1920, 1080, len(b'owned test clip'))
    assert first['url'].startswith('/mvp-api/videos/20/original?revision=')
    assert first['fallback_url'] == '/demo/media/20'
    assert first['thumbnail'] == '/demo/thumbnail?media_id=20&timestamp=0'
    assert first['duration'] == indexed['duration']
    assert read_media(project, 20) == indexed
    assert read_catalog_snapshot(project)[0]['width'] == 64
    assert 'location' not in first


def test_missing_original_and_unmatched_id_keep_indexed_playback(tmp_path):
    manager, project, job = fixture(tmp_path)
    jobs = manager.jobs()
    jobs[0]['media_id'] = 10  # An old job must not attach an original to a reused ID.
    videos = list_video_catalog(project, jobs, tmp_path)['videos']
    assert all('fallback_url' not in v for v in videos)
    assert videos[1]['name'] == 'keep.mp4'
    original_path(job, tmp_path).unlink()
    videos = list_video_catalog(project, manager.jobs(), tmp_path)['videos']
    assert videos[0]['url'] == '/demo/media/20' and videos[0]['width'] == 64


def test_original_metadata_rotation_and_replacement_invalidates_cache(tmp_path):
    source = tmp_path/'source.mov'; source.write_bytes(b'first')
    with patch('scripts.original_media.subprocess.run', return_value=probe_result(rotation=-90)) as probe:
        assert original_metadata(source, tmp_path) == {'width':1080, 'height':1920, 'size_in_bytes':5}
        original_metadata(source, tmp_path)
        assert probe.call_count == 1
        source.write_bytes(b'replaced original')
        assert original_metadata(source, tmp_path)['size_in_bytes'] == 17
        assert probe.call_count == 2


def test_unreadable_metadata_does_not_show_compressed_resolution(tmp_path):
    manager, project, _ = fixture(tmp_path)
    with patch('scripts.original_media.subprocess.run', side_effect=OSError('probe unavailable')):
        item = list_video_catalog(project, manager.jobs(), tmp_path)['videos'][0]
    assert item['width'] is None and item['height'] is None
    assert item['size_in_bytes'] == len(b'owned test clip')
    assert item['fallback_url'] == '/demo/media/20'


def test_original_http_range_head_and_identity(tmp_path):
    manager, project, job = fixture(tmp_path)
    app = FastAPI()
    @app.api_route('/original/{mid}', methods=['GET', 'HEAD'])
    def original(mid: int, revision: str):
        return original_response(mid, revision, project, manager.jobs(), tmp_path)
    client = TestClient(app)
    token = list_video_catalog(project, manager.jobs(), tmp_path)['videos'][0]['revision']
    url = f'/original/20?revision={token}'
    body = b'owned test clip'
    response = client.get(url)
    assert response.content == body and response.headers['content-type'] == 'video/mp4'
    assert response.headers['accept-ranges'] == 'bytes'
    assert client.head(url).content == b''
    assert int(client.head(url).headers['content-length']) == len(body)
    for value, expected in [('bytes=2-6', body[2:7]), ('bytes=-4', body[-4:]), ('bytes=5-', body[5:])]:
        response = client.get(url, headers={'Range':value})
        assert response.status_code == 206 and response.content == expected
    assert client.get(url, headers={'Range':'bytes=999-'}).status_code == 416
    assert client.get('/original/20?revision=stale').status_code == 409
    assert client.get('/original/999?revision=stale').status_code == 404
    original_path(job, tmp_path).unlink()
    assert client.get(url).status_code == 404


def test_original_symlink_cannot_serve_outside_upload(tmp_path):
    manager, project, job = fixture(tmp_path)
    source = original_path(job, tmp_path)
    outside = tmp_path/'outside.mp4'; outside.write_bytes(b'not an upload')
    source.unlink(); source.symlink_to(outside)
    assert original_path(job, tmp_path) is None


def test_catalog_new_publication_updates_original_url(tmp_path):
    manager, project, _ = fixture(tmp_path)
    first = list_video_catalog(project, manager.jobs(), tmp_path)['videos'][0]
    stage = tmp_path/'projects/releases/next/demo'; shutil.copytree(project, stage)
    second = list_video_catalog(stage, manager.jobs(), tmp_path)['videos'][0]
    assert first['url'] != second['url']
    assert first['fallback_url'] == second['fallback_url']
