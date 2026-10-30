import json
import os
import shutil

import pytest

from scripts import release_retention as retention


def project(root, name):
    path = root / 'projects/releases' / name / 'demo'
    path.mkdir(parents=True)
    (path / 'index.faiss').write_bytes(b'index data')
    return path


def initial(root):
    path = project(root, 'initial')
    (root / 'projects/demo').symlink_to(path)
    return path


def names(root):
    return {p.name for p in (root / 'projects/releases').iterdir()}


def test_publications_keep_actual_predecessor_and_ignore_drafts_and_external_data(tmp_path):
    old = initial(tmp_path)
    draft = project(tmp_path, 'unfinished')
    video = tmp_path / 'data/uploads/source.mp4'
    video.parent.mkdir(parents=True); video.write_bytes(b'original video')
    runtime = tmp_path / '.venv/lib/model'
    runtime.parent.mkdir(parents=True); runtime.write_bytes(b'keep runtime')
    previous = old
    for number in range(4):
        stage = project(tmp_path, str(number))
        retention.publish_pointer(tmp_path, stage)
        # File timestamps must not determine which historical release survives.
        os.utime(old.parent, (2000000000, 2000000000)) if old.exists() else None
        retention.prune_releases(tmp_path)
        assert names(tmp_path) == {'unfinished', stage.parent.name, previous.parent.name}
        assert (previous / 'index.faiss').read_bytes() == b'index data'
        previous = stage
    assert draft.exists() and video.read_bytes() == b'original video'
    assert runtime.read_bytes() == b'keep runtime'


def test_failed_pointer_swap_never_schedules_active_history_for_deletion(tmp_path, monkeypatch):
    old = initial(tmp_path)
    stage = project(tmp_path, 'new')
    replace = retention.os.replace
    def fail_pointer(source, destination):
        if destination == tmp_path / 'projects/demo':
            raise OSError('injected publication failure')
        replace(source, destination)
    monkeypatch.setattr(retention.os, 'replace', fail_pointer)
    with pytest.raises(OSError, match='publication failure'):
        retention.publish_pointer(tmp_path, stage)
    retention.prune_releases(tmp_path)
    assert (tmp_path / 'projects/demo').resolve() == old
    assert old.exists() and stage.exists()


def test_cleanup_retries_after_partial_failure_and_next_publication(tmp_path, monkeypatch):
    old = initial(tmp_path)
    second = project(tmp_path, 'second'); retention.publish_pointer(tmp_path, second)
    third = project(tmp_path, 'third'); retention.publish_pointer(tmp_path, third)
    remove = retention.shutil.rmtree
    def fail_partway(path):
        (path / 'demo/index.faiss').unlink()
        raise OSError('injected cleanup failure')
    monkeypatch.setattr(retention.shutil, 'rmtree', fail_partway)
    assert retention.prune_releases(tmp_path) == []
    assert retention.read_history(third.parent)['obsolete'] == ['initial']
    # Simulate restart: all retry information is on disk, not in a Python object.
    fourth = project(tmp_path, 'fourth'); retention.publish_pointer(tmp_path, fourth)
    monkeypatch.setattr(retention.shutil, 'rmtree', remove)
    assert set(retention.prune_releases(tmp_path)) == {'initial', 'second'}
    assert names(tmp_path) == {'third', 'fourth'}
    assert retention.prune_releases(tmp_path) == []


def test_pending_video_deletion_defers_retention(tmp_path):
    old = initial(tmp_path)
    second = project(tmp_path, 'second'); retention.publish_pointer(tmp_path, second)
    third = project(tmp_path, 'third'); retention.publish_pointer(tmp_path, third)
    record = third / 'deletion.json'
    record.write_text(json.dumps({'cleanup_complete': False}))
    assert retention.prune_releases(tmp_path) == [] and old.exists()
    record.write_text(json.dumps({'cleanup_complete': True}))
    assert retention.prune_releases(tmp_path) == ['initial']


@pytest.mark.parametrize('bad_name', ['../uploads', '.', 'third', 'second'])
def test_corrupt_cleanup_record_cannot_remove_protected_paths(tmp_path, bad_name):
    initial(tmp_path)
    second = project(tmp_path, 'second'); retention.publish_pointer(tmp_path, second)
    third = project(tmp_path, 'third'); retention.publish_pointer(tmp_path, third)
    retention.write_history(third.parent, 'second', [bad_name])
    assert retention.prune_releases(tmp_path) == []
    assert names(tmp_path) == {'initial', 'second', 'third'}


def test_symlink_replaced_obsolete_release_is_not_followed(tmp_path):
    old = initial(tmp_path)
    second = project(tmp_path, 'second'); retention.publish_pointer(tmp_path, second)
    third = project(tmp_path, 'third'); retention.publish_pointer(tmp_path, third)
    outside = tmp_path / 'external'; outside.mkdir(); (outside / 'video.mp4').write_bytes(b'keep')
    shutil.rmtree(old.parent); old.parent.symlink_to(outside)
    assert retention.prune_releases(tmp_path) == []
    assert (outside / 'video.mp4').read_bytes() == b'keep'


def test_reviewed_legacy_history_is_adopted_without_using_directory_mtime(tmp_path):
    current = initial(tmp_path)
    previous = project(tmp_path, 'previous')
    obsolete = project(tmp_path, 'obsolete')
    retention.adopt_history(tmp_path, previous, [obsolete])
    assert retention.prune_releases(tmp_path) == ['obsolete']
    new = project(tmp_path, 'new'); retention.publish_pointer(tmp_path, new)
    assert retention.prune_releases(tmp_path) == ['previous']
    assert names(tmp_path) == {'initial', 'new'}


def test_publication_outside_managed_releases_is_rejected(tmp_path):
    current = initial(tmp_path)
    outside = tmp_path / 'data/demo'; outside.mkdir(parents=True)
    with pytest.raises(ValueError): retention.publish_pointer(tmp_path, outside)
    assert (tmp_path / 'projects/demo').resolve() == current
