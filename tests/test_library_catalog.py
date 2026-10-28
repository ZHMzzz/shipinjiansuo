from unittest.mock import patch
import os
import shutil
from scripts.library_catalog import list_video_catalog
from scripts.video_deletion import prune_project
from test_video_deletion import fixture


def test_native_only_catalog_retains_segment_counts(tmp_path):
    from scripts.native_qwen import prepare_native_qwen_project
    manager,old,_=fixture(tmp_path)
    before=list_video_catalog(old,manager.jobs())
    stage=prepare_native_qwen_project(tmp_path)
    after=list_video_catalog(stage,manager.jobs())
    assert [v['action_segments'] for v in before['videos']]==[v['action_segments'] for v in after['videos']]==[1,1]
    assert not (stage/'action-index').exists()


def test_repeated_queries_do_not_rescan_vector_metadata(tmp_path):
    manager,project,_=fixture(tmp_path);jobs=manager.jobs()
    first=list_video_catalog(project,jobs)
    with patch('scripts.library_catalog.sqlite3.connect',side_effect=AssertionError('query rescanned immutable vector metadata')):
        assert list_video_catalog(project,jobs)==first


def test_new_publication_invalidates_catalog_and_job_names_stay_live(tmp_path):
    manager,old,_=fixture(tmp_path);pointer=tmp_path/'projects/demo';jobs=manager.jobs()
    first=list_video_catalog(pointer,jobs);assert first['count']==2
    jobs[0]['filename']='最新显示名称.mp4'
    assert list_video_catalog(pointer,jobs)['videos'][0]['name']=='最新显示名称.mp4'
    stage=tmp_path/'projects/releases/next/demo';shutil.copytree(old,stage);prune_project(stage,20)
    temporary=tmp_path/'projects/next';temporary.symlink_to(stage);os.replace(temporary,pointer)
    next_catalog=list_video_catalog(pointer,[])
    assert next_catalog['count']==1 and next_catalog['videos'][0]['id']==10
    assert next_catalog['videos'][0]['revision']!=first['videos'][1]['revision']


def test_returned_rows_do_not_mutate_snapshot_cache(tmp_path):
    manager,project,_=fixture(tmp_path)
    first=list_video_catalog(project,manager.jobs());first['videos'][0]['name']='modified';first['videos'][0]['features'].clear()
    second=list_video_catalog(project,manager.jobs())
    assert second['videos'][0]['name']=='delete.mp4' and second['videos'][0]['features']
