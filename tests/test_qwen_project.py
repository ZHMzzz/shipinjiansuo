import sqlite3
from pathlib import Path
import pytest
from scripts.qwen_project import prepare_qwen_project, retained_fingerprint, FACE, RETIRED, OWL
from test_action_search import project, save


def fixture(root):
    path=project(root/'projects'); save(path)
    with sqlite3.connect(path/'metadata/internal.db') as db:
        db.executescript('''CREATE TABLE vector_metadata_insightface(vector_id INTEGER REFERENCES vectors(id) ON DELETE CASCADE,score REAL);
            CREATE TABLE vector_metadata_owlv2(vector_id INTEGER REFERENCES vectors(id) ON DELETE CASCADE,score REAL);
            CREATE TABLE featured(vector_id INTEGER REFERENCES vectors(id) ON DELETE CASCADE);''')
        db.executemany('INSERT INTO vectors(id,feature_extractor_id) VALUES(?,?)',enumerate((FACE,*RETIRED,OWL),101))
        db.execute('INSERT INTO vector_metadata_insightface VALUES(101,.9)')
        db.execute('INSERT INTO vector_metadata_owlv2 VALUES(103,.8)')
        db.executemany('INSERT INTO featured VALUES(?)',[(101,),(102,),(103,)])
    for model in (FACE,*RETIRED,OWL):
        folder=path/'store'/model/'features';folder.mkdir(parents=True)
        (folder/'vectors.bin').write_bytes(model.encode())
    (path/'thumbs.db').write_bytes(b'preserve thumbnails')
    return path


def test_migration_preserves_data_and_old_release_and_is_idempotent(tmp_path):
    path=fixture(tmp_path); before=retained_fingerprint(path)
    stage=prepare_qwen_project(tmp_path)
    old=tmp_path/'projects/releases/initial/demo'
    assert (tmp_path/'projects/demo').is_symlink() and stage!=old
    assert before==retained_fingerprint(stage)==retained_fingerprint(old)
    assert all((old/'store'/m).exists() and not (stage/'store'/m).exists() for m in RETIRED)
    with sqlite3.connect(stage/'metadata/internal.db') as db:
        assert db.execute('SELECT id,feature_extractor_id FROM vectors WHERE id>=101').fetchall()==[(101,FACE),(103,OWL)]
        assert db.execute('SELECT * FROM featured').fetchall()==[(101,),(103,)]
    assert prepare_qwen_project(tmp_path)==stage


def test_incomplete_qwen_index_blocks_migration(tmp_path):
    path=fixture(tmp_path)
    with sqlite3.connect(path/'metadata/internal.db') as db:
        db.execute("INSERT INTO media VALUES(19,'new.mp4',3,1,'VIDEO')")
    with pytest.raises(ValueError,match='缺少动作索引'): prepare_qwen_project(tmp_path)
    assert all((tmp_path/'projects/demo/store'/m).exists() for m in RETIRED)


def test_failed_validation_does_not_publish_partial_version(tmp_path,monkeypatch):
    from scripts import qwen_project
    path=fixture(tmp_path)
    monkeypatch.setattr(qwen_project,'validate_project',lambda _: (_ for _ in ()).throw(ValueError('test failure')))
    with pytest.raises(ValueError,match='test failure'): prepare_qwen_project(tmp_path)
    assert (tmp_path/'projects/demo').resolve()==tmp_path/'projects/releases/initial/demo'
    assert not list((tmp_path/'projects/releases').glob('qwen-*'))
