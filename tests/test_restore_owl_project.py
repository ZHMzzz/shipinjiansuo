import sqlite3
import faiss
import numpy as np
import pytest
from scripts.qwen_project import validate_project, OWL, FACE
from scripts.restore_owl_project import validate_owl


def test_owl_validation_rejects_stale_id_mapping_even_when_counts_match(tmp_path):
    (tmp_path/'metadata').mkdir()
    with sqlite3.connect(tmp_path/'metadata/internal.db') as db:
        db.executescript('CREATE TABLE vectors(id INTEGER,feature_extractor_id TEXT); CREATE TABLE vector_metadata_owlv2(vector_id INTEGER);')
        db.execute('INSERT INTO vectors VALUES(7,?)',(OWL,));db.execute('INSERT INTO vector_metadata_owlv2 VALUES(7)')
    features=tmp_path/'store'/OWL/'features';features.mkdir(parents=True)
    folder=tmp_path/'store'/OWL/'index';folder.mkdir()
    index=faiss.IndexIDMap2(faiss.IndexFlatIP(2));index.add_with_ids(np.array([[1.,0.]],dtype='float32'),np.array([7],dtype='int64'))
    faiss.write_index(index,str(features/'video-000000.faiss'));faiss.write_index(index,str(folder/'video-IndexFlatIP.faiss'))
    assert validate_owl(tmp_path)==1
    wrong=faiss.IndexIDMap2(faiss.IndexFlatIP(2));wrong.add_with_ids(np.array([[1.,0.]],dtype='float32'),np.array([8],dtype='int64'))
    faiss.write_index(wrong,str(folder/'video-IndexFlatIP.faiss'))
    with pytest.raises(ValueError,match='ID 不一致'):validate_owl(tmp_path)


def test_runtime_requires_owl_but_allows_face_and_owl(tmp_path):
    (tmp_path/'metadata').mkdir()
    with sqlite3.connect(tmp_path/'metadata/internal.db') as db:
        db.execute('CREATE TABLE vectors(feature_extractor_id TEXT)');db.executemany('INSERT INTO vectors VALUES(?)',[(FACE,),(OWL,)])
    with pytest.raises(ValueError,match='缺少 OWLv2'):validate_project(tmp_path,require_owl=True)
    index=tmp_path/'store'/OWL/'index/video-IndexFlatIP.faiss';index.parent.mkdir(parents=True);index.write_bytes(b'checked separately by validate_owl')
    validate_project(tmp_path,require_owl=True)


def test_publish_refuses_to_overwrite_uploads_added_during_build(tmp_path):
    import json
    from scripts.restore_owl_project import publish
    (tmp_path/'logs').mkdir();(tmp_path/'projects').mkdir()
    current=tmp_path/'projects/current';current.mkdir()
    stage=tmp_path/'projects/staged';stage.mkdir()
    pointer=tmp_path/'projects/demo';pointer.symlink_to(current)
    (stage/'owl-restoration.json').write_text(json.dumps({'base':str(tmp_path/'projects/older')}))
    with pytest.raises(RuntimeError,match='视频库已更新'):publish(tmp_path,stage)
    assert pointer.resolve()==current
    assert stage.exists()
