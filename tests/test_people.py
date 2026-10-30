import io
import sqlite3
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from PIL import Image
from wise.api.dependencies import get_embedding_service

from scripts.people_api import create_people_router
from scripts import people_store as store

MODEL = 'test-face'


def picture(color):
    stream = io.BytesIO()
    Image.new('RGB', (12, 12), color).save(stream, 'PNG')
    return stream.getvalue()


def vector(index=0):
    result = np.zeros((1, 512), dtype=np.float32)
    result[0, index] = 1
    return result


@pytest.fixture
def enrollment(tmp_path):
    path = tmp_path / 'people.sqlite'
    store.initialize_people(path)
    extractor = NS(preprocess_image=lambda images: images,
                   extract_image_features=Mock(return_value=[NS(vectors=vector())]))
    app = FastAPI()
    app.include_router(create_people_router(path, MODEL))
    app.dependency_overrides[get_embedding_service] = lambda: NS(feature_extractors={MODEL: extractor})
    with TestClient(app) as client:
        yield client, path, extractor


def upload(client, colors, name='测试人物', field='photos'):
    return client.post('/mvp-api/people', data={'name': name},
                       files=[(field, (f'{i}.png', picture(color), 'image/png')) for i, color in enumerate(colors)])


def test_migration_preserves_legacy_bytes_and_is_idempotent(tmp_path):
    path = tmp_path / 'people.sqlite'
    photo = picture('red'); feature = vector().tobytes()
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE people(name TEXT PRIMARY KEY,vector BLOB,photo BLOB,model TEXT)')
        db.execute('INSERT INTO people VALUES(?,?,?,?)', ('旧登记', feature, photo, MODEL))
    store.initialize_people(path)
    first = store.list_people(path)
    store.initialize_people(path)
    assert store.list_people(path) == first
    assert first[0]['photo_count'] == 1
    assert store.read_photo(path, '旧登记') == photo
    assert store.load_person_vectors(path, '旧登记', MODEL).tobytes() == feature
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT vector,photo FROM people').fetchone() == (feature, photo)


def test_append_batch_legacy_single_field_and_duplicate_retry(enrollment):
    client, path, extractor = enrollment
    first = upload(client, ['red'], field='photo').json()
    assert first['photo_count'] == first['added'] == 1
    extractor.extract_image_features.side_effect = [[NS(vectors=vector(1))], [NS(vectors=vector(2))]]
    added = upload(client, ['blue', 'green']).json()
    assert added['added'] == 2 and added['photo_count'] == 3
    assert np.argmax(store.load_person_vectors(path, '测试人物', MODEL), axis=1).tolist() == [0, 1, 2]
    extractor.extract_image_features.side_effect = None
    repeated = upload(client, ['blue', 'green']).json()
    assert repeated['added'] == 0 and repeated['duplicates'] == 2 and repeated['photo_count'] == 3
    assert client.get('/mvp-api/people').json() == ['测试人物']
    details = client.get('/mvp-api/people?details=true').json()[0]
    assert len(details['photos']) == 3
    for photo in details['photos']:
        response = client.get(photo['url'])
        assert response.status_code == 200 and response.headers['content-type'] == 'image/jpeg'


@pytest.mark.parametrize('bad_count', [0, 2])
def test_invalid_face_rejects_entire_batch_without_losing_old_registration(enrollment, bad_count):
    client, path, extractor = enrollment
    upload(client, ['red'])
    before = store.list_people(path)
    extractor.extract_image_features.side_effect = [[NS(vectors=vector(1))], [NS(vectors=np.zeros((bad_count, 512)))]]
    result = upload(client, ['green', 'blue'])
    assert result.status_code == 400 and '第 2 张' in result.json()['detail']
    assert store.list_people(path) == before
    assert np.argmax(store.load_person_vectors(path, '测试人物', MODEL)) == 0


def test_invalid_file_and_invalid_vector_leave_no_registration(enrollment):
    client, path, extractor = enrollment
    response = client.post('/mvp-api/people', data={'name':'新人物'}, files={'photo':('bad.jpg', b'not an image')})
    assert response.status_code == 400
    extractor.extract_image_features.return_value = [NS(vectors=np.full((1,512), np.nan))]
    assert upload(client, ['red']).status_code == 503
    assert store.list_people(path) == []


def test_capacity_and_duplicate_in_same_batch(enrollment):
    client, path, _ = enrollment
    colors = [(i, 10, 20) for i in range(10)]
    assert upload(client, colors).json()['photo_count'] == 10
    assert upload(client, [(30,10,20)]).status_code == 400
    assert store.list_people(path)[0]['photo_count'] == 10
    result = upload(client, ['red','red'], name='另一人').json()
    assert result['added'] == 1 and result['duplicates'] == 1
    assert upload(client, colors + ['blue'], name='超限').status_code == 400


def test_delete_photo_is_scoped_preserves_last_and_updates_representative(enrollment):
    client, path, _ = enrollment
    upload(client, ['red', 'blue']); upload(client, ['green'], name='另一人')
    people = {p['name']:p for p in client.get('/mvp-api/people?details=true').json()}
    photos = people['测试人物']['photos']; other = people['另一人']['photos'][0]
    assert client.delete('/mvp-api/people/测试人物/photos/' + other['id']).status_code == 404
    remaining = client.get(photos[1]['url']).content
    assert client.delete(photos[0]['url']).json()['photo_count'] == 1
    assert client.get(photos[0]['url']).status_code == 404
    assert client.get('/mvp-api/people/测试人物/photo').content == remaining
    assert client.delete(photos[1]['url']).status_code == 409
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT photo FROM people WHERE name=?', ('测试人物',)).fetchone()[0] == remaining
    assert client.delete('/mvp-api/people/测试人物').status_code == 200
    store.initialize_people(path)
    assert client.get('/mvp-api/people').json() == ['另一人']
    assert client.get(photos[1]['url']).status_code == 404


def test_wrong_model_rejects_append_and_query(enrollment):
    client, path, _ = enrollment
    upload(client, ['red'])
    with sqlite3.connect(path) as db:
        db.execute("UPDATE person_photos SET model='old-model'")
    assert upload(client, ['blue']).status_code == 409
    with pytest.raises(HTTPException) as exc:
        store.load_person_vectors(path, '测试人物', MODEL)
    assert exc.value.status_code == 409
