"""Multiple enrollment photos per name; retain the legacy representative row."""
import hashlib
import sqlite3
import uuid
from contextlib import contextmanager

import numpy as np
from fastapi import HTTPException

MAX_PHOTOS = 10


@contextmanager
def connect(path):
    db = sqlite3.connect(path, timeout=30)
    db.execute('PRAGMA foreign_keys=ON')
    try:
        with db:
            yield db
    finally:
        db.close()


def initialize_people(path):
    with connect(path) as db:
        db.execute('BEGIN IMMEDIATE')
        db.execute('CREATE TABLE IF NOT EXISTS people (name TEXT PRIMARY KEY, vector BLOB NOT NULL, photo BLOB NOT NULL, model TEXT NOT NULL)')
        db.execute('''CREATE TABLE IF NOT EXISTS person_photos (
            id TEXT PRIMARY KEY, name TEXT NOT NULL REFERENCES people(name) ON DELETE CASCADE,
            vector BLOB NOT NULL, photo BLOB NOT NULL, model TEXT NOT NULL,
            source_hash TEXT NOT NULL, UNIQUE(name,source_hash))''')
        # Copy bytes, not re-encoded faces. Repeated startup does not add photos.
        for name, vector, photo, model in db.execute('''SELECT name,vector,photo,model FROM people
                WHERE NOT EXISTS (SELECT 1 FROM person_photos p WHERE p.name=people.name)''').fetchall():
            db.execute('INSERT INTO person_photos VALUES(?,?,?,?,?,?)',
                       (uuid.uuid4().hex, name, vector, photo, model, hashlib.sha256(photo).hexdigest()))


def list_people(path):
    with connect(path) as db:
        rows = db.execute('SELECT name,id FROM person_photos ORDER BY name,rowid').fetchall()
    grouped = {}
    for name, photo_id in rows:
        grouped.setdefault(name, []).append(photo_id)
    return [{'name': name, 'photo_count': len(ids), 'photo_ids': ids} for name, ids in grouped.items()]


def load_person_vectors(path, name, model):
    with connect(path) as db:
        rows = db.execute('SELECT vector,model FROM person_photos WHERE name=? ORDER BY rowid', (name,)).fetchall()
    if not rows:
        raise HTTPException(404, '未登记此姓名，请先在人脸维护中登记')
    if any(row[1] != model for row in rows):
        raise HTTPException(409, '人脸模型已改变，请重新登记')
    vectors = [np.frombuffer(row[0], dtype=np.float32) for row in rows]
    if any(v.shape != (512,) or not np.isfinite(v).all() or np.linalg.norm(v) == 0 for v in vectors):
        raise HTTPException(409, '登记的人脸特征无效，请重新登记')
    return np.stack(vectors)


def add_photos(path, name, photos, model):
    """All validated photos commit together; duplicate files do not consume slots."""
    with connect(path) as db:
        db.execute('BEGIN IMMEDIATE')
        existing = db.execute('SELECT source_hash,model FROM person_photos WHERE name=?', (name,)).fetchall()
        if any(row[1] != model for row in existing):
            raise HTTPException(409, '人脸模型已改变，请先删除原登记后重新登记')
        known = {row[0] for row in existing}
        additions = []
        for vector, photo, digest in photos:
            if digest not in known:
                additions.append((vector, photo, digest))
                known.add(digest)
        if len(existing) + len(additions) > MAX_PHOTOS:
            raise HTTPException(400, f'每人最多登记 {MAX_PHOTOS} 张照片，当前已有 {len(existing)} 张')
        if not existing and additions:
            vector, photo, _ = additions[0]
            db.execute('INSERT INTO people VALUES(?,?,?,?)', (name, vector, photo, model))
        for vector, photo, digest in additions:
            db.execute('INSERT INTO person_photos VALUES(?,?,?,?,?,?)',
                       (uuid.uuid4().hex, name, vector, photo, model, digest))
    return {'name': name, 'dimensions': 512, 'added': len(additions),
            'duplicates': len(photos) - len(additions), 'photo_count': len(existing) + len(additions)}


def read_photo(path, name, photo_id=None):
    with connect(path) as db:
        if photo_id is None:
            row = db.execute('SELECT photo FROM person_photos WHERE name=? ORDER BY rowid LIMIT 1', (name,)).fetchone()
        else:
            row = db.execute('SELECT photo FROM person_photos WHERE name=? AND id=?', (name, photo_id)).fetchone()
    if row is None:
        raise HTTPException(404, '此登记照片不存在')
    return row[0]


def delete_photo(path, name, photo_id):
    with connect(path) as db:
        db.execute('BEGIN IMMEDIATE')
        ids = [r[0] for r in db.execute('SELECT id FROM person_photos WHERE name=? ORDER BY rowid', (name,))]
        if photo_id not in ids:
            raise HTTPException(404, '此登记照片不存在')
        if len(ids) == 1:
            raise HTTPException(409, '至少保留一张照片；如不再需要此人物，请删除整个人物登记')
        db.execute('DELETE FROM person_photos WHERE name=? AND id=?', (name, photo_id))
        vector, photo, model = db.execute('SELECT vector,photo,model FROM person_photos WHERE name=? ORDER BY rowid LIMIT 1', (name,)).fetchone()
        db.execute('UPDATE people SET vector=?,photo=?,model=? WHERE name=?', (vector, photo, model, name))
    return {'name': name, 'deleted_photo': photo_id, 'photo_count': len(ids) - 1}


def delete_person(path, name):
    with connect(path) as db:
        db.execute('DELETE FROM people WHERE name=?', (name,))
    return {'deleted': name}
