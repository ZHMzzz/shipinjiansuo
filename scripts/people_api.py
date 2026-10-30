"""Enrollment API, independently testable without starting video/model services."""
import hashlib
import io
from urllib.parse import quote

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError
from wise.api.dependencies import EmbeddingServiceDep

if __package__:
    from . import people_store as store
    from .query_image import decode_query_image
else:
    import people_store as store
    from query_image import decode_query_image


def create_people_router(db_path, face_model):
    router = APIRouter(prefix='/mvp-api/people')

    @router.get('')
    def people(details: bool = False):
        records = store.list_people(db_path)
        if not details:
            return [record['name'] for record in records]
        return [{'name': record['name'], 'photo_count': record['photo_count'],
                 'photos': [{'id': pid, 'url': f'/mvp-api/people/{quote(record["name"], safe="")}/photos/{pid}'}
                            for pid in record['photo_ids']]} for record in records]

    @router.get('/{name}/photo')
    def person_photo(name: str):
        return Response(store.read_photo(db_path, name), media_type='image/jpeg', headers={'Cache-Control': 'no-store'})

    @router.get('/{name}/photos/{photo_id}')
    def photo_by_id(name: str, photo_id: str):
        return Response(store.read_photo(db_path, name, photo_id), media_type='image/jpeg', headers={'Cache-Control': 'no-store'})

    @router.post('')
    def register(embedding: EmbeddingServiceDep, name: str = Form(...),
                 photo: UploadFile | None = File(None), photos: list[UploadFile] | None = File(None)):
        name = name.strip()
        if not 1 <= len(name) <= 60:
            raise HTTPException(400, '姓名须为 1～60 个字符')
        uploads = ([photo] if photo is not None else []) + (photos or [])
        if not 1 <= len(uploads) <= store.MAX_PHOTOS:
            raise HTTPException(400, f'请选择 1～{store.MAX_PHOTOS} 张人脸照片')
        prepared = []
        extractor = embedding.feature_extractors[face_model]
        for number, upload in enumerate(uploads, 1):
            prefix = f'第 {number} 张照片：'
            raw = upload.file.read(10 * 1024 * 1024 + 1)
            if len(raw) > 10 * 1024 * 1024:
                raise HTTPException(400, prefix + '照片应小于等于 10 MB')
            try:
                image = decode_query_image(raw)
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError):
                raise HTTPException(400, prefix + '无法读取图片')
            features = extractor.extract_image_features(extractor.preprocess_image([image]))[0]
            if len(features.vectors) != 1:
                raise HTTPException(400, prefix + f'检测到 {len(features.vectors)} 张人脸，请使用清晰的单人照片；本次未保存任何照片')
            vector = np.asarray(features.vectors, dtype=np.float32)
            if vector.shape != (1, 512) or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
                raise HTTPException(503, prefix + '人脸特征无效，请更换照片重试')
            output = io.BytesIO()
            image.save(output, 'JPEG', quality=90)
            prepared.append((vector.tobytes(), output.getvalue(), hashlib.sha256(raw).hexdigest()))
        return store.add_photos(db_path, name, prepared, face_model)

    @router.delete('/{name}/photos/{photo_id}')
    def remove_photo(name: str, photo_id: str):
        return store.delete_photo(db_path, name, photo_id)

    @router.delete('/{name}')
    def remove_person(name: str):
        return store.delete_person(db_path, name)

    return router
