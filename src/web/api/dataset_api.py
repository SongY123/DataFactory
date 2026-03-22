from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse

from utils.auth_guard import assert_login
from ..entity.request import (
    DatasetCreateRequest,
    DatasetQueryRequest,
    DatasetSqlQueryRequest,
    DatasetUpdateRequest,
    HuggingFaceDatasetImportRequest,
)
from ..service import DatasetService


router = APIRouter(prefix="/datasets", tags=["datasets"])

_dataset_service = DatasetService()


def _ok(data=None, message: str = "ok", meta=None):
    payload = {
        "success": True,
        "message": message,
        "data": data,
    }
    if meta is not None:
        payload["meta"] = meta
    return payload


@router.get("")
def list_datasets(request: Request):
    try:
        user_id = assert_login(request)
        return _ok(data=_dataset_service.list_datasets(user_id=user_id), message="datasets fetched")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/search")
def search_datasets(request: Request, body: DatasetQueryRequest):
    try:
        user_id = assert_login(request)
        result = _dataset_service.search_datasets(user_id=user_id, filters=body.model_dump())
        return _ok(
            data=result.get("items", []),
            message="datasets fetched",
            meta={
                "total_count": result.get("total_count", 0),
                "filtered_count": result.get("filtered_count", 0),
                "importing_count": result.get("importing_count", 0),
                "generated_count": result.get("generated_count", 0),
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("")
def create_dataset(request: Request, body: DatasetCreateRequest):
    try:
        user_id = assert_login(request)
        data = _dataset_service.create_dataset(user_id=user_id, payload=body.model_dump())
        return _ok(data=data, message="dataset created")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/upload")
async def upload_dataset(
    request: Request,
    file: UploadFile | None = File(default=None),
    files: list[UploadFile] | None = File(default=None),
    cover: UploadFile | None = File(default=None),
    name: str = Form(""),
    type: str = Form("instruction"),
    language: str = Form("multi"),
    source: str | None = Form(default=None),
    note: str | None = Form(default=None),
):
    try:
        user_id = assert_login(request)
        data = await _dataset_service.upload_dataset(
            user_id=user_id,
            file=file,
            files=files,
            cover=cover,
            name=name,
            dataset_type=type,
            language=language,
            source=source,
            note=note,
        )
        return _ok(data=data, message="dataset uploaded")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/import/huggingface")
def import_huggingface_dataset(request: Request, body: HuggingFaceDatasetImportRequest):
    try:
        user_id = assert_login(request)
        data = _dataset_service.import_huggingface_dataset(
            user_id=user_id,
            repo_id=body.repo_id,
            revision=body.revision,
            name=body.name,
            note=body.note,
        )
        return _ok(data=data, message="huggingface dataset import started")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/{dataset_id}")
def get_dataset(request: Request, dataset_id: int):
    try:
        user_id = assert_login(request)
        data = _dataset_service.get_dataset(user_id=user_id, dataset_id=dataset_id)
        if data is None:
            raise HTTPException(status_code=404, detail=f"dataset not found: dataset_id={dataset_id}")
        return _ok(data=data, message="dataset fetched")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/{dataset_id}/readme")
def get_dataset_readme(request: Request, dataset_id: int):
    try:
        user_id = assert_login(request)
        data = _dataset_service.get_dataset_readme(user_id=user_id, dataset_id=dataset_id)
        if data is None:
            raise HTTPException(status_code=404, detail=f"dataset not found: dataset_id={dataset_id}")
        return _ok(data=data, message="dataset README fetched")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/{dataset_id}/files")
def get_dataset_files(request: Request, dataset_id: int):
    try:
        user_id = assert_login(request)
        data = _dataset_service.get_dataset_files(user_id=user_id, dataset_id=dataset_id)
        if data is None:
            raise HTTPException(status_code=404, detail=f"dataset not found: dataset_id={dataset_id}")
        return _ok(data=data, message="dataset files fetched")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/{dataset_id}/preview")
def get_dataset_preview(
    request: Request,
    dataset_id: int,
    path: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=200),
):
    try:
        user_id = assert_login(request)
        data = _dataset_service.get_dataset_preview(user_id=user_id, dataset_id=dataset_id, path=path, limit=limit)
        return _ok(data=data, message="dataset preview fetched")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/{dataset_id}/query")
def query_dataset_sql(request: Request, dataset_id: int, body: DatasetSqlQueryRequest):
    try:
        user_id = assert_login(request)
        data = _dataset_service.query_dataset_sql(
            user_id=user_id,
            dataset_id=dataset_id,
            path=body.path,
            sql=body.sql,
            limit=body.limit,
        )
        return _ok(data=data, message="dataset query executed")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/{dataset_id}/cover")
def get_dataset_cover(request: Request, dataset_id: int):
    try:
        user_id = assert_login(request)
        cover_path = _dataset_service.get_cover_path(user_id=user_id, dataset_id=dataset_id)
        if not cover_path:
            raise HTTPException(status_code=404, detail=f"cover not found: dataset_id={dataset_id}")
        path = Path(cover_path)
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail=f"cover file not found: dataset_id={dataset_id}")
        return FileResponse(path)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.put("/{dataset_id}")
def update_dataset(request: Request, dataset_id: int, body: DatasetUpdateRequest):
    try:
        user_id = assert_login(request)
        data = _dataset_service.update_dataset(user_id=user_id, dataset_id=dataset_id, payload=body.model_dump(exclude_unset=True))
        if data is None:
            raise HTTPException(status_code=404, detail=f"dataset not found: dataset_id={dataset_id}")
        return _ok(data=data, message="dataset updated")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.put("/{dataset_id}/cover")
async def update_dataset_cover(request: Request, dataset_id: int, cover: UploadFile = File(...)):
    try:
        user_id = assert_login(request)
        data = await _dataset_service.update_cover(user_id=user_id, dataset_id=dataset_id, cover=cover)
        if data is None:
            raise HTTPException(status_code=404, detail=f"dataset not found: dataset_id={dataset_id}")
        return _ok(data=data, message="dataset cover updated")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/{dataset_id}")
def delete_dataset(request: Request, dataset_id: int):
    try:
        user_id = assert_login(request)
        ok = _dataset_service.delete_dataset(user_id=user_id, dataset_id=dataset_id)
        if not ok:
            raise HTTPException(status_code=404, detail=f"dataset not found: dataset_id={dataset_id}")
        return _ok(message="dataset deleted")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
