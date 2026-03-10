from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from ..entity.request import DatasetCreateRequest, DatasetUpdateRequest
from ..service import DatasetService


router = APIRouter(prefix="/datasets", tags=["datasets"])

_dataset_service = DatasetService()


def _ok(data=None, message: str = "ok"):
    return {
        "success": True,
        "message": message,
        "data": data,
    }


@router.get("")
def list_datasets():
    try:
        return _ok(data=_dataset_service.list_datasets(), message="datasets fetched")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/{dataset_id}")
def get_dataset(dataset_id: int):
    try:
        data = _dataset_service.get_dataset(dataset_id=dataset_id)
        if data is None:
            raise HTTPException(status_code=404, detail=f"dataset not found: dataset_id={dataset_id}")
        return _ok(data=data, message="dataset fetched")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/{dataset_id}/cover")
def get_dataset_cover(dataset_id: int):
    try:
        cover_path = _dataset_service.get_cover_path(dataset_id=dataset_id)
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


@router.post("")
def create_dataset(body: DatasetCreateRequest):
    try:
        data = _dataset_service.create_dataset(body.model_dump())
        return _ok(data=data, message="dataset created")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.put("/{dataset_id}")
def update_dataset(dataset_id: int, body: DatasetUpdateRequest):
    try:
        data = _dataset_service.update_dataset(dataset_id=dataset_id, payload=body.model_dump(exclude_unset=True))
        if data is None:
            raise HTTPException(status_code=404, detail=f"dataset not found: dataset_id={dataset_id}")
        return _ok(data=data, message="dataset updated")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.put("/{dataset_id}/cover")
async def update_dataset_cover(dataset_id: int, cover: UploadFile = File(...)):
    try:
        data = await _dataset_service.update_cover(dataset_id=dataset_id, cover=cover)
        if data is None:
            raise HTTPException(status_code=404, detail=f"dataset not found: dataset_id={dataset_id}")
        return _ok(data=data, message="dataset cover updated")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: int):
    try:
        ok = _dataset_service.delete_dataset(dataset_id=dataset_id)
        if not ok:
            raise HTTPException(status_code=404, detail=f"dataset not found: dataset_id={dataset_id}")
        return _ok(message="dataset deleted")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/upload")
async def upload_dataset(
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
        data = await _dataset_service.upload_dataset(
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
