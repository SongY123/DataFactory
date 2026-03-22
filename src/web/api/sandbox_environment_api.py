from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from utils.auth_guard import assert_login
from ..service.sandbox_environment_service import SandboxEnvironmentService


router = APIRouter(prefix="/sandbox-environments", tags=["sandbox-environments"])

_service = SandboxEnvironmentService()


class SandboxEnvironmentCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    python_path: str = Field(..., min_length=1, max_length=1000)


def _ok(data=None, message: str = "ok"):
    return {
        "success": True,
        "message": message,
        "data": data,
    }


@router.get("")
def list_sandbox_environments(request: Request):
    try:
        assert_login(request)
        return _ok(data=_service.list_environments(), message="sandbox environments fetched")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("")
def create_sandbox_environment(request: Request, body: SandboxEnvironmentCreateRequest):
    try:
        assert_login(request)
        item = _service.create_environment(name=body.name, python_path=body.python_path)
        payload = _service.list_environments()
        payload["selected"] = item
        return _ok(data=payload, message="sandbox environment created")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/{environment_id}")
def delete_sandbox_environment(request: Request, environment_id: str):
    try:
        assert_login(request)
        data = _service.delete_environment(environment_id)
        return _ok(data=data, message="sandbox environment deleted")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
