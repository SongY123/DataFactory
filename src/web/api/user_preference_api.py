from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from utils.auth_guard import assert_login
from ..entity.request import UserPreferenceUpdateRequest
from ..service.user_preference_service import UserPreferenceService


router = APIRouter(prefix="/preferences", tags=["preferences"])

_service = UserPreferenceService()


def _ok(data=None, message: str = "ok"):
    return {
        "success": True,
        "message": message,
        "data": data,
    }


@router.get("/{preference_key}")
def get_user_preference(request: Request, preference_key: str):
    try:
        user_id = assert_login(request)
        data = _service.get_preference(user_id=user_id, preference_key=preference_key)
        return _ok(data=data, message="preference fetched")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.put("/{preference_key}")
def save_user_preference(request: Request, preference_key: str, body: UserPreferenceUpdateRequest):
    try:
        user_id = assert_login(request)
        data = _service.save_preference(user_id=user_id, preference_key=preference_key, value=body.value)
        return _ok(data=data, message="preference saved")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
