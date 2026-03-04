from __future__ import annotations

from threading import RLock
from typing import Any, Dict, Optional

from fastapi import HTTPException, Request

from utils.config_loader import get_config


SESSION_COOKIE_KEY = "datafactory_session_id"
_AUTO_LOGIN_LOCK = RLock()
_AUTO_LOGIN_SESSION_ID: Optional[str] = None


def _get_session_service():
    # Delay import to avoid circular imports caused by package-level side effects.
    from web.service.session_service import get_global_session_service

    return get_global_session_service()


def _is_auto_login_as_admin_enabled() -> bool:
    try:
        raw = get_config("auth.auto_login_as_admin", False)
    except Exception:
        return False

    if isinstance(raw, bool):
        return raw
    if raw is None:
        return False
    if isinstance(raw, (int, float)):
        return bool(raw)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _normalize_login_user(payload: Dict[str, Any], fallback_session_id: str) -> Dict[str, Any]:
    user = payload.get("user") or {}
    user_id = user.get("id")
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid session.")

    role = str(user.get("role") or "user").strip().lower()
    if role not in {"user", "admin"}:
        role = "user"
    status = str(user.get("status") or "active").strip().lower()
    if status not in {"active", "disabled"}:
        status = "active"
    if status != "active":
        raise HTTPException(status_code=403, detail="User is disabled.")

    return {
        "id": int(user_id),
        "username": str(user.get("username") or ""),
        "role": role,
        "status": status,
        "session_id": str(payload.get("session_id") or fallback_session_id),
    }


def _resolve_auto_login_payload() -> Dict[str, Any]:
    global _AUTO_LOGIN_SESSION_ID
    session_service = _get_session_service()

    with _AUTO_LOGIN_LOCK:
        if _AUTO_LOGIN_SESSION_ID:
            payload = session_service.get_session(_AUTO_LOGIN_SESSION_ID)
            if payload:
                return payload
            _AUTO_LOGIN_SESSION_ID = None

        # Delay import to avoid unnecessary dependency initialization at module import time.
        from web.service.user_service import UserService

        login_data = UserService().login(username="admin", password="admin")
        session_id = str(login_data.get("session_id") or "").strip()
        if not session_id:
            raise RuntimeError("auto login failed to create a session id")

        _AUTO_LOGIN_SESSION_ID = session_id
        payload = session_service.get_session(session_id)
        if not payload:
            raise RuntimeError("auto login session was not found after creation")
        return payload


def _get_auto_login_user() -> Dict[str, Any]:
    try:
        payload = _resolve_auto_login_payload()
        sid = str(payload.get("session_id") or "")
        return _normalize_login_user(payload, sid)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Auto login as admin failed: {exc}",
        )


def get_login_user(request: Request) -> Dict[str, Any]:
    session_id = str(request.cookies.get(SESSION_COOKIE_KEY) or "").strip()
    session_service = _get_session_service()
    if session_id:
        payload = session_service.get_session(session_id=session_id)
        if payload:
            return _normalize_login_user(payload, session_id)

    if _is_auto_login_as_admin_enabled():
        return _get_auto_login_user()

    raise HTTPException(status_code=401, detail="Not logged in.")


def assert_login(request: Request) -> int:
    return int(get_login_user(request).get("id"))


def assert_admin_user(request: Request) -> Dict[str, Any]:
    user = get_login_user(request)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin role required.")
    return user
