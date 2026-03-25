from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from utils.auth_guard import assert_login
from ..entity.request import WorkflowAssistantChatRequest
from ..service.workflow_assistant_service import WorkflowAssistantService


router = APIRouter(prefix="/workflow-assistant", tags=["workflow-assistant"])
_service = WorkflowAssistantService()


def _ok(data=None, message: str = "ok"):
    return {
        "success": True,
        "message": message,
        "data": data,
    }


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat")
def workflow_assistant_chat(request: Request, body: WorkflowAssistantChatRequest):
    try:
        user_id = assert_login(request)
        data = _service.chat(
            page_key=body.page_key,
            session_id=body.session_id,
            messages=[item.model_dump() for item in body.messages],
            page_context=body.page_context or {},
            user_id=user_id,
        )
        return _ok(data=data, message="workflow assistant response")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/chat/stream")
def workflow_assistant_chat_stream(request: Request, body: WorkflowAssistantChatRequest):
    user_id = assert_login(request)

    def event_stream():
        try:
            for event in _service.stream_chat(
                page_key=body.page_key,
                session_id=body.session_id,
                messages=[item.model_dump() for item in body.messages],
                page_context=body.page_context or {},
                user_id=user_id,
            ):
                yield _sse(str(event.get("event") or "message"), event.get("data") or {})
        except Exception as exc:
            yield _sse("error", {"message": str(exc)})
            yield _sse("done", {"ok": False})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
