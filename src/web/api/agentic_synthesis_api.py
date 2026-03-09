from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from utils.auth_guard import assert_admin_user
from ..entity.request import AgenticSynthesisStartRequest
from ..service import AgenticSynthesisService


router = APIRouter(prefix="/agentic-synthesis", tags=["agentic-synthesis"])

_service = AgenticSynthesisService()


def _ok(data=None, message: str = "ok"):
    return {
        "success": True,
        "message": message,
        "data": data,
    }


@router.post("/tasks")
def start_agentic_synthesis_task(request: Request, body: AgenticSynthesisStartRequest):
    # assert_admin_user(request)
    try:
        data = _service.start_task(
            prompt=body.prompt,
            action_tags=body.action_tags,
            llm_api_key=body.llm_api_key,
            llm_base_url=body.llm_base_url,
            llm_model_name=body.llm_model_name,
            datasets=[x.model_dump() for x in body.datasets],
            output_file_path=body.output_file_path,
        )
        return _ok(data=data, message="agentic synthesis task started")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/tasks")
def list_agentic_synthesis_tasks(request: Request, limit: int = Query(default=20, ge=1, le=200)):
    # assert_admin_user(request)
    try:
        data = _service.list_tasks(limit=limit)
        return _ok(data=data, message="agentic synthesis tasks fetched")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/tasks/{task_id}")
def get_agentic_synthesis_task(request: Request, task_id: int):
    # assert_admin_user(request)
    try:
        data = _service.get_task(task_id=task_id)
        if data is None:
            raise HTTPException(status_code=404, detail=f"task not found: task_id={task_id}")
        return _ok(data=data, message="agentic synthesis task fetched")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
