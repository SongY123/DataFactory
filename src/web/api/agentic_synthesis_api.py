from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from utils.auth_guard import assert_login
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
    try:
        user_id = assert_login(request)
        data = _service.start_task(
            user_id=user_id,
            dataset_id=body.dataset_id,
            prompt=body.prompt,
            action_tags=body.action_tags,
            llm_api_key=body.llm_api_key,
            llm_base_url=body.llm_base_url,
            llm_model_name=body.llm_model_name,
            parallelism=body.parallelism,
            save_path=body.save_path,
            save_path_key=body.save_path_key,
            sandbox_environment_id=body.sandbox_environment_id,
            llm_params_json=body.llm_params_json,
        )
        return _ok(data=data, message="agentic synthesis task started")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/output-path-options")
def list_agentic_synthesis_output_path_options(request: Request):
    try:
        assert_login(request)
        data = _service.list_output_path_options()
        return _ok(data=data, message="agentic synthesis output path options fetched")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/tasks")
def list_agentic_synthesis_tasks(request: Request, limit: int = Query(default=20, ge=1, le=200)):
    try:
        user_id = assert_login(request)
        data = _service.list_tasks(user_id=user_id, limit=limit)
        return _ok(data=data, message="agentic synthesis tasks fetched")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/tasks/{task_id}")
def get_agentic_synthesis_task(request: Request, task_id: int):
    try:
        user_id = assert_login(request)
        data = _service.get_task(task_id=task_id, user_id=user_id)
        if data is None:
            raise HTTPException(status_code=404, detail=f"task not found: task_id={task_id}")
        return _ok(data=data, message="agentic synthesis task fetched")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/tasks/{task_id}/results")
def list_agentic_synthesis_results(request: Request, task_id: int, limit: int = Query(default=200, ge=1, le=1000)):
    try:
        user_id = assert_login(request)
        data = _service.list_results(user_id=user_id, task_id=task_id, limit=limit)
        return _ok(data=data, message="agentic synthesis results fetched")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/results/{result_id}")
def get_agentic_synthesis_result(request: Request, result_id: int):
    try:
        user_id = assert_login(request)
        data = _service.get_result(user_id=user_id, result_id=result_id)
        if data is None:
            raise HTTPException(status_code=404, detail=f"result not found: result_id={result_id}")
        return _ok(data=data, message="agentic synthesis result fetched")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
