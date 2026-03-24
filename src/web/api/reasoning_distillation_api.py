from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from utils.auth_guard import assert_login
from ..entity.request import ReasoningDistillationStartRequest
from ..service import ReasoningDistillationService


router = APIRouter(prefix="/reasoning-distillation", tags=["reasoning-distillation"])

_service = ReasoningDistillationService()


def _ok(data=None, message: str = "ok"):
    return {
        "success": True,
        "message": message,
        "data": data,
    }


@router.post("/tasks")
def start_reasoning_distillation_task(request: Request, body: ReasoningDistillationStartRequest):
    try:
        user_id = assert_login(request)
        data = _service.start_task(
            user_id=user_id,
            source_type=body.source_type,
            source_dataset_id=body.source_dataset_id,
            source_task_id=body.source_task_id,
            selected_file_paths=body.selected_file_paths,
            file_mappings=[item.model_dump() for item in body.file_mappings],
            prompt_field=body.prompt_field,
            completion_field=body.completion_field,
            prompt=body.prompt,
            strategy=body.strategy,
            target_max_tokens=body.target_max_tokens,
            compression_ratio=body.compression_ratio,
            keep_tool_trace=body.keep_tool_trace,
            note=body.note,
            llm_api_key=body.llm_api_key,
            llm_base_url=body.llm_base_url,
            llm_model_name=body.llm_model_name,
            parallelism=body.parallelism,
            save_path=body.save_path,
            save_path_key=body.save_path_key,
            llm_params_json=body.llm_params_json,
        )
        return _ok(data=data, message="reasoning distillation task started")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/output-path-options")
def list_reasoning_distillation_output_path_options(request: Request):
    try:
        assert_login(request)
        data = _service.list_output_path_options(task_namespace="reasoning_distillation")
        return _ok(data=data, message="reasoning distillation output path options fetched")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/tasks")
def list_reasoning_distillation_tasks(request: Request, limit: int = Query(default=20, ge=1, le=200)):
    try:
        user_id = assert_login(request)
        data = _service.list_tasks(user_id=user_id, limit=limit)
        return _ok(data=data, message="reasoning distillation tasks fetched")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/tasks/{task_id}")
def get_reasoning_distillation_task(request: Request, task_id: int):
    try:
        user_id = assert_login(request)
        data = _service.get_task(user_id=user_id, task_id=task_id)
        if data is None:
            raise HTTPException(status_code=404, detail=f"task not found: task_id={task_id}")
        return _ok(data=data, message="reasoning distillation task fetched")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/tasks/{task_id}/results")
def list_reasoning_distillation_results(request: Request, task_id: int, limit: int = Query(default=200, ge=1, le=1000)):
    try:
        user_id = assert_login(request)
        data = _service.list_results(user_id=user_id, task_id=task_id, limit=limit)
        return _ok(data=data, message="reasoning distillation results fetched")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/results/{result_id}")
def get_reasoning_distillation_result(request: Request, result_id: int):
    try:
        user_id = assert_login(request)
        data = _service.get_result(user_id=user_id, result_id=result_id)
        if data is None:
            raise HTTPException(status_code=404, detail=f"result not found: result_id={result_id}")
        return _ok(data=data, message="reasoning distillation result fetched")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
