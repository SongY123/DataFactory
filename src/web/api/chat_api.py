from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from contextlib import suppress
from typing import Any, AsyncGenerator

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from agents.context import set_event_bus, set_python_interpreter, set_workspace
from agents.event_bus import EventBus
from utils.auth_guard import get_login_user
from utils.config_loader import get_config
from utils.logger import logger
from utils.model_factory import reset_model_override, set_model_override
from web.entity.request import AssetImportRequest, ChatRequest
from web.service.agent_asset_service import AgentAssetService
from web.service.sandbox_environment_service import SandboxEnvironmentService

router = APIRouter(tags=["agent-interaction"])
_asset_service = AgentAssetService()
_sandbox_environment_service = SandboxEnvironmentService()
_WEB_ROOT = Path(__file__).resolve().parents[1]
_BACKEND_ROOT = Path(__file__).resolve().parents[3]
_ARTIFACT_FALLBACK_BASES = (_WEB_ROOT, _BACKEND_ROOT, Path.cwd())


class FolderCreateBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    parent_path: str = Field(default="")


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _resolve_under_base(base_dir: Path, relative_path: str) -> Path | None:
    base_resolved = base_dir.resolve()
    target = (base_resolved / relative_path).resolve()
    if target != base_resolved and base_resolved not in target.parents:
        return None
    if not target.exists() or not target.is_file():
        return None
    return target


def _resolve_chat_artifact_file(user_id: int, workspace_name: str | None, artifact_path: str) -> Path:
    normalized = _asset_service.normalize_artifact_path(artifact_path)

    if workspace_name:
        try:
            runtime_target = _asset_service.resolve_runtime_artifact(user_id, workspace_name, normalized)
            if runtime_target.exists() and runtime_target.is_file():
                return runtime_target
        except ValueError:
            pass

    checked_bases: set[str] = set()
    for base_dir in _ARTIFACT_FALLBACK_BASES:
        base_key = str(base_dir.resolve())
        if base_key in checked_bases:
            continue
        checked_bases.add(base_key)
        candidate = _resolve_under_base(base_dir, normalized)
        if candidate is not None:
            return candidate

    raise ValueError(f"Artifact file not found: {normalized}")


def _build_effective_query(query: str, selected_file_path: str | None, workspace_dir) -> tuple[str, str | None]:
    raw_query = str(query or "").strip()
    if not selected_file_path:
        return raw_query, None

    normalized_path = _asset_service.normalize_asset_path(selected_file_path, allow_empty=False)
    resolved_path = _asset_service.resolve_file_under(workspace_dir, normalized_path)
    resolved_path_str = resolved_path.as_posix()

    instruction = (
        "User has explicitly selected a file for this question. The file already exists in the workspace.\n"
        f"- Selected file relative path: {normalized_path}\n"
        f"- Selected file absolute path: {resolved_path_str}\n"
        f"- Call create_data_analyst_worker directly with file_paths=['{resolved_path_str}']\n"
        "- Do not ask the user to provide or confirm file paths.\n"
        "- Unless the user explicitly asks for cross-file analysis, do not expand to other files.\n\n"
        "User question:\n"
        f"{raw_query}"
    )
    return instruction, normalized_path


def _resolve_selected_file_paths(
    selected_file_path: str | None,
    selected_file_paths: list[str] | None,
    workspace_dir,
) -> list[str]:
    raw_paths: list[str] = []
    if str(selected_file_path or "").strip():
        raw_paths.append(str(selected_file_path))

    if isinstance(selected_file_paths, (list, tuple, set)):
        raw_paths.extend(str(item) for item in selected_file_paths)
    elif isinstance(selected_file_paths, str) and selected_file_paths.strip():
        raw_paths.append(selected_file_paths)

    normalized_paths: list[str] = []
    seen: set[str] = set()
    for raw_path in raw_paths:
        normalized = _asset_service.normalize_asset_path(raw_path, allow_empty=False)
        _asset_service.resolve_file_under(workspace_dir, normalized)
        if normalized in seen:
            continue
        seen.add(normalized)
        normalized_paths.append(normalized)

    return normalized_paths


def _resolve_effective_provider(selected_model_override: dict[str, Any] | None) -> str:
    if selected_model_override and selected_model_override.get("provider"):
        return str(selected_model_override["provider"]).strip().lower()
    return str(get_config("model.provider", "ollama") or "ollama").strip().lower()


def _normalize_ollama_endpoint(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    normalized = text.rstrip("/")
    if normalized.lower().endswith("/v1"):
        normalized = normalized[:-3].rstrip("/")
    return normalized


def _build_model_debug_info(selected_model_override: dict[str, Any] | None) -> dict[str, str]:
    provider = _resolve_effective_provider(selected_model_override)
    model_name = ""
    endpoint = "server default"

    if selected_model_override:
        model_name = str(selected_model_override.get("model_name") or "").strip()
        if provider == "ollama":
            endpoint = _normalize_ollama_endpoint(str(selected_model_override.get("host") or get_config("model.ollama.host", "server default") or "server default").strip())
        elif provider == "dashscope":
            endpoint = str(selected_model_override.get("base_url") or get_config("model.dashscope.base_url", "DashScope default endpoint") or "DashScope default endpoint").strip()
        else:
            endpoint = str(selected_model_override.get("base_url") or get_config("model.openai.base_url", "OpenAI default endpoint") or "OpenAI default endpoint").strip()
    else:
        if provider == "ollama":
            model_name = str(get_config("model.ollama.model_name", "") or "").strip()
            endpoint = _normalize_ollama_endpoint(str(get_config("model.ollama.host", "server default") or "server default").strip())
        elif provider == "dashscope":
            model_name = str(get_config("model.dashscope.model_name", "") or "").strip()
            endpoint = str(get_config("model.dashscope.base_url", "DashScope default endpoint") or "DashScope default endpoint").strip()
        else:
            model_name = str(get_config("model.openai.model_name", "") or "").strip()
            endpoint = str(get_config("model.openai.base_url", "OpenAI default endpoint") or "OpenAI default endpoint").strip()

    return {
        "provider": provider or "unknown",
        "model_name": model_name or "unknown",
        "endpoint": endpoint or "unknown",
    }


def _format_model_execution_error(exc: Exception, model_debug: dict[str, str]) -> str:
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)

    provider = model_debug.get("provider", "unknown")
    model_name = model_debug.get("model_name", "unknown")
    endpoint = model_debug.get("endpoint", "unknown")

    if status_code == 502:
        return (
            f"Upstream model service returned 502. provider={provider}, model={model_name}, endpoint={endpoint}. "
            "Check whether the selected API/base URL is reachable and the model name is valid, or switch the model selector back to Server Default."
        )

    if status_code:
        return (
            f"Model request failed with status {status_code}. provider={provider}, model={model_name}, endpoint={endpoint}. "
            f"Original error: {exc}"
        )

    return str(exc or "Model request failed")


def _extract_tool_response_text(response: Any) -> str:
    content = getattr(response, "content", None)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = getattr(item, "text", None)
            if isinstance(text, str) and text.strip():
                parts.append(text)
                continue
            if isinstance(item, dict):
                dict_text = item.get("text")
                if isinstance(dict_text, str) and dict_text.strip():
                    parts.append(dict_text)
        if parts:
            return "\n\n".join(parts)

    if isinstance(response, str):
        return response

    return str(response or "")


def _build_selected_file_analysis_task(query: str, selected_file_path: str) -> str:
    requested_focus = str(query or "").strip() or "Please perform a comprehensive analysis of the selected dataset."
    return (
        "The user has selected a specific file. Analyze that file directly and do not ask for a file path.\n"
        f"Selected file: {selected_file_path}\n\n"
        "Produce a rich analysis rather than a minimal acknowledgement. Your analysis must cover:\n"
        "1. Dataset structure, schema, and field meanings\n"
        "2. Data quality checks such as missing values, duplicates, invalid values, and suspicious patterns\n"
        "3. Key distributions, descriptive statistics, and notable segments\n"
        "4. Important anomalies, risks, and limitations\n"
        "5. Concrete findings that answer the user's request\n"
        "6. Practical next-step suggestions when appropriate\n\n"
        "Start by reading the selected file, then continue the analysis until you have meaningful findings.\n"
        "If the user's request is broad, still provide a substantive exploratory analysis instead of a short acknowledgement.\n\n"
        "User request:\n"
        f"{requested_focus}"
    )


def _is_meaningful_analysis_output(text: str, original_query: str) -> bool:
    normalized = str(text or "").strip()
    if not normalized:
        return False
    if normalized.lower().startswith("error:"):
        return False

    empty_markers = (
        "当前目录下未找到任何数据文件",
        "请提供明确的数据文件路径",
        "需要先读取这个数据文件来了解其结构和内容",
    )
    if any(marker in normalized for marker in empty_markers):
        return False

    query = str(original_query or "").strip()
    if query and normalized == query:
        return False
    if query and len(normalized) < max(24, len(query) + 8) and query in normalized:
        return False
    if len(normalized) < 16 and "\n" not in normalized:
        return False

    return True


def _append_analysis_section(sections: list[str], title: str, content: str, original_query: str) -> bool:
    cleaned = str(content or "").strip()
    if not _is_meaningful_analysis_output(cleaned, original_query):
        return False
    sections.append(f"## {title}\n{cleaned}")
    return True


def _should_run_visualization(query: str) -> bool:
    lowered = str(query or "").lower()
    keywords = (
        "chart",
        "plot",
        "graph",
        "visual",
        "visualize",
        "visualise",
        "dashboard",
        "figure",
        "图",
        "可视化",
        "绘图",
        "画图",
        "图表",
        "趋势图",
        "柱状图",
        "折线图",
        "散点图",
        "饼图",
    )
    return any(keyword in lowered for keyword in keywords)


@router.get("/assets/tree")
def get_asset_tree(request: Request):
    user = get_login_user(request)
    data = _asset_service.list_asset_tree(int(user["id"]))
    return {
        "ok": True,
        **data,
        "user": {
            "id": int(user["id"]),
            "username": str(user.get("username") or ""),
        },
    }


@router.post("/assets/import")
def import_platform_asset(body: AssetImportRequest, request: Request):
    user = get_login_user(request)
    try:
        payload = _asset_service.import_platform_object(
            int(user["id"]),
            body.source_type,
            int(body.source_id),
            body.target_folder_path,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "ok": True,
        "item": payload,
        "message": "platform object imported into assets",
    }


@router.post("/folders")
def create_folder(body: FolderCreateBody, request: Request):
    user = get_login_user(request)
    try:
        folder = _asset_service.create_folder(int(user["id"]), body.name, body.parent_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "ok": True,
        "folder": folder,
        "message": "folder created",
    }


@router.delete("/folders")
def delete_folder(request: Request, path: str = Query(...), force: bool = Query(False)):
    user = get_login_user(request)
    try:
        _asset_service.delete_folder(int(user["id"]), path, force=force)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "ok": True,
        "message": "folder deleted",
        "path": path,
    }


@router.delete("/files")
def delete_file(request: Request, path: str = Query(...)):
    user = get_login_user(request)
    try:
        _asset_service.delete_file(int(user["id"]), path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "ok": True,
        "message": "file deleted",
        "path": path,
    }


@router.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...), folder_path: str = Form("")):
    user = get_login_user(request)
    try:
        payload = await _asset_service.upload_file(int(user["id"]), file, folder_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "ok": True,
        **payload,
        "message": "file uploaded",
    }


@router.get("/chat/artifact")
def get_chat_artifact(request: Request, path: str = Query(...), workspace: str | None = Query(default=None)):
    user = get_login_user(request)
    try:
        artifact = _resolve_chat_artifact_file(int(user["id"]), workspace, path)
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if 'not found' in message.lower() else 400
        raise HTTPException(status_code=status_code, detail=message) from exc

    return FileResponse(artifact)


@router.post("/chat")
async def chat(req: ChatRequest, request: Request):
    user = get_login_user(request)
    user_id = int(user["id"])
    username = str(user.get("username") or "")
    context_items = [item.model_dump(exclude_none=True) for item in (req.context_items or [])]

    if not _asset_service.has_files(user_id) and not context_items:
        raise HTTPException(status_code=400, detail="Please upload at least one context file first.")

    try:
        runtime = _asset_service.prepare_runtime_workspace(user_id, req.workspace or f"agent-ws-{uuid.uuid4().hex[:12]}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        sandbox_environment = _sandbox_environment_service.resolve_python_executable(req.sandbox_environment_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    workspace_name = str(runtime["workspace_name"])
    workspace_dir = runtime["workspace_dir"]

    try:
        staged_context_items = _asset_service.stage_context_items(user_id, workspace_dir, context_items)
        selected_file_paths = _resolve_selected_file_paths(
            req.selected_file_path,
            req.selected_file_paths,
            workspace_dir,
        )
        effective_query, _ = _build_effective_query(req.query, None, workspace_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    primary_selected_file_path = selected_file_paths[0] if selected_file_paths else None

    try:
        from agents.orchestrator import create_orchestrator
        from agents.iterative_file_analysis import run_iterative_file_analysis
        from agentscope.message import Msg
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Agent interaction dependencies are missing. "
                f"Install backend requirements for AgentInteraction first: {exc}"
            ),
        ) from exc

    event_bus = EventBus(maxsize=1000)
    selected_model_override = req.selected_model.model_dump(exclude_none=True) if req.selected_model else None

    session_id = req.request_id or f"session_{uuid.uuid4().hex}"

    model_debug = _build_model_debug_info(selected_model_override)

    logger.info(
        "AgentInteraction chat request received. user_id=%s username=%s session_id=%s workspace=%s selected_files=%s provider=%s model=%s endpoint=%s sandbox=%s",
        user_id,
        username,
        session_id,
        workspace_dir,
        ",".join(selected_file_paths),
        model_debug["provider"],
        model_debug["model_name"],
        model_debug["endpoint"],
        sandbox_environment.get("python_path"),
    )

    async def event_stream() -> AsyncGenerator[str, None]:
        runner_task: asyncio.Task | None = None
        subscriber = event_bus.subscribe()
        model_override_token = None

        set_event_bus(event_bus)
        set_workspace(str(workspace_dir))
        set_python_interpreter(str(sandbox_environment.get("python_path") or ""))
        model_override_token = set_model_override(selected_model_override)

        try:
            yield _sse(
                "opened",
                {
                    "session_id": session_id,
                    "user_id": user_id,
                    "username": username,
                    "workspace": workspace_name,
                    "selected_file_path": primary_selected_file_path,
                    "selected_file_paths": selected_file_paths,
                    "context_items": staged_context_items,
                    "model": selected_model_override,
                    "sandbox_environment": sandbox_environment,
                },
            )

            if selected_file_paths:
                async def run_selected_file_analysis():
                    total_files = len(selected_file_paths)
                    completed_files = 0
                    last_error: Exception | None = None

                    try:
                        for index, current_file_path in enumerate(selected_file_paths, start=1):
                            try:
                                final_summary = str(
                                    await run_iterative_file_analysis(
                                        req.query,
                                        current_file_path,
                                        workspace_dir,
                                        file_index=index,
                                        total_files=total_files,
                                    )
                                    or ""
                                ).strip()
                                if final_summary:
                                    completed_files += 1
                            except asyncio.CancelledError:
                                logger.warning(
                                    "AgentInteraction selected-file analysis cancelled during file run. session_id=%s file=%s",
                                    session_id,
                                    current_file_path,
                                )
                                raise
                            except Exception as exc:
                                last_error = exc
                                logger.error(
                                    "Iterative selected-file analysis failed. session_id=%s file=%s error=%s",
                                    session_id,
                                    current_file_path,
                                    exc,
                                    exc_info=True,
                                )
                                if total_files == 1:
                                    raise
                    except asyncio.CancelledError:
                        logger.warning("AgentInteraction selected-file analysis cancelled. session_id=%s", session_id)
                        raise
                    if completed_files == 0 and last_error is not None:
                        raise last_error

                runner_task = asyncio.create_task(run_selected_file_analysis())
            else:
                orchestrator = create_orchestrator()
                user_msg = Msg("user", effective_query, "user")

                async def run_orchestrator():
                    try:
                        await orchestrator(user_msg)
                    except asyncio.CancelledError:
                        logger.warning("AgentInteraction orchestrator cancelled. session_id=%s", session_id)
                        raise

                runner_task = asyncio.create_task(run_orchestrator())

            next_event_task: asyncio.Task | None = asyncio.create_task(anext(subscriber))

            while True:
                if await request.is_disconnected():
                    logger.warning("Client disconnected during AgentInteraction stream. session_id=%s", session_id)
                    if runner_task:
                        runner_task.cancel()
                    break

                if runner_task and runner_task.done() and event_bus._queue.empty():
                    break

                if next_event_task is None:
                    next_event_task = asyncio.create_task(anext(subscriber))

                done, _ = await asyncio.wait({next_event_task}, timeout=0.1)
                if not done:
                    continue

                try:
                    event = next_event_task.result()
                except StopAsyncIteration:
                    next_event_task = None
                    break

                next_event_task = asyncio.create_task(anext(subscriber))

                yield _sse(
                    "delta",
                    {
                        "agent": event.agent_name,
                        "type": event.event_type.value,
                        "content": event.data,
                        "is_final": event.is_final,
                        "timestamp": event.timestamp.isoformat(),
                        "metadata": event.metadata,
                    },
                )

            if runner_task:
                await runner_task
            yield _sse("done", {"ok": True, "session_id": session_id})
        except asyncio.CancelledError:
            yield _sse("done", {"ok": False, "reason": "cancelled", "session_id": session_id})
            raise
        except Exception as exc:
            error_message = _format_model_execution_error(exc, model_debug)
            logger.error(
                "AgentInteraction execution failed. session_id=%s provider=%s model=%s endpoint=%s error=%s",
                session_id,
                model_debug["provider"],
                model_debug["model_name"],
                model_debug["endpoint"],
                exc,
                exc_info=True,
            )
            yield _sse("error", {"message": error_message})
            yield _sse("done", {"ok": False, "session_id": session_id})
        finally:
            next_event_task = locals().get("next_event_task")
            if next_event_task is not None and not next_event_task.done():
                next_event_task.cancel()
                with suppress(asyncio.CancelledError, StopAsyncIteration):
                    await next_event_task
            await subscriber.aclose()
            await event_bus.close()
            set_event_bus(None)
            set_workspace("")
            set_python_interpreter("")
            if model_override_token is not None:
                reset_model_override(model_override_token)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
