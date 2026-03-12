from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import suppress
from typing import Any, AsyncGenerator

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agents.context import set_event_bus, set_workspace
from agents.event_bus import EventBus, create_agent_finish_event
from utils.auth_guard import get_login_user
from utils.config_loader import get_config
from utils.logger import logger
from utils.model_factory import reset_model_override, set_model_override
from web.entity.request import ChatRequest
from web.service.agent_asset_service import AgentAssetService

router = APIRouter(tags=["agent-interaction"])
_asset_service = AgentAssetService()


class FolderCreateBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    parent_path: str = Field(default="")


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


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


@router.post("/chat")
async def chat(req: ChatRequest, request: Request):
    user = get_login_user(request)
    user_id = int(user["id"])
    username = str(user.get("username") or "")

    if not _asset_service.has_files(user_id):
        raise HTTPException(status_code=400, detail="Please upload at least one context file first.")

    try:
        runtime = _asset_service.prepare_runtime_workspace(user_id, req.workspace or f"agent-ws-{uuid.uuid4().hex[:12]}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    workspace_name = str(runtime["workspace_name"])
    workspace_dir = runtime["workspace_dir"]

    try:
        effective_query, selected_file_path = _build_effective_query(req.query, req.selected_file_path, workspace_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        from agents.business_consultant_worker import create_business_consultant_worker
        from agents.data_analyst_worker import create_data_analyst_worker
        from agents.orchestrator import create_orchestrator
        from agents.visualization_worker import create_visualization_worker
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
        "AgentInteraction chat request received. user_id=%s username=%s session_id=%s workspace=%s selected_file=%s provider=%s model=%s endpoint=%s",
        user_id,
        username,
        session_id,
        workspace_dir,
        selected_file_path or "",
        model_debug["provider"],
        model_debug["model_name"],
        model_debug["endpoint"],
    )

    async def event_stream() -> AsyncGenerator[str, None]:
        runner_task: asyncio.Task | None = None
        subscriber = event_bus.subscribe()
        model_override_token = None

        set_event_bus(event_bus)
        set_workspace(str(workspace_dir))
        model_override_token = set_model_override(selected_model_override)

        try:
            yield _sse(
                "opened",
                {
                    "session_id": session_id,
                    "user_id": user_id,
                    "username": username,
                    "workspace": workspace_name,
                    "selected_file_path": selected_file_path,
                    "model": selected_model_override,
                },
            )

            if selected_file_path:
                async def run_selected_file_analysis():
                    try:
                        collected_sections: list[str] = []

                        analyst_response = await create_data_analyst_worker(
                            task_description=_build_selected_file_analysis_task(req.query, selected_file_path),
                            file_paths=[selected_file_path],
                            display_task_description=req.query or f"Analyze {selected_file_path}",
                        )
                        analyst_output = _extract_tool_response_text(analyst_response).strip()
                        has_analyst_output = _append_analysis_section(
                            collected_sections,
                            "Data Analyst",
                            analyst_output,
                            req.query,
                        )

                        if has_analyst_output:
                            consultant_response = await create_business_consultant_worker(
                                task_description=(
                                    "Continue the analysis based on the completed dataset findings.\n"
                                    f"Selected file: {selected_file_path}\n"
                                    "Provide deeper interpretation, business meaning, risks, and practical next actions.\n"
                                    "Build on the actual analysis results instead of repeating boilerplate.\n\n"
                                    "Original user request:\n"
                                    f"{req.query}"
                                ),
                                analysis_data=analyst_output,
                            )
                            consultant_output = _extract_tool_response_text(consultant_response).strip()
                            if not _append_analysis_section(
                                collected_sections,
                                "Business Consultant",
                                consultant_output,
                                req.query,
                            ):
                                logger.info(
                                    "Skipping empty BusinessConsultant output. session_id=%s selected_file=%s",
                                    session_id,
                                    selected_file_path,
                                )

                            if _should_run_visualization(req.query):
                                visualization_response = await create_visualization_worker(
                                    analysis_data=analyst_output,
                                    custom_requirements=req.query,
                                )
                                visualization_output = _extract_tool_response_text(visualization_response).strip()
                                if not _append_analysis_section(
                                    collected_sections,
                                    "Visualization",
                                    visualization_output,
                                    req.query,
                                ):
                                    logger.info(
                                        "Skipping empty Visualization output. session_id=%s selected_file=%s",
                                        session_id,
                                        selected_file_path,
                                    )
                        else:
                            logger.warning(
                                "Selected-file analysis produced no meaningful DataAnalyst output. session_id=%s selected_file=%s",
                                session_id,
                                selected_file_path,
                            )

                        final_summary = "\n\n".join(collected_sections).strip()
                        if final_summary:
                            await event_bus.publish(await create_agent_finish_event(
                                "Orchestrator",
                                result=final_summary,
                            ))
                    except asyncio.CancelledError:
                        logger.warning("AgentInteraction selected-file analysis cancelled. session_id=%s", session_id)
                        raise

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
