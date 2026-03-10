from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..entity.request import AgentChatRequest, AgentReviseRequest
from ..service import AgentReportService


router = APIRouter(prefix="/agent", tags=["agent"])

_agent_service = AgentReportService()


def _ok(data=None, message: str = "ok"):
    return {
        "success": True,
        "message": message,
        "data": data,
    }


@router.get("/models")
def list_models():
    try:
        return _ok(data={"models": _agent_service.list_models()}, message="agent models fetched")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/report")
async def generate_report(
    file: UploadFile = File(...),
    model: str = Form(default=""),
    prompt: str = Form(default=""),
    llm_provider: str = Form(default=""),
    llm_endpoint: str = Form(default=""),
    llm_api_key: str = Form(default=""),
    llm_model_name: str = Form(default=""),
):
    try:
        data = await _agent_service.analyze_upload(
            file=file,
            model=model,
            prompt=prompt,
            llm_provider=llm_provider,
            llm_endpoint=llm_endpoint,
            llm_api_key=llm_api_key,
            llm_model_name=llm_model_name,
        )
        return _ok(data=data, message="report generated")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/report/revise")
def revise_report(body: AgentReviseRequest):
    try:
        data = _agent_service.revise_report(
            session_id=body.session_id,
            prompt=body.prompt,
            model=body.model,
            llm_provider=body.llm_provider,
            llm_endpoint=body.llm_endpoint,
            llm_api_key=body.llm_api_key,
            llm_model_name=body.llm_model_name,
        )
        return _ok(data=data, message="report revised")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/chat")
def chat(body: AgentChatRequest):
    try:
        data = _agent_service.chat(
            message=body.message,
            model=body.model,
            session_id=body.session_id,
            report=body.report,
            llm_provider=body.llm_provider,
            llm_endpoint=body.llm_endpoint,
            llm_api_key=body.llm_api_key,
            llm_model_name=body.llm_model_name,
        )
        return _ok(data=data, message="chat response")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
