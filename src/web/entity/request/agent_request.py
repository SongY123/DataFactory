from __future__ import annotations

from pydantic import BaseModel, Field


class AgentReviseRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    prompt: str = Field(min_length=1, max_length=8000)
    model: str | None = Field(default=None, max_length=128)
    llm_provider: str | None = Field(default=None, max_length=16)
    llm_endpoint: str | None = Field(default=None, max_length=1024)
    llm_api_key: str | None = Field(default=None, max_length=2048)
    llm_model_name: str | None = Field(default=None, max_length=128)


class AgentChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    history: list[dict] | None = None
    session_id: str | None = Field(default=None, max_length=128)
    report: str | None = None
    model: str | None = Field(default=None, max_length=128)
    llm_provider: str | None = Field(default=None, max_length=16)
    llm_endpoint: str | None = Field(default=None, max_length=1024)
    llm_api_key: str | None = Field(default=None, max_length=2048)
    llm_model_name: str | None = Field(default=None, max_length=128)
