from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatModelConfig(BaseModel):
    name: str | None = Field(default=None, max_length=128, description="Display name for the selected model config")
    mode: Literal["local", "api"] | None = Field(default=None, description="UI-level config mode")
    provider: Literal["ollama", "openai", "dashscope"] | None = Field(default=None, description="Model provider override")
    model_name: str = Field(..., min_length=1, max_length=200, description="Target model name")
    host: str | None = Field(default=None, max_length=500, description="Local model host, used for ollama")
    api_key: str | None = Field(default=None, max_length=500, description="API key for remote model providers")
    base_url: str | None = Field(default=None, max_length=500, description="Optional OpenAI-compatible base URL")
    organization: str | None = Field(default=None, max_length=200, description="Optional OpenAI organization")
    client_type: Literal["openai", "azure"] | None = Field(default=None, description="Optional OpenAI client type")
    enable_thinking: bool | None = Field(default=None, description="Optional model-specific thinking flag")


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8000, description="User prompt")
    request_id: str | None = Field(default=None, max_length=128, description="Conversation/session id")
    workspace: str = Field(default="default", min_length=1, max_length=128, description="Workspace name")
    user_id: str | None = Field(default=None, max_length=128, description="Optional user id")
    selected_file_path: str | None = Field(
        default=None,
        max_length=500,
        description="Optional relative file path explicitly selected by the user",
    )
    selected_model: ChatModelConfig | None = Field(
        default=None,
        alias="model_config",
        description="Optional per-request model override",
    )
