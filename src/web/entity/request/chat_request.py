from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatModelConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

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


class ChatContextItem(BaseModel):
    type: Literal["asset_file", "dataset", "trajectory_task", "distillation_task"]
    path: str | None = Field(default=None, max_length=500)
    ref_id: int | None = Field(default=None, ge=1)


class AssetImportRequest(BaseModel):
    source_type: Literal["dataset", "trajectory_task", "distillation_task"]
    source_id: int = Field(..., ge=1)
    target_folder_path: str = Field(default="")


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
    selected_file_paths: list[str] | None = Field(
        default=None,
        description="Optional relative file paths explicitly selected by the user for sequential analysis",
    )
    selected_model: ChatModelConfig | None = Field(
        default=None,
        alias="model_config",
        description="Optional per-request model override",
    )
    context_items: list[ChatContextItem] | None = Field(
        default=None,
        description="Optional platform objects or asset files to stage into the runtime workspace",
    )
