from __future__ import annotations

from typing import Any
from typing import Literal

from pydantic import BaseModel, Field


WORKFLOW_ASSISTANT_PAGE_KEYS = (
    "dataset_management",
    "reasoning_data_synthesis",
    "agentic_trajectory_synthesis",
)


class WorkflowAssistantMessage(BaseModel):
    role: Literal["user", "assistant"] = Field(...)
    content: str = Field(..., min_length=1, max_length=12000)


class WorkflowAssistantChatRequest(BaseModel):
    page_key: Literal["dataset_management", "reasoning_data_synthesis", "agentic_trajectory_synthesis"] = Field(...)
    session_id: str | None = Field(default=None, max_length=128)
    messages: list[WorkflowAssistantMessage] = Field(default_factory=list, min_length=1, max_length=40)
    page_context: dict[str, Any] = Field(default_factory=dict)
