from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class ReasoningDistillationStartRequest(BaseModel):
    source_type: Literal["dataset", "trajectory_task"]
    source_dataset_id: Optional[int] = Field(default=None, ge=1)
    source_task_id: Optional[int] = Field(default=None, ge=1)
    prompt: Optional[str] = Field(default=None, max_length=12000)
    strategy: str = Field(..., min_length=1, max_length=64)
    target_max_tokens: int = Field(default=1024, ge=1)
    compression_ratio: float = Field(default=0.5, gt=0, le=1)
    keep_tool_trace: bool = Field(default=False)
    note: Optional[str] = Field(default=None, max_length=2000)
    llm_api_key: str = Field(..., min_length=1)
    llm_base_url: str = Field(..., min_length=1)
    llm_model_name: str = Field(..., min_length=1, max_length=128)
    parallelism: int = Field(default=1, ge=1, le=32)
    save_path: Optional[str] = Field(default=None, max_length=2000)
    save_path_key: Optional[str] = Field(default=None, max_length=128)
    llm_params_json: Optional[str] = Field(default=None, max_length=16000)

    @field_validator("strategy", "llm_api_key", "llm_base_url", "llm_model_name")
    @classmethod
    def _validate_non_empty(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("field must not be empty")
        return normalized

    @field_validator("prompt", "save_path", "save_path_key", "llm_params_json")
    @classmethod
    def _normalize_optional_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = str(value or "").strip()
        return normalized or None

    @model_validator(mode="after")
    def _validate_source_ref(self):
        if self.source_type == "dataset":
            if self.source_dataset_id is None:
                raise ValueError("source_dataset_id is required when source_type=dataset")
            self.source_task_id = None
        elif self.source_type == "trajectory_task":
            if self.source_task_id is None:
                raise ValueError("source_task_id is required when source_type=trajectory_task")
            self.source_dataset_id = None
        return self
