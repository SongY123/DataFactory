from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class ReasoningDatasetFileMapping(BaseModel):
    path: str = Field(..., min_length=1, max_length=2000)
    placeholder_mappings: dict[str, str] = Field(default_factory=dict)
    prompt_field: Optional[str] = Field(default=None, max_length=256)
    completion_field: Optional[str] = Field(default=None, max_length=256)

    @field_validator("path")
    @classmethod
    def _normalize_path(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("path must not be empty")
        return normalized

    @field_validator("prompt_field", "completion_field")
    @classmethod
    def _normalize_optional_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = str(value or "").strip()
        return normalized or None

    @field_validator("placeholder_mappings", mode="before")
    @classmethod
    def _normalize_placeholder_mappings(cls, value) -> dict[str, str]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("placeholder_mappings must be an object")

        normalized: dict[str, str] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key or "").strip()
            mapped = str(raw_value or "").strip()
            if not key or not mapped:
                continue
            normalized[key] = mapped
        return normalized


class ReasoningDistillationStartRequest(BaseModel):
    source_type: Literal["dataset", "trajectory_task"]
    source_dataset_id: Optional[int] = Field(default=None, ge=1)
    source_task_id: Optional[int] = Field(default=None, ge=1)
    selected_file_paths: list[str] = Field(default_factory=list)
    file_mappings: list[ReasoningDatasetFileMapping] = Field(default_factory=list)
    prompt_field: Optional[str] = Field(default=None, max_length=256)
    completion_field: Optional[str] = Field(default=None, max_length=256)
    prompt: Optional[str] = Field(default=None, max_length=12000)
    evaluation_enabled: bool = Field(default=False)
    evaluation_prompt: Optional[str] = Field(default=None, max_length=12000)
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

    @field_validator("prompt", "evaluation_prompt", "save_path", "save_path_key", "llm_params_json", "prompt_field", "completion_field")
    @classmethod
    def _normalize_optional_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = str(value or "").strip()
        return normalized or None

    @field_validator("selected_file_paths", mode="before")
    @classmethod
    def _normalize_selected_file_paths(cls, value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple, set)):
            raise ValueError("selected_file_paths must be a list of file paths")

        normalized: list[str] = []
        seen = set()
        for item in value:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            normalized.append(text)
        return normalized

    @field_validator("file_mappings", mode="before")
    @classmethod
    def _normalize_file_mappings(cls, value):
        if value is None:
            return []
        if not isinstance(value, (list, tuple)):
            raise ValueError("file_mappings must be a list")
        return list(value)

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
        if self.evaluation_enabled and not self.evaluation_prompt:
            raise ValueError("evaluation_prompt is required when evaluation_enabled=true")
        return self
