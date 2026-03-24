from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class AgenticSynthesisStartRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    dataset_ids: List[int] = Field(..., min_length=1)
    action_tags: List[str] = Field(default_factory=list)
    llm_api_key: str = Field(..., min_length=1)
    llm_base_url: str = Field(..., min_length=1)
    llm_model_name: str = Field(..., min_length=1)
    parallelism: int = Field(default=1, ge=1, le=32)
    save_path: Optional[str] = Field(default=None, max_length=2000)
    save_path_key: Optional[str] = Field(default=None, max_length=128)
    sandbox_environment_id: Optional[str] = Field(default=None, max_length=128)
    llm_params_json: Optional[str] = Field(default=None, max_length=16000)

    @model_validator(mode="before")
    @classmethod
    def _compat_single_dataset_id(cls, data):
        if not isinstance(data, dict):
            return data
        if data.get("dataset_ids") is None and data.get("dataset_id") is not None:
            data = dict(data)
            data["dataset_ids"] = [data.get("dataset_id")]
        return data

    @field_validator("prompt", "llm_api_key", "llm_base_url", "llm_model_name")
    @classmethod
    def _validate_non_empty(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("field must not be empty")
        return normalized

    @field_validator("save_path", "save_path_key", "sandbox_environment_id", "llm_params_json")
    @classmethod
    def _validate_optional_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = str(value or "").strip()
        return normalized or None

    @field_validator("action_tags")
    @classmethod
    def _validate_action_tags(cls, value: List[str]) -> List[str]:
        result: List[str] = []
        for item in value or []:
            cleaned = str(item or "").strip()
            if cleaned:
                result.append(cleaned)
        return result

    @field_validator("dataset_ids", mode="before")
    @classmethod
    def _validate_dataset_ids(cls, value) -> List[int]:
        if value is None:
            raise ValueError("dataset_ids must not be empty")
        if isinstance(value, int):
            value = [value]
        if not isinstance(value, list):
            raise ValueError("dataset_ids must be a list of integers")
        result: List[int] = []
        seen = set()
        for item in value:
            dataset_id = int(item)
            if dataset_id < 1:
                raise ValueError("dataset_ids contains invalid id")
            if dataset_id in seen:
                continue
            seen.add(dataset_id)
            result.append(dataset_id)
        if not result:
            raise ValueError("dataset_ids must not be empty")
        return result

class AgenticSynthesisTaskQueryRequest(BaseModel):
    limit: int = Field(default=20, ge=1, le=200)
