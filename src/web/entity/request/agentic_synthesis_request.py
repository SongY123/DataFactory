from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class AgenticDatasetRequest(BaseModel):
    id: Optional[int] = Field(default=None)
    name: Optional[str] = Field(default=None)
    uri: str = Field(..., min_length=1)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @field_validator("uri")
    @classmethod
    def _validate_uri(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("dataset uri must not be empty")
        return normalized


class AgenticSynthesisStartRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    action_tags: List[str] = Field(default_factory=list)
    llm_api_key: str = Field(..., min_length=1)
    llm_base_url: str = Field(..., min_length=1)
    llm_model_name: str = Field(..., min_length=1)
    datasets: List[AgenticDatasetRequest] = Field(..., min_length=1)
    output_file_path: Optional[str] = Field(default=None)

    @field_validator("prompt", "llm_api_key", "llm_base_url", "llm_model_name")
    @classmethod
    def _validate_non_empty(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("field must not be empty")
        return normalized

    @field_validator("action_tags")
    @classmethod
    def _validate_action_tags(cls, value: List[str]) -> List[str]:
        result: List[str] = []
        for item in value or []:
            cleaned = str(item or "").strip()
            if cleaned:
                result.append(cleaned)
        return result

    @field_validator("datasets")
    @classmethod
    def _validate_datasets(cls, value: List[AgenticDatasetRequest]) -> List[AgenticDatasetRequest]:
        if not value:
            raise ValueError("datasets must contain at least one valid path")
        return value


class AgenticSynthesisTaskQueryRequest(BaseModel):
    limit: int = Field(default=20, ge=1, le=200)
