from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class DatasetCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    type: str = Field(default="instruction", min_length=1, max_length=32)
    source: Optional[str] = Field(default=None, max_length=1024)
    language: str = Field(default="multi", min_length=1, max_length=16)
    size: int = Field(default=0, ge=0)
    note: Optional[str] = Field(default=None, max_length=2000)
    sample_data: Optional[List[dict]] = Field(default=None)

    @field_validator("type")
    @classmethod
    def _validate_type(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in {"instruction", "conversation", "evaluation", "tool-trace"}:
            raise ValueError("type must be one of: instruction, conversation, evaluation, tool-trace")
        return normalized

    @field_validator("language")
    @classmethod
    def _validate_language(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in {"zh", "en", "multi"}:
            raise ValueError("language must be one of: zh, en, multi")
        return normalized


class DatasetUpdateRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=128)
    type: Optional[str] = Field(default=None, min_length=1, max_length=32)
    source: Optional[str] = Field(default=None, max_length=1024)
    language: Optional[str] = Field(default=None, min_length=1, max_length=16)
    status: Optional[str] = Field(default=None, min_length=1, max_length=32)
    note: Optional[str] = Field(default=None, max_length=2000)

    @field_validator("type")
    @classmethod
    def _validate_type(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = str(value or "").strip().lower()
        if normalized not in {"instruction", "conversation", "evaluation", "tool-trace"}:
            raise ValueError("type must be one of: instruction, conversation, evaluation, tool-trace")
        return normalized

    @field_validator("language")
    @classmethod
    def _validate_language(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = str(value or "").strip().lower()
        if normalized not in {"zh", "en", "multi"}:
            raise ValueError("language must be one of: zh, en, multi")
        return normalized
