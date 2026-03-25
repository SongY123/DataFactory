from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field, field_validator


_SUPPORTED_DATASET_TYPES = {"instruction", "conversation", "evaluation", "tool-trace", "trajectory", "reasoning"}
_SUPPORTED_DATASET_LANGUAGES = {"zh", "en", "multi"}
_SUPPORTED_DATASET_SOURCE_KINDS = {"upload", "generated", "huggingface"}
_SUPPORTED_DATASET_SIZE_LEVELS = {"kb", "mb", "gb"}


def _normalize_tag_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    raw_items = value
    if isinstance(raw_items, str):
        raw_items = [item.strip() for item in raw_items.split(",")]
    if not isinstance(raw_items, (list, tuple, set)):
        raise ValueError("tag field must be a list of strings")

    result: List[str] = []
    seen = set()
    for item in raw_items:
        normalized = str(item or "").strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _normalize_keyword(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value or "").strip()
    return text or None


def _normalize_choice_list(value: Any) -> List[str]:
    normalized = _normalize_tag_list(value)
    return normalized or []


class DatasetCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    type: str = Field(default="instruction", min_length=1, max_length=32)
    source: Optional[str] = Field(default=None, max_length=1024)
    language: str = Field(default="multi", min_length=1, max_length=16)
    size: int = Field(default=0, ge=0)
    note: Optional[str] = Field(default=None, max_length=2000)
    sample_data: Optional[List[dict]] = Field(default=None)
    source_kind: str = Field(default="upload", min_length=1, max_length=32)
    modality_tags: Optional[List[str]] = Field(default=None)
    format_tags: Optional[List[str]] = Field(default=None)
    language_tags: Optional[List[str]] = Field(default=None)
    license_tag: Optional[str] = Field(default=None, max_length=128)

    @field_validator("type")
    @classmethod
    def _validate_type(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in _SUPPORTED_DATASET_TYPES:
            raise ValueError("type must be one of: instruction, conversation, evaluation, tool-trace, trajectory, reasoning")
        return normalized

    @field_validator("language")
    @classmethod
    def _validate_language(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in _SUPPORTED_DATASET_LANGUAGES:
            raise ValueError("language must be one of: zh, en, multi")
        return normalized

    @field_validator("source_kind")
    @classmethod
    def _validate_source_kind(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in _SUPPORTED_DATASET_SOURCE_KINDS:
            raise ValueError("source_kind must be one of: upload, generated, huggingface")
        return normalized

    @field_validator("modality_tags", "format_tags", "language_tags", mode="before")
    @classmethod
    def _validate_tag_fields(cls, value):
        return _normalize_tag_list(value)


class DatasetUpdateRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=128)
    type: Optional[str] = Field(default=None, min_length=1, max_length=32)
    source: Optional[str] = Field(default=None, max_length=1024)
    language: Optional[str] = Field(default=None, min_length=1, max_length=16)
    status: Optional[str] = Field(default=None, min_length=1, max_length=32)
    note: Optional[str] = Field(default=None, max_length=2000)
    modality_tags: Optional[List[str]] = Field(default=None)
    format_tags: Optional[List[str]] = Field(default=None)
    language_tags: Optional[List[str]] = Field(default=None)
    license_tag: Optional[str] = Field(default=None, max_length=128)

    @field_validator("type")
    @classmethod
    def _validate_type(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = str(value or "").strip().lower()
        if normalized not in _SUPPORTED_DATASET_TYPES:
            raise ValueError("type must be one of: instruction, conversation, evaluation, tool-trace, trajectory, reasoning")
        return normalized

    @field_validator("language")
    @classmethod
    def _validate_language(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = str(value or "").strip().lower()
        if normalized not in _SUPPORTED_DATASET_LANGUAGES:
            raise ValueError("language must be one of: zh, en, multi")
        return normalized

    @field_validator("modality_tags", "format_tags", "language_tags", mode="before")
    @classmethod
    def _validate_tag_fields(cls, value):
        return _normalize_tag_list(value)


class HuggingFaceDatasetImportRequest(BaseModel):
    repo_id: str = Field(..., min_length=1, max_length=256)
    revision: Optional[str] = Field(default=None, max_length=128)
    name: Optional[str] = Field(default=None, max_length=128)
    note: Optional[str] = Field(default=None, max_length=2000)

    @field_validator("repo_id")
    @classmethod
    def _validate_repo_id(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("repo_id is required")
        return normalized


class DatasetSqlQueryRequest(BaseModel):
    path: Optional[str] = Field(default=None, max_length=1024)
    sql: str = Field(..., min_length=1, max_length=8000)
    limit: int = Field(default=100, ge=1, le=500)


class DatasetQueryRequest(BaseModel):
    name_keyword: Optional[str] = Field(default=None, max_length=256)
    format_tags: List[str] = Field(default_factory=list)
    language_tags: List[str] = Field(default_factory=list)
    size_levels: List[str] = Field(default_factory=list)
    min_size_bytes: Optional[int] = Field(default=None, ge=0)
    statuses: List[str] = Field(default_factory=list)

    @field_validator("name_keyword", mode="before")
    @classmethod
    def _validate_name_keyword(cls, value: Any) -> Optional[str]:
        return _normalize_keyword(value)

    @field_validator("format_tags", "language_tags", "statuses", mode="before")
    @classmethod
    def _validate_choice_lists(cls, value: Any) -> List[str]:
        return _normalize_choice_list(value)

    @field_validator("size_levels", mode="before")
    @classmethod
    def _validate_size_levels(cls, value: Any) -> List[str]:
        values = _normalize_choice_list(value)
        invalid = [item for item in values if item not in _SUPPORTED_DATASET_SIZE_LEVELS]
        if invalid:
            raise ValueError("size_levels must only contain: kb, mb, gb")
        return values
