from __future__ import annotations

import json
from datetime import datetime
from typing import Dict

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, text
from sqlalchemy.orm import validates

from .base import Base


SUPPORTED_DATASET_TYPES = {"instruction", "conversation", "evaluation", "tool-trace", "trajectory", "reasoning"}
SUPPORTED_DATASET_LANGUAGES = {"zh", "en", "multi"}
SUPPORTED_DATASET_SOURCE_KINDS = {"upload", "generated", "huggingface"}


def _utc_now() -> datetime:
    return datetime.utcnow()


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(128), nullable=False, index=True)
    type = Column(String(32), nullable=False, default="instruction", server_default=text("'instruction'"))
    source = Column(String(1024), nullable=True)
    language = Column(String(16), nullable=False, default="multi", server_default=text("'multi'"))
    size = Column(Integer, nullable=False, default=0, server_default=text("0"))
    status = Column(String(32), nullable=False, default="uploaded", server_default=text("'uploaded'"), index=True)
    note = Column(Text, nullable=True)
    file_name = Column(String(256), nullable=True)
    file_path = Column(String(1024), nullable=True)
    cover_path = Column(String(1024), nullable=True)
    sample_data = Column(Text, nullable=True)
    source_kind = Column(String(32), nullable=False, default="upload", server_default=text("'upload'"))
    hf_repo_id = Column(String(256), nullable=True)
    hf_revision = Column(String(128), nullable=True)
    readme_text = Column(Text, nullable=True)
    modality_tags_json = Column(Text, nullable=True, default="[]", server_default=text("'[]'"))
    format_tags_json = Column(Text, nullable=True, default="[]", server_default=text("'[]'"))
    language_tags_json = Column(Text, nullable=True, default="[]", server_default=text("'[]'"))
    license_tag = Column(String(128), nullable=True)
    import_progress = Column(Integer, nullable=False, default=100, server_default=text("100"))
    import_total_files = Column(Integer, nullable=False, default=0, server_default=text("0"))
    import_downloaded_files = Column(Integer, nullable=False, default=0, server_default=text("0"))
    import_error_message = Column(Text, nullable=True)
    origin_stage = Column(String(64), nullable=True)
    origin_dataset_id = Column(Integer, nullable=True)
    origin_task_type = Column(String(64), nullable=True)
    origin_task_id = Column(Integer, nullable=True)
    generation_meta_json = Column(Text, nullable=True)
    insert_time = Column(DateTime, nullable=False, default=_utc_now, server_default=text("CURRENT_TIMESTAMP"))
    update_time = Column(DateTime, nullable=False, default=_utc_now, onupdate=_utc_now, server_default=text("CURRENT_TIMESTAMP"))

    @validates("name")
    def _validate_name(self, key, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("dataset name must not be empty")
        return normalized

    @validates("type")
    def _validate_type(self, key, value: str) -> str:
        normalized = str(value or "instruction").strip().lower()
        if normalized not in SUPPORTED_DATASET_TYPES:
            raise ValueError("dataset type is invalid")
        return normalized

    @validates("language")
    def _validate_language(self, key, value: str) -> str:
        normalized = str(value or "multi").strip().lower()
        if normalized not in SUPPORTED_DATASET_LANGUAGES:
            raise ValueError("dataset language is invalid")
        return normalized

    @validates("size")
    def _validate_size(self, key, value: int) -> int:
        size = int(value or 0)
        if size < 0:
            raise ValueError("dataset size must be >= 0")
        return size

    @validates("source_kind")
    def _validate_source_kind(self, key, value: str) -> str:
        normalized = str(value or "upload").strip().lower()
        if normalized not in SUPPORTED_DATASET_SOURCE_KINDS:
            raise ValueError("dataset source_kind is invalid")
        return normalized

    def to_dict(self, include_internal: bool = False) -> Dict:
        generation_meta = {}
        try:
            parsed = json.loads(self.generation_meta_json or "{}")
            if isinstance(parsed, dict):
                generation_meta = parsed
        except Exception:
            generation_meta = {}

        data = {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "type": self.type,
            "source": self.source,
            "language": self.language,
            "size": self.size,
            "status": self.status,
            "note": self.note,
            "sample_data": self.sample_data,
            "source_kind": self.source_kind,
            "hf_repo_id": self.hf_repo_id,
            "hf_revision": self.hf_revision,
            "readme_text": self.readme_text,
            "modality_tags_json": self.modality_tags_json,
            "format_tags_json": self.format_tags_json,
            "language_tags_json": self.language_tags_json,
            "license_tag": self.license_tag,
            "import_progress": self.import_progress,
            "import_total_files": self.import_total_files,
            "import_downloaded_files": self.import_downloaded_files,
            "import_error_message": self.import_error_message,
            "origin_stage": self.origin_stage,
            "origin_dataset_id": self.origin_dataset_id,
            "origin_task_type": self.origin_task_type,
            "origin_task_id": self.origin_task_id,
            "generation_meta": generation_meta,
            "generated_output": generation_meta,
            "insert_time": self.insert_time.isoformat() if self.insert_time else None,
            "update_time": self.update_time.isoformat() if self.update_time else None,
        }
        if include_internal:
            data["file_name"] = self.file_name
            data["file_path"] = self.file_path
            data["cover_path"] = self.cover_path
            data["generation_meta_json"] = self.generation_meta_json
        return data
