from __future__ import annotations

from datetime import datetime
from typing import Dict

from sqlalchemy import Column, DateTime, Integer, String, Text, text
from sqlalchemy.orm import validates

from .base import Base


SUPPORTED_DATASET_TYPES = {"instruction", "conversation", "evaluation", "tool-trace"}
SUPPORTED_DATASET_LANGUAGES = {"zh", "en", "multi"}


def _utc_now() -> datetime:
    return datetime.utcnow()


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, autoincrement=True)
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

    def to_dict(self, include_internal: bool = False) -> Dict:
        data = {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "source": self.source,
            "language": self.language,
            "size": self.size,
            "status": self.status,
            "note": self.note,
            "sample_data": self.sample_data,
            "insert_time": self.insert_time.isoformat() if self.insert_time else None,
            "update_time": self.update_time.isoformat() if self.update_time else None,
        }
        if include_internal:
            data["file_name"] = self.file_name
            data["file_path"] = self.file_path
            data["cover_path"] = self.cover_path
        return data
