from __future__ import annotations

import json
from datetime import datetime
from typing import Dict

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, text
from sqlalchemy.orm import validates

from .base import Base


SUPPORTED_RESULT_STATUSES = {"pending", "completed", "failed"}


def _utc_now() -> datetime:
    return datetime.utcnow()


class AgenticSynthesisResult(Base):
    __tablename__ = "agentic_synthesis_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(Integer, ForeignKey("agentic_synthesis_tasks.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False, index=True)
    workspace_name = Column(String(256), nullable=False)
    question = Column(Text, nullable=False)
    trajectory = Column(Text, nullable=False)
    evaluation_json = Column(Text, nullable=False, default="{}", server_default=text("'{}'"))
    status = Column(String(16), nullable=False, default="completed", server_default=text("'completed'"), index=True)
    error_message = Column(Text, nullable=True, default=None)
    insert_time = Column(DateTime, nullable=False, default=_utc_now, server_default=text("CURRENT_TIMESTAMP"))
    update_time = Column(DateTime, nullable=False, default=_utc_now, onupdate=_utc_now, server_default=text("CURRENT_TIMESTAMP"))

    @validates("workspace_name", "question", "trajectory")
    def _validate_non_empty(self, key, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError(f"{key} must not be empty")
        return normalized

    @validates("status")
    def _validate_status(self, key, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in SUPPORTED_RESULT_STATUSES:
            raise ValueError("status must be one of: pending, completed, failed")
        return normalized

    def to_dict(self) -> Dict:
        evaluation = {}
        try:
            parsed = json.loads(self.evaluation_json or "{}")
            if isinstance(parsed, dict):
                evaluation = parsed
        except Exception:
            evaluation = {}

        return {
            "id": self.id,
            "task_id": self.task_id,
            "user_id": self.user_id,
            "dataset_id": self.dataset_id,
            "workspace_name": self.workspace_name,
            "question": self.question,
            "trajectory": self.trajectory,
            "evaluation": evaluation,
            "status": self.status,
            "error_message": self.error_message,
            "insert_time": self.insert_time.isoformat() if self.insert_time else None,
            "update_time": self.update_time.isoformat() if self.update_time else None,
        }
