from __future__ import annotations

import json
from datetime import datetime
from typing import Dict

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, text
from sqlalchemy.orm import validates

from .base import Base


SUPPORTED_REASONING_RESULT_STATUSES = {"pending", "completed", "failed"}
EVALUATION_DIMENSIONS = (
    "clarity",
    "coherence",
    "completeness",
    "complexity",
    "correctness",
    "meaningfulness",
    "difficulty",
)


def _utc_now() -> datetime:
    return datetime.utcnow()


class ReasoningDistillationResult(Base):
    __tablename__ = "reasoning_distillation_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(Integer, ForeignKey("reasoning_distillation_tasks.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    source_type = Column(String(32), nullable=False)
    source_ref_id = Column(Integer, nullable=False)
    item_key = Column(String(256), nullable=False)
    prompt_text = Column(Text, nullable=False)
    reasoning_text = Column(Text, nullable=False)
    answer_text = Column(Text, nullable=False)
    record_json = Column(Text, nullable=False, default="{}", server_default=text("'{}'"))
    evaluation_json = Column(Text, nullable=True)
    evaluation_raw_text = Column(Text, nullable=True)
    evaluation_error_message = Column(Text, nullable=True, default=None)
    token_count = Column(Integer, nullable=False, default=0, server_default=text("0"))
    status = Column(String(16), nullable=False, default="completed", server_default=text("'completed'"), index=True)
    error_message = Column(Text, nullable=True, default=None)
    insert_time = Column(DateTime, nullable=False, default=_utc_now, server_default=text("CURRENT_TIMESTAMP"))
    update_time = Column(DateTime, nullable=False, default=_utc_now, onupdate=_utc_now, server_default=text("CURRENT_TIMESTAMP"))

    @validates("source_type", "item_key", "prompt_text", "reasoning_text", "answer_text")
    def _validate_non_empty(self, key, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError(f"{key} must not be empty")
        return normalized

    @validates("status")
    def _validate_status(self, key, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in SUPPORTED_REASONING_RESULT_STATUSES:
            raise ValueError("status must be one of: pending, completed, failed")
        return normalized

    def to_dict(self) -> Dict:
        record = {}
        evaluation = {}
        try:
            parsed = json.loads(self.record_json or "{}")
            if isinstance(parsed, dict):
                record = parsed
        except Exception:
            record = {}
        try:
            parsed = json.loads(self.evaluation_json or "{}")
            if isinstance(parsed, dict):
                evaluation = parsed
        except Exception:
            evaluation = {}

        payload = {
            "id": self.id,
            "task_id": self.task_id,
            "user_id": self.user_id,
            "source_type": self.source_type,
            "source_ref_id": self.source_ref_id,
            "item_key": self.item_key,
            "prompt_text": self.prompt_text,
            "reasoning_text": self.reasoning_text,
            "answer_text": self.answer_text,
            "record": record,
            "evaluation": evaluation,
            "evaluation_raw_text": self.evaluation_raw_text,
            "evaluation_error_message": self.evaluation_error_message,
            "token_count": self.token_count,
            "status": self.status,
            "error_message": self.error_message,
            "insert_time": self.insert_time.isoformat() if self.insert_time else None,
            "update_time": self.update_time.isoformat() if self.update_time else None,
        }
        for key in EVALUATION_DIMENSIONS:
            payload[key] = evaluation.get(key)
        return payload
