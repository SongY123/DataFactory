from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, List

from sqlalchemy import Column, DateTime, Integer, String, Text, text
from sqlalchemy.orm import validates

from .base import Base


def _utc_now() -> datetime:
    return datetime.utcnow()


SUPPORTED_SYNTHESIS_TASK_STATUSES = {
    "pending",
    "running",
    "completed",
    "failed",
}


class AgenticSynthesisTask(Base):
    __tablename__ = "agentic_synthesis_tasks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prompt_text = Column(Text, nullable=False)
    action_tags_json = Column(Text, nullable=False, default="[]", server_default=text("'[]'"))
    llm_api_key = Column(Text, nullable=False)
    llm_base_url = Column(Text, nullable=False)
    llm_model_name = Column(String(128), nullable=False)
    dataset_paths_json = Column(Text, nullable=False, default="[]", server_default=text("'[]'"))
    output_file_path = Column(Text, nullable=False)
    status = Column(
        String(16),
        nullable=False,
        default="pending",
        server_default=text("'pending'"),
        index=True,
    )
    total_files = Column(Integer, nullable=False, default=0, server_default=text("0"))
    processed_files = Column(Integer, nullable=False, default=0, server_default=text("0"))
    success_files = Column(Integer, nullable=False, default=0, server_default=text("0"))
    failed_files = Column(Integer, nullable=False, default=0, server_default=text("0"))
    started_time = Column(DateTime, nullable=True, default=None)
    finished_time = Column(DateTime, nullable=True, default=None)
    error_message = Column(Text, nullable=True, default=None)
    insert_time = Column(DateTime, nullable=False, default=_utc_now, server_default=text("CURRENT_TIMESTAMP"))
    update_time = Column(DateTime, nullable=False, default=_utc_now, onupdate=_utc_now, server_default=text("CURRENT_TIMESTAMP"))

    @validates("prompt_text")
    def _validate_prompt_text(self, key, value: str) -> str:
        prompt = str(value or "").strip()
        if not prompt:
            raise ValueError("prompt_text must not be empty")
        return prompt

    @validates("llm_api_key")
    def _validate_llm_api_key(self, key, value: str) -> str:
        token = str(value or "").strip()
        if not token:
            raise ValueError("llm_api_key must not be empty")
        return token

    @validates("llm_base_url")
    def _validate_llm_base_url(self, key, value: str) -> str:
        base_url = str(value or "").strip()
        if not base_url:
            raise ValueError("llm_base_url must not be empty")
        return base_url

    @validates("llm_model_name")
    def _validate_llm_model_name(self, key, value: str) -> str:
        model_name = str(value or "").strip()
        if not model_name:
            raise ValueError("llm_model_name must not be empty")
        return model_name

    @validates("status")
    def _validate_status(self, key, value: str) -> str:
        status = str(value or "").strip().lower()
        if status not in SUPPORTED_SYNTHESIS_TASK_STATUSES:
            raise ValueError("status must be one of: pending, running, completed, failed")
        return status

    def to_dict(self) -> Dict:
        action_tags = self._safe_json_list(self.action_tags_json)
        dataset_paths = self._safe_json_list(self.dataset_paths_json)
        return {
            "id": self.id,
            "prompt_text": self.prompt_text,
            "action_tags": action_tags,
            "llm_api_key": self.llm_api_key,
            "llm_base_url": self.llm_base_url,
            "llm_model_name": self.llm_model_name,
            "dataset_paths": dataset_paths,
            "output_file_path": self.output_file_path,
            "status": self.status,
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "success_files": self.success_files,
            "failed_files": self.failed_files,
            "started_time": self.started_time.isoformat() if self.started_time else None,
            "finished_time": self.finished_time.isoformat() if self.finished_time else None,
            "error_message": self.error_message,
            "insert_time": self.insert_time.isoformat() if self.insert_time else None,
            "update_time": self.update_time.isoformat() if self.update_time else None,
        }

    @staticmethod
    def _safe_json_list(value: str) -> List[str]:
        try:
            loaded = json.loads(value or "[]")
            if isinstance(loaded, list):
                return [str(x) for x in loaded]
        except Exception:
            pass
        return []
