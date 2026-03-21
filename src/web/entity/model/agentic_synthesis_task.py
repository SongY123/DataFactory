from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, List

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, text
from sqlalchemy.orm import validates

from .base import Base


def _utc_now() -> datetime:
    return datetime.utcnow()


class AgenticSynthesisTask(Base):
    __tablename__ = "agentic_synthesis_tasks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False, index=True)
    prompt_text = Column(Text, nullable=False)
    action_tags_json = Column(Text, nullable=False, default="[]", server_default=text("'[]'"))
    llm_api_key = Column(Text, nullable=False)
    llm_base_url = Column(Text, nullable=False)
    llm_model_name = Column(String(128), nullable=False)
    output_file_path = Column(Text, nullable=False)
    generated_dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True)
    total_workspaces = Column(Integer, nullable=False, default=0, server_default=text("0"))
    processed_workspaces = Column(Integer, nullable=False, default=0, server_default=text("0"))
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

    def to_dict(self) -> Dict:
        action_tags = self._safe_json_list(self.action_tags_json)
        return {
            "id": self.id,
            "user_id": self.user_id,
            "dataset_id": self.dataset_id,
            "prompt_text": self.prompt_text,
            "action_tags": action_tags,
            "llm_api_key": self.llm_api_key,
            "llm_base_url": self.llm_base_url,
            "llm_model_name": self.llm_model_name,
            "output_file_path": self.output_file_path,
            "generated_dataset_id": self.generated_dataset_id,
            "total_workspaces": self.total_workspaces,
            "processed_workspaces": self.processed_workspaces,
            "status": self._derive_status(),
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

    def _derive_status(self) -> str:
        if self.finished_time and self.error_message:
            return "failed"
        if self.finished_time:
            return "completed"
        if self.started_time:
            return "running"
        return "pending"
