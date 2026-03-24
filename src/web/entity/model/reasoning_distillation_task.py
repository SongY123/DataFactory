from __future__ import annotations

import json
from datetime import datetime
from typing import Dict

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text, text
from sqlalchemy.orm import validates

from .base import Base


SUPPORTED_DISTILLATION_SOURCE_TYPES = {"dataset", "trajectory_task"}


def _utc_now() -> datetime:
    return datetime.utcnow()


class ReasoningDistillationTask(Base):
    __tablename__ = "reasoning_distillation_tasks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    source_type = Column(String(32), nullable=False)
    source_dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True, index=True)
    source_task_id = Column(Integer, ForeignKey("agentic_synthesis_tasks.id"), nullable=True, index=True)
    prompt_text = Column(Text, nullable=True)
    evaluation_enabled = Column(Integer, nullable=False, default=0, server_default=text("0"))
    evaluation_prompt_text = Column(Text, nullable=True)
    strategy = Column(String(64), nullable=False)
    target_max_tokens = Column(Integer, nullable=False, default=1024, server_default=text("1024"))
    compression_ratio = Column(Float, nullable=False, default=0.5, server_default=text("0.5"))
    keep_tool_trace = Column(Integer, nullable=False, default=0, server_default=text("0"))
    note = Column(Text, nullable=True)
    llm_api_key = Column(Text, nullable=False)
    llm_base_url = Column(Text, nullable=False)
    llm_model_name = Column(String(128), nullable=False)
    parallelism = Column(Integer, nullable=False, default=1, server_default=text("1"))
    llm_params_json = Column(Text, nullable=True)
    output_file_path = Column(Text, nullable=False)
    generated_dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True)
    total_items = Column(Integer, nullable=False, default=0, server_default=text("0"))
    processed_items = Column(Integer, nullable=False, default=0, server_default=text("0"))
    distilled_samples = Column(Integer, nullable=False, default=0, server_default=text("0"))
    avg_tokens = Column(Integer, nullable=False, default=0, server_default=text("0"))
    started_time = Column(DateTime, nullable=True, default=None)
    finished_time = Column(DateTime, nullable=True, default=None)
    error_message = Column(Text, nullable=True, default=None)
    insert_time = Column(DateTime, nullable=False, default=_utc_now, server_default=text("CURRENT_TIMESTAMP"))
    update_time = Column(DateTime, nullable=False, default=_utc_now, onupdate=_utc_now, server_default=text("CURRENT_TIMESTAMP"))

    @validates("source_type")
    def _validate_source_type(self, key, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in SUPPORTED_DISTILLATION_SOURCE_TYPES:
            raise ValueError("source_type must be one of: dataset, trajectory_task")
        return normalized

    @validates("strategy", "llm_api_key", "llm_base_url", "llm_model_name", "output_file_path")
    def _validate_non_empty(self, key, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError(f"{key} must not be empty")
        return normalized

    @validates("target_max_tokens")
    def _validate_target_max_tokens(self, key, value: int) -> int:
        normalized = int(value or 0)
        if normalized < 1:
            raise ValueError("target_max_tokens must be >= 1")
        return normalized

    @validates("compression_ratio")
    def _validate_compression_ratio(self, key, value: float) -> float:
        normalized = float(value or 0)
        if normalized <= 0 or normalized > 1:
            raise ValueError("compression_ratio must be > 0 and <= 1")
        return normalized

    def to_dict(self) -> Dict:
        llm_params = self._safe_json_dict(self.llm_params_json)
        return {
            "id": self.id,
            "user_id": self.user_id,
            "source_type": self.source_type,
            "source_dataset_id": self.source_dataset_id,
            "source_task_id": self.source_task_id,
            "prompt_text": self.prompt_text,
            "evaluation_enabled": bool(self.evaluation_enabled),
            "evaluation_prompt_text": self.evaluation_prompt_text,
            "strategy": self.strategy,
            "target_max_tokens": self.target_max_tokens,
            "compression_ratio": self.compression_ratio,
            "keep_tool_trace": bool(self.keep_tool_trace),
            "note": self.note,
            "llm_api_key": self.llm_api_key,
            "llm_base_url": self.llm_base_url,
            "llm_model_name": self.llm_model_name,
            "parallelism": self.parallelism,
            "llm_params_json": self.llm_params_json,
            "llm_params": llm_params,
            "output_file_path": self.output_file_path,
            "generated_dataset_id": self.generated_dataset_id,
            "total_items": self.total_items,
            "processed_items": self.processed_items,
            "distilled_samples": self.distilled_samples,
            "avg_tokens": self.avg_tokens,
            "status": self._derive_status(),
            "started_time": self.started_time.isoformat() if self.started_time else None,
            "finished_time": self.finished_time.isoformat() if self.finished_time else None,
            "error_message": self.error_message,
            "insert_time": self.insert_time.isoformat() if self.insert_time else None,
            "update_time": self.update_time.isoformat() if self.update_time else None,
        }

    def _derive_status(self) -> str:
        if self.finished_time and self.error_message:
            return "failed"
        if self.finished_time:
            return "completed"
        if self.started_time:
            return "running"
        return "pending"

    @staticmethod
    def _safe_json_dict(value: str | None) -> Dict:
        try:
            loaded = json.loads(value or "{}")
            if isinstance(loaded, dict):
                return loaded
        except Exception:
            pass
        return {}
