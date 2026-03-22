from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import select

from .base_dao import BaseDAO
from ..entity.model import ReasoningDistillationTask


class ReasoningDistillationTaskDAO(BaseDAO):
    def insert_task(self, payload: dict) -> ReasoningDistillationTask:
        with self.session_scope() as session:
            task = ReasoningDistillationTask(**payload)
            session.add(task)
            session.flush()
            session.refresh(task)
            return task

    def get_task_by_id(self, task_id: int, user_id: Optional[int] = None) -> Optional[ReasoningDistillationTask]:
        with self.session_scope() as session:
            stmt = select(ReasoningDistillationTask).where(ReasoningDistillationTask.id == int(task_id))
            if user_id is not None:
                stmt = stmt.where(ReasoningDistillationTask.user_id == int(user_id))
            return session.execute(stmt).scalars().first()

    def list_tasks(self, limit: int = 20, user_id: Optional[int] = None):
        safe_limit = max(1, min(int(limit), 200))
        with self.session_scope() as session:
            stmt = select(ReasoningDistillationTask).order_by(ReasoningDistillationTask.id.desc()).limit(safe_limit)
            if user_id is not None:
                stmt = stmt.where(ReasoningDistillationTask.user_id == int(user_id))
            return list(session.execute(stmt).scalars().all())

    def mark_started(self, task_id: int) -> Optional[ReasoningDistillationTask]:
        with self.session_scope() as session:
            task = session.get(ReasoningDistillationTask, int(task_id))
            if task is None:
                return None
            now = datetime.utcnow()
            task.started_time = now
            task.update_time = now
            session.add(task)
            session.flush()
            session.refresh(task)
            return task

    def update_output_file_path(self, task_id: int, output_file_path: str) -> Optional[ReasoningDistillationTask]:
        with self.session_scope() as session:
            task = session.get(ReasoningDistillationTask, int(task_id))
            if task is None:
                return None
            task.output_file_path = str(output_file_path)
            task.update_time = datetime.utcnow()
            session.add(task)
            session.flush()
            session.refresh(task)
            return task

    def update_progress(
        self,
        task_id: int,
        *,
        processed_items: int,
        distilled_samples: int,
        avg_tokens: int,
        error_message: Optional[str] = None,
    ) -> Optional[ReasoningDistillationTask]:
        with self.session_scope() as session:
            task = session.get(ReasoningDistillationTask, int(task_id))
            if task is None:
                return None
            now = datetime.utcnow()
            task.processed_items = int(processed_items)
            task.distilled_samples = int(distilled_samples)
            task.avg_tokens = int(avg_tokens)
            task.error_message = error_message
            task.update_time = now
            session.add(task)
            session.flush()
            session.refresh(task)
            return task

    def update_generated_dataset(self, task_id: int, generated_dataset_id: int) -> Optional[ReasoningDistillationTask]:
        with self.session_scope() as session:
            task = session.get(ReasoningDistillationTask, int(task_id))
            if task is None:
                return None
            task.generated_dataset_id = int(generated_dataset_id)
            task.update_time = datetime.utcnow()
            session.add(task)
            session.flush()
            session.refresh(task)
            return task

    def mark_finished(
        self,
        task_id: int,
        *,
        processed_items: int,
        distilled_samples: int,
        avg_tokens: int,
        error_message: Optional[str] = None,
    ) -> Optional[ReasoningDistillationTask]:
        with self.session_scope() as session:
            task = session.get(ReasoningDistillationTask, int(task_id))
            if task is None:
                return None
            now = datetime.utcnow()
            task.processed_items = int(processed_items)
            task.distilled_samples = int(distilled_samples)
            task.avg_tokens = int(avg_tokens)
            task.finished_time = now
            task.error_message = error_message
            task.update_time = now
            session.add(task)
            session.flush()
            session.refresh(task)
            return task
