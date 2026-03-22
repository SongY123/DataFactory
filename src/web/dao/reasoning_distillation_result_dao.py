from __future__ import annotations

from typing import Optional

from sqlalchemy import select

from .base_dao import BaseDAO
from ..entity.model import ReasoningDistillationResult


class ReasoningDistillationResultDAO(BaseDAO):
    def insert_result(self, payload: dict) -> ReasoningDistillationResult:
        with self.session_scope() as session:
            item = ReasoningDistillationResult(**payload)
            session.add(item)
            session.flush()
            session.refresh(item)
            return item

    def list_results_by_task(self, task_id: int, user_id: int, limit: int = 200):
        safe_limit = max(1, min(int(limit), 1000))
        with self.session_scope() as session:
            stmt = (
                select(ReasoningDistillationResult)
                .where(
                    ReasoningDistillationResult.task_id == int(task_id),
                    ReasoningDistillationResult.user_id == int(user_id),
                )
                .order_by(ReasoningDistillationResult.id.asc())
                .limit(safe_limit)
            )
            return list(session.execute(stmt).scalars().all())

    def get_result_by_id(self, result_id: int, user_id: Optional[int] = None) -> Optional[ReasoningDistillationResult]:
        with self.session_scope() as session:
            stmt = select(ReasoningDistillationResult).where(ReasoningDistillationResult.id == int(result_id))
            if user_id is not None:
                stmt = stmt.where(ReasoningDistillationResult.user_id == int(user_id))
            return session.execute(stmt).scalars().first()

    def count_results_by_task(self, task_id: int, user_id: int) -> int:
        with self.session_scope() as session:
            stmt = select(ReasoningDistillationResult).where(
                ReasoningDistillationResult.task_id == int(task_id),
                ReasoningDistillationResult.user_id == int(user_id),
            )
            return len(list(session.execute(stmt).scalars().all()))
