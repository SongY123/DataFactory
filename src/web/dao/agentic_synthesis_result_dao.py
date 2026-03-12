from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from .base_dao import BaseDAO
from ..entity.model import AgenticSynthesisResult


class AgenticSynthesisResultDAO(BaseDAO):
    def insert_result(self, payload: Dict[str, Any]) -> AgenticSynthesisResult:
        with self.session_scope() as session:
            item = AgenticSynthesisResult(**payload)
            session.add(item)
            session.flush()
            session.refresh(item)
            return item

    def get_result_by_id(self, result_id: int, user_id: Optional[int] = None) -> Optional[AgenticSynthesisResult]:
        with self.session_scope() as session:
            stmt = select(AgenticSynthesisResult).where(AgenticSynthesisResult.id == int(result_id))
            if user_id is not None:
                stmt = stmt.where(AgenticSynthesisResult.user_id == int(user_id))
            return session.execute(stmt).scalars().first()

    def list_results_by_task(self, task_id: int, user_id: int, limit: int = 100) -> List[AgenticSynthesisResult]:
        safe_limit = max(1, min(int(limit), 1000))
        with self.session_scope() as session:
            stmt = (
                select(AgenticSynthesisResult)
                .where(AgenticSynthesisResult.task_id == int(task_id), AgenticSynthesisResult.user_id == int(user_id))
                .order_by(AgenticSynthesisResult.id.asc())
                .limit(safe_limit)
            )
            return list(session.execute(stmt).scalars().all())

    def count_results_by_task(self, task_id: int, user_id: int) -> int:
        with self.session_scope() as session:
            stmt = select(AgenticSynthesisResult).where(
                AgenticSynthesisResult.task_id == int(task_id),
                AgenticSynthesisResult.user_id == int(user_id),
            )
            return len(list(session.execute(stmt).scalars().all()))

    @staticmethod
    def to_payload(row: AgenticSynthesisResult) -> Dict[str, Any]:
        data = row.to_dict()
        return {
            **data,
            "evaluation": data.get("evaluation") if isinstance(data.get("evaluation"), dict) else json.loads("{}"),
        }
