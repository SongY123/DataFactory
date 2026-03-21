from __future__ import annotations

import json
from datetime import datetime
from typing import List, Optional

from sqlalchemy import select

from .base_dao import BaseDAO
from ..entity.model import AgenticSynthesisTask


class AgenticSynthesisTaskDAO(BaseDAO):
    def insert_task(
        self,
        user_id: int,
        dataset_id: int,
        prompt_text: str,
        action_tags: List[str],
        llm_api_key: str,
        llm_base_url: str,
        llm_model_name: str,
        parallelism: int,
        llm_params_json: Optional[str],
        output_file_path: str,
        total_workspaces: int,
    ) -> AgenticSynthesisTask:
        with self.session_scope() as session:
            task = AgenticSynthesisTask(
                user_id=int(user_id),
                dataset_id=int(dataset_id),
                prompt_text=prompt_text,
                action_tags_json=json.dumps(action_tags, ensure_ascii=False),
                llm_api_key=llm_api_key,
                llm_base_url=llm_base_url,
                llm_model_name=llm_model_name,
                parallelism=int(parallelism),
                llm_params_json=llm_params_json,
                output_file_path=output_file_path,
                total_workspaces=int(total_workspaces),
                processed_workspaces=0,
            )
            session.add(task)
            session.flush()
            session.refresh(task)
            return task

    def get_task_by_id(self, task_id: int, user_id: Optional[int] = None) -> Optional[AgenticSynthesisTask]:
        with self.session_scope() as session:
            stmt = select(AgenticSynthesisTask).where(AgenticSynthesisTask.id == int(task_id))
            if user_id is not None:
                stmt = stmt.where(AgenticSynthesisTask.user_id == int(user_id))
            return session.execute(stmt).scalars().first()

    def list_tasks(self, limit: int = 20, user_id: Optional[int] = None) -> List[AgenticSynthesisTask]:
        safe_limit = max(1, min(int(limit), 200))
        with self.session_scope() as session:
            stmt = select(AgenticSynthesisTask).order_by(AgenticSynthesisTask.id.desc()).limit(safe_limit)
            if user_id is not None:
                stmt = stmt.where(AgenticSynthesisTask.user_id == int(user_id))
            return list(session.execute(stmt).scalars().all())

    def mark_started(self, task_id: int) -> Optional[AgenticSynthesisTask]:
        with self.session_scope() as session:
            task = session.get(AgenticSynthesisTask, int(task_id))
            if task is None:
                return None
            now = datetime.utcnow()
            task.started_time = now
            task.update_time = now
            session.add(task)
            session.flush()
            session.refresh(task)
            return task

    def update_output_file_path(self, task_id: int, output_file_path: str) -> Optional[AgenticSynthesisTask]:
        with self.session_scope() as session:
            task = session.get(AgenticSynthesisTask, int(task_id))
            if task is None:
                return None
            task.output_file_path = str(output_file_path)
            task.update_time = datetime.utcnow()
            session.add(task)
            session.flush()
            session.refresh(task)
            return task

    def update_generated_dataset(self, task_id: int, generated_dataset_id: int) -> Optional[AgenticSynthesisTask]:
        with self.session_scope() as session:
            task = session.get(AgenticSynthesisTask, int(task_id))
            if task is None:
                return None
            task.generated_dataset_id = int(generated_dataset_id)
            task.update_time = datetime.utcnow()
            session.add(task)
            session.flush()
            session.refresh(task)
            return task

    def update_progress(
        self,
        task_id: int,
        processed_workspaces: int,
        error_message: Optional[str] = None,
    ) -> Optional[AgenticSynthesisTask]:
        with self.session_scope() as session:
            task = session.get(AgenticSynthesisTask, int(task_id))
            if task is None:
                return None
            now = datetime.utcnow()
            task.processed_workspaces = int(processed_workspaces)
            task.error_message = error_message
            task.update_time = now
            session.add(task)
            session.flush()
            session.refresh(task)
            return task

    def mark_finished(
        self,
        task_id: int,
        processed_workspaces: int,
        error_message: Optional[str] = None,
    ) -> Optional[AgenticSynthesisTask]:
        with self.session_scope() as session:
            task = session.get(AgenticSynthesisTask, int(task_id))
            if task is None:
                return None
            now = datetime.utcnow()
            task.processed_workspaces = int(processed_workspaces)
            task.finished_time = now
            task.error_message = error_message
            task.update_time = now
            session.add(task)
            session.flush()
            session.refresh(task)
            return task
