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
        prompt_text: str,
        action_tags: List[str],
        llm_api_key: str,
        llm_base_url: str,
        llm_model_name: str,
        dataset_paths: List[str],
        output_file_path: str,
        total_files: int,
    ) -> AgenticSynthesisTask:
        with self.session_scope() as session:
            task = AgenticSynthesisTask(
                prompt_text=prompt_text,
                action_tags_json=json.dumps(action_tags, ensure_ascii=False),
                llm_api_key=llm_api_key,
                llm_base_url=llm_base_url,
                llm_model_name=llm_model_name,
                dataset_paths_json=json.dumps(dataset_paths, ensure_ascii=False),
                output_file_path=output_file_path,
                status="pending",
                total_files=int(total_files),
                processed_files=0,
                success_files=0,
                failed_files=0,
            )
            session.add(task)
            session.flush()
            session.refresh(task)
            return task

    def get_task_by_id(self, task_id: int) -> Optional[AgenticSynthesisTask]:
        with self.session_scope() as session:
            return session.get(AgenticSynthesisTask, int(task_id))

    def list_tasks(self, limit: int = 20) -> List[AgenticSynthesisTask]:
        safe_limit = max(1, min(int(limit), 200))
        with self.session_scope() as session:
            stmt = select(AgenticSynthesisTask).order_by(AgenticSynthesisTask.id.desc()).limit(safe_limit)
            return list(session.execute(stmt).scalars().all())

    def mark_running(self, task_id: int) -> Optional[AgenticSynthesisTask]:
        with self.session_scope() as session:
            task = session.get(AgenticSynthesisTask, int(task_id))
            if task is None:
                return None
            now = datetime.utcnow()
            task.status = "running"
            task.started_time = now
            task.update_time = now
            session.add(task)
            session.flush()
            session.refresh(task)
            return task

    def update_progress(
        self,
        task_id: int,
        processed_files: int,
        success_files: int,
        failed_files: int,
        error_message: Optional[str] = None,
    ) -> Optional[AgenticSynthesisTask]:
        with self.session_scope() as session:
            task = session.get(AgenticSynthesisTask, int(task_id))
            if task is None:
                return None
            now = datetime.utcnow()
            task.processed_files = int(processed_files)
            task.success_files = int(success_files)
            task.failed_files = int(failed_files)
            task.error_message = error_message
            task.update_time = now
            session.add(task)
            session.flush()
            session.refresh(task)
            return task

    def mark_finished(
        self,
        task_id: int,
        status: str,
        processed_files: int,
        success_files: int,
        failed_files: int,
        error_message: Optional[str] = None,
    ) -> Optional[AgenticSynthesisTask]:
        with self.session_scope() as session:
            task = session.get(AgenticSynthesisTask, int(task_id))
            if task is None:
                return None
            now = datetime.utcnow()
            task.status = status
            task.processed_files = int(processed_files)
            task.success_files = int(success_files)
            task.failed_files = int(failed_files)
            task.finished_time = now
            task.error_message = error_message
            task.update_time = now
            session.add(task)
            session.flush()
            session.refresh(task)
            return task
