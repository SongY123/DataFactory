from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..dao import AgenticSynthesisResultDAO, AgenticSynthesisTaskDAO
from ..dao.dataset_dao import DatasetDAO
from utils.logger import logger


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_QUESTIONS = [
    "How many heads of the departments are older than 56 ?",
    "List all department heads and summarize their age distribution.",
]


class AgenticSynthesisService:
    def __init__(
        self,
        task_dao: Optional[AgenticSynthesisTaskDAO] = None,
        dataset_dao: Optional[DatasetDAO] = None,
        result_dao: Optional[AgenticSynthesisResultDAO] = None,
    ) -> None:
        self.task_dao = task_dao or AgenticSynthesisTaskDAO()
        self.dataset_dao = dataset_dao or DatasetDAO()
        self.result_dao = result_dao or AgenticSynthesisResultDAO()
        self._lock = threading.RLock()
        self._running_threads: Dict[int, threading.Thread] = {}

    def start_task(
        self,
        user_id: int,
        dataset_id: int,
        prompt: str,
        action_tags: List[str],
        llm_api_key: str,
        llm_base_url: str,
        llm_model_name: str,
    ) -> Dict:
        dataset = self.dataset_dao.get_dataset_by_id(dataset_id=dataset_id, user_id=user_id)
        if dataset is None:
            raise ValueError("dataset not found for current user")
        dataset_root = self._resolve_path(str(dataset.file_path or ""))
        workspaces = self._collect_direct_workspaces(dataset_root)
        if not workspaces:
            raise ValueError("dataset has no direct workspace folders")

        task = self.task_dao.insert_task(
            user_id=user_id,
            dataset_id=dataset_id,
            prompt_text=prompt,
            action_tags=action_tags,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            llm_model_name=llm_model_name,
            output_file_path="__pending__",
            total_workspaces=len(workspaces),
        )
        output_path = self._resolve_output_path(user_id=user_id, task_id=int(task.id))
        updated_task = self.task_dao.update_output_file_path(task_id=int(task.id), output_file_path=str(output_path))
        if updated_task is not None:
            task = updated_task

        thread = threading.Thread(
            target=self._run_task,
            args=(
                int(task.id),
                int(user_id),
                int(dataset_id),
                workspaces,
                output_path,
            ),
            daemon=True,
            name=f"agentic-synthesis-task-{task.id}",
        )
        with self._lock:
            self._running_threads[int(task.id)] = thread
        thread.start()

        return self._sanitize_task_payload(task.to_dict())

    def get_task(self, task_id: int, user_id: int) -> Optional[Dict]:
        task = self.task_dao.get_task_by_id(task_id=task_id, user_id=user_id)
        if task is None:
            return None
        payload = self._sanitize_task_payload(task.to_dict())
        payload["result_count"] = self.result_dao.count_results_by_task(task_id=task_id, user_id=user_id)
        return payload

    def list_tasks(self, user_id: int, limit: int = 20) -> List[Dict]:
        rows = self.task_dao.list_tasks(limit=limit, user_id=user_id)
        result: List[Dict] = []
        for row in rows:
            payload = self._sanitize_task_payload(row.to_dict())
            payload["result_count"] = self.result_dao.count_results_by_task(task_id=int(row.id), user_id=user_id)
            result.append(payload)
        return result

    def list_results(self, user_id: int, task_id: int, limit: int = 200) -> List[Dict]:
        task = self.task_dao.get_task_by_id(task_id=task_id, user_id=user_id)
        if task is None:
            raise ValueError("task not found")
        rows = self.result_dao.list_results_by_task(task_id=task_id, user_id=user_id, limit=limit)
        return [x.to_dict() for x in rows]

    def get_result(self, user_id: int, result_id: int) -> Optional[Dict]:
        row = self.result_dao.get_result_by_id(result_id=result_id, user_id=user_id)
        if row is None:
            return None
        return row.to_dict()

    def _run_task(
        self,
        task_id: int,
        user_id: int,
        dataset_id: int,
        workspaces: List[Path],
        output_path: Path,
    ) -> None:
        processed_workspaces = 0

        try:
            self.task_dao.mark_started(task_id=task_id)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fixed_template = self._load_fixed_result_template()
            with output_path.open("w", encoding="utf-8") as writer:
                for workspace in workspaces:
                    workspace_name = workspace.name
                    for question in DEFAULT_QUESTIONS:
                        record = self._build_fixed_result_record(
                            task_id=task_id,
                            dataset_id=dataset_id,
                            workspace_name=workspace_name,
                            question=question,
                            fixed_template=fixed_template,
                        )
                        self.result_dao.insert_result(
                            {
                                "task_id": task_id,
                                "user_id": user_id,
                                "dataset_id": dataset_id,
                                "workspace_name": workspace_name,
                                "question": record["question"],
                                "trajectory": record["trajectory"],
                                "evaluation_json": json.dumps(record["evaluation"], ensure_ascii=False),
                                "status": record["status"],
                                "error_message": record.get("error"),
                            }
                        )
                        writer.write(json.dumps(record, ensure_ascii=False) + "\n")

                    processed_workspaces += 1
                    self.task_dao.update_progress(
                        task_id=task_id,
                        processed_workspaces=processed_workspaces,
                    )

            self.task_dao.mark_finished(
                task_id=task_id,
                processed_workspaces=processed_workspaces,
                error_message=None,
            )
        except Exception as exc:
            logger.exception("Agentic synthesis task failed. task_id=%s", task_id)
            self.task_dao.mark_finished(
                task_id=task_id,
                processed_workspaces=processed_workspaces,
                error_message=str(exc),
            )
        finally:
            with self._lock:
                self._running_threads.pop(task_id, None)

    @staticmethod
    def _resolve_path(path_str: str) -> Path:
        raw = str(path_str or "").strip()
        if not raw:
            raise ValueError("dataset path must not be empty")
        path = Path(raw)
        if path.is_absolute():
            resolved = path
        else:
            resolved = PROJECT_ROOT / path
        if not resolved.exists():
            raise FileNotFoundError(f"dataset path does not exist: {resolved}")
        return resolved

    @staticmethod
    def _collect_direct_workspaces(dataset_root: Path) -> List[Path]:
        if dataset_root.is_file():
            raise ValueError("dataset root must be a directory containing workspace folders")
        children = sorted([x for x in dataset_root.iterdir() if x.is_dir()])
        if children:
            return children
        if any(x.is_file() for x in dataset_root.iterdir()):
            return [dataset_root]
        return []

    @staticmethod
    def _resolve_output_path(user_id: int, task_id: int) -> Path:
        output_dir = PROJECT_ROOT / "output" / str(int(user_id)) / str(int(task_id))
        output_dir.mkdir(parents=True, exist_ok=True)
        return (output_dir / "result.jsonl").resolve()

    @staticmethod
    def _sanitize_task_payload(payload: Dict) -> Dict:
        masked = dict(payload or {})
        if "llm_api_key" in masked:
            masked["llm_api_key"] = "***"
        return masked

    @staticmethod
    def _load_fixed_result_template() -> Dict[str, Any]:
        sample_path = PROJECT_ROOT / "tmp" / "result_example.json"
        if sample_path.exists() and sample_path.is_file():
            try:
                parsed = json.loads(sample_path.read_text(encoding="utf-8"))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        return {
            "status": "completed",
            "question": "How many heads of the departments are older than 56 ?",
            "trajectory": "<Analyze>Fixed trajectory placeholder.</Analyze>",
            "evaluation": {
                "difficulty": 4,
                "quality": 5,
                "ability": "Data Analysis",
            },
            "error": None,
        }

    @staticmethod
    def _build_fixed_result_record(
        task_id: int,
        dataset_id: int,
        workspace_name: str,
        question: str,
        fixed_template: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "task_id": task_id,
            "dataset_id": dataset_id,
            "workspace_name": workspace_name,
            "status": str(fixed_template.get("status") or "completed"),
            "question": question,
            "trajectory": str(fixed_template.get("trajectory") or ""),
            "evaluation": fixed_template.get("evaluation") if isinstance(fixed_template.get("evaluation"), dict) else {},
            "error": fixed_template.get("error"),
        }
