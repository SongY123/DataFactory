from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import request as urllib_request

from ..dao import AgenticSynthesisTaskDAO
from utils.logger import logger


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class AgenticSynthesisService:
    def __init__(self, task_dao: Optional[AgenticSynthesisTaskDAO] = None) -> None:
        self.task_dao = task_dao or AgenticSynthesisTaskDAO()
        self._lock = threading.RLock()
        self._running_threads: Dict[int, threading.Thread] = {}

    def start_task(
        self,
        prompt: str,
        action_tags: List[str],
        llm_api_key: str,
        llm_base_url: str,
        llm_model_name: str,
        datasets: List[Dict[str, Any]],
        output_file_path: Optional[str] = None,
    ) -> Dict:
        dataset_uris = self._extract_dataset_uris(datasets)
        dataset_paths = [self._resolve_path(p) for p in dataset_uris]
        file_list = self._collect_dataset_files(dataset_paths)

        output_path = self._resolve_output_path(output_file_path)
        task = self.task_dao.insert_task(
            prompt_text=prompt,
            action_tags=action_tags,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            llm_model_name=llm_model_name,
            dataset_paths=[str(x) for x in dataset_paths],
            output_file_path=str(output_path),
            total_files=len(file_list),
        )

        thread = threading.Thread(
            target=self._run_task,
            args=(
                int(task.id),
                file_list,
                prompt,
                action_tags,
                llm_api_key,
                llm_base_url,
                llm_model_name,
                output_path,
            ),
            daemon=True,
            name=f"agentic-synthesis-task-{task.id}",
        )
        with self._lock:
            self._running_threads[int(task.id)] = thread
        thread.start()

        return self._sanitize_task_payload(task.to_dict())

    def get_task(self, task_id: int) -> Optional[Dict]:
        task = self.task_dao.get_task_by_id(task_id)
        if task is None:
            return None
        payload = self._sanitize_task_payload(task.to_dict())
        payload["is_running"] = self._is_task_running(task_id)
        return payload

    def list_tasks(self, limit: int = 20) -> List[Dict]:
        rows = self.task_dao.list_tasks(limit=limit)
        result: List[Dict] = []
        for row in rows:
            payload = self._sanitize_task_payload(row.to_dict())
            payload["is_running"] = self._is_task_running(int(row.id))
            result.append(payload)
        return result

    def _run_task(
        self,
        task_id: int,
        file_list: List[Path],
        prompt: str,
        action_tags: List[str],
        llm_api_key: str,
        llm_base_url: str,
        llm_model_name: str,
        output_path: Path,
    ) -> None:
        results: List[Dict] = []
        processed = 0
        success = 0
        failed = 0
        last_error_message: Optional[str] = None

        try:
            self.task_dao.mark_running(task_id=task_id)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            for file_path in file_list:
                processed += 1
                try:
                    source_content = self._read_dataset_file(file_path)
                    synthesized = self._call_llm(
                        prompt=prompt,
                        action_tags=action_tags,
                        source_file_path=file_path,
                        source_content=source_content,
                        api_key=llm_api_key,
                        base_url=llm_base_url,
                        model_name=llm_model_name,
                    )
                    results.append(
                        {
                            "task_id": task_id,
                            "source_file": str(file_path),
                            "synthesized_data": synthesized,
                        }
                    )
                    success += 1
                except Exception as exc:
                    failed += 1
                    last_error_message = str(exc)
                    logger.exception("Agentic synthesis file process failed. task_id=%s file=%s", task_id, file_path)
                    results.append(
                        {
                            "task_id": task_id,
                            "source_file": str(file_path),
                            "error": str(exc),
                        }
                    )
                finally:
                    self.task_dao.update_progress(
                        task_id=task_id,
                        processed_files=processed,
                        success_files=success,
                        failed_files=failed,
                    )

            with output_path.open("w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            final_status = "completed"
            final_error_message = None
            if failed > 0 and success == 0:
                final_status = "failed"
                final_error_message = f"all files failed. last_error={last_error_message or 'unknown'}"

            self.task_dao.mark_finished(
                task_id=task_id,
                status=final_status,
                processed_files=processed,
                success_files=success,
                failed_files=failed,
                error_message=final_error_message,
            )
        except Exception as exc:
            logger.exception("Agentic synthesis task failed. task_id=%s", task_id)
            self.task_dao.mark_finished(
                task_id=task_id,
                status="failed",
                processed_files=processed,
                success_files=success,
                failed_files=failed,
                error_message=str(exc),
            )
        finally:
            with self._lock:
                self._running_threads.pop(task_id, None)

    def _is_task_running(self, task_id: int) -> bool:
        with self._lock:
            thread = self._running_threads.get(int(task_id))
            if thread is None:
                return False
            return bool(thread.is_alive())

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
    def _extract_dataset_uris(datasets: List[Dict[str, Any]]) -> List[str]:
        uris: List[str] = []
        for item in datasets or []:
            if not isinstance(item, dict):
                raise ValueError("datasets item must be an object")
            raw_uri = str(item.get("uri") or "").strip()
            if not raw_uri:
                raise ValueError("datasets item uri must not be empty")
            uris.append(raw_uri)

        if not uris:
            raise ValueError("datasets must contain at least one valid uri")
        return uris

    @staticmethod
    def _collect_dataset_files(dataset_paths: List[Path]) -> List[Path]:
        all_files: List[Path] = []
        for dataset_path in dataset_paths:
            if dataset_path.is_file():
                all_files.append(dataset_path)
                continue
            for child in sorted(dataset_path.rglob("*")):
                if child.is_file():
                    all_files.append(child)
        if not all_files:
            raise ValueError("no files found in datasets")
        return all_files

    @staticmethod
    def _resolve_output_path(output_file_path: Optional[str]) -> Path:
        if str(output_file_path or "").strip():
            path = Path(str(output_file_path).strip())
            if path.is_absolute():
                return path
            return PROJECT_ROOT / path

        default_dir = PROJECT_ROOT / "logs" / "agentic_synthesis"
        default_dir.mkdir(parents=True, exist_ok=True)
        return default_dir / f"synthesis_{uuid.uuid4().hex}.json"

    @staticmethod
    def _sanitize_task_payload(payload: Dict) -> Dict:
        masked = dict(payload or {})
        if "llm_api_key" in masked:
            masked["llm_api_key"] = "***"
        return masked

    @staticmethod
    def _read_dataset_file(file_path: Path) -> str:
        try:
            return file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return file_path.read_text(encoding="utf-8", errors="ignore")

    def _call_llm(
        self,
        prompt: str,
        action_tags: List[str],
        source_file_path: Path,
        source_content: str,
        api_key: str,
        base_url: str,
        model_name: str,
    ) -> str:
        endpoint = self._build_chat_endpoint(base_url)
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an agentic data synthesis assistant.",
                },
                {
                    "role": "user",
                    "content": self._build_user_prompt(
                        prompt=prompt,
                        action_tags=action_tags,
                        source_file_path=source_file_path,
                        source_content=source_content,
                    ),
                },
            ],
            "temperature": 0.2,
        }

        req = urllib_request.Request(
            url=endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        with urllib_request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8", errors="ignore")

        parsed = json.loads(body)
        choices = parsed.get("choices") or []
        if not choices:
            raise RuntimeError("LLM response has no choices")

        first = choices[0] or {}
        message = first.get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            content = "\n".join(str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in content)

        text_content = str(content or "").strip()
        if not text_content:
            raise RuntimeError("LLM response content is empty")
        return text_content

    @staticmethod
    def _build_chat_endpoint(base_url: str) -> str:
        normalized = str(base_url or "").strip().rstrip("/")
        if normalized.endswith("/chat/completions"):
            return normalized
        if normalized.endswith("/v1"):
            return f"{normalized}/chat/completions"
        return f"{normalized}/v1/chat/completions"

    @staticmethod
    def _build_user_prompt(
        prompt: str,
        action_tags: List[str],
        source_file_path: Path,
        source_content: str,
    ) -> str:
        return (
            f"{prompt}\n\n" 
            f"Source file path:\n{source_file_path}\n\n"
            f"Source file content:\n{source_content}"
        )
