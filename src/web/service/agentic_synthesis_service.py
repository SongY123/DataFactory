from __future__ import annotations

import base64
import ast
import csv
import json
import mimetypes
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple
from urllib import request as urllib_request
from urllib.error import URLError

from ..dao import AgenticSynthesisResultDAO, AgenticSynthesisTaskDAO
from ..dao.dataset_dao import DatasetDAO
from utils.logger import logger
from utils.pdf_support import get_pdf_reader
from .dataset_service import DatasetService


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_QUESTION_COUNT = 3
MAX_FILE_OBSERVATIONS = 120
MAX_TEXT_PREVIEW_CHARS = 2000
MAX_SANDBOX_OUTPUT_CHARS = 4000
MAX_TRAJECTORY_STEPS = 6
OPEN_ENDED_RATIO = 0.5
MAX_MODEL_AUDIT_CHARS = 2000
FIXED_TASK_PROMPT = (
    "Generate one grounded data-analysis question for the current workspace, "
    "solve it via executable Python steps, and provide trajectory plus evaluation."
)
FIXED_ACTION_TAGS = ["Analyze", "Understand", "Code", "Execute", "Answer"]
SANDBOX_ALLOWED_LIBS = [
    "python-stdlib",
    "json",
    "csv",
    "sqlite3",
    "re",
    "math",
    "statistics",
    "pandas",
    "numpy",
    "openpyxl",
    "pyarrow",
    "pypdf",
]


class ModelOutputParseError(ValueError):
    def __init__(self, message: str, raw_output: str = "") -> None:
        super().__init__(message)
        self.raw_output = str(raw_output or "")


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
        self.dataset_service = DatasetService(dataset_dao=self.dataset_dao)
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
        effective_prompt = str(prompt or "").strip() or FIXED_TASK_PROMPT
        effective_action_tags = [str(item or "").strip() for item in (action_tags or []) if str(item or "").strip()]
        if not effective_action_tags:
            effective_action_tags = list(FIXED_ACTION_TAGS)
        dataset_root = self._resolve_path(str(dataset.file_path or ""))
        workspaces = self._collect_direct_workspaces(dataset_root)
        if not workspaces:
            raise ValueError("dataset has no direct workspace folders")

        task = self.task_dao.insert_task(
            user_id=user_id,
            dataset_id=dataset_id,
            prompt_text=effective_prompt,
            action_tags=effective_action_tags,
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
                effective_prompt,
                effective_action_tags,
                str(llm_api_key or "").strip(),
                str(llm_base_url or "").strip(),
                str(llm_model_name or "").strip(),
                output_path,
            ),
            daemon=True,
            name=f"agentic-synthesis-task-{task.id}",
        )
        with self._lock:
            self._running_threads[int(task.id)] = thread
        thread.start()

        return self._enrich_task_payload(task.to_dict(), user_id=int(user_id))

    def get_task(self, task_id: int, user_id: int) -> Optional[Dict]:
        task = self.task_dao.get_task_by_id(task_id=task_id, user_id=user_id)
        if task is None:
            return None
        payload = self._enrich_task_payload(task.to_dict(), user_id=user_id)
        payload["result_count"] = self.result_dao.count_results_by_task(task_id=task_id, user_id=user_id)
        return payload

    def list_tasks(self, user_id: int, limit: int = 20) -> List[Dict]:
        rows = self.task_dao.list_tasks(limit=limit, user_id=user_id)
        result: List[Dict] = []
        for row in rows:
            payload = self._enrich_task_payload(row.to_dict(), user_id=user_id)
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
        prompt: str,
        action_tags: List[str],
        llm_api_key: str,
        llm_base_url: str,
        llm_model_name: str,
        output_path: Path,
    ) -> None:
        processed_workspaces = 0
        generated_samples: List[Dict[str, Any]] = []
        generated_records = 0

        try:
            self.task_dao.mark_started(task_id=task_id)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            dataset_output_path = output_path.with_name("trajectory_dataset.jsonl")
            manifest_path = output_path.with_name("manifest.json")
            with output_path.open("w", encoding="utf-8") as writer, dataset_output_path.open("w", encoding="utf-8") as dataset_writer:
                for workspace in workspaces:
                    tmp_dir = self._prepare_workspace_copy(task_id=task_id, workspace=workspace)
                    try:
                        workspace_name = workspace.name
                        workspace_context = self._build_workspace_context(tmp_dir)
                        workspace_audit: Optional[str] = None
                        try:
                            questions = self._generate_questions(
                                prompt=prompt,
                                action_tags=action_tags,
                                workspace_name=workspace_name,
                                workspace_context=workspace_context,
                                api_key=llm_api_key,
                                base_url=llm_base_url,
                                model_name=llm_model_name,
                            )
                        except Exception as q_exc:
                            workspace_audit = self._build_audit_text("question_generation", q_exc)
                            logger.warning(
                                "Question generation failed, use fallback. task_id=%s workspace=%s err=%s",
                                task_id,
                                workspace_name,
                                q_exc,
                            )
                            if workspace_audit:
                                logger.warning(
                                    "Question generation raw output (truncated). task_id=%s workspace=%s audit=%s",
                                    task_id,
                                    workspace_name,
                                    workspace_audit,
                                )
                            questions = self._build_fallback_questions(workspace_context=workspace_context)

                        if not questions:
                            questions = self._build_fallback_questions(workspace_context=workspace_context)

                        for question in questions:
                            trajectory, answer_text, solve_error, solve_audit = self._synthesize_trajectory(
                                question=question,
                                workspace_name=workspace_name,
                                workspace_context=workspace_context,
                                workspace_dir=tmp_dir,
                                api_key=llm_api_key,
                                base_url=llm_base_url,
                                model_name=llm_model_name,
                            )
                            evaluation = self._evaluate_result(
                                question=question,
                                trajectory=trajectory,
                                answer_text=answer_text,
                                workspace_name=workspace_name,
                                workspace_context=workspace_context,
                                api_key=llm_api_key,
                                base_url=llm_base_url,
                                model_name=llm_model_name,
                            )

                            status = "completed"
                            error_text = None
                            if solve_error:
                                status = "failed"
                                error_text = solve_error

                            audit_text = solve_audit or workspace_audit

                            record = {
                                "task_id": task_id,
                                "dataset_id": dataset_id,
                                "workspace_name": workspace_name,
                                "status": status,
                                "question": question,
                                "trajectory": trajectory,
                                "evaluation": evaluation,
                                "error": error_text,
                                "model_output_audit": audit_text,
                            }

                            self.result_dao.insert_result(
                                {
                                    "task_id": task_id,
                                    "user_id": user_id,
                                    "dataset_id": dataset_id,
                                    "workspace_name": workspace_name,
                                    "question": question,
                                    "trajectory": trajectory,
                                    "evaluation_json": json.dumps(evaluation, ensure_ascii=False),
                                    "status": status,
                                    "error_message": error_text,
                                    "model_output_audit": audit_text,
                                }
                            )
                            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                            writer.flush()
                            if status == "completed":
                                dataset_record = {
                                    "question": question,
                                    "trajectory": trajectory,
                                    "evaluation": evaluation,
                                    "metadata": {
                                        "task_id": task_id,
                                        "dataset_id": dataset_id,
                                        "workspace_name": workspace_name,
                                        "source_question": question,
                                    },
                                }
                                dataset_writer.write(json.dumps(dataset_record, ensure_ascii=False) + "\n")
                                dataset_writer.flush()
                                generated_records += 1
                                if len(generated_samples) < DatasetService.SAMPLE_PREVIEW_LIMIT:
                                    generated_samples.append(dataset_record)
                    except Exception as ws_exc:
                        logger.exception(
                            "Workspace synthesis failed. task_id=%s workspace=%s",
                            task_id,
                            workspace.name,
                        )
                        failed_record = {
                            "task_id": task_id,
                            "dataset_id": dataset_id,
                            "workspace_name": workspace.name,
                            "status": "failed",
                            "question": "Workspace-level synthesis failure",
                            "trajectory": self._wrap_tag("Analyze", "Workspace pipeline failed before question synthesis")
                            + "\n"
                            + self._wrap_tag("Execute", str(ws_exc))
                            + "\n"
                            + self._wrap_tag("Answer", ""),
                            "evaluation": {
                                "difficulty": 1,
                                "quality": 1,
                                "ability": "error-handling",
                            },
                            "error": str(ws_exc),
                                "model_output_audit": self._build_audit_text("workspace_pipeline", ws_exc),
                        }
                        self.result_dao.insert_result(
                            {
                                "task_id": task_id,
                                "user_id": user_id,
                                "dataset_id": dataset_id,
                                "workspace_name": workspace.name,
                                "question": failed_record["question"],
                                "trajectory": failed_record["trajectory"],
                                "evaluation_json": json.dumps(failed_record["evaluation"], ensure_ascii=False),
                                "status": "failed",
                                "error_message": str(ws_exc),
                                "model_output_audit": failed_record.get("model_output_audit"),
                            }
                        )
                        writer.write(json.dumps(failed_record, ensure_ascii=False) + "\n")
                        writer.flush()
                    finally:
                        self._cleanup_workspace_copy(tmp_dir)

                    processed_workspaces += 1
                    self.task_dao.update_progress(
                        task_id=task_id,
                        processed_workspaces=processed_workspaces,
                    )

            source_dataset = self.dataset_dao.get_dataset_by_id(dataset_id=dataset_id, user_id=user_id)
            if generated_records > 0 and source_dataset is not None:
                manifest = {
                    "task_id": task_id,
                    "source_dataset_id": dataset_id,
                    "source_dataset_name": source_dataset.name,
                    "generated_records": generated_records,
                    "total_workspaces": len(workspaces),
                    "processed_workspaces": processed_workspaces,
                    "task_output_path": str(output_path),
                    "dataset_output_path": str(dataset_output_path),
                }
                manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
                generated_dataset = self.dataset_service.register_generated_dataset(
                    user_id=user_id,
                    name=f"{source_dataset.name} Trajectory Synthesized T{task_id}",
                    dataset_type="trajectory",
                    language=str(source_dataset.language or "multi"),
                    source=f"generated://agentic-synthesis/tasks/{task_id}",
                    note=f"Synthesized trajectories generated from dataset {source_dataset.name}",
                    file_path=str(dataset_output_path),
                    file_name=dataset_output_path.name,
                    size=dataset_output_path.stat().st_size,
                    sample_data=generated_samples,
                    origin_stage="trajectory_synthesis",
                    origin_dataset_id=int(source_dataset.id),
                    origin_task_type="trajectory_task",
                    origin_task_id=task_id,
                    generation_meta={
                        **manifest,
                        "manifest_path": str(manifest_path),
                    },
                    status="ready",
                )
                self.task_dao.update_generated_dataset(task_id=task_id, generated_dataset_id=int(generated_dataset["id"]))

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

        # 兼容上传目录结构 uploads/user_id/dataset_id/dataset_name/*
        # 若最外层只有一个目录且没有文件，则自动下钻到该目录继续识别workspace。
        normalized_root = dataset_root
        while True:
            dirs = sorted([x for x in normalized_root.iterdir() if x.is_dir()])
            has_files = any(x.is_file() for x in normalized_root.iterdir())
            if has_files:
                break
            if len(dirs) == 1:
                normalized_root = dirs[0]
                continue
            break

        children = sorted([x for x in normalized_root.iterdir() if x.is_dir()])
        if children:
            return children
        if any(x.is_file() for x in normalized_root.iterdir()):
            return [normalized_root]
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

    def _enrich_task_payload(self, payload: Dict, user_id: int) -> Dict:
        masked = self._sanitize_task_payload(payload)
        generated_dataset_id = int(masked.get("generated_dataset_id") or 0)
        if generated_dataset_id > 0:
            dataset = self.dataset_dao.get_dataset_by_id(dataset_id=generated_dataset_id, user_id=user_id)
            if dataset is not None:
                masked["generated_dataset"] = dataset.to_dict()
        return masked

    @staticmethod
    def _prepare_workspace_copy(task_id: int, workspace: Path) -> Path:
        runtime_root = PROJECT_ROOT / "tmp" / "agentic_synthesis_runtime" / str(int(task_id))
        runtime_root.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"ws_{workspace.name}_", dir=str(runtime_root)))
        if workspace.is_dir():
            dst = tmp_dir / workspace.name
            shutil.copytree(str(workspace), str(dst), dirs_exist_ok=True)
            return dst
        dst = tmp_dir / workspace.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(workspace), str(dst))
        return dst

    @staticmethod
    def _cleanup_workspace_copy(path_value: Path) -> None:
        try:
            shutil.rmtree(str(path_value.parent), ignore_errors=True)
        except Exception:
            return

    def _build_workspace_context(self, workspace_dir: Path) -> Dict[str, Any]:
        files: List[Path] = []
        if workspace_dir.is_dir():
            for child in workspace_dir.rglob("*"):
                if child.is_file():
                    files.append(child)
                if len(files) >= MAX_FILE_OBSERVATIONS:
                    break
        elif workspace_dir.is_file():
            files = [workspace_dir]

        observations: List[Dict[str, Any]] = []
        for path in files:
            rel = self._normalize_relpath(path, workspace_dir)
            observations.append(self._extract_file_observation(path, rel))

        return {
            "workspace_name": workspace_dir.name,
            "workspace_path": str(workspace_dir),
            "file_count": len(files),
            "files": observations,
        }

    @staticmethod
    def _normalize_relpath(path: Path, root: Path) -> str:
        try:
            rel = path.relative_to(root)
            return str(PurePosixPath(*rel.parts))
        except Exception:
            return path.name

    def _extract_file_observation(self, path: Path, rel_path: str) -> Dict[str, Any]:
        suffix = path.suffix.lower()
        size = int(path.stat().st_size)
        mime, _ = mimetypes.guess_type(str(path))

        obs: Dict[str, Any] = {
            "path": rel_path,
            "size": size,
            "suffix": suffix,
            "mime": mime,
        }

        try:
            if suffix in {".txt", ".md", ".py", ".sql", ".yaml", ".yml", ".json", ".jsonl", ".csv", ".log", ".xml"}:
                text = path.read_text(encoding="utf-8", errors="ignore")
                obs["preview"] = text[:MAX_TEXT_PREVIEW_CHARS]
                if suffix == ".csv":
                    rows = []
                    reader = csv.reader(text.splitlines())
                    for row in reader:
                        rows.append(row)
                        if len(rows) >= 5:
                            break
                    obs["table_preview"] = rows
                if suffix == ".json":
                    parsed = json.loads(text) if text.strip() else None
                    obs["json_type"] = type(parsed).__name__ if parsed is not None else "empty"
                if suffix == ".jsonl":
                    records = []
                    for line in text.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            records.append(json.loads(line))
                        except Exception:
                            records.append({"raw": line[:200]})
                        if len(records) >= 5:
                            break
                    obs["jsonl_preview"] = records
                return obs

            if suffix in {".db", ".sqlite", ".sqlite3"}:
                obs["sqlite"] = self._inspect_sqlite(path)
                return obs

            if suffix == ".pdf":
                obs["pdf"] = self._inspect_pdf(path)
                return obs

            binary = path.read_bytes()[:128]
            obs["binary_head_base64"] = base64.b64encode(binary).decode("ascii")
            return obs
        except Exception as exc:
            obs["error"] = str(exc)
            return obs

    @staticmethod
    def _inspect_sqlite(path: Path) -> Dict[str, Any]:
        result: Dict[str, Any] = {"tables": []}
        conn = sqlite3.connect(str(path))
        try:
            cur = conn.cursor()
            rows = cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name LIMIT 20").fetchall()
            table_names = [str(x[0]) for x in rows]
            result["tables"] = table_names
            samples: Dict[str, Any] = {}
            for table in table_names[:5]:
                try:
                    cols = cur.execute(f"PRAGMA table_info('{table}')").fetchall()
                    col_names = [str(c[1]) for c in cols]
                    vals = cur.execute(f"SELECT * FROM '{table}' LIMIT 3").fetchall()
                    samples[table] = {
                        "columns": col_names,
                        "rows": [list(v) for v in vals],
                    }
                except Exception as table_exc:
                    samples[table] = {"error": str(table_exc)}
            result["samples"] = samples
            return result
        finally:
            conn.close()

    @staticmethod
    def _inspect_pdf(path: Path) -> Dict[str, Any]:
        PdfReader = get_pdf_reader()
        reader = PdfReader(str(path))
        page_count = len(reader.pages)
        snippets: List[str] = []
        for page in reader.pages[:2]:
            text = page.extract_text() or ""
            snippets.append(text[:500])
        return {
            "page_count": page_count,
            "snippets": snippets,
        }

    def _generate_questions(
        self,
        prompt: str,
        action_tags: List[str],
        workspace_name: str,
        workspace_context: Dict[str, Any],
        api_key: str,
        base_url: str,
        model_name: str,
    ) -> List[str]:
        required_open_count = self._required_open_question_count(DEFAULT_QUESTION_COUNT)
        system_prompt = (
            "You are a data synthesis question generator. "
            "Generate diverse, solvable questions based only on observed workspace files. "
            "Return strict JSON: {\"questions\": [\"...\"]}."
        )
        user_prompt = {
            "task_prompt": prompt,
            "action_tags": action_tags,
            "workspace_name": workspace_name,
            "workspace_context": workspace_context,
            "required_count": DEFAULT_QUESTION_COUNT,
            "required_open_ended_count": required_open_count,
            "constraints": [
                "Questions must be answerable using workspace files.",
                "Questions must reflect filename and directory-name semantics (e.g., infer likely content from names).",
                "Each question should reference at least one concrete filename or subdirectory name when possible.",
                f"At least {required_open_count} of {DEFAULT_QUESTION_COUNT} questions must be open-ended analysis tasks.",
                "Output only bare question text. Never include prefixes like 'Question:' or sections like 'Trajectory:'.",
                "Use concise English questions.",
                "Do not include markdown.",
            ],
        }
        content = self._chat_completion(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            system_prompt=system_prompt,
            user_payload=user_prompt,
        )
        questions_raw: Optional[List[Any]] = None
        try:
            payload = self._extract_json_object(content)
            parsed = payload.get("questions") if isinstance(payload, dict) else None
            if isinstance(parsed, list):
                questions_raw = parsed
        except ModelOutputParseError:
            questions_raw = self._parse_questions_from_plain_text(content)

        if not isinstance(questions_raw, list):
            return self._ensure_question_diversity([], workspace_context=workspace_context)
        cleaned: List[str] = []
        for q in questions_raw:
            text = self._clean_question_text(str(q or ""))
            if text:
                cleaned.append(text)
            if len(cleaned) >= DEFAULT_QUESTION_COUNT:
                break
        return self._ensure_question_diversity(cleaned, workspace_context=workspace_context)

    def _build_fallback_questions(self, workspace_context: Dict[str, Any]) -> List[str]:
        files = workspace_context.get("files") or []
        file_names = [str((x or {}).get("path") or "") for x in files[:6] if isinstance(x, dict)]
        anchor = ", ".join([x for x in file_names if x]) or "available files"
        workspace_name = str(workspace_context.get("workspace_name") or "")
        open_ended_pool = [
            f"Analyze cross-file consistency patterns across {anchor}, and provide actionable insights with concrete remediation priorities.",
            f"Compare structural and semantic differences across {anchor}, then propose an open-ended improvement strategy.",
            f"Identify potential anomalies or data quality risks across {anchor}, and explain why they matter with recommended fixes.",
            f"Provide an open-ended analysis of schema alignment across {anchor}, including hypothesis-driven root-cause discussion.",
        ]
        closed_pool = [
            f"Design one verifiable metric from {anchor} and compute it with Python.",
            f"Compute one reproducible cross-format validation check using {anchor} and report the result.",
            f"Calculate one summary statistic from {anchor} that can be used for monitoring data quality over time.",
        ]

        seed = abs(hash(workspace_name + "|" + anchor))
        open_shift = seed % max(1, len(open_ended_pool))
        closed_shift = (seed // 7) % max(1, len(closed_pool))
        open_ended_pool = open_ended_pool[open_shift:] + open_ended_pool[:open_shift]
        closed_pool = closed_pool[closed_shift:] + closed_pool[:closed_shift]

        required_open = self._required_open_question_count(DEFAULT_QUESTION_COUNT)
        result: List[str] = []
        result.extend(open_ended_pool[:required_open])

        mixed = []
        for i in range(max(len(open_ended_pool), len(closed_pool))):
            if i < len(closed_pool):
                mixed.append(closed_pool[i])
            if i < len(open_ended_pool):
                mixed.append(open_ended_pool[i])
        for q in mixed:
            if len(result) >= DEFAULT_QUESTION_COUNT:
                break
            if q not in result:
                result.append(q)

        return result[:DEFAULT_QUESTION_COUNT]

    def _ensure_question_diversity(self, questions: List[str], workspace_context: Dict[str, Any]) -> List[str]:
        result = [str(x).strip() for x in questions if str(x).strip()]
        fallback = self._build_fallback_questions(workspace_context=workspace_context)

        # 先补足总数量
        for item in fallback:
            if len(result) >= DEFAULT_QUESTION_COUNT:
                break
            if str(item or "").strip():
                result.append(str(item).strip())

        required_open = self._required_open_question_count(DEFAULT_QUESTION_COUNT)
        open_count = sum(1 for x in result if self._is_open_ended_question(x))
        if open_count < required_open:
            open_fallback = [x for x in fallback if self._is_open_ended_question(x)]
            for item in open_fallback:
                if open_count >= required_open:
                    break
                # 优先替换非开放问题
                replaced = False
                for idx, q in enumerate(result):
                    if not self._is_open_ended_question(q):
                        result[idx] = item
                        open_count += 1
                        replaced = True
                        break
                if not replaced and len(result) < DEFAULT_QUESTION_COUNT:
                    result.append(item)
                    open_count += 1

        # 去重并截断
        uniq: List[str] = []
        seen = set()
        for q in result:
            k = q.lower()
            if k in seen:
                continue
            seen.add(k)
            uniq.append(q)
            if len(uniq) >= DEFAULT_QUESTION_COUNT:
                break

        # 去重后再次保证开放题数量
        required_open = self._required_open_question_count(len(uniq) or DEFAULT_QUESTION_COUNT)
        open_count = sum(1 for x in uniq if self._is_open_ended_question(x))
        if open_count < required_open:
            for item in [x for x in fallback if self._is_open_ended_question(x)]:
                if open_count >= required_open:
                    break
                for idx, q in enumerate(uniq):
                    if not self._is_open_ended_question(q):
                        uniq[idx] = item
                        open_count += 1
                        break
        return uniq

    @staticmethod
    def _required_open_question_count(total_count: int) -> int:
        total = max(1, int(total_count or 1))
        return max(1, int(total * OPEN_ENDED_RATIO + 0.999999))

    @staticmethod
    def _is_open_ended_question(question: str) -> bool:
        q = str(question or "").lower()
        keys = [
            "insight",
            "pattern",
            "why",
            "recommend",
            "strategy",
            "open-ended",
            "actionable",
            "compare",
            "anomaly",
            "hypothesis",
        ]
        return any(k in q for k in keys)

    @staticmethod
    def _clean_question_text(raw_question: str) -> str:
        text = str(raw_question or "").strip()
        if not text:
            return ""

        lowered = text.lower()
        if lowered.startswith("question:"):
            text = text.split(":", 1)[1].strip()

        for marker in ["\ntrajectory:", "trajectory:", "<analyze>", "<understand>", "<code>", "<execute>", "<answer>"]:
            idx = text.lower().find(marker)
            if idx >= 0:
                text = text[:idx].strip()

        text = " ".join(text.split())
        if not text:
            return ""
        return text

    def _parse_questions_from_plain_text(self, content: str) -> List[str]:
        raw = str(content or "")
        lines = [x.strip() for x in raw.splitlines() if str(x or "").strip()]
        questions: List[str] = []
        for line in lines:
            line = re.sub(r"^[-*\d\.)\s]+", "", line).strip()
            line = re.sub(r"^(question|q)\s*[:：]\s*", "", line, flags=re.IGNORECASE).strip()
            cleaned = self._clean_question_text(line)
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered.startswith("trajectory"):
                continue
            if "?" in cleaned or any(
                lowered.startswith(k)
                for k in ("what", "how", "why", "which", "analyze", "compare", "identify", "design", "please")
            ):
                questions.append(cleaned)
            if len(questions) >= DEFAULT_QUESTION_COUNT:
                break
        return questions

    def _synthesize_trajectory(
        self,
        question: str,
        workspace_name: str,
        workspace_context: Dict[str, Any],
        workspace_dir: Path,
        api_key: str,
        base_url: str,
        model_name: str,
    ) -> Tuple[str, str, Optional[str], Optional[str]]:
        system_prompt = (
            "You are an autonomous data analyst with a python execution tool. "
            "You must solve the question step-by-step and may request python code execution. "
            "You must answer ONLY the exact question provided in user field 'question'. "
            "Never replace it with another task (for example, do not switch to counting gender unless explicitly asked). "
            "You MUST return exactly one valid JSON object (UTF-8), no markdown, no code fences, no prose outside JSON. "
            "Return strict JSON with keys: analyze, understand, next_action, code, final_answer. "
            "All five keys must always exist in every response. "
            "next_action must be one of ['python','final']. "
            "If next_action='python', provide executable python code in code and set final_answer to ''. "
            "If next_action='final', provide final_answer and set code to ''. "
            "final_answer must be a single natural-language string containing the final response text. "
            "Do NOT make final_answer a JSON object or JSON array. "
            "If structured content is needed (e.g., table/json), embed it as text within the string. "
            "Do not add any extra keys. "
            "Do not wrap JSON in backticks. "
            "Before responding, self-check that your output can be parsed by Python json.loads. "
            "Never fabricate execute results; execution is provided by tool outputs. "
            "When writing python code, only use allowed libraries: "
            + ", ".join(SANDBOX_ALLOWED_LIBS)
            + "."
        )

        steps: List[Dict[str, Any]] = []
        last_execute: Optional[Dict[str, Any]] = None
        final_answer = ""
        final_error: Optional[str] = None

        try:
            for step_idx in range(1, MAX_TRAJECTORY_STEPS + 1):
                user_prompt = {
                    "workspace_name": workspace_name,
                    "question": question,
                    "question_text_verbatim": question,
                    "workspace_context": workspace_context,
                    "workspace_cwd": str(workspace_dir),
                    "step_index": step_idx,
                    "previous_steps": steps,
                    "last_execute": last_execute,
                    "requirements": {
                        "must_be_grounded": True,
                        "no_hallucinated_files": True,
                        "prefer_python_tool_for_verifiable_computation": True,
                        "prioritize_filename_semantics": True,
                        "answer_exact_question_only": True,
                    },
                    "output_contract": {
                        "format": "single_json_object_only",
                        "no_markdown": True,
                        "no_code_fence": True,
                        "no_text_outside_json": True,
                        "required_keys": ["analyze", "understand", "next_action", "code", "final_answer"],
                        "next_action_allowed": ["python", "final"],
                        "python_mode": "next_action=python => final_answer must be empty string",
                        "final_mode": "next_action=final => code must be empty string",
                        "final_answer_type": "string",
                        "final_answer_style": "natural_language_text",
                        "final_answer_no_top_level_json": True,
                        "extra_keys_forbidden": True,
                        "must_be_json_loads_parseable": True,
                    },
                    "output_example": {
                        "analyze": "I should inspect file schema first.",
                        "understand": "Need one python step before final answer.",
                        "next_action": "python",
                        "code": "import pandas as pd\nprint('ok')",
                        "final_answer": "",
                    },
                    "allowed_python_libraries": SANDBOX_ALLOWED_LIBS,
                }

                content = self._chat_completion(
                    api_key=api_key,
                    base_url=base_url,
                    model_name=model_name,
                    system_prompt=system_prompt,
                    user_payload=user_prompt,
                )
                payload: Dict[str, Any] = {}
                try:
                    payload = self._extract_json_object(content)
                except ModelOutputParseError:
                    fallback_answer = self._extract_final_answer_from_text(content)
                    if fallback_answer:
                        final_answer = fallback_answer
                        steps.append(
                            {
                                "analyze": "Model returned non-JSON output.",
                                "code": "",
                                "execute": "",
                                "understand": "Recovered final answer from raw text output.",
                            }
                        )
                        break

                    extracted_code = self._extract_python_code_from_text(content)
                    if extracted_code:
                        payload = {
                            "analyze": "Model returned non-JSON output.",
                            "understand": "Recovered python code from raw text output.",
                            "next_action": "python",
                            "code": extracted_code,
                        }
                    else:
                        payload = {
                            "analyze": "Model returned non-JSON output.",
                            "understand": "No executable code found; continue with fallback step.",
                            "next_action": "python",
                            "code": "print('Unable to parse structured model response in this step.')",
                        }

                analyze = str(payload.get("analyze") or "").strip()
                understand = str(payload.get("understand") or "").strip()
                next_action = str(payload.get("next_action") or "python").strip().lower()
                code = str(payload.get("code") or "").strip()

                if next_action not in {"python", "final"}:
                    next_action = "python"

                if next_action == "final":
                    final_answer = str(payload.get("final_answer") or "").strip()
                    if not final_answer:
                        for alt_key in ("answer", "final", "result", "output"):
                            alt = str(payload.get(alt_key) or "").strip()
                            if alt:
                                final_answer = alt
                                break
                    if not final_answer:
                        final_answer = self._extract_final_answer_from_text(content)
                    final_answer = self._normalize_final_answer_text(final_answer)
                    steps.append(
                        {
                            "analyze": analyze,
                            "code": "",
                            "execute": "",
                            "understand": understand,
                        }
                    )
                    break

                if not code:
                    code = "print('No code generated by model for this step.')"

                execute_result = self._run_python_in_sandbox(workspace_dir=workspace_dir, code=code)
                execute_text = self._format_execute_result(execute_result)
                last_execute = execute_result
                steps.append(
                    {
                        "analyze": analyze,
                        "code": code,
                        "execute": execute_text,
                        "understand": understand,
                    }
                )

                if execute_result.get("timeout"):
                    final_error = "python execution timeout"

            if not final_answer:
                final_answer = self._finalize_answer_from_steps(
                    question=question,
                    steps=steps,
                    api_key=api_key,
                    base_url=base_url,
                    model_name=model_name,
                )
                if not final_answer:
                    final_answer = self._fallback_answer_from_steps(question=question, steps=steps)
                if not final_answer and final_error is None:
                    final_error = "solver did not produce final answer within step limit"
            final_answer = self._normalize_final_answer_text(final_answer)

            trajectory = self._compose_trajectory(steps=steps, final_answer=final_answer)
            return trajectory, final_answer, final_error, None
        except Exception as exc:
            fallback = (
                f"<Analyze>Model call failed while solving question.</Analyze>\n"
                f"<Code></Code>\n<Execute>[Error]: {exc}</Execute>\n"
                f"<Understand>Unable to continue synthesis due to model failure.</Understand>\n"
                f"<Answer></Answer>"
            )
            return fallback, "", str(exc), self._build_audit_text("trajectory_generation", exc)

    @staticmethod
    def _extract_python_code_from_text(content: str) -> str:
        raw = str(content or "")
        m = re.search(r"```python\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return str(m.group(1) or "").strip()
        m = re.search(r"```\s*(.*?)```", raw, flags=re.DOTALL)
        if m:
            return str(m.group(1) or "").strip()
        return ""

    @staticmethod
    def _extract_final_answer_from_text(content: str) -> str:
        raw = str(content or "").strip()
        if not raw:
            return ""

        # 若是JSON文本，优先提取常见答案字段，避免把整个JSON对象原样写入<Answer>
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                for key in ("final_answer", "answer", "result", "final", "output"):
                    val = parsed.get(key)
                    text_val = str(val or "").strip()
                    if text_val:
                        return text_val
                return ""
        except Exception:
            pass

        # 文本中夹杂JSON对象时，尝试提取并解析
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            frag = str(match.group(0) or "").strip()
            try:
                parsed = json.loads(frag)
                if isinstance(parsed, dict):
                    for key in ("final_answer", "answer", "result", "final", "output"):
                        val = parsed.get(key)
                        text_val = str(val or "").strip()
                        if text_val:
                            return text_val
                    return ""
            except Exception:
                pass

        # JSON不合法时，兜底从文本里提取常见答案字段值（尽量避免返回整个JSON对象）
        key_patterns = ["final_answer", "answer", "result", "final", "output"]
        for key in key_patterns:
            # 匹配如: "final_answer": "..." 或 'final_answer': '...'
            pat = rf"['\"]{key}['\"]\s*:\s*['\"]([\s\S]*?)['\"]\s*(,|\}}|$)"
            m = re.search(pat, raw, flags=re.IGNORECASE)
            if m:
                candidate = str(m.group(1) or "")
                candidate = candidate.encode("utf-8", errors="ignore").decode("unicode_escape", errors="ignore")
                candidate = candidate.strip()
                if candidate:
                    return candidate

        for marker in ["final answer:", "answer:", "conclusion:"]:
            idx = raw.lower().find(marker)
            if idx >= 0:
                answer = raw[idx + len(marker) :].strip()
                if answer:
                    return answer

        cleaned = re.sub(r"```[\s\S]*?```", "", raw).strip()
        if cleaned.startswith("{") and cleaned.endswith("}"):
            return ""
        if cleaned:
            return cleaned
        return ""

    @staticmethod
    def _normalize_final_answer_text(answer_text: str) -> str:
        raw = str(answer_text or "").strip()
        if not raw:
            return ""

        # 若回答本身是JSON对象，优先抽取常见答案字段，避免<Answer>里直接塞整段JSON
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                for key in ("final_answer", "answer", "result", "final", "output"):
                    val = str(parsed.get(key) or "").strip()
                    if val:
                        return val
                # 兜底：转成可读文本而非原始JSON
                pairs = []
                for k, v in parsed.items():
                    text_v = str(v).strip()
                    if text_v:
                        pairs.append(f"{k}: {text_v}")
                return "\n".join(pairs).strip()
            if isinstance(parsed, list):
                return json.dumps(parsed, ensure_ascii=False)
        except Exception:
            return raw
        return raw

    @staticmethod
    def _build_audit_text(phase: str, exc: Exception) -> Optional[str]:
        raw = ""
        if isinstance(exc, ModelOutputParseError):
            raw = exc.raw_output
        if not raw:
            return None
        text = str(raw or "")[:MAX_MODEL_AUDIT_CHARS]
        return f"[{phase}] {text}"

    def _finalize_answer_from_steps(
        self,
        question: str,
        steps: List[Dict[str, Any]],
        api_key: str,
        base_url: str,
        model_name: str,
    ) -> str:
        if not steps:
            return ""
        system_prompt = (
            "You are a concise answer finisher. "
            "Given a question and executed steps, return strict JSON: {\"final_answer\":\"...\"}. "
            "The value of final_answer must be a single natural-language string, not a JSON object/array."
        )
        user_payload = {
            "question": question,
            "steps": steps[-3:],
            "constraints": [
                "Do not include tags.",
                "Return only final answer text.",
                "final_answer must be plain text string.",
                "Do not return a top-level JSON object/array inside final_answer.",
            ],
        }
        try:
            content = self._chat_completion(
                api_key=api_key,
                base_url=base_url,
                model_name=model_name,
                system_prompt=system_prompt,
                user_payload=user_payload,
            )
            payload = self._extract_json_object(content)
            answer = str(payload.get("final_answer") or "").strip()
            if not answer:
                answer = self._extract_final_answer_from_text(content)
            return answer
        except Exception:
            return ""

    @staticmethod
    def _fallback_answer_from_steps(question: str, steps: List[Dict[str, Any]]) -> str:
        if not steps:
            return ""

        last_step = steps[-1]
        understand = str(last_step.get("understand") or "").strip()
        execute_text = str(last_step.get("execute") or "").strip()

        stdout = ""
        stderr = ""
        returncode = ""
        if execute_text:
            try:
                payload = json.loads(execute_text)
                if isinstance(payload, dict):
                    stdout = str(payload.get("stdout") or "").strip()
                    stderr = str(payload.get("stderr") or "").strip()
                    returncode = str(payload.get("returncode") or "").strip()
            except Exception:
                stdout = execute_text

        parts: List[str] = [
            "A complete model-generated final answer was not returned within step limit.",
            f"Question: {str(question or '').strip()}",
        ]
        if understand:
            parts.append(f"Latest reasoning summary: {understand}")
        if returncode:
            parts.append(f"Latest execution return code: {returncode}")
        if stdout:
            parts.append(f"Latest execution stdout (truncated): {stdout[:600]}")
        if stderr:
            parts.append(f"Latest execution stderr (truncated): {stderr[:600]}")

        parts.append("Please review the execution evidence above; it reflects the best available intermediate result.")
        return "\n".join(parts).strip()

    @staticmethod
    def _compose_trajectory(steps: List[Dict[str, Any]], final_answer: str) -> str:
        parts: List[str] = []
        for step in steps:
            analyze = str(step.get("analyze") or "").strip()
            code = str(step.get("code") or "").strip()
            execute = str(step.get("execute") or "").strip()
            understand = str(step.get("understand") or "").strip()

            if analyze:
                parts.append(AgenticSynthesisService._wrap_tag("Analyze", analyze))
            if code:
                parts.append(AgenticSynthesisService._wrap_tag("Code", code))
            if execute:
                parts.append(AgenticSynthesisService._wrap_tag("Execute", execute))
            if understand:
                parts.append(AgenticSynthesisService._wrap_tag("Understand", understand))

        parts.append(AgenticSynthesisService._wrap_tag("Answer", str(final_answer or "").strip()))
        return "\n".join(parts)

    @staticmethod
    def _wrap_tag(tag_name: str, content: str) -> str:
        text = str(content or "")
        return f"<{tag_name}>\n{text}\n</{tag_name}>"

    @staticmethod
    def _format_execute_result(result: Dict[str, Any]) -> str:
        payload = {
            "returncode": result.get("returncode"),
            "timeout": bool(result.get("timeout")),
            "stdout": str(result.get("stdout") or "")[:MAX_SANDBOX_OUTPUT_CHARS],
            "stderr": str(result.get("stderr") or "")[:MAX_SANDBOX_OUTPUT_CHARS],
        }
        return json.dumps(payload, ensure_ascii=False)

    def _run_python_in_sandbox(self, workspace_dir: Path, code: str, timeout_seconds: int = 25) -> Dict[str, Any]:
        runner_dir = workspace_dir / ".agentic_runtime"
        runner_dir.mkdir(parents=True, exist_ok=True)
        script_path = runner_dir / "sandbox_exec.py"

        transformed_code = self._transform_code_for_output(code)

        sandbox_prelude = (
            "import os\n"
            "import socket\n"
            "import subprocess\n"
            "import builtins\n"
            "import json\n"
            "\n"
            "def _deny(*args, **kwargs):\n"
            "    raise RuntimeError('sandbox restricted operation')\n"
            "\n"
            "socket.create_connection = _deny\n"
            "socket.getaddrinfo = _deny\n"
            "os.system = _deny\n"
            "os.popen = _deny\n"
        )
        script_path.write_text(sandbox_prelude + "\n" + transformed_code, encoding="utf-8")

        env = {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PATH": os.environ.get("PATH", ""),
        }
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
            if key in os.environ:
                env[key] = ""

        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script_path)],
                cwd=str(workspace_dir),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=env,
            )
            return {
                "returncode": int(proc.returncode),
                "timeout": False,
                "stdout": str(proc.stdout or ""),
                "stderr": str(proc.stderr or ""),
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "returncode": -1,
                "timeout": True,
                "stdout": str((exc.stdout or "") if isinstance(exc.stdout, str) else ""),
                "stderr": str((exc.stderr or "") if isinstance(exc.stderr, str) else ""),
            }

    @staticmethod
    def _transform_code_for_output(code: str) -> str:
        raw = str(code or "")
        try:
            tree = ast.parse(raw)
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last_expr = tree.body[-1].value
                tree.body[-1] = ast.Assign(
                    targets=[ast.Name(id="__agentic_last", ctx=ast.Store())],
                    value=last_expr,
                )
                postlude = ast.parse(
                    "\n"
                    "try:\n"
                    "    print(json.dumps({'__agentic_last__': __agentic_last}, ensure_ascii=False))\n"
                    "except Exception:\n"
                    "    try:\n"
                    "        print(repr(__agentic_last))\n"
                    "    except Exception:\n"
                    "        pass\n"
                )
                tree.body.extend(postlude.body)
            tree = ast.fix_missing_locations(tree)
            return ast.unparse(tree)
        except Exception:
            return raw

    def _evaluate_result(
        self,
        question: str,
        trajectory: str,
        answer_text: str,
        workspace_name: str,
        workspace_context: Dict[str, Any],
        api_key: str,
        base_url: str,
        model_name: str,
    ) -> Dict[str, Any]:
        system_prompt = (
            "You are a strict evaluator. Score the quality of question and trajectory. "
            "Return strict JSON: {\"difficulty\": int(1-5), \"quality\": int(1-5), \"ability\": string}."
        )
        user_prompt = {
            "workspace_name": workspace_name,
            "question": question,
            "trajectory": trajectory,
            "answer": answer_text,
            "workspace_context_summary": {
                "file_count": workspace_context.get("file_count"),
                "files": workspace_context.get("files", [])[:20],
            },
            "rubric": {
                "difficulty": "complexity of question",
                "quality": "faithfulness, correctness, clarity",
                "ability": "main capability demonstrated",
            },
        }
        try:
            content = self._chat_completion(
                api_key=api_key,
                base_url=base_url,
                model_name=model_name,
                system_prompt=system_prompt,
                user_payload=user_prompt,
            )
            payload = self._extract_json_object(content)
            difficulty = int(payload.get("difficulty") or 3)
            quality = int(payload.get("quality") or 3)
            ability = str(payload.get("ability") or "Data Analysis").strip() or "Data Analysis"
            difficulty = max(1, min(5, difficulty))
            quality = max(1, min(5, quality))
            return {
                "difficulty": difficulty,
                "quality": quality,
                "ability": ability,
            }
        except Exception:
            return {
                "difficulty": 3,
                "quality": 3,
                "ability": "Data Analysis",
            }

    def _chat_completion(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        system_prompt: str,
        user_payload: Dict[str, Any],
    ) -> str:
        endpoint = self._build_chat_endpoint(base_url)
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
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

        try:
            with urllib_request.urlopen(req, timeout=180) as resp:
                body = resp.read().decode("utf-8", errors="ignore")
        except URLError as exc:
            raise RuntimeError(f"llm request failed: {exc}") from exc

        parsed = json.loads(body)
        choices = parsed.get("choices") or []
        if not choices:
            raise RuntimeError("llm response has no choices")
        first = choices[0] or {}
        message = first.get("message") or {}
        content = message.get("content")
        if content is None:
            content = first.get("text")
        if isinstance(content, list):
            content = "\n".join(str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in content)
        text_content = str(content or "").strip()
        if not text_content:
            raise RuntimeError("llm response content is empty")
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
    def _extract_json_object(text: str) -> Dict[str, Any]:
        raw = str(text or "").strip()
        if raw.startswith("```"):
            raw = raw.strip("`").strip()
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # 提取首个平衡的大括号JSON对象（忽略字符串中的括号）
        start = -1
        depth = 0
        in_str = False
        escape = False
        for idx, ch in enumerate(raw):
            if ch == '"' and not escape:
                in_str = not in_str
            if ch == "\\" and in_str:
                escape = not escape
            else:
                escape = False

            if in_str:
                continue
            if ch == "{":
                if depth == 0:
                    start = idx
                depth += 1
            elif ch == "}" and depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    frag = raw[start : idx + 1]
                    try:
                        parsed = json.loads(frag)
                        if isinstance(parsed, dict):
                            return parsed
                    except Exception:
                        start = -1
                        continue
        raise ModelOutputParseError("failed to parse json object from model response", raw_output=raw)
