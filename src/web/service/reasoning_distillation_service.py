from __future__ import annotations

import csv
import io
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..dao import (
    AgenticSynthesisResultDAO,
    AgenticSynthesisTaskDAO,
    DatasetDAO,
    ReasoningDistillationResultDAO,
    ReasoningDistillationTaskDAO,
)
from utils.logger import logger
from .agentic_synthesis_service import AgenticSynthesisService
from .dataset_service import DatasetService


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MAX_SOURCE_ITEMS = 200
MAX_TEXT_CHARS = 4000
PROMPT_PLACEHOLDER_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
EVALUATION_DIMENSIONS = (
    "clarity",
    "coherence",
    "completeness",
    "complexity",
    "correctness",
    "meaningfulness",
    "difficulty",
)
DEFAULT_REASONING_PROMPT = (
    "Generate one training record for reasoning synthesis. "
    "The reasoning field must always be wrapped in exactly one pair of <think></think> tags. "
    "Do not add markdown fences."
)
DEFAULT_EVALUATION_PROMPT = """You are a single-sample quality evaluator. Based on the input below, evaluate the quality of the output relative to the ground truth.

question:
{question}

output:
{output}

ground truth:
{completion}

Please score the completion on the following 7 dimensions (1-10) and provide a brief reason for each score:
- clarity: whether the expression is clear and easy to understand
- coherence: whether the logic is internally consistent and flows naturally
- completeness: whether the core information is fully covered without major omissions
- complexity: whether it demonstrates an appropriate level of reasoning depth and information structure
- correctness: whether the content is accurate and faithful to the ground truth
- meaningfulness: whether the output is informative, useful, and aligned with the task goal
- difficulty: the intrinsic difficulty of this question itself

Requirements:
- Be strict and avoid inflated scoring.
- If the output deviates from the ground truth, lower correctness and completeness accordingly.
- Difficulty refers to the difficulty of the sample itself, not the quality of the output.
- Return valid JSON only.
- Do not output any extra text.

JSON schema:
{
  "clarity": 0,
  "coherence": 0,
  "completeness": 0,
  "complexity": 0,
  "correctness": 0,
  "meaningfulness": 0,
  "difficulty": 0
}"""


class ReasoningDistillationService(AgenticSynthesisService):
    def __init__(
        self,
        task_dao: Optional[ReasoningDistillationTaskDAO] = None,
        result_dao: Optional[ReasoningDistillationResultDAO] = None,
        dataset_dao: Optional[DatasetDAO] = None,
        source_task_dao: Optional[AgenticSynthesisTaskDAO] = None,
        source_result_dao: Optional[AgenticSynthesisResultDAO] = None,
        dataset_service: Optional[DatasetService] = None,
    ) -> None:
        super().__init__(dataset_dao=dataset_dao, result_dao=source_result_dao)
        self.distillation_task_dao = task_dao or ReasoningDistillationTaskDAO()
        self.distillation_result_dao = result_dao or ReasoningDistillationResultDAO()
        self.source_task_dao = source_task_dao or AgenticSynthesisTaskDAO()
        self.source_result_dao = source_result_dao or AgenticSynthesisResultDAO()
        self.dataset_service = dataset_service or DatasetService(dataset_dao=self.dataset_dao)
        self._distillation_lock = threading.RLock()
        self._distillation_threads: Dict[int, threading.Thread] = {}

    def start_task(
        self,
        *,
        user_id: int,
        source_type: str,
        source_dataset_id: Optional[int],
        source_task_id: Optional[int],
        selected_file_paths: Optional[List[str]],
        file_mappings: Optional[List[Dict[str, Any]]],
        prompt_field: Optional[str],
        completion_field: Optional[str],
        prompt: Optional[str],
        evaluation_enabled: bool,
        evaluation_prompt: Optional[str],
        strategy: str,
        target_max_tokens: int,
        compression_ratio: float,
        keep_tool_trace: bool,
        note: Optional[str],
        llm_api_key: str,
        llm_base_url: str,
        llm_model_name: str,
        parallelism: int = 1,
        save_path: Optional[str] = None,
        save_path_key: Optional[str] = None,
        llm_params_json: Optional[str] = None,
    ) -> Dict:
        source_config = self._normalize_dataset_source_config(
            selected_file_paths=selected_file_paths,
            file_mappings=file_mappings,
            prompt_field=prompt_field,
            completion_field=completion_field,
        )
        source_context = self._build_source_context(
            user_id=user_id,
            source_type=source_type,
            source_dataset_id=source_dataset_id,
            source_task_id=source_task_id,
            source_config=source_config,
        )
        source_items = source_context["items"]
        if not source_items:
            raise ValueError("no source items available for distillation")
        llm_params = self._parse_llm_params_json(llm_params_json)
        normalized_parallelism = self._normalize_parallelism(parallelism, len(source_items))

        task = self.distillation_task_dao.insert_task(
            {
                "user_id": int(user_id),
                "source_type": source_type,
                "source_dataset_id": int(source_dataset_id) if source_dataset_id else None,
                "source_task_id": int(source_task_id) if source_task_id else None,
                "prompt_text": str(prompt or "").strip() or DEFAULT_REASONING_PROMPT,
                "evaluation_enabled": 1 if evaluation_enabled else 0,
                "evaluation_prompt_text": str(evaluation_prompt or "").strip() or None,
                "strategy": str(strategy).strip(),
                "target_max_tokens": int(target_max_tokens),
                "compression_ratio": float(compression_ratio),
                "keep_tool_trace": 1 if keep_tool_trace else 0,
                "note": str(note or "").strip() or None,
                "llm_api_key": str(llm_api_key or "").strip(),
                "llm_base_url": str(llm_base_url or "").strip(),
                "llm_model_name": str(llm_model_name or "").strip(),
                "parallelism": normalized_parallelism,
                "llm_params_json": json.dumps(llm_params, ensure_ascii=False) if llm_params else None,
                "output_file_path": str(
                    self._resolve_task_output_path(
                        user_id=user_id,
                        task_id=0,
                        save_path=save_path,
                        save_path_key=save_path_key,
                    )
                ),
                "generated_dataset_id": None,
                "total_items": len(source_items),
                "processed_items": 0,
                "distilled_samples": 0,
                "avg_tokens": 0,
            }
        )
        output_path = self._resolve_task_output_path(
            user_id=user_id,
            task_id=int(task.id),
            save_path=save_path,
            save_path_key=save_path_key,
        )
        self.distillation_task_dao.update_output_file_path(int(task.id), str(output_path))
        task = self.distillation_task_dao.get_task_by_id(int(task.id), user_id=int(user_id)) or task

        thread = threading.Thread(
            target=self._run_task,
            args=(
                int(task.id),
                int(user_id),
                source_context,
                str(prompt or "").strip() or DEFAULT_REASONING_PROMPT,
                bool(evaluation_enabled),
                str(evaluation_prompt or "").strip() or None,
                str(strategy).strip(),
                int(target_max_tokens),
                float(compression_ratio),
                bool(keep_tool_trace),
                str(note or "").strip() or None,
                str(llm_api_key or "").strip(),
                str(llm_base_url or "").strip(),
                str(llm_model_name or "").strip(),
                normalized_parallelism,
                llm_params,
                output_path,
            ),
            daemon=True,
            name=f"reasoning-distillation-task-{task.id}",
        )
        with self._distillation_lock:
            self._distillation_threads[int(task.id)] = thread
        thread.start()
        return self._enrich_task_payload(task.to_dict(), user_id=int(user_id))

    def list_tasks(self, user_id: int, limit: int = 20) -> List[Dict]:
        rows = self.distillation_task_dao.list_tasks(limit=limit, user_id=user_id)
        return [self._enrich_task_payload(row.to_dict(), user_id=user_id) for row in rows]

    def get_task(self, user_id: int, task_id: int) -> Optional[Dict]:
        row = self.distillation_task_dao.get_task_by_id(task_id=task_id, user_id=user_id)
        if row is None:
            return None
        payload = self._enrich_task_payload(row.to_dict(), user_id=user_id)
        payload["result_count"] = self.distillation_result_dao.count_results_by_task(task_id=task_id, user_id=user_id)
        return payload

    def list_results(self, user_id: int, task_id: int, limit: int = 200) -> List[Dict]:
        task = self.distillation_task_dao.get_task_by_id(task_id=task_id, user_id=user_id)
        if task is None:
            raise ValueError("task not found")
        return [row.to_dict() for row in self.distillation_result_dao.list_results_by_task(task_id=task_id, user_id=user_id, limit=limit)]

    def get_result(self, user_id: int, result_id: int) -> Optional[Dict]:
        row = self.distillation_result_dao.get_result_by_id(result_id=result_id, user_id=user_id)
        return row.to_dict() if row else None

    def _run_task(
        self,
        task_id: int,
        user_id: int,
        source_context: Dict[str, Any],
        prompt: str,
        evaluation_enabled: bool,
        evaluation_prompt: Optional[str],
        strategy: str,
        target_max_tokens: int,
        compression_ratio: float,
        keep_tool_trace: bool,
        note: Optional[str],
        llm_api_key: str,
        llm_base_url: str,
        llm_model_name: str,
        parallelism: int,
        llm_params: Dict[str, Any],
        output_path: Path,
    ) -> None:
        source_items = source_context["items"]
        processed_items = 0
        distilled_samples = 0
        token_total = 0
        first_failure_message: Optional[str] = None
        sample_records: List[Dict[str, Any]] = []
        effective_parallelism = self._normalize_parallelism(parallelism, len(source_items))

        dataset_output_path = output_path.with_name("reasoning_dataset.jsonl")
        manifest_path = output_path.with_name("manifest.json")

        try:
            self.distillation_task_dao.mark_started(task_id)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as task_writer, dataset_output_path.open("w", encoding="utf-8") as dataset_writer:
                if effective_parallelism > 1:
                    with ThreadPoolExecutor(max_workers=effective_parallelism, thread_name_prefix=f"reasoning-{task_id}") as executor:
                        futures = [
                            executor.submit(
                                self._process_source_item,
                                task_id=task_id,
                                user_id=user_id,
                                source_context=source_context,
                                source_item=item,
                                prompt=prompt,
                                evaluation_enabled=evaluation_enabled,
                                evaluation_prompt=evaluation_prompt,
                                strategy=strategy,
                                target_max_tokens=target_max_tokens,
                                compression_ratio=compression_ratio,
                                keep_tool_trace=keep_tool_trace,
                                note=note,
                                llm_api_key=llm_api_key,
                                llm_base_url=llm_base_url,
                                llm_model_name=llm_model_name,
                                llm_params=llm_params,
                            )
                            for item in source_items
                        ]
                        for future in as_completed(futures):
                            item_result = future.result()
                            processed_items, distilled_samples, token_total = self._persist_source_item_result(
                                task_id=task_id,
                                processed_items=processed_items,
                                distilled_samples=distilled_samples,
                                token_total=token_total,
                                item_result=item_result,
                                task_writer=task_writer,
                                dataset_writer=dataset_writer,
                                sample_records=sample_records,
                            )
                            if (
                                first_failure_message is None
                                and str(item_result.get("status") or "").lower() == "failed"
                            ):
                                first_failure_message = str(
                                    (item_result.get("record_payload") or {}).get("error_message") or ""
                                ).strip() or None
                            avg_tokens = int(token_total / distilled_samples) if distilled_samples else 0
                            self.distillation_task_dao.update_progress(
                                task_id,
                                processed_items=processed_items,
                                distilled_samples=distilled_samples,
                                avg_tokens=avg_tokens,
                            )
                else:
                    for item in source_items:
                        item_result = self._process_source_item(
                            task_id=task_id,
                            user_id=user_id,
                            source_context=source_context,
                            source_item=item,
                            prompt=prompt,
                            evaluation_enabled=evaluation_enabled,
                            evaluation_prompt=evaluation_prompt,
                            strategy=strategy,
                            target_max_tokens=target_max_tokens,
                            compression_ratio=compression_ratio,
                            keep_tool_trace=keep_tool_trace,
                            note=note,
                            llm_api_key=llm_api_key,
                            llm_base_url=llm_base_url,
                            llm_model_name=llm_model_name,
                            llm_params=llm_params,
                        )
                        processed_items, distilled_samples, token_total = self._persist_source_item_result(
                            task_id=task_id,
                            processed_items=processed_items,
                            distilled_samples=distilled_samples,
                            token_total=token_total,
                            item_result=item_result,
                            task_writer=task_writer,
                            dataset_writer=dataset_writer,
                            sample_records=sample_records,
                        )
                        if (
                            first_failure_message is None
                            and str(item_result.get("status") or "").lower() == "failed"
                        ):
                            first_failure_message = str(
                                (item_result.get("record_payload") or {}).get("error_message") or ""
                            ).strip() or None
                        avg_tokens = int(token_total / distilled_samples) if distilled_samples else 0
                        self.distillation_task_dao.update_progress(
                            task_id,
                            processed_items=processed_items,
                            distilled_samples=distilled_samples,
                            avg_tokens=avg_tokens,
                        )

            avg_tokens = int(token_total / distilled_samples) if distilled_samples else 0
            manifest = {
                "task_id": task_id,
                "source_type": source_context["source_type"],
                "source_ref_id": source_context["source_ref_id"],
                "source_label": source_context["source_label"],
                "source_config": source_context.get("source_config") or {},
                "evaluation_enabled": bool(evaluation_enabled),
                "evaluation_prompt": str(evaluation_prompt or "").strip() or None,
                "strategy": strategy,
                "target_max_tokens": target_max_tokens,
                "compression_ratio": compression_ratio,
                "keep_tool_trace": keep_tool_trace,
                "distilled_samples": distilled_samples,
                "processed_items": processed_items,
                "avg_tokens": avg_tokens,
                "dataset_output_path": str(dataset_output_path),
                "task_output_path": str(output_path),
            }
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

            error_message = None
            if distilled_samples > 0:
                generated = self.dataset_service.register_generated_dataset(
                    user_id=user_id,
                    name=self._build_generated_dataset_name(source_context, task_id),
                    dataset_type="reasoning",
                    language=source_context.get("language") or "multi",
                    source=f"generated://reasoning-distillation/tasks/{task_id}",
                    note=note or f"Generated from {source_context['source_label']} with {strategy}",
                    file_path=str(dataset_output_path),
                    file_name=dataset_output_path.name,
                    size=dataset_output_path.stat().st_size,
                    sample_data=sample_records,
                    origin_stage="reasoning_distillation",
                    origin_dataset_id=source_context.get("origin_dataset_id"),
                    origin_task_type=source_context.get("origin_task_type"),
                    origin_task_id=source_context.get("origin_task_id"),
                    generation_meta={
                        **manifest,
                        "manifest_path": str(manifest_path),
                    },
                    status="ready",
                )
                self.distillation_task_dao.update_generated_dataset(task_id, int(generated["id"]))
            else:
                error_message = first_failure_message or "no reasoning samples were generated"

            self.distillation_task_dao.mark_finished(
                task_id,
                processed_items=processed_items,
                distilled_samples=distilled_samples,
                avg_tokens=avg_tokens,
                error_message=error_message,
            )
        except Exception as exc:
            logger.exception("Reasoning distillation task failed. task_id=%s", task_id)
            avg_tokens = int(token_total / distilled_samples) if distilled_samples else 0
            self.distillation_task_dao.mark_finished(
                task_id,
                processed_items=processed_items,
                distilled_samples=distilled_samples,
                avg_tokens=avg_tokens,
                error_message=str(exc),
            )
        finally:
            with self._distillation_lock:
                self._distillation_threads.pop(task_id, None)

    def _process_source_item(
        self,
        *,
        task_id: int,
        user_id: int,
        source_context: Dict[str, Any],
        source_item: Dict[str, Any],
        prompt: str,
        evaluation_enabled: bool,
        evaluation_prompt: Optional[str],
        strategy: str,
        target_max_tokens: int,
        compression_ratio: float,
        keep_tool_trace: bool,
        note: Optional[str],
        llm_api_key: str,
        llm_base_url: str,
        llm_model_name: str,
        llm_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            distilled = self._distill_item(
                source_context=source_context,
                source_item=source_item,
                prompt=prompt,
                strategy=strategy,
                target_max_tokens=target_max_tokens,
                compression_ratio=compression_ratio,
                keep_tool_trace=keep_tool_trace,
                note=note,
                llm_api_key=llm_api_key,
                llm_base_url=llm_base_url,
                llm_model_name=llm_model_name,
                llm_params=llm_params,
            )
            evaluation_scores = None
            evaluation_raw_text = None
            evaluation_error_message = None
            if evaluation_enabled:
                evaluation_result = self._evaluate_item(
                    source_context=source_context,
                    source_item=source_item,
                    distilled=distilled,
                    evaluation_prompt=evaluation_prompt or DEFAULT_EVALUATION_PROMPT,
                    llm_api_key=llm_api_key,
                    llm_base_url=llm_base_url,
                    llm_model_name=llm_model_name,
                    llm_params=llm_params,
                )
                evaluation_scores = evaluation_result["scores"]
                evaluation_raw_text = evaluation_result["raw_text"]
                distilled["record"]["evaluation"] = evaluation_scores
            token_count = int(distilled.get("token_count") or 0)
            return {
                "status": "completed",
                "record_payload": {
                    "task_id": task_id,
                    "user_id": user_id,
                    "source_type": source_context["source_type"],
                    "source_ref_id": int(source_context["source_ref_id"]),
                    "item_key": distilled["item_key"],
                    "prompt_text": distilled["prompt_text"],
                    "reasoning_text": distilled["reasoning_text"],
                    "answer_text": distilled["answer_text"],
                    "record_json": json.dumps(distilled["record"], ensure_ascii=False),
                    "evaluation_json": json.dumps(evaluation_scores, ensure_ascii=False) if evaluation_scores else None,
                    "evaluation_raw_text": evaluation_raw_text,
                    "evaluation_error_message": evaluation_error_message,
                    "token_count": token_count,
                    "status": "completed",
                    "error_message": None,
                },
                "dataset_record": distilled["record"],
                "token_count": token_count,
            }
        except Exception as exc:
            logger.exception("Reasoning distillation item failed. task_id=%s item=%s", task_id, source_item.get("item_key"))
            distilled_record = locals().get("distilled")
            failure_record = dict((distilled_record or {}).get("record") or {})
            if failure_record:
                failure_record["evaluation_error"] = str(exc)
            return {
                "status": "failed",
                "record_payload": {
                    "task_id": task_id,
                    "user_id": user_id,
                    "source_type": source_context["source_type"],
                    "source_ref_id": int(source_context["source_ref_id"]),
                    "item_key": str(source_item.get("item_key") or "item"),
                    "prompt_text": str((distilled_record or {}).get("prompt_text") or source_item.get("prompt_text") or self._derive_prompt_text(source_item) or "Distillation source item"),
                    "reasoning_text": str((distilled_record or {}).get("reasoning_text") or "Distillation failed for this item."),
                    "answer_text": str((distilled_record or {}).get("answer_text") or "Distillation failed."),
                    "record_json": json.dumps(
                        failure_record
                        or {
                            "error": str(exc),
                            "item_key": str(source_item.get("item_key") or "item"),
                            "source_path": source_item.get("source_path"),
                        },
                        ensure_ascii=False,
                    ),
                    "evaluation_json": None,
                    "evaluation_raw_text": None,
                    "evaluation_error_message": str(exc),
                    "token_count": 0,
                    "status": "failed",
                    "error_message": str(exc),
                },
                "dataset_record": None,
                "token_count": 0,
            }

    def _persist_source_item_result(
        self,
        *,
        task_id: int,
        processed_items: int,
        distilled_samples: int,
        token_total: int,
        item_result: Dict[str, Any],
        task_writer,
        dataset_writer,
        sample_records: List[Dict[str, Any]],
    ) -> tuple[int, int, int]:
        processed_items += 1
        record_payload = dict(item_result.get("record_payload") or {})
        saved = self.distillation_result_dao.insert_result(record_payload)
        status = str(item_result.get("status") or record_payload.get("status") or "unknown")
        task_writer.write(json.dumps({**saved.to_dict(), "status": status}, ensure_ascii=False) + "\n")
        task_writer.flush()

        dataset_record = item_result.get("dataset_record")
        token_total += int(item_result.get("token_count") or 0)
        if dataset_record:
            distilled_samples += 1
            dataset_writer.write(json.dumps(dataset_record, ensure_ascii=False) + "\n")
            dataset_writer.flush()
            if len(sample_records) < DatasetService.SAMPLE_PREVIEW_LIMIT:
                sample_records.append(dataset_record)

        return processed_items, distilled_samples, token_total

    def _build_source_context(
        self,
        *,
        user_id: int,
        source_type: str,
        source_dataset_id: Optional[int],
        source_task_id: Optional[int],
        source_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        normalized_type = str(source_type or "").strip().lower()
        normalized_source_config = dict(source_config or {})
        if normalized_type == "dataset":
            dataset = self.dataset_dao.get_dataset_by_id(dataset_id=int(source_dataset_id or 0), user_id=user_id)
            if dataset is None:
                raise ValueError("source dataset not found")
            dataset_row = dataset.to_dict(include_internal=True)
            items = self._load_dataset_source_items(
                user_id=user_id,
                dataset_id=int(dataset.id),
                dataset_row=dataset_row,
                source_config=normalized_source_config,
            )
            return {
                "source_type": "dataset",
                "source_ref_id": int(dataset.id),
                "source_label": str(dataset.name),
                "language": str(dataset.language or "multi"),
                "origin_dataset_id": int(dataset.id),
                "origin_task_type": None,
                "origin_task_id": None,
                "source_config": normalized_source_config,
                "items": items,
            }

        if normalized_type == "trajectory_task":
            task = self.source_task_dao.get_task_by_id(task_id=int(source_task_id or 0), user_id=user_id)
            if task is None:
                raise ValueError("source trajectory task not found")
            results = self.source_result_dao.list_results_by_task(task_id=int(task.id), user_id=user_id, limit=1000)
            if not results:
                raise ValueError("source trajectory task has no results")
            dataset = self.dataset_dao.get_dataset_by_id(dataset_id=int(task.dataset_id), user_id=user_id)
            items = self._load_trajectory_source_items(results)
            return {
                "source_type": "trajectory_task",
                "source_ref_id": int(task.id),
                "source_label": f"Trajectory Task {task.id}",
                "language": str(dataset.language if dataset else "multi"),
                "origin_dataset_id": int(task.generated_dataset_id or task.dataset_id or 0) or None,
                "origin_task_type": "trajectory_task",
                "origin_task_id": int(task.id),
                "source_config": {},
                "items": items,
            }

        raise ValueError(f"unsupported source_type: {source_type}")

    def _load_dataset_source_items(
        self,
        *,
        user_id: int,
        dataset_id: int,
        dataset_row: Dict[str, Any],
        source_config: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        normalized_source_config = dict(source_config or {})
        selected_file_paths = [
            str(item or "").strip()
            for item in (normalized_source_config.get("selected_file_paths") or [])
            if str(item or "").strip()
        ]
        default_completion_field = str(normalized_source_config.get("completion_field") or "").strip() or None
        file_mapping_by_path = dict(normalized_source_config.get("file_mapping_by_path") or {})

        files_payload = self.dataset_service.get_dataset_files(user_id=user_id, dataset_id=dataset_id) or {}
        data_files = files_payload.get("data_files") or []
        available_paths = [str(item.get("path") or "").strip() for item in data_files if str(item.get("path") or "").strip()]

        if selected_file_paths:
            invalid_paths = [path for path in selected_file_paths if path not in available_paths]
            if invalid_paths:
                raise ValueError(f"selected dataset files not found: {', '.join(invalid_paths[:3])}")
            candidate_paths = selected_file_paths
        else:
            candidate_paths = available_paths

        items: List[Dict[str, Any]] = []
        for relative_path in candidate_paths:
            if len(items) >= MAX_SOURCE_ITEMS:
                break
            file_mapping = file_mapping_by_path.get(relative_path) or {}
            completion_field = str(file_mapping.get("completion_field") or default_completion_field or "").strip() or None
            placeholder_mappings = {
                str(key or "").strip(): str(value or "").strip()
                for key, value in (file_mapping.get("placeholder_mappings") or {}).items()
                if str(key or "").strip() and str(value or "").strip()
            }
            preview = self.dataset_service.get_dataset_preview(
                user_id=user_id,
                dataset_id=dataset_id,
                path=relative_path,
                limit=max(1, min(100, MAX_SOURCE_ITEMS - len(items))),
                _row_override=dataset_row,
            )
            for index, row in enumerate(preview.get("rows") or []):
                if len(items) >= MAX_SOURCE_ITEMS:
                    break
                item = self._make_dataset_item(
                    record=row,
                    source_path=relative_path,
                    row_index=index,
                    placeholder_mappings=placeholder_mappings,
                    completion_field=completion_field,
                )
                if item:
                    items.append(item)

        if items:
            return items

        file_path = str(dataset_row.get("file_path") or "").strip()
        if not file_path:
            fallback_rows = dataset_row.get("sample_data")
            if isinstance(fallback_rows, str):
                try:
                    fallback_rows = json.loads(fallback_rows)
                except Exception:
                    fallback_rows = []
            rows = fallback_rows if isinstance(fallback_rows, list) else []
            fallback_items: List[Dict[str, Any]] = []
            for index, row in enumerate(rows[:MAX_SOURCE_ITEMS]):
                item = self._make_dataset_item(
                    record=row,
                    source_path="sample_data",
                    row_index=index,
                    placeholder_mappings={},
                    completion_field=default_completion_field,
                )
                if item:
                    fallback_items.append(item)
            return fallback_items

        root = Path(file_path)
        if not root.exists():
            raise FileNotFoundError(f"dataset path does not exist: {root}")

        files: List[Path] = []
        if root.is_file():
            files = [root]
        else:
            files = [candidate for candidate in sorted(root.rglob("*")) if candidate.is_file()]

        items: List[Dict[str, Any]] = []
        for file in files:
            if len(items) >= MAX_SOURCE_ITEMS:
                break
            items.extend(
                self._parse_dataset_file_to_items(
                    file,
                    limit=MAX_SOURCE_ITEMS - len(items),
                    placeholder_mappings={},
                    completion_field=default_completion_field,
                )
            )
        return items[:MAX_SOURCE_ITEMS]

    def _parse_dataset_file_to_items(
        self,
        file: Path,
        limit: int,
        *,
        placeholder_mappings: Optional[Dict[str, str]] = None,
        completion_field: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        suffix = file.suffix.lower()
        if limit <= 0:
            return []
        if suffix == ".csv":
            text = file.read_text(encoding="utf-8", errors="ignore")
            reader = csv.DictReader(io.StringIO(text))
            rows: List[Dict[str, Any]] = []
            for index, row in zip(range(limit), reader):
                item = self._make_dataset_item(
                    record=row,
                    source_path=str(file),
                    row_index=index,
                    placeholder_mappings=placeholder_mappings,
                    completion_field=completion_field,
                )
                if item:
                    rows.append(item)
            return rows
        if suffix == ".json":
            text = file.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                return []
            parsed = json.loads(text)
            rows = parsed if isinstance(parsed, list) else [parsed]
            items: List[Dict[str, Any]] = []
            for index, row in enumerate(rows[:limit]):
                item = self._make_dataset_item(
                    record=row,
                    source_path=str(file),
                    row_index=index,
                    placeholder_mappings=placeholder_mappings,
                    completion_field=completion_field,
                )
                if item:
                    items.append(item)
            return items
        if suffix == ".jsonl":
            rows: List[Dict[str, Any]] = []
            for index, line in enumerate(file.read_text(encoding="utf-8", errors="ignore").splitlines()):
                if len(rows) >= limit:
                    break
                text = line.strip()
                if not text:
                    continue
                try:
                    parsed = json.loads(text)
                except Exception:
                    parsed = {"text": text}
                item = self._make_dataset_item(
                    record=parsed,
                    source_path=str(file),
                    row_index=index,
                    placeholder_mappings=placeholder_mappings,
                    completion_field=completion_field,
                )
                if item:
                    rows.append(item)
            return rows
        if suffix in {".txt", ".md"}:
            rows: List[Dict[str, Any]] = []
            for index, line in enumerate(file.read_text(encoding="utf-8", errors="ignore").splitlines()):
                if len(rows) >= limit:
                    break
                text = line.strip()
                if not text:
                    continue
                item = self._make_dataset_item(
                    record={"text": text},
                    source_path=str(file),
                    row_index=index,
                    placeholder_mappings=placeholder_mappings,
                    completion_field=completion_field,
                )
                if item:
                    rows.append(item)
            return rows
        return []

    @staticmethod
    def _make_dataset_item(
        record: Any,
        source_path: str,
        row_index: int,
        *,
        placeholder_mappings: Optional[Dict[str, str]] = None,
        completion_field: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        normalized_record = record if isinstance(record, dict) else {"value": record}
        normalized_placeholder_mappings = {
            str(key or "").strip(): str(value or "").strip()
            for key, value in (placeholder_mappings or {}).items()
            if str(key or "").strip() and str(value or "").strip()
        }
        placeholder_values = {
            placeholder: ReasoningDistillationService._extract_record_field_text(normalized_record, field_name)
            for placeholder, field_name in normalized_placeholder_mappings.items()
        }
        completion_value = ReasoningDistillationService._extract_record_field_text(normalized_record, completion_field)

        if normalized_placeholder_mappings and any(not value for value in placeholder_values.values()):
            return None
        if completion_field and not completion_value:
            return None

        working_record = dict(normalized_record)
        if placeholder_values:
            working_record["__placeholder_values__"] = placeholder_values
        if completion_value:
            working_record["__mapped_completion__"] = completion_value
        if normalized_placeholder_mappings or completion_field:
            working_record["__field_mapping__"] = {
                "placeholder_mappings": normalized_placeholder_mappings,
                "completion_field": completion_field,
            }

        return {
            "item_key": f"{Path(str(source_path)).name}:{row_index + 1}",
            "source_path": str(source_path),
            "record": working_record,
            "prompt_text": ReasoningDistillationService._derive_prompt_text({"record": working_record}) or f"Dataset item {row_index + 1}",
            "placeholder_mappings": normalized_placeholder_mappings,
            "completion_field": completion_field,
        }

    @staticmethod
    def _load_trajectory_source_items(results: List[Any]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for result in results[:MAX_SOURCE_ITEMS]:
            row = result.to_dict() if hasattr(result, "to_dict") else dict(result)
            items.append(
                {
                    "item_key": f"trajectory-result-{row.get('id')}",
                    "source_path": f"trajectory_task_result:{row.get('id')}",
                    "record": row,
                    "prompt_text": str(row.get("question") or f"Trajectory result {row.get('id')}"),
                }
            )
        return items

    def _distill_item(
        self,
        *,
        source_context: Dict[str, Any],
        source_item: Dict[str, Any],
        prompt: str,
        strategy: str,
        target_max_tokens: int,
        compression_ratio: float,
        keep_tool_trace: bool,
        note: Optional[str],
        llm_api_key: str,
        llm_base_url: str,
        llm_model_name: str,
        llm_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        rendered_prompt = self._render_prompt_template(prompt, source_item["record"])
        if rendered_prompt:
            source_item = {
                **source_item,
                "record": {
                    **dict(source_item.get("record") or {}),
                    "__rendered_prompt__": rendered_prompt,
                },
                "prompt_text": rendered_prompt,
            }
        base_messages = self._derive_messages(source_context["source_type"], source_item["record"])
        answer_seed = self._derive_answer_text(source_context["source_type"], source_item["record"])
        system_prompt = (
            "You distill training data for a data-analysis model. "
            "Return exactly one JSON object with keys: messages, reasoning, answer. "
            "messages must be an array of {role, content}. reasoning and answer must be strings. "
            "Keep reasoning concise but useful for model training, and wrap reasoning in exactly one pair of <think></think> tags. "
            "Do not include markdown fences or extra keys."
        )
        user_payload = {
            "prompt": rendered_prompt or str(prompt or "").strip() or DEFAULT_REASONING_PROMPT,
            "source_type": source_context["source_type"],
            "strategy": strategy,
            "target_max_tokens": target_max_tokens,
            "compression_ratio": compression_ratio,
            "keep_tool_trace": keep_tool_trace,
            "note": note or "",
            "source_label": source_context["source_label"],
            "source_item_key": source_item["item_key"],
            "source_record": source_item["record"],
            "derived_messages": base_messages,
            "answer_seed": answer_seed,
        }
        content = self._chat_completion(
            api_key=llm_api_key,
            base_url=llm_base_url,
            model_name=llm_model_name,
            system_prompt=system_prompt,
            user_payload=user_payload,
            llm_params=llm_params,
        )
        try:
            payload = self._extract_json_object(content)
        except Exception:
            payload = {}

        messages = self._normalize_messages(payload.get("messages"), fallback=base_messages)
        reasoning = str(payload.get("reasoning") or "").strip()
        answer = str(payload.get("answer") or "").strip()

        if not reasoning:
            reasoning = self._fallback_reasoning(source_context["source_type"], source_item["record"], keep_tool_trace=keep_tool_trace)
        reasoning = self._ensure_think_tags(reasoning)
        if not answer:
            answer = answer_seed or self._fallback_answer(source_context["source_type"], source_item["record"])

        record = {
            "messages": messages,
            "reasoning": reasoning,
            "answer": answer,
            "metadata": {
                "source_type": source_context["source_type"],
                "source_ref_id": source_context["source_ref_id"],
                "source_label": source_context["source_label"],
                "item_key": source_item["item_key"],
                "source_path": source_item.get("source_path"),
                "placeholder_mappings": source_item.get("placeholder_mappings") or {},
                "completion_field": source_item.get("completion_field"),
                "strategy": strategy,
                "target_max_tokens": target_max_tokens,
                "compression_ratio": compression_ratio,
                "keep_tool_trace": keep_tool_trace,
            },
        }
        token_count = self._estimate_tokens(messages=messages, reasoning=reasoning, answer=answer)
        return {
            "item_key": str(source_item["item_key"]),
            "prompt_text": self._derive_prompt_text(source_item),
            "reasoning_text": reasoning,
            "answer_text": answer,
            "record": record,
            "token_count": token_count,
        }

    def _evaluate_item(
        self,
        *,
        source_context: Dict[str, Any],
        source_item: Dict[str, Any],
        distilled: Dict[str, Any],
        evaluation_prompt: str,
        llm_api_key: str,
        llm_base_url: str,
        llm_model_name: str,
        llm_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        record = dict(source_item.get("record") or {})
        placeholder_values = record.get("__placeholder_values__") if isinstance(record.get("__placeholder_values__"), dict) else {}
        evaluation_values = {
            **placeholder_values,
            "question": placeholder_values.get("question") or self._derive_prompt_text(source_item),
            "output": str(distilled.get("answer_text") or "").strip(),
            "completion": str(record.get("__mapped_completion__") or self._derive_answer_text(source_context["source_type"], record) or "").strip(),
        }
        render_record = {
            **record,
            "__placeholder_values__": evaluation_values,
        }
        rendered_prompt = self._render_prompt_template(evaluation_prompt or DEFAULT_EVALUATION_PROMPT, render_record).strip()
        if not rendered_prompt:
            raise ValueError("evaluation prompt rendered to empty content")

        system_prompt = (
            "You evaluate one generated sample. "
            "Return exactly one JSON object with numeric scores only. "
            "Do not add markdown fences or extra text."
        )
        user_payload = {
            "evaluation_prompt": rendered_prompt,
            "source_type": source_context["source_type"],
            "source_label": source_context["source_label"],
            "source_item_key": source_item["item_key"],
        }
        raw_text = self._chat_completion(
            api_key=llm_api_key,
            base_url=llm_base_url,
            model_name=llm_model_name,
            system_prompt=system_prompt,
            user_payload=user_payload,
            llm_params=llm_params,
        )
        parsed = self._extract_json_object(raw_text)
        scores = self._normalize_evaluation_scores(parsed)
        return {
            "scores": scores,
            "raw_text": str(raw_text or ""),
        }

    @staticmethod
    def _derive_messages(source_type: str, record: Any) -> List[Dict[str, str]]:
        if source_type == "trajectory_task":
            question = str((record or {}).get("question") or "").strip()
            if question:
                return [{"role": "user", "content": question}]

        if isinstance(record, dict):
            rendered_prompt = str(record.get("__rendered_prompt__") or "").strip()
            if rendered_prompt:
                return [{"role": "user", "content": rendered_prompt[:MAX_TEXT_CHARS]}]
            raw_messages = record.get("messages")
            if isinstance(raw_messages, list) and raw_messages:
                normalized_messages = []
                for item in raw_messages:
                    if isinstance(item, dict):
                        role = str(item.get("role") or item.get("speaker") or "user").strip() or "user"
                        content = str(item.get("content") or item.get("text") or item.get("message") or "").strip()
                        if content:
                            normalized_messages.append({"role": role, "content": content[:MAX_TEXT_CHARS]})
                if normalized_messages:
                    return normalized_messages

            for key in ("question", "prompt", "instruction", "input", "query", "user"):
                value = str(record.get(key) or "").strip()
                if value:
                    return [{"role": "user", "content": value[:MAX_TEXT_CHARS]}]

        return [{"role": "user", "content": json.dumps(record, ensure_ascii=False)[:MAX_TEXT_CHARS]}]

    @staticmethod
    def _derive_answer_text(source_type: str, record: Any) -> str:
        if source_type == "trajectory_task":
            trajectory = str((record or {}).get("trajectory") or "").strip()
            answer = str((record or {}).get("answer") or "").strip()
            if answer:
                return answer[:MAX_TEXT_CHARS]
            extracted = AgenticSynthesisService._extract_final_answer_from_text(trajectory)
            return extracted[:MAX_TEXT_CHARS]

        if isinstance(record, dict):
            mapped_completion = str(record.get("__mapped_completion__") or "").strip()
            if mapped_completion:
                return mapped_completion[:MAX_TEXT_CHARS]
            for key in ("answer", "response", "output", "assistant", "target", "label"):
                value = str(record.get(key) or "").strip()
                if value:
                    return value[:MAX_TEXT_CHARS]
        return ""

    @staticmethod
    def _derive_prompt_text(source_item: Dict[str, Any]) -> str:
        record = source_item.get("record") if isinstance(source_item, dict) else source_item
        if isinstance(record, dict):
            rendered_prompt = str(record.get("__rendered_prompt__") or "").strip()
            if rendered_prompt:
                return rendered_prompt[:MAX_TEXT_CHARS]
            for key in ("question", "prompt", "instruction", "input", "query", "user", "text"):
                value = str(record.get(key) or "").strip()
                if value:
                    return value[:MAX_TEXT_CHARS]
        return str(source_item.get("prompt_text") or "").strip()[:MAX_TEXT_CHARS]

    @staticmethod
    def _fallback_reasoning(source_type: str, record: Any, *, keep_tool_trace: bool) -> str:
        if source_type == "trajectory_task":
            trajectory = str((record or {}).get("trajectory") or "").strip()
            if not keep_tool_trace:
                trajectory = trajectory.replace("<Code>", "").replace("</Code>", "").replace("<Execute>", "").replace("</Execute>", "")
            return trajectory[:MAX_TEXT_CHARS] or "Distilled concise reasoning from the source trajectory."
        if isinstance(record, dict):
            keys = ", ".join(list(record.keys())[:6])
            if keys:
                return f"Summarized from source fields: {keys}"
        return "Summarized from the source record."

    @staticmethod
    def _fallback_answer(source_type: str, record: Any) -> str:
        if source_type == "trajectory_task":
            return AgenticSynthesisService._extract_final_answer_from_text(str((record or {}).get("trajectory") or ""))
        if isinstance(record, dict):
            return json.dumps(record, ensure_ascii=False)[:MAX_TEXT_CHARS]
        return str(record or "")[:MAX_TEXT_CHARS]

    @staticmethod
    def _normalize_messages(value: Any, *, fallback: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if isinstance(value, list):
            normalized: List[Dict[str, str]] = []
            for item in value:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role") or "user").strip() or "user"
                content = str(item.get("content") or "").strip()
                if content:
                    normalized.append({"role": role, "content": content[:MAX_TEXT_CHARS]})
            if normalized:
                return normalized
        return fallback

    @staticmethod
    def _estimate_tokens(*, messages: List[Dict[str, str]], reasoning: str, answer: str) -> int:
        text_parts = [json.dumps(messages, ensure_ascii=False), str(reasoning or ""), str(answer or "")]
        total_chars = sum(len(part) for part in text_parts)
        return max(1, int(total_chars / 4))

    def _resolve_task_output_path(
        self,
        user_id: int,
        task_id: int,
        *,
        save_path: Optional[str] = None,
        save_path_key: Optional[str] = None,
    ) -> Path:
        output_root = self._resolve_selected_output_root(
            save_path=save_path,
            save_path_key=save_path_key,
            task_namespace="reasoning_distillation",
        )
        output_dir = output_root / str(int(user_id)) / str(int(task_id))
        output_dir.mkdir(parents=True, exist_ok=True)
        return (output_dir / "task_results.jsonl").resolve()

    def _enrich_task_payload(self, payload: Dict[str, Any], user_id: int) -> Dict[str, Any]:
        result = dict(payload or {})
        generated_dataset_id = int(result.get("generated_dataset_id") or 0)
        if generated_dataset_id > 0:
            dataset = self.dataset_dao.get_dataset_by_id(dataset_id=generated_dataset_id, user_id=user_id)
            if dataset is not None:
                result["generated_dataset"] = dataset.to_dict()
        result["source_label"] = self._resolve_source_label(payload=result, user_id=user_id)
        if "llm_api_key" in result:
            result["llm_api_key"] = "***"
        return result

    def _resolve_source_label(self, payload: Dict[str, Any], user_id: int) -> str:
        source_type = str(payload.get("source_type") or "").strip()
        if source_type == "dataset" and payload.get("source_dataset_id"):
            dataset = self.dataset_dao.get_dataset_by_id(dataset_id=int(payload["source_dataset_id"]), user_id=user_id)
            if dataset is not None:
                return str(dataset.name)
        if source_type == "trajectory_task" and payload.get("source_task_id"):
            return f"Trajectory Task {payload['source_task_id']}"
        return source_type or "Unknown source"

    @staticmethod
    def _build_generated_dataset_name(source_context: Dict[str, Any], task_id: int) -> str:
        source_label = str(source_context.get("source_label") or "source").strip()
        return f"{source_label} Reasoning Distilled T{task_id}"

    @staticmethod
    def _normalize_dataset_source_config(
        *,
        selected_file_paths: Optional[List[str]],
        file_mappings: Optional[List[Dict[str, Any]]],
        prompt_field: Optional[str],
        completion_field: Optional[str],
    ) -> Dict[str, Any]:
        normalized_paths = []
        seen = set()
        for item in selected_file_paths or []:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            normalized_paths.append(text)
        mapping_by_path: Dict[str, Dict[str, Any]] = {}
        for item in file_mappings or []:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path") or "").strip()
            if not path:
                continue
            placeholder_mappings = {
                str(key or "").strip(): str(value or "").strip()
                for key, value in (item.get("placeholder_mappings") or {}).items()
                if str(key or "").strip() and str(value or "").strip()
            }
            mapping_by_path[path] = {
                "placeholder_mappings": placeholder_mappings,
                "prompt_field": str(item.get("prompt_field") or "").strip() or None,
                "completion_field": str(item.get("completion_field") or "").strip() or None,
            }
        return {
            "selected_file_paths": normalized_paths,
            "file_mapping_by_path": mapping_by_path,
            "prompt_field": str(prompt_field or "").strip() or None,
            "completion_field": str(completion_field or "").strip() or None,
        }

    @staticmethod
    def _extract_record_field_text(record: Dict[str, Any], field_name: Optional[str]) -> str:
        key = str(field_name or "").strip()
        if not key or not isinstance(record, dict) or key not in record:
            return ""
        value = record.get(key)
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()[:MAX_TEXT_CHARS]
        try:
            return json.dumps(value, ensure_ascii=False)[:MAX_TEXT_CHARS]
        except Exception:
            return str(value).strip()[:MAX_TEXT_CHARS]

    @staticmethod
    def _ensure_think_tags(value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return "<think></think>"
        if text.startswith("<think>") and text.endswith("</think>"):
            return text
        return f"<think>{text}</think>"

    @staticmethod
    def _normalize_evaluation_scores(value: Any) -> Dict[str, float]:
        if not isinstance(value, dict):
            raise ValueError("evaluation result must be a JSON object")
        scores: Dict[str, float] = {}
        for key in EVALUATION_DIMENSIONS:
            raw = value.get(key)
            if raw is None or str(raw).strip() == "":
                raise ValueError(f"evaluation score is missing: {key}")
            try:
                numeric = float(raw)
            except Exception as exc:
                raise ValueError(f"evaluation score is invalid: {key}") from exc
            scores[key] = max(0.0, min(10.0, round(numeric, 2)))
        return scores

    @staticmethod
    def _render_prompt_template(template: Optional[str], record: Any) -> str:
        prompt_template = str(template or "").strip()
        if not prompt_template:
            return ""
        if not isinstance(record, dict):
            return prompt_template

        placeholder_values = record.get("__placeholder_values__") if isinstance(record.get("__placeholder_values__"), dict) else {}
        if not placeholder_values:
            return prompt_template

        def replace(match: re.Match[str]) -> str:
            key = str(match.group(1) or "").strip()
            if not key:
                return match.group(0)
            if not PROMPT_PLACEHOLDER_NAME_PATTERN.fullmatch(key):
                return match.group(0)
            value = placeholder_values.get(key)
            if value is None:
                return match.group(0)
            return str(value)

        return re.sub(r"\{([^{}]+)\}", replace, prompt_template)
