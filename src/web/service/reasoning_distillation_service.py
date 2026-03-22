from __future__ import annotations

import csv
import io
import json
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
DEFAULT_REASONING_PROMPT = (
    "Generate one training record for reasoning synthesis. "
    "The reasoning field must always be wrapped in exactly one pair of <think></think> tags. "
    "Do not add markdown fences."
)


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
        prompt: Optional[str],
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
        source_context = self._build_source_context(
            user_id=user_id,
            source_type=source_type,
            source_dataset_id=source_dataset_id,
            source_task_id=source_task_id,
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
                error_message = "no reasoning samples were generated"

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
                    "token_count": token_count,
                    "status": "completed",
                    "error_message": None,
                },
                "dataset_record": distilled["record"],
                "token_count": token_count,
            }
        except Exception as exc:
            logger.exception("Reasoning distillation item failed. task_id=%s item=%s", task_id, source_item.get("item_key"))
            return {
                "status": "failed",
                "record_payload": {
                    "task_id": task_id,
                    "user_id": user_id,
                    "source_type": source_context["source_type"],
                    "source_ref_id": int(source_context["source_ref_id"]),
                    "item_key": str(source_item.get("item_key") or "item"),
                    "prompt_text": str(source_item.get("prompt_text") or self._derive_prompt_text(source_item) or "Distillation source item"),
                    "reasoning_text": "Distillation failed for this item.",
                    "answer_text": "",
                    "record_json": json.dumps({}, ensure_ascii=False),
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
    ) -> Dict[str, Any]:
        normalized_type = str(source_type or "").strip().lower()
        if normalized_type == "dataset":
            dataset = self.dataset_dao.get_dataset_by_id(dataset_id=int(source_dataset_id or 0), user_id=user_id)
            if dataset is None:
                raise ValueError("source dataset not found")
            dataset_row = dataset.to_dict(include_internal=True)
            items = self._load_dataset_source_items(dataset_row)
            return {
                "source_type": "dataset",
                "source_ref_id": int(dataset.id),
                "source_label": str(dataset.name),
                "language": str(dataset.language or "multi"),
                "origin_dataset_id": int(dataset.id),
                "origin_task_type": None,
                "origin_task_id": None,
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
                "items": items,
            }

        raise ValueError(f"unsupported source_type: {source_type}")

    def _load_dataset_source_items(self, dataset_row: Dict[str, Any]) -> List[Dict[str, Any]]:
        file_path = str(dataset_row.get("file_path") or "").strip()
        if not file_path:
            fallback_rows = dataset_row.get("sample_data")
            if isinstance(fallback_rows, str):
                try:
                    fallback_rows = json.loads(fallback_rows)
                except Exception:
                    fallback_rows = []
            rows = fallback_rows if isinstance(fallback_rows, list) else []
            return [self._make_dataset_item(record=row, source_path="sample_data", row_index=index) for index, row in enumerate(rows[:MAX_SOURCE_ITEMS])]

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
            items.extend(self._parse_dataset_file_to_items(file, limit=MAX_SOURCE_ITEMS - len(items)))
        return items[:MAX_SOURCE_ITEMS]

    def _parse_dataset_file_to_items(self, file: Path, limit: int) -> List[Dict[str, Any]]:
        suffix = file.suffix.lower()
        if limit <= 0:
            return []
        if suffix == ".csv":
            text = file.read_text(encoding="utf-8", errors="ignore")
            reader = csv.DictReader(io.StringIO(text))
            return [self._make_dataset_item(record=row, source_path=str(file), row_index=index) for index, row in zip(range(limit), reader)]
        if suffix == ".json":
            text = file.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                return []
            parsed = json.loads(text)
            rows = parsed if isinstance(parsed, list) else [parsed]
            return [self._make_dataset_item(record=row, source_path=str(file), row_index=index) for index, row in enumerate(rows[:limit])]
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
                rows.append(self._make_dataset_item(record=parsed, source_path=str(file), row_index=index))
            return rows
        if suffix in {".txt", ".md"}:
            rows: List[Dict[str, Any]] = []
            for index, line in enumerate(file.read_text(encoding="utf-8", errors="ignore").splitlines()):
                if len(rows) >= limit:
                    break
                text = line.strip()
                if not text:
                    continue
                rows.append(self._make_dataset_item(record={"text": text}, source_path=str(file), row_index=index))
            return rows
        return []

    @staticmethod
    def _make_dataset_item(record: Any, source_path: str, row_index: int) -> Dict[str, Any]:
        prompt_text = ReasoningDistillationService._derive_prompt_text({"record": record}) or f"Dataset item {row_index + 1}"
        return {
            "item_key": f"{Path(str(source_path)).name}:{row_index + 1}",
            "source_path": str(source_path),
            "record": record if isinstance(record, dict) else {"value": record},
            "prompt_text": prompt_text,
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
            "prompt": str(prompt or "").strip() or DEFAULT_REASONING_PROMPT,
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

    @staticmethod
    def _derive_messages(source_type: str, record: Any) -> List[Dict[str, str]]:
        if source_type == "trajectory_task":
            question = str((record or {}).get("question") or "").strip()
            if question:
                return [{"role": "user", "content": question}]

        if isinstance(record, dict):
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
            for key in ("answer", "response", "output", "assistant", "target", "label"):
                value = str(record.get(key) or "").strip()
                if value:
                    return value[:MAX_TEXT_CHARS]
        return ""

    @staticmethod
    def _derive_prompt_text(source_item: Dict[str, Any]) -> str:
        record = source_item.get("record") if isinstance(source_item, dict) else source_item
        if isinstance(record, dict):
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
    def _ensure_think_tags(value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return "<think></think>"
        if text.startswith("<think>") and text.endswith("</think>"):
            return text
        return f"<think>{text}</think>"
