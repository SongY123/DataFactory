from __future__ import annotations

import json
import re
from typing import Any
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

from utils.config_loader import get_config
from ..service.dataset_service import DatasetService

_GLOBAL_SYSTEM_PROMPT = (
    "Global response policy for DataAgentFactory Factory Agent: "
    "Always answer in English. Do not answer in Chinese. "
    "If the user's message is not written in English, infer the intent internally and still reply in English. "
    "Keep dataset names, field names, code, tags, placeholders, and literal UI labels unchanged when they are part of the answer."
)

_PAGE_PROMPTS = {
    "dataset_management": (
        "You are the DataAgentFactory assistant for Dataset Management. "
        "Help the user inspect datasets, understand metadata, import flows, tags, and data preparation decisions. "
        "If dataset search results are provided, summarize only the datasets that satisfy the request, do not mention non-matching datasets, "
        "and stay concise. When you recommend datasets, explicitly mention the exact dataset names. "
        "The UI will render a single View button after the answer when matching datasets exist, so do not invent custom button markup."
    ),
    "reasoning_data_synthesis": (
        "You are the DataAgentFactory assistant for Reasoning Data Synthesis. "
        "Assume the user's request and prompt drafts are written in English unless they explicitly say otherwise, and always answer in English. "
        "Help the user design prompts, placeholder mappings, evaluation strategies, dataset configuration, and synthesis workflow decisions. "
        "Be concrete and implementation-oriented."
    ),
    "agentic_trajectory_synthesis": (
        "You are the DataAgentFactory assistant for Agentic Trajectory Synthesis. "
        "Assume the user's request and prompt drafts are written in English unless they explicitly say otherwise, and always answer in English. "
        "Help the user design synthesis prompts, dataset selection, action trajectories, environment choices, and task configuration tradeoffs. "
        "Be concise and grounded in the workflow."
    ),
}

_DATASET_SEARCH_VERBS = (
    "find",
    "search",
    "look for",
    "looking for",
    "need",
    "want",
    "recommend",
    "show me",
    "list",
    "match",
    "suitable",
    "帮我找",
    "搜索",
    "查找",
    "推荐",
    "需要",
    "想找",
    "有没有",
)
_DATASET_SEARCH_NOUNS = ("dataset", "datasets", "data set", "数据集")
_FORMAT_ALIASES = {
    "csv": {"csv"},
    "tsv": {"tsv"},
    "json": {"json"},
    "jsonl": {"jsonl"},
    "parquet": {"parquet"},
    "excel": {"excel", "xlsx", "xls"},
    "xlsx": {"excel", "xlsx"},
    "xls": {"excel", "xls"},
    "sqlite": {"sqlite"},
    "text": {"text", "txt", "markdown", "md"},
}
_LANGUAGE_ALIASES = {
    "zh": {"zh", "chinese", "中文", "汉语"},
    "en": {"en", "english", "英文", "英语"},
    "multi": {"multi", "multilingual", "bilingual", "多语言"},
}
_STATUS_ALIASES = {
    "uploaded": {"uploaded", "ready", "available", "已上传", "可用"},
    "downloading": {"downloading", "importing", "processing", "下载中", "导入中"},
    "failed": {"failed", "error", "失败", "错误"},
}
_SIZE_ALIASES = {
    "kb": {"kb"},
    "mb": {"mb"},
    "gb": {"gb"},
}
_NOISE_TERMS = {
    "dataset",
    "datasets",
    "data",
    "for",
    "with",
    "that",
    "about",
    "need",
    "want",
    "find",
    "search",
    "look",
    "show",
    "me",
    "please",
    "suitable",
    "recommended",
    "推荐",
    "数据集",
    "查找",
    "搜索",
    "需要",
    "想要",
}
_NON_NAME_TERMS = {
    "bigger",
    "larger",
    "greater",
    "smaller",
    "less",
    "more",
    "than",
    "over",
    "under",
    "least",
    "most",
}


class WorkflowAssistantService:
    def __init__(self, dataset_service: DatasetService | None = None) -> None:
        self.dataset_service = dataset_service or DatasetService()

    def _prepare_chat_state(
        self,
        *,
        page_key: str,
        messages: list[dict[str, Any]],
        session_id: str | None = None,
        page_context: dict[str, Any] | None = None,
        user_id: int | None = None,
    ) -> dict[str, Any]:
        normalized_page = str(page_key or "").strip().lower()
        if normalized_page not in _PAGE_PROMPTS:
            raise ValueError(f"unsupported page_key: {page_key}")

        normalized_messages: list[dict[str, str]] = []
        for item in messages or []:
            role = str((item or {}).get("role") or "").strip().lower()
            content = str((item or {}).get("content") or "").strip()
            if role not in {"user", "assistant"} or not content:
                continue
            normalized_messages.append({"role": role, "content": content})

        if not normalized_messages:
            raise ValueError("messages must contain at least one user/assistant message")

        api_key = str(get_config("model.dashscope.api_key", "") or "").strip()
        base_url = str(get_config("model.dashscope.base_url", "") or "").strip()
        model_name = str(get_config("model.dashscope.model_name", "qwen-max") or "qwen-max").strip()
        if not api_key:
            raise ValueError("DashScope API key is not configured")
        if not base_url:
            raise ValueError("DashScope base_url is not configured")
        if not model_name:
            raise ValueError("DashScope model_name is not configured")

        tool_context_message = None
        dataset_matches: list[dict[str, Any]] = []
        dataset_search_meta: dict[str, Any] | None = None
        prompt_recommendation: dict[str, Any] | None = None
        latest_user_message = next(
            (item.get("content", "") for item in reversed(normalized_messages) if item.get("role") == "user"),
            "",
        )
        normalized_page_context = page_context if isinstance(page_context, dict) else {}

        if normalized_page == "dataset_management" and user_id is not None:
            search_context = self._maybe_search_datasets(user_id=user_id, query=latest_user_message)
            dataset_matches = search_context.get("dataset_matches", [])
            dataset_search_meta = search_context.get("search_meta")
            tool_context_message = search_context.get("tool_message")
            dataset_candidates = search_context.get("candidate_items", [])
        else:
            dataset_candidates = []
        if normalized_page == "dataset_management":
            tool_context_message = self._merge_tool_messages(
                tool_context_message,
                self._build_dataset_management_context_message(normalized_page_context),
            )

        if normalized_page == "reasoning_data_synthesis":
            tool_context_message = self._merge_tool_messages(
                tool_context_message,
                self._build_reasoning_context_message(normalized_page_context),
            )
        if normalized_page == "agentic_trajectory_synthesis":
            tool_context_message = self._merge_tool_messages(
                tool_context_message,
                self._build_trajectory_context_message(normalized_page_context),
            )

        payload_messages = [
            {"role": "system", "content": _GLOBAL_SYSTEM_PROMPT},
            {"role": "system", "content": _PAGE_PROMPTS[normalized_page]},
        ]
        if tool_context_message:
            payload_messages.append({"role": "system", "content": tool_context_message})
        payload_messages.extend(normalized_messages)

        return {
            "page_key": normalized_page,
            "session_id": str(session_id or "").strip() or None,
            "messages": normalized_messages,
            "payload_messages": payload_messages,
            "page_context": normalized_page_context,
            "latest_user_message": latest_user_message,
            "dataset_matches": dataset_matches,
            "dataset_search": dataset_search_meta,
            "dataset_candidates": dataset_candidates,
            "api_key": api_key,
            "base_url": base_url,
            "model_name": model_name,
            "prompt_optimization": self._is_prompt_optimization_intent(latest_user_message),
        }

    def chat(
        self,
        *,
        page_key: str,
        messages: list[dict[str, Any]],
        session_id: str | None = None,
        page_context: dict[str, Any] | None = None,
        user_id: int | None = None,
    ) -> dict[str, Any]:
        state = self._prepare_chat_state(
            page_key=page_key,
            messages=messages,
            session_id=session_id,
            page_context=page_context,
            user_id=user_id,
        )

        if state["prompt_optimization"]:
            try:
                if state["page_key"] == "reasoning_data_synthesis":
                    structured = self._optimize_reasoning_prompt(
                        base_url=state["base_url"],
                        api_key=state["api_key"],
                        model_name=state["model_name"],
                        messages=state["payload_messages"],
                        query=state["latest_user_message"],
                        page_context=state["page_context"],
                    )
                    answer = structured.get("answer") or ""
                    prompt_recommendation = structured.get("prompt_recommendation")
                elif state["page_key"] == "agentic_trajectory_synthesis":
                    structured = self._optimize_trajectory_prompt(
                        base_url=state["base_url"],
                        api_key=state["api_key"],
                        model_name=state["model_name"],
                        messages=state["payload_messages"],
                        query=state["latest_user_message"],
                        page_context=state["page_context"],
                    )
                    answer = structured.get("answer") or ""
                    prompt_recommendation = structured.get("prompt_recommendation")
                else:
                    raise ValueError("unsupported prompt optimization page")
            except Exception:
                answer = self._chat_completion(
                    base_url=state["base_url"],
                    api_key=state["api_key"],
                    model_name=state["model_name"],
                    messages=state["payload_messages"],
                )
        else:
            answer = self._chat_completion(
                base_url=state["base_url"],
                api_key=state["api_key"],
                model_name=state["model_name"],
                messages=state["payload_messages"],
            )
        dataset_view_items = self._resolve_dataset_view_items(
            answer=answer,
            dataset_candidates=state.get("dataset_candidates") or [],
        )
        return {
            "page_key": state["page_key"],
            "session_id": state["session_id"],
            "answer": answer,
            "provider": "dashscope",
            "model_name": state["model_name"],
            "dataset_matches": state["dataset_matches"],
            "dataset_search": state["dataset_search"],
            "dataset_view_items": dataset_view_items,
            "prompt_recommendation": prompt_recommendation,
        }

    def stream_chat(
        self,
        *,
        page_key: str,
        messages: list[dict[str, Any]],
        session_id: str | None = None,
        page_context: dict[str, Any] | None = None,
        user_id: int | None = None,
    ):
        state = self._prepare_chat_state(
            page_key=page_key,
            messages=messages,
            session_id=session_id,
            page_context=page_context,
            user_id=user_id,
        )

        yield {
            "event": "opened",
            "data": {
                "page_key": state["page_key"],
                "session_id": state["session_id"],
                "provider": "dashscope",
                "model_name": state["model_name"],
            },
        }

        answer_parts: list[str] = []
        for chunk in self._chat_completion_stream(
            base_url=state["base_url"],
            api_key=state["api_key"],
            model_name=state["model_name"],
            messages=state["payload_messages"],
        ):
            if not chunk:
                continue
            answer_parts.append(chunk)
            yield {
                "event": "delta",
                "data": {
                    "content": chunk,
                },
            }

        answer = "".join(answer_parts).strip()
        prompt_recommendation = None
        if state["prompt_optimization"]:
            try:
                if state["page_key"] == "reasoning_data_synthesis":
                    structured = self._optimize_reasoning_prompt(
                        base_url=state["base_url"],
                        api_key=state["api_key"],
                        model_name=state["model_name"],
                        messages=state["payload_messages"],
                        query=state["latest_user_message"],
                        page_context=state["page_context"],
                    )
                elif state["page_key"] == "agentic_trajectory_synthesis":
                    structured = self._optimize_trajectory_prompt(
                        base_url=state["base_url"],
                        api_key=state["api_key"],
                        model_name=state["model_name"],
                        messages=state["payload_messages"],
                        query=state["latest_user_message"],
                        page_context=state["page_context"],
                    )
                else:
                    structured = {}
                if not answer:
                    answer = str(structured.get("answer") or "").strip()
                prompt_recommendation = structured.get("prompt_recommendation")
            except Exception:
                prompt_recommendation = None

        if not answer:
            raise ValueError("workflow assistant returned empty content")

        dataset_view_items = self._resolve_dataset_view_items(
            answer=answer,
            dataset_candidates=state.get("dataset_candidates") or [],
        )

        yield {
            "event": "done",
            "data": {
                "ok": True,
                "page_key": state["page_key"],
                "session_id": state["session_id"],
                "provider": "dashscope",
                "model_name": state["model_name"],
                "answer": answer,
                "dataset_matches": state["dataset_matches"],
                "dataset_search": state["dataset_search"],
                "dataset_view_items": dataset_view_items,
                "prompt_recommendation": prompt_recommendation,
            },
        }

    @staticmethod
    def _merge_tool_messages(primary: str | None, secondary: str | None) -> str | None:
        first = str(primary or "").strip()
        second = str(secondary or "").strip()
        if first and second:
            return f"{first}\n\n{second}"
        return first or second or None

    def _maybe_search_datasets(self, *, user_id: int, query: str) -> dict[str, Any]:
        normalized_query = str(query or "").strip()
        if not normalized_query:
            return {}
        if not self._is_dataset_search_intent(normalized_query):
            return {}

        filters = self._extract_dataset_filters(normalized_query)
        search_result = self.dataset_service.search_datasets(user_id=user_id, filters=filters)
        items = list(search_result.get("items") or [])
        limited_items = [self._summarize_dataset_item(item) for item in items[:5]]
        tool_lines = [
            "Dataset search tool result:",
            f"- query: {normalized_query}",
            f"- filters: {json.dumps(filters, ensure_ascii=False)}",
            f"- total_matches: {len(items)}",
            "- Only the listed matches satisfy the current request. Do not mention datasets outside this result set.",
        ]
        if limited_items:
            tool_lines.append("- matches:")
            for item in limited_items:
                tool_lines.append(
                    "  - "
                    f"#{item['id']} | {item['name']} | type={item['type']} | source={item['source_kind']} | "
                    f"status={item['status']} | format={','.join(item['format_tags']) or 'n/a'} | "
                    f"language={','.join(item['language_tags']) or 'n/a'} | size={item['size']} | "
                    f"note={item['note'] or 'n/a'}"
                )
        else:
            tool_lines.append("- matches: none")

        return {
            "dataset_matches": limited_items,
            "candidate_items": items,
            "search_meta": {
                "query": normalized_query,
                "filters": filters,
                "total_matches": len(items),
                "shown_matches": len(limited_items),
                "matched_dataset_ids": [int(item.get("id") or 0) for item in items if int(item.get("id") or 0) > 0],
            },
            "tool_message": "\n".join(tool_lines),
        }

    @staticmethod
    def _resolve_dataset_view_items(*, answer: str, dataset_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        candidates = [item for item in dataset_candidates if isinstance(item, dict)]
        if not candidates:
            return []
        normalized_answer = str(answer or "").strip().lower()
        if not normalized_answer:
            return candidates

        matched_items = []
        for item in sorted(candidates, key=lambda value: len(str(value.get("name") or "")), reverse=True):
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            if name.lower() in normalized_answer:
                matched_items.append(item)

        if matched_items:
            matched_names = {str(item.get("name") or "").strip().lower() for item in matched_items}
            return [item for item in candidates if str(item.get("name") or "").strip().lower() in matched_names]

        return candidates

    @staticmethod
    def _is_dataset_search_intent(query: str) -> bool:
        text = str(query or "").strip().lower()
        if not text:
            return False
        has_search_verb = any(WorkflowAssistantService._query_contains_alias(text, token) for token in _DATASET_SEARCH_VERBS)
        has_dataset_noun = any(WorkflowAssistantService._query_contains_alias(text, token) for token in _DATASET_SEARCH_NOUNS)
        has_filter_clue = any(
            WorkflowAssistantService._query_contains_alias(text, alias)
            for values in (*_FORMAT_ALIASES.values(), *_LANGUAGE_ALIASES.values(), *_STATUS_ALIASES.values(), *_SIZE_ALIASES.values())
            for alias in values
        )
        if WorkflowAssistantService._extract_min_size_bytes(text) is not None:
            has_filter_clue = True
        has_name_keyword = bool(WorkflowAssistantService._extract_name_keyword(text))
        return (has_search_verb and (has_dataset_noun or has_filter_clue or has_name_keyword)) or (has_dataset_noun and has_filter_clue)

    @staticmethod
    def _extract_dataset_filters(query: str) -> dict[str, Any]:
        text = str(query or "").strip()
        lowered = text.lower()
        min_size_bytes = WorkflowAssistantService._extract_min_size_bytes(text)
        format_tags = sorted(
            {
                tag
                for canonical, aliases in _FORMAT_ALIASES.items()
                if any(WorkflowAssistantService._query_contains_alias(lowered, alias) for alias in aliases)
                for tag in {canonical}
            }
        )
        language_tags = sorted(
            {
                canonical
                for canonical, aliases in _LANGUAGE_ALIASES.items()
                if any(WorkflowAssistantService._query_contains_alias(lowered, alias) for alias in aliases)
            }
        )
        statuses = sorted(
            {
                canonical
                for canonical, aliases in _STATUS_ALIASES.items()
                if any(WorkflowAssistantService._query_contains_alias(lowered, alias) for alias in aliases)
            }
        )
        size_levels = sorted(
            {
                canonical
                for canonical, aliases in _SIZE_ALIASES.items()
                if any(WorkflowAssistantService._query_contains_alias(lowered, alias) for alias in aliases)
            }
        )
        name_keyword = WorkflowAssistantService._extract_name_keyword(text)
        return {
            "name_keyword": name_keyword,
            "format_tags": format_tags,
            "language_tags": language_tags,
            "size_levels": [] if min_size_bytes is not None else size_levels,
            "min_size_bytes": min_size_bytes,
            "statuses": statuses,
        }

    @staticmethod
    def _extract_name_keyword(query: str) -> str | None:
        text = str(query or "").strip()
        if not text:
            return None
        quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', text)
        for left, right in quoted:
            candidate = (left or right or "").strip()
            if candidate:
                return candidate[:256]

        named_match = re.search(r"(?:named|called|dataset)\s+([A-Za-z0-9._/-]{2,})", text, flags=re.IGNORECASE)
        if named_match:
            candidate = named_match.group(1).strip()
            if candidate and candidate.lower() not in _NON_NAME_TERMS:
                return candidate[:256]

        topical_match = re.search(r"(?:about|for|on)\s+([A-Za-z0-9._/-]{2,})", text, flags=re.IGNORECASE)
        if topical_match:
            candidate = topical_match.group(1).strip()
            if candidate and candidate.lower() not in _NON_NAME_TERMS:
                return candidate[:256]

        tokens = re.findall(r"[A-Za-z0-9._/-]+", text)
        meaningful = [token for token in tokens if token.lower() not in _NOISE_TERMS and len(token) >= 2]
        if len(meaningful) == 1 and len(meaningful[0]) <= 64:
            return meaningful[0][:256]
        return None

    @staticmethod
    def _extract_min_size_bytes(query: str) -> int | None:
        text = str(query or "").strip().lower()
        if not text:
            return None
        match = re.search(
            r"(?:bigger than|larger than|greater than|more than|over|at least|>=)\s*(\d+(?:\.\d+)?)\s*(kb|mb|gb)\b",
            text,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        value = float(match.group(1))
        unit = str(match.group(2) or "").strip().lower()
        multipliers = {
            "kb": 1024,
            "mb": 1024 * 1024,
            "gb": 1024 * 1024 * 1024,
        }
        multiplier = multipliers.get(unit)
        if multiplier is None:
            return None
        return max(0, int(value * multiplier))

    @staticmethod
    def _query_contains_alias(query: str, alias: str) -> bool:
        text = str(query or "").strip().lower()
        token = str(alias or "").strip().lower()
        if not text or not token:
            return False
        if re.fullmatch(r"[a-z0-9._/-]+", token):
            pattern = rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])"
            return re.search(pattern, text) is not None
        return token in text

    @staticmethod
    def _summarize_dataset_item(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": int(item.get("id") or 0),
            "name": str(item.get("name") or "").strip(),
            "type": str(item.get("type") or "").strip(),
            "source_kind": str(item.get("source_kind") or "").strip(),
            "status": str(item.get("status") or "").strip(),
            "format_tags": list(item.get("format_tags") or []),
            "language_tags": list(item.get("language_tags") or []),
            "size": int(item.get("size") or 0),
            "note": str(item.get("note") or "").strip(),
        }

    @staticmethod
    def _is_prompt_optimization_intent(query: str) -> bool:
        text = str(query or "").strip().lower()
        if not text:
            return False
        keywords = (
            "prompt",
            "improve",
            "improved",
            "optimize",
            "better",
            "rewrite",
            "revise",
            "adjust",
            "refine",
            "polish",
            "tune",
            "recommend",
            "modify",
            "make it better",
            "优化",
            "调整",
            "改进",
            "修改",
            "重写",
            "润色",
            "推荐",
        )
        return any(token in text for token in keywords)

    @staticmethod
    def _determine_reasoning_prompt_target(query: str, page_context: dict[str, Any]) -> str:
        text = str(query or "").strip().lower()
        if "evaluation prompt" in text or "eval prompt" in text:
            return "evaluation"
        if "synthesis prompt" in text:
            return "synthesis"
        if "评估" in text:
            return "evaluation"
        if "合成" in text:
            return "synthesis"
        active = str((page_context or {}).get("active_prompt_tab") or "").strip().lower()
        return "evaluation" if active == "evaluation" else "synthesis"

    @staticmethod
    def _build_reasoning_context_message(page_context: dict[str, Any]) -> str | None:
        if not page_context:
            return None
        active_prompt_tab = str(page_context.get("active_prompt_tab") or "synthesis").strip().lower()
        synthesis_prompt = str(page_context.get("synthesis_prompt") or "").strip()
        evaluation_prompt = str(page_context.get("evaluation_prompt") or "").strip()
        default_synthesis_prompt = str(page_context.get("default_synthesis_prompt") or "").strip()
        default_evaluation_prompt = str(page_context.get("default_evaluation_prompt") or "").strip()
        evaluation_enabled = bool(page_context.get("evaluation_enabled"))
        selected_task = page_context.get("selected_task") if isinstance(page_context.get("selected_task"), dict) else {}
        evaluation_summary = page_context.get("evaluation_summary") if isinstance(page_context.get("evaluation_summary"), dict) else {}
        task_result_preview = page_context.get("task_result_preview")
        task_result_preview = task_result_preview if isinstance(task_result_preview, list) else []

        lines = [
            "Reasoning Data Synthesis page context:",
            f"- active_prompt_tab: {active_prompt_tab or 'synthesis'}",
            f"- evaluation_enabled: {str(evaluation_enabled).lower()}",
            f"- current_synthesis_prompt:\n{synthesis_prompt or '(empty)'}",
            f"- current_evaluation_prompt:\n{evaluation_prompt or '(empty)'}",
            f"- default_synthesis_prompt:\n{default_synthesis_prompt or '(empty)'}",
            f"- default_evaluation_prompt:\n{default_evaluation_prompt or '(empty)'}",
        ]

        if selected_task:
            lines.extend([
                "- selected_task:",
                f"  - id: {selected_task.get('id')}",
                f"  - status: {selected_task.get('status')}",
                f"  - source_label: {selected_task.get('source_label')}",
                f"  - evaluation_enabled: {selected_task.get('evaluation_enabled')}",
                f"  - progress: {selected_task.get('progress')}",
            ])

        if evaluation_summary:
            lines.append(f"- evaluation_summary: {json.dumps(evaluation_summary, ensure_ascii=False)}")
        if task_result_preview:
            lines.append(f"- task_result_preview: {json.dumps(task_result_preview[:5], ensure_ascii=False)}")

        return "\n".join(lines)

    @staticmethod
    def _build_dataset_management_context_message(page_context: dict[str, Any]) -> str | None:
        if not page_context:
            return None
        dataset_count = int(page_context.get("dataset_count") or 0)
        importing_count = int(page_context.get("importing_count") or 0)
        generated_count = int(page_context.get("generated_count") or 0)
        current_filters = page_context.get("current_filters") if isinstance(page_context.get("current_filters"), dict) else {}
        visible_datasets = page_context.get("visible_datasets")
        visible_datasets = visible_datasets if isinstance(visible_datasets, list) else []

        lines = [
            "Dataset Management page context:",
            f"- dataset_count: {dataset_count}",
            f"- importing_count: {importing_count}",
            f"- generated_count: {generated_count}",
        ]
        if current_filters:
            lines.append(f"- current_filters: {json.dumps(current_filters, ensure_ascii=False)}")
        if visible_datasets:
            lines.append(f"- visible_datasets: {json.dumps(visible_datasets[:20], ensure_ascii=False)}")
        return "\n".join(lines)

    @staticmethod
    def _build_trajectory_context_message(page_context: dict[str, Any]) -> str | None:
        if not page_context:
            return None
        synthesis_prompt = str(page_context.get("synthesis_prompt") or "").strip()
        default_synthesis_prompt = str(page_context.get("default_synthesis_prompt") or "").strip()
        action_tags = page_context.get("default_action_tags")
        action_tags = action_tags if isinstance(action_tags, list) else []
        selected_dataset_names = page_context.get("selected_dataset_names")
        selected_dataset_names = selected_dataset_names if isinstance(selected_dataset_names, list) else []
        selected_task = page_context.get("selected_task") if isinstance(page_context.get("selected_task"), dict) else {}

        lines = [
            "Agentic Trajectory Synthesis page context:",
            f"- current_synthesis_prompt:\n{synthesis_prompt or '(empty)'}",
            f"- default_synthesis_prompt:\n{default_synthesis_prompt or '(empty)'}",
            f"- default_action_tags: {json.dumps(action_tags, ensure_ascii=False)}",
            f"- selected_datasets: {json.dumps(selected_dataset_names, ensure_ascii=False)}",
        ]

        if selected_task:
            lines.extend([
                "- selected_task:",
                f"  - id: {selected_task.get('id')}",
                f"  - status: {selected_task.get('status')}",
                f"  - dataset_name: {selected_task.get('dataset_name')}",
                f"  - progress: {selected_task.get('progress')}",
            ])

        return "\n".join(lines)

    def _optimize_reasoning_prompt(
        self,
        *,
        base_url: str,
        api_key: str,
        model_name: str,
        messages: list[dict[str, str]],
        query: str,
        page_context: dict[str, Any],
    ) -> dict[str, Any]:
        target = self._determine_reasoning_prompt_target(query, page_context)
        instruction = {
            "role": "system",
            "content": (
                "You are improving prompts for DataAgentFactory Reasoning Data Synthesis. "
                "Answer in English. Use the current prompt, the default template, and any task evaluation summary as references. "
                "Return valid JSON only with this schema: "
                '{"answer":"string","prompt_recommendation":{"target":"synthesis|evaluation","prompt":"string","changes":["string"]}}. '
                "The recommended prompt must be directly usable in the UI textarea. "
                f"Prefer the {target} prompt unless the user explicitly asks for the other prompt."
            ),
        }
        raw = self._chat_completion(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            messages=[instruction, *messages],
        )
        parsed = self._extract_json_payload(raw)
        answer = str(parsed.get("answer") or "").strip()
        recommendation = parsed.get("prompt_recommendation")
        normalized_recommendation = self._normalize_prompt_recommendation(recommendation, fallback_target=target)
        if not answer:
            answer = "I recommended an updated prompt based on the current tab and available task feedback."
            if normalized_recommendation and normalized_recommendation.get("changes"):
                answer = f"{answer} Main changes: " + "; ".join(normalized_recommendation["changes"][:3])
        return {
            "answer": answer,
            "prompt_recommendation": normalized_recommendation,
        }

    def _optimize_trajectory_prompt(
        self,
        *,
        base_url: str,
        api_key: str,
        model_name: str,
        messages: list[dict[str, str]],
        query: str,
        page_context: dict[str, Any],
    ) -> dict[str, Any]:
        default_action_tags = page_context.get("default_action_tags")
        action_tags = default_action_tags if isinstance(default_action_tags, list) and default_action_tags else [
            "Analyze",
            "Understand",
            "Code",
            "Execute",
            "Answer",
        ]
        instruction = {
            "role": "system",
            "content": (
                "You are improving prompts for DataAgentFactory Agentic Trajectory Synthesis. "
                "Answer in English. Use the current synthesis prompt, the default template, and the user's task intent as references. "
                "Recommend an action sequence and an updated synthesis prompt that stays compatible with the existing workflow. "
                "Return valid JSON only with this schema: "
                '{"answer":"string","prompt_recommendation":{"target":"synthesis","prompt":"string","changes":["string"],"action_sequence":["Analyze","Understand","Code","Execute","Answer"]}}. '
                "The recommended prompt must be directly usable in the UI textarea. "
                f"Prefer this canonical action vocabulary when possible: {json.dumps(action_tags, ensure_ascii=False)}."
            ),
        }
        raw = self._chat_completion(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            messages=[instruction, *messages],
        )
        parsed = self._extract_json_payload(raw)
        recommendation = self._normalize_prompt_recommendation(
            parsed.get("prompt_recommendation"),
            fallback_target="synthesis",
        )
        answer = str(parsed.get("answer") or "").strip()
        if not answer:
            answer = "I recommended an updated synthesis prompt and action sequence for the current trajectory task."
            if recommendation and recommendation.get("changes"):
                answer = f"{answer} Main changes: " + "; ".join(recommendation["changes"][:3])
        return {
            "answer": answer,
            "prompt_recommendation": recommendation,
        }

    @staticmethod
    def _extract_json_payload(raw: str) -> dict[str, Any]:
        text = str(raw or "").strip()
        if not text:
            raise ValueError("workflow assistant returned empty content")
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        raise ValueError("workflow assistant returned invalid prompt recommendation JSON")

    @staticmethod
    def _normalize_prompt_recommendation(value: Any, fallback_target: str) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            return None
        prompt_text = str(value.get("prompt") or "").strip()
        if not prompt_text:
            return None
        target = str(value.get("target") or fallback_target or "synthesis").strip().lower()
        if target not in {"synthesis", "evaluation"}:
            target = fallback_target if fallback_target in {"synthesis", "evaluation"} else "synthesis"
        raw_changes = value.get("changes")
        changes = []
        if isinstance(raw_changes, list):
            for item in raw_changes:
                normalized = str(item or "").strip()
                if normalized:
                    changes.append(normalized)
        raw_action_sequence = value.get("action_sequence")
        action_sequence = []
        if isinstance(raw_action_sequence, list):
            for item in raw_action_sequence:
                normalized = str(item or "").strip()
                if normalized:
                    action_sequence.append(normalized)
        return {
            "target": target,
            "prompt": prompt_text,
            "changes": changes[:6],
            "action_sequence": action_sequence[:12],
        }

    @staticmethod
    def _build_chat_endpoint(base_url: str) -> str:
        normalized = str(base_url or "").strip().rstrip("/")
        if not normalized:
            raise ValueError("base_url is required")
        if normalized.endswith("/chat/completions"):
            return normalized
        if normalized.endswith("/v1"):
            return f"{normalized}/chat/completions"
        return f"{normalized}/v1/chat/completions"

    def _chat_completion(self, *, base_url: str, api_key: str, model_name: str, messages: list[dict[str, str]]) -> str:
        endpoint = self._build_chat_endpoint(base_url)
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.2,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib_request.Request(
            endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=180) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise ValueError(f"workflow assistant model request failed ({exc.code}): {detail or exc.reason}") from exc
        except URLError as exc:
            raise ValueError(f"workflow assistant model connection failed: {exc}") from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("workflow assistant returned invalid JSON") from exc

        choices = data.get("choices") or []
        if not choices:
            raise ValueError("workflow assistant returned no choices")
        content = (((choices[0] or {}).get("message") or {}).get("content") or "").strip()
        if not content:
            raise ValueError("workflow assistant returned empty content")
        return content

    def _chat_completion_stream(
        self,
        *,
        base_url: str,
        api_key: str,
        model_name: str,
        messages: list[dict[str, str]],
    ):
        endpoint = self._build_chat_endpoint(base_url)
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.2,
            "stream": True,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib_request.Request(
            endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=180) as resp:
                data_lines: list[str] = []
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line:
                        if not data_lines:
                            continue
                        data_text = "\n".join(data_lines).strip()
                        data_lines = []
                        if not data_text:
                            continue
                        if data_text == "[DONE]":
                            break
                        try:
                            data = json.loads(data_text)
                        except json.JSONDecodeError:
                            continue
                        chunk = self._extract_stream_chunk(data)
                        if chunk:
                            yield chunk
                        continue
                    if line.startswith("data:"):
                        data_lines.append(line[5:].strip())
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise ValueError(f"workflow assistant model request failed ({exc.code}): {detail or exc.reason}") from exc
        except URLError as exc:
            raise ValueError(f"workflow assistant model connection failed: {exc}") from exc

    @staticmethod
    def _extract_stream_chunk(payload: dict[str, Any]) -> str:
        choices = payload.get("choices") or []
        if not choices:
            return ""
        delta = ((choices[0] or {}).get("delta") or {})
        content = delta.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = str(item.get("text") or item.get("content") or "").strip()
                    if text:
                        parts.append(text)
            return "".join(parts)
        reasoning_content = delta.get("reasoning_content")
        if isinstance(reasoning_content, str):
            return reasoning_content
        return ""
