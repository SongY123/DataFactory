from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Any
from urllib import error, request
from uuid import uuid4

from fastapi import UploadFile

from utils.config_loader import get_config
from utils.logger import logger
from .file_parser import AgentFileParser, FileParseError, ParsedFileContent


class AgentReportService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, dict[str, Any]] = {}
        self._models = self._resolve_models()
        self._file_parser = AgentFileParser()

    @staticmethod
    def _resolve_models() -> list[str]:
        configured = get_config("agent.models", None)
        if isinstance(configured, list):
            models = [str(x).strip() for x in configured if str(x).strip()]
            if models:
                return models
        return ["gpt-4o-mini"]

    @staticmethod
    def _trim_text(value: str | None) -> str:
        return str(value or "").strip()

    def list_models(self) -> list[dict[str, str]]:
        return [{"id": x, "label": x} for x in self._models]

    def _select_model(self, model: str | None) -> str:
        candidate = self._trim_text(model)
        if candidate:
            return candidate
        return self._models[0]

    def _build_runtime_llm(
        self,
        *,
        llm_provider: str | None,
        llm_endpoint: str | None,
        llm_api_key: str | None,
        llm_model_name: str | None,
    ) -> dict[str, str]:
        provider = self._trim_text(llm_provider).lower()
        if provider not in {"api", "local"}:
            provider = ""

        endpoint = self._trim_text(llm_endpoint)
        api_key = self._trim_text(llm_api_key)
        model_name = self._trim_text(llm_model_name)

        if provider in {"api", "local"} and not endpoint:
            raise ValueError(f"{provider} mode requires model endpoint/address")

        if not provider and not endpoint:
            endpoint = self._trim_text(get_config("agent.llm.endpoint", ""))
            api_key = api_key or self._trim_text(get_config("agent.llm.api_key", ""))
            model_name = model_name or self._trim_text(get_config("agent.llm.model_name", ""))
            provider = "api" if endpoint else ""

        return {
            "provider": provider,
            "endpoint": endpoint,
            "api_key": api_key,
            "model_name": model_name,
        }

    @staticmethod
    def _extract_llm_text(data: Any) -> str | None:
        if isinstance(data, str) and data.strip():
            return data.strip()

        if isinstance(data, dict):
            choices = data.get("choices")
            first_choice = choices[0] if isinstance(choices, list) and choices else {}
            message = first_choice.get("message") if isinstance(first_choice, dict) else {}

            candidates = [
                message.get("content") if isinstance(message, dict) else None,
                data.get("response"),
                data.get("answer"),
                data.get("output"),
                data.get("text"),
                data.get("message"),
            ]
            for item in candidates:
                if isinstance(item, str) and item.strip():
                    return item.strip()

        return None

    @staticmethod
    def _extract_error_detail(raw: str) -> str:
        text = str(raw or "").strip()
        if not text:
            return ""

        try:
            body = json.loads(text)
        except json.JSONDecodeError:
            return text[:240]

        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                for key in ("message", "detail", "msg", "error"):
                    value = err.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()[:240]
            for key in ("detail", "message", "error", "msg"):
                value = body.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()[:240]

        return text[:240]

    def _explain_llm_failure(self, *, exc: Exception, endpoint: str) -> str:
        if isinstance(exc, TimeoutError):
            return "Request timed out. Check model service latency or increase timeout."

        if isinstance(exc, error.HTTPError):
            status = int(getattr(exc, "code", 0) or 0)
            raw = ""
            try:
                raw = exc.read().decode("utf-8", errors="ignore")
            except Exception:
                raw = ""
            detail = self._extract_error_detail(raw)

            if status == 404:
                return f"Endpoint not found (404): {endpoint}. Check that it ends with /v1/chat/completions."
            if status in {401, 403}:
                base = "Authentication failed. Check API key or access policy."
                return f"{base} Detail: {detail}" if detail else base
            if status == 400:
                base = "Bad request from model service. Check model name and payload format."
                return f"{base} Detail: {detail}" if detail else base

            base = f"Model service returned HTTP {status}."
            return f"{base} Detail: {detail}" if detail else base

        if isinstance(exc, error.URLError):
            reason = getattr(exc, "reason", exc)
            return f"Network error while connecting to model endpoint: {reason}"

        if isinstance(exc, json.JSONDecodeError):
            return "Model service returned invalid JSON. Check if endpoint is OpenAI-compatible."

        return f"Model request failed: {exc}"

    def _invoke_llm(self, *, model: str, payload: dict[str, Any], llm: dict[str, str]) -> tuple[str | None, str | None]:
        endpoint = self._trim_text(llm.get("endpoint"))
        if not endpoint:
            return None, None

        model_name = self._trim_text(llm.get("model_name")) or model
        api_key = self._trim_text(llm.get("api_key"))
        timeout = int(get_config("agent.llm.timeout_seconds", 60) or 60)

        llm_payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a senior data analyst. Use only parsed file context to answer. "
                        "Never output raw binary, archive signatures, base64, hash digests, or debug traces."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
            ],
            "temperature": 0.2,
            "stream": False,
        }

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        req = request.Request(
            endpoint,
            data=json.dumps(llm_payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        provider = llm.get("provider") or "default"
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
        except (TimeoutError, error.URLError, error.HTTPError) as exc:
            reason = self._explain_llm_failure(exc=exc, endpoint=endpoint)
            logger.warning("LLM call failed (%s): %s", provider, reason)
            return None, reason

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            reason = self._explain_llm_failure(exc=exc, endpoint=endpoint)
            logger.warning("LLM response decode failed (%s): %s", provider, reason)
            return None, reason

        text = self._extract_llm_text(data)
        if text:
            return text, None

        reason = "Model service returned empty or incompatible response format."
        logger.warning("LLM response extraction failed (%s): %s", provider, reason)
        return None, reason

    @staticmethod
    def _table_structure_lines(parsed: ParsedFileContent) -> list[str]:
        lines: list[str] = []
        for table in parsed.table_structures[:8]:
            name = str(table.get("name") or "table")
            columns = table.get("columns") or []
            sample_rows = table.get("sample_rows") or []
            lines.append(f"- {name}: {len(columns)} columns, {len(sample_rows)} sampled rows")
        return lines

    def _build_local_report(
        self,
        *,
        parsed: ParsedFileContent,
        prompt: str,
        model: str,
        previous_report: str | None = None,
    ) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        structure_lines = self._table_structure_lines(parsed)
        structure_block = structure_lines if structure_lines else ["- No table structure detected; analyzed as plain text document."]

        if previous_report:
            parts = [
                "# Updated File Analysis",
                f"- Model: `{model}`",
                f"- Time: `{now}`",
                "",
                "## New Question",
                prompt,
                "",
                "## Updated Conclusion",
                "1. Re-analyzed using parsed file context and your new question.",
                "2. Output intentionally excludes raw binary and debug internals.",
                "",
                "## File Profile",
                f"- Name: `{parsed.file_name}`",
                f"- Type: `{parsed.file_type}` ({parsed.mime_type})",
                f"- Size: `{parsed.size_bytes}` bytes",
                f"- Parser: `{parsed.parser}`",
                "",
                "## Structure",
                *structure_block,
                "",
                "## Recommended Next Steps",
                "1. Validate high-risk fields first.",
                "2. Run schema/null/outlier checks before downstream tasks.",
            ]
            return "\n".join(parts).strip()

        parts = [
            "# File Analysis Result",
            f"- Model: `{model}`",
            f"- Time: `{now}`",
            "",
            "## User Question",
            prompt,
            "",
            "## File Profile",
            f"- Name: `{parsed.file_name}`",
            f"- Type: `{parsed.file_type}` ({parsed.mime_type})",
            f"- Size: `{parsed.size_bytes}` bytes",
            f"- Parser: `{parsed.parser}`",
            "",
            "## Structure",
            *structure_block,
            "",
            "## Key Findings",
            "1. File content was parsed successfully and analyzed from structured context.",
            "2. Model input excludes binary payload and corrupted text sequences.",
            "3. You can continue with follow-up prompts for deeper analysis.",
            "",
            "## Risk and Actions",
            "1. Validate schema and missing values.",
            "2. Check outliers and inconsistent field formats.",
            "3. Add quality gates before training/evaluation steps.",
        ]
        return "\n".join(parts).strip()

    def _compose_report(
        self,
        *,
        parsed: ParsedFileContent,
        prompt: str,
        model: str,
        previous_report: str | None,
        llm: dict[str, str],
    ) -> str:
        safe_prompt = self._trim_text(prompt) or "Analyze this file and provide actionable conclusions."
        payload = {
            "task": "revise_report" if previous_report else "initial_report",
            "user_question": safe_prompt,
            "file_context": parsed.to_prompt_payload(),
            "previous_report": previous_report or "",
        }

        llm_text, llm_error = self._invoke_llm(model=model, payload=payload, llm=llm)
        if llm_text:
            return llm_text

        if self._trim_text(llm.get("endpoint")):
            raise ValueError(f"Model is not correctly connected. {llm_error or 'Unknown model service error.'}")

        return self._build_local_report(
            parsed=parsed,
            prompt=safe_prompt,
            model=model,
            previous_report=previous_report,
        )

    async def analyze_upload(
        self,
        *,
        file: UploadFile,
        model: str | None,
        prompt: str | None,
        llm_provider: str | None = None,
        llm_endpoint: str | None = None,
        llm_api_key: str | None = None,
        llm_model_name: str | None = None,
    ) -> dict[str, Any]:
        if file is None or not file.filename:
            raise ValueError("file is required")

        raw = await file.read()
        if not raw:
            raise ValueError("uploaded file is empty")

        try:
            parsed = self._file_parser.parse(
                file_name=file.filename,
                content_type=file.content_type,
                raw=raw,
            )
        except FileParseError as exc:
            raise ValueError(str(exc)) from exc

        llm = self._build_runtime_llm(
            llm_provider=llm_provider,
            llm_endpoint=llm_endpoint,
            llm_api_key=llm_api_key,
            llm_model_name=llm_model_name,
        )

        chosen_model = self._select_model(model)
        report = self._compose_report(
            parsed=parsed,
            prompt=self._trim_text(prompt),
            model=chosen_model,
            previous_report=None,
            llm=llm,
        )

        session_id = uuid4().hex
        with self._lock:
            self._sessions[session_id] = {
                "session_id": session_id,
                "file_name": parsed.file_name,
                "file_type": parsed.file_type,
                "parsed": parsed,
                "model": chosen_model,
                "llm": llm,
                "report": report,
                "history": [
                    {
                        "type": "init",
                        "prompt": self._trim_text(prompt),
                        "report": report,
                        "at": datetime.now(timezone.utc).isoformat(),
                    }
                ],
            }

        return {
            "session_id": session_id,
            "model": chosen_model,
            "llm": llm,
            "file_name": parsed.file_name,
            "file_type": parsed.file_type,
            "report": report,
        }

    def revise_report(
        self,
        *,
        session_id: str,
        prompt: str,
        model: str | None = None,
        llm_provider: str | None = None,
        llm_endpoint: str | None = None,
        llm_api_key: str | None = None,
        llm_model_name: str | None = None,
    ) -> dict[str, Any]:
        sid = self._trim_text(session_id)
        if not sid:
            raise ValueError("session_id is required")

        user_prompt = self._trim_text(prompt)
        if not user_prompt:
            raise ValueError("prompt is required")

        with self._lock:
            current = self._sessions.get(sid)
            if not current:
                raise ValueError(f"session not found: {sid}")

            llm = self._build_runtime_llm(
                llm_provider=llm_provider or (current.get("llm") or {}).get("provider"),
                llm_endpoint=llm_endpoint or (current.get("llm") or {}).get("endpoint"),
                llm_api_key=llm_api_key or (current.get("llm") or {}).get("api_key"),
                llm_model_name=llm_model_name or (current.get("llm") or {}).get("model_name"),
            )

            chosen_model = self._select_model(model or current.get("model"))
            parsed: ParsedFileContent = current.get("parsed")
            if not parsed:
                raise ValueError("session parsed content missing")

            updated_report = self._compose_report(
                parsed=parsed,
                prompt=user_prompt,
                model=chosen_model,
                previous_report=str(current.get("report") or ""),
                llm=llm,
            )

            current["model"] = chosen_model
            current["llm"] = llm
            current["report"] = updated_report
            current.setdefault("history", []).append(
                {
                    "type": "revise",
                    "prompt": user_prompt,
                    "report": updated_report,
                    "at": datetime.now(timezone.utc).isoformat(),
                }
            )

            return {
                "session_id": sid,
                "model": chosen_model,
                "llm": llm,
                "file_name": current.get("file_name"),
                "file_type": current.get("file_type"),
                "report": updated_report,
                "history_count": len(current.get("history") or []),
            }

    def chat(
        self,
        *,
        message: str,
        model: str | None = None,
        session_id: str | None = None,
        report: str | None = None,
        llm_provider: str | None = None,
        llm_endpoint: str | None = None,
        llm_api_key: str | None = None,
        llm_model_name: str | None = None,
    ) -> dict[str, Any]:
        user_text = self._trim_text(message)
        if not user_text:
            raise ValueError("message is required")

        sid = self._trim_text(session_id)
        if sid and sid in self._sessions:
            revised = self.revise_report(
                session_id=sid,
                prompt=user_text,
                model=model,
                llm_provider=llm_provider,
                llm_endpoint=llm_endpoint,
                llm_api_key=llm_api_key,
                llm_model_name=llm_model_name,
            )
            return {
                "session_id": revised["session_id"],
                "model": revised["model"],
                "answer": revised["report"],
                "llm": revised["llm"],
            }

        llm = self._build_runtime_llm(
            llm_provider=llm_provider,
            llm_endpoint=llm_endpoint,
            llm_api_key=llm_api_key,
            llm_model_name=llm_model_name,
        )

        chosen_model = self._select_model(model)
        if self._trim_text(report):
            report_text = self._trim_text(report)
            pseudo = ParsedFileContent(
                file_name="existing_report.md",
                file_type="md",
                mime_type="text/markdown",
                size_bytes=len(report_text.encode("utf-8")),
                parser="inline-report",
                text_summary=report_text[:4000],
                table_structures=[],
                sample_content=[],
                warnings=[],
                metadata={"source": "chat_report"},
            )
            answer = self._compose_report(
                parsed=pseudo,
                prompt=user_text,
                model=chosen_model,
                previous_report=report_text,
                llm=llm,
            )
            return {"model": chosen_model, "answer": answer, "llm": llm}

        llm_answer, llm_error = self._invoke_llm(
            model=chosen_model,
            payload={"task": "chat", "user_question": user_text},
            llm=llm,
        )
        if llm_answer:
            return {"model": chosen_model, "answer": llm_answer, "llm": llm}

        if self._trim_text(llm.get("endpoint")):
            raise ValueError(f"Model is not correctly connected. {llm_error or 'Unknown model service error.'}")

        return {
            "model": chosen_model,
            "llm": llm,
            "answer": "No report session found. Upload a file first, then revise via prompts.",
        }


