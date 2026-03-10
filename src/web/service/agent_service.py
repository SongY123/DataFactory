from __future__ import annotations

import csv
import hashlib
import io
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request
from uuid import uuid4

from fastapi import UploadFile

from utils.config_loader import get_config
from utils.logger import logger


class AgentReportService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, dict[str, Any]] = {}
        self._models = self._resolve_models()

    @staticmethod
    def _resolve_models() -> list[str]:
        configured = get_config("agent.models", None)
        if isinstance(configured, list):
            models = [str(x).strip() for x in configured if str(x).strip()]
            if models:
                return models
        return ["gpt-4o-mini", "gpt-4.1-mini", "deepseek-chat", "qwen-max"]

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

    @staticmethod
    def _decode_text(raw: bytes) -> str:
        for enc in ("utf-8", "utf-8-sig", "gbk", "latin-1"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="ignore")

    @staticmethod
    def _extract_pdf_preview(raw: bytes) -> dict[str, Any] | None:
        try:
            from pypdf import PdfReader
        except Exception as exc:
            return {
                "page_count": 0,
                "preview": f"[PDF support missing: failed to import pypdf: {exc}]",
                "extractor": "none",
            }

        try:
            reader = PdfReader(io.BytesIO(raw))
            total_pages = len(reader.pages)
            collected: list[str] = []

            for page in reader.pages[:8]:
                text = (page.extract_text() or "").strip()
                if text:
                    collected.append(text)

            if not collected:
                return {
                    "page_count": total_pages,
                    "preview": "[PDF text extraction returned empty content. This PDF may be image-only and requires OCR.]",
                    "extractor": "pypdf",
                }

            merged = "\n\n".join(collected)
            return {
                "page_count": total_pages,
                "preview": merged[:3000],
                "extractor": "pypdf",
            }
        except Exception as exc:
            logger.warning("Failed to parse PDF via pypdf: %s", exc)
            return {
                "page_count": 0,
                "preview": f"[PDF parsing failed: {exc}]",
                "extractor": "pypdf",
            }

    def _summarize_file(self, file_name: str, raw: bytes) -> dict[str, Any]:
        suffix = Path(file_name).suffix.lower()
        digest = hashlib.sha256(raw).hexdigest()
        base: dict[str, Any] = {
            "file_name": Path(file_name).name,
            "file_ext": suffix or "unknown",
            "size_bytes": len(raw),
            "sha256": digest,
        }

        try:
            if suffix == ".pdf":
                pdf = self._extract_pdf_preview(raw)
                if pdf is not None:
                    base["preview"] = str(pdf.get("preview") or "")
                    base["page_count"] = int(pdf.get("page_count") or 0)
                    base["extractor"] = str(pdf.get("extractor") or "pypdf")
                    preview_text = base["preview"]
                    if preview_text:
                        lines = [line for line in preview_text.splitlines() if line.strip()]
                        base["line_preview"] = lines[:15]
                    return base

                base["preview"] = "[PDF parsing unavailable due to unknown runtime issue.]"
                return base

            if suffix == ".csv":
                text = self._decode_text(raw)
                reader = csv.reader(io.StringIO(text))
                rows = [row for _, row in zip(range(6), reader)]
                base["columns"] = rows[0] if rows else []
                base["row_preview"] = rows[1:6] if len(rows) > 1 else []
                base["preview"] = text[:1200]
                return base

            if suffix in {".json", ".geojson"}:
                parsed = json.loads(self._decode_text(raw).strip() or "{}")
                if isinstance(parsed, list):
                    sample = parsed[:3]
                    base["record_count_preview"] = len(sample)
                    base["sample"] = sample
                    if sample and isinstance(sample[0], dict):
                        base["keys"] = sorted(sample[0].keys())[:30]
                elif isinstance(parsed, dict):
                    base["keys"] = sorted(parsed.keys())[:50]
                    base["sample"] = parsed
                else:
                    base["sample"] = parsed
                base["preview"] = json.dumps(base.get("sample"), ensure_ascii=False)[:1200]
                return base

            if suffix == ".jsonl":
                rows: list[Any] = []
                for line in self._decode_text(raw).splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
                    if len(rows) >= 3:
                        break
                base["sample"] = rows
                if rows and isinstance(rows[0], dict):
                    base["keys"] = sorted(rows[0].keys())[:30]
                base["preview"] = json.dumps(rows, ensure_ascii=False)[:1200]
                return base

            text_preview = self._decode_text(raw)[:2000]
            base["preview"] = text_preview
            if text_preview:
                lines = [line for line in text_preview.splitlines() if line.strip()]
                base["line_preview"] = lines[:10]
            return base
        except Exception as exc:
            logger.warning("Failed to parse uploaded file as structured text: %s", exc)
            base["preview"] = self._decode_text(raw)[:800]
            return base

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
            candidates = [
                data.get("choices", [{}])[0].get("message", {}).get("content") if isinstance(data.get("choices"), list) else None,
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

    def _invoke_llm(self, *, model: str, user_prompt: str, llm: dict[str, str]) -> str | None:
        endpoint = self._trim_text(llm.get("endpoint"))
        if not endpoint:
            return None

        model_name = self._trim_text(llm.get("model_name")) or model
        api_key = self._trim_text(llm.get("api_key"))
        timeout = int(get_config("agent.llm.timeout_seconds", 60) or 60)

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a data analyst agent. Return a concise markdown report."},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "stream": False,
        }

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        req = request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
            data = json.loads(raw)
            return self._extract_llm_text(data)
        except (TimeoutError, error.URLError, error.HTTPError, json.JSONDecodeError) as exc:
            logger.warning("LLM call failed (%s), fallback to local report generation: %s", llm.get("provider") or "default", exc)
            return None

    def _build_local_report(
        self,
        *,
        summary: dict[str, Any],
        prompt: str,
        model: str,
        previous_report: str | None = None,
    ) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        file_name = summary.get("file_name", "uploaded_file")
        file_ext = summary.get("file_ext", "unknown")
        size_bytes = int(summary.get("size_bytes") or 0)
        preview = str(summary.get("preview", "")).strip()

        preview_block = "```\n" + preview[:1500] + "\n```"
        if previous_report:
            return "\n".join(
                [
                    "# Revised Analysis Report",
                    f"- Model: `{model}`",
                    f"- Revised At: `{now}`",
                    f"- File: `{file_name}` ({file_ext}, {size_bytes} bytes)",
                    "",
                    "## Revision Prompt",
                    prompt,
                    "",
                    "## Updated Report",
                    "The report has been revised according to your prompt. Key updates:",
                    f"1. Focus aligned to: {prompt}",
                    "2. Risk and quality checks retained and tightened.",
                    "3. Action plan reprioritized for immediate execution.",
                    "",
                    "## Current File Snapshot",
                    preview_block,
                    "",
                    "## Prior Report (for traceability)",
                    str(previous_report)[:1800],
                ]
            ).strip()

        return "\n".join(
            [
                "# Analysis Report",
                f"- Model: `{model}`",
                f"- Generated At: `{now}`",
                f"- File: `{file_name}` ({file_ext}, {size_bytes} bytes)",
                "",
                "## User Intent",
                prompt,
                "",
                "## File Overview",
                f"- File hash (SHA-256): `{summary.get('sha256', '-')}`",
                f"- Parsed keys/columns: `{summary.get('keys') or summary.get('columns') or []}`",
                "",
                "## Core Observations",
                "1. The file is readable and has extractable structured/textual content.",
                "2. The sample preview suggests potential fields suitable for feature engineering and validation.",
                "3. A schema consistency check should be run before downstream training/evaluation.",
                "",
                "## Risk And Quality Checks",
                "- Missing values and outliers should be profiled.",
                "- Field type conflicts should be normalized.",
                "- Data leakage-sensitive columns should be reviewed.",
                "",
                "## Recommended Next Actions",
                "1. Add schema + null-rate profiling job.",
                "2. Define train/eval split policy and leakage guard.",
                "3. Create task-specific feature extraction checklist.",
                "",
                "## Content Preview",
                preview_block,
            ]
        ).strip()

    def _compose_report(
        self,
        *,
        summary: dict[str, Any],
        prompt: str,
        model: str,
        previous_report: str | None,
        llm: dict[str, str],
    ) -> str:
        safe_prompt = self._trim_text(prompt) or "Please generate a structured analysis report for this file."
        prompt_payload = {
            "task": "revise_report" if previous_report else "initial_report",
            "user_prompt": safe_prompt,
            "file_summary": summary,
            "previous_report": previous_report,
        }

        llm_text = self._invoke_llm(
            model=model,
            user_prompt=json.dumps(prompt_payload, ensure_ascii=False),
            llm=llm,
        )
        if llm_text:
            return llm_text

        return self._build_local_report(
            summary=summary,
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

        summary = self._summarize_file(file.filename, raw)
        llm = self._build_runtime_llm(
            llm_provider=llm_provider,
            llm_endpoint=llm_endpoint,
            llm_api_key=llm_api_key,
            llm_model_name=llm_model_name,
        )
        chosen_model = self._select_model(model)
        report = self._compose_report(
            summary=summary,
            prompt=self._trim_text(prompt),
            model=chosen_model,
            previous_report=None,
            llm=llm,
        )

        session_id = uuid4().hex
        with self._lock:
            self._sessions[session_id] = {
                "session_id": session_id,
                "file_name": summary.get("file_name"),
                "summary": summary,
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
            "file_name": summary.get("file_name"),
            "summary": summary,
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
            updated_report = self._compose_report(
                summary=current.get("summary") or {},
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
            return {"session_id": revised["session_id"], "model": revised["model"], "answer": revised["report"], "llm": revised["llm"]}

        llm = self._build_runtime_llm(
            llm_provider=llm_provider,
            llm_endpoint=llm_endpoint,
            llm_api_key=llm_api_key,
            llm_model_name=llm_model_name,
        )

        chosen_model = self._select_model(model)
        if self._trim_text(report):
            report_text = str(report)
            pseudo_summary = {
                "file_name": "existing_report.md",
                "file_ext": ".md",
                "size_bytes": len(report_text.encode("utf-8")),
                "preview": report_text[:1500],
                "sha256": hashlib.sha256(report_text.encode("utf-8")).hexdigest(),
            }
            answer = self._compose_report(
                summary=pseudo_summary,
                prompt=user_text,
                model=chosen_model,
                previous_report=report_text,
                llm=llm,
            )
            return {"model": chosen_model, "answer": answer, "llm": llm}

        llm_answer = self._invoke_llm(model=chosen_model, user_prompt=user_text, llm=llm)
        if llm_answer:
            return {"model": chosen_model, "answer": llm_answer, "llm": llm}

        return {
            "model": chosen_model,
            "llm": llm,
            "answer": "No report session found. Upload a file first, then revise via prompts.",
        }
