from __future__ import annotations

import asyncio
import json
import re
import sys
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator

from pydantic import BaseModel, Field, ValidationError

from agents.context import get_event_bus, get_python_interpreter
from agents.event_bus import create_agent_error_event, create_agent_finish_event, create_agent_start_event, create_stream_event
from agents.result_utils import extract_agent_result_text
from utils.config_loader import get_config
from utils.logger import logger
from utils.model_factory import create_model
from web.service.file_parser import AgentFileParser, ParsedFileContent

ANALYZE_STAGE = "Analyze"
UNDERSTAND_STAGE = "Understand"
CODE_STAGE = "Code"
EXECUTE_STAGE = "Execute"
ANSWER_STAGE = "Answer"

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_BANNED_CODE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("subprocess", "subprocess execution is not allowed"),
    ("os.system", "shell execution is not allowed"),
    ("powershell", "shell execution is not allowed"),
    ("cmd.exe", "shell execution is not allowed"),
    ("requests", "network access is not allowed"),
    ("httpx", "network access is not allowed"),
    ("urllib.request", "network access is not allowed"),
    ("socket", "network access is not allowed"),
    ("shutil.rmtree", "deleting files is not allowed"),
    (".unlink(", "deleting files is not allowed"),
    (".rmdir(", "deleting files is not allowed"),
    (".write_text(", "writing files is not allowed"),
    (".write_bytes(", "writing files is not allowed"),
    (".to_csv(", "writing files is not allowed"),
    (".to_excel(", "writing files is not allowed"),
    ("savefig(", "writing files is not allowed"),
    ("eval(", "dynamic code execution is not allowed"),
    ("exec(", "dynamic code execution is not allowed"),
    ("__import__", "dynamic imports are not allowed"),
)

_SYSTEM_PROMPT = """
You are a senior data analyst operating inside a closed analysis loop.

Workflow rules:
- The API is responsible for thinking, generating hypotheses, and generating Python when more evidence is needed.
- The execution environment is solely responsible for running code.
- Never claim that code ran unless an execution result is explicitly provided to you.
- Base every conclusion on the parsed file context and the execution results that are given to you.
- When you are asked for JSON, return only valid JSON. Do not wrap it in markdown fences.
- Match the user's language unless they explicitly ask for another language.
- Any generated Python must be read-only, self-contained, and focused on the selected file.
- Generated Python must not use network access, shell commands, subprocesses, or file writes.
- Prefer concise evidence over long narration.
""".strip()


class AnalyzeResponse(BaseModel):
    analyze_summary: str = Field(..., min_length=1)
    key_observations: list[str] = Field(default_factory=list)
    next_focus: str = Field(..., min_length=1)


class UnderstandResponse(BaseModel):
    understanding: str = Field(..., min_length=1)
    evidence_gaps: list[str] = Field(default_factory=list)
    ready_to_answer: bool = Field(default=False)
    needs_code_execution: bool = Field(default=False)
    code_purpose: str | None = None
    python_code: str | None = None


@dataclass(slots=True)
class ExecutionResult:
    round_index: int
    purpose: str
    code: str
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    rejected: bool = False
    rejection_reason: str = ""
    script_path: str = ""

    def to_model_payload(self) -> dict[str, Any]:
        return {
            "round": self.round_index,
            "purpose": self.purpose,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "rejected": self.rejected,
            "rejection_reason": self.rejection_reason,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "script_path": self.script_path,
        }


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _truncate_text(value: Any, limit: int) -> str:
    text = _clean_text(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}\n...[truncated]"


def _first_non_empty(values: list[str]) -> str:
    for value in values:
        if _clean_text(value):
            return _clean_text(value)
    return ""


def _format_markdown_list(items: list[str]) -> str:
    cleaned = [_clean_text(item) for item in items if _clean_text(item)]
    if not cleaned:
        return "- None"
    return "\n".join(f"- {item}" for item in cleaned)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    source = _clean_text(text)
    if not source:
        return None

    decoder = json.JSONDecoder()
    candidates = [source]
    candidates.extend(match.group(1).strip() for match in _JSON_BLOCK_RE.finditer(source))

    for candidate in candidates:
        stripped = candidate.strip()
        if not stripped:
            continue

        try:
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

        for match in re.finditer(r"\{", stripped):
            try:
                payload, end = decoder.raw_decode(stripped[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and match.start() + end <= len(stripped):
                return payload

    return None


def _search_schema_payload(value: Any, schema: type[BaseModel]) -> BaseModel | None:
    if value is None:
        return None

    if isinstance(value, schema):
        return value

    if isinstance(value, BaseModel):
        try:
            return schema.model_validate(value.model_dump())
        except ValidationError:
            pass

    if isinstance(value, dict):
        try:
            return schema.model_validate(value)
        except ValidationError:
            pass
        for child in value.values():
            parsed = _search_schema_payload(child, schema)
            if parsed is not None:
                return parsed
        return None

    if isinstance(value, (list, tuple)):
        for item in value:
            parsed = _search_schema_payload(item, schema)
            if parsed is not None:
                return parsed
        return None

    if isinstance(value, str):
        payload = _extract_json_object(value)
        if payload is None:
            return None
        try:
            return schema.model_validate(payload)
        except ValidationError:
            return None

    for attr_name in ("metadata", "content", "text"):
        attr_value = getattr(value, attr_name, None)
        if attr_value is None:
            continue
        parsed = _search_schema_payload(attr_value, schema)
        if parsed is not None:
            return parsed

    return None


class IterativeFileAnalyzer:
    def __init__(
        self,
        *,
        query: str,
        selected_file_path: str,
        workspace_dir: Path,
        file_index: int = 1,
        total_files: int = 1,
    ) -> None:
        self.query = _clean_text(query)
        self.selected_file_path = _clean_text(selected_file_path)
        self.workspace_dir = Path(workspace_dir).resolve()
        self.file_index = max(1, int(file_index or 1))
        self.total_files = max(1, int(total_files or 1))
        self.file_path = (self.workspace_dir / self.selected_file_path).resolve()
        self.file_name = self.file_path.name or self.selected_file_path
        self.event_bus = get_event_bus()
        self.python_executable = str(get_python_interpreter() or sys.executable)
        self.parser = AgentFileParser()
        self.model = create_model(stream=False)
        self.streaming_text_model = create_model(stream=True)
        self.max_rounds = max(1, int(get_config("agent.single_file.max_rounds", 4) or 4))
        self.execution_timeout_seconds = max(
            5,
            int(get_config("agent.single_file.execution_timeout_seconds", 45) or 45),
        )
        self.max_execution_output_chars = max(
            1000,
            int(get_config("agent.single_file.max_execution_output_chars", 6000) or 6000),
        )
        self.execution_live_read_bytes = max(
            64,
            int(get_config("agent.single_file.execution_live_read_bytes", 256) or 256),
        )
        self.max_model_retries = max(1, int(get_config("agent.single_file.max_model_retries", 2) or 2))
        self.stream_chunk_chars = max(6, int(get_config("agent.single_file.stream_chunk_chars", 140) or 140))
        self.stream_chunk_delay_seconds = max(
            0.1,
            float(get_config("agent.single_file.stream_chunk_delay_seconds", 0.02) or 0.02),
        )
        self.current_stage = ANALYZE_STAGE

    @property
    def file_heading(self) -> str:
        return f"`{self.file_name}`"

    def _build_prompt_file_context(self, parsed: ParsedFileContent) -> dict[str, Any]:
        payload = parsed.to_prompt_payload()
        payload["text_summary"] = _truncate_text(payload.get("text_summary"), 6000)
        payload["sample_content"] = list(payload.get("sample_content") or [])[:8]
        payload["table_structures"] = list(payload.get("table_structures") or [])[:4]
        payload["warnings"] = list(payload.get("warnings") or [])[:8]
        return {
            "relative_path": self.selected_file_path,
            "file_name": parsed.file_name,
            "file_type": parsed.file_type,
            "mime_type": parsed.mime_type,
            "size_bytes": parsed.size_bytes,
            "parser_payload": payload,
        }

    async def _publish_start(self, stage: str, task_description: str) -> None:
        if self.event_bus is None:
            return
        await self.event_bus.publish(await create_agent_start_event(stage, task_description=task_description))

    async def _publish_finish(self, stage: str, result: str) -> None:
        if self.event_bus is None:
            return
        await self.event_bus.publish(await create_agent_finish_event(stage, result=result))

    async def _publish_error(self, stage: str, exc: Exception) -> None:
        if self.event_bus is None:
            return
        await self.event_bus.publish(await create_agent_error_event(stage, exc))

    async def _publish_stream_event(self, stage: str, content: str, *, is_final: bool = False) -> None:
        if self.event_bus is None:
            return
        await self.event_bus.publish(
            await create_stream_event(
                agent_name=stage,
                chunk={"result": str(content or "")},
                is_final=is_final,
            )
        )

    async def _publish_stream(self, stage: str, content: str) -> None:
        if self.event_bus is None:
            return

        text = str(content or "")
        if not text:
            return

        chunks: list[str] = []
        buffer = ""
        for line in text.splitlines(keepends=True):
            if len(buffer) + len(line) <= self.stream_chunk_chars:
                buffer += line
                continue
            if buffer:
                chunks.append(buffer)
                buffer = ""
            if len(line) <= self.stream_chunk_chars:
                buffer = line
                continue
            for start in range(0, len(line), self.stream_chunk_chars):
                chunks.append(line[start : start + self.stream_chunk_chars])
        if buffer:
            chunks.append(buffer)
        if not chunks:
            chunks = [text]

        cumulative = ""
        last_index = len(chunks) - 1
        for index, chunk in enumerate(chunks):
            cumulative += chunk
            await self._publish_stream_event(stage, cumulative, is_final=index == last_index)
            if self.stream_chunk_delay_seconds > 0 and index != last_index:
                await asyncio.sleep(self.stream_chunk_delay_seconds)

    async def _publish_streamed_finish(self, stage: str, result: str) -> None:
        await self._publish_stream(stage, result)
        await self._publish_finish(stage, result)

    async def _invoke_structured(self, *, schema: type[BaseModel], user_prompt: str) -> BaseModel:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        last_error: Exception | None = None

        for attempt in range(1, self.max_model_retries + 1):
            response = None
            raw_text = ""

            try:
                response = await self.model(messages=messages, structured_model=schema)
                parsed = _search_schema_payload(response, schema)
                if parsed is not None:
                    return parsed
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Structured model call failed for %s on %s (attempt %s/%s): %s",
                    schema.__name__,
                    self.selected_file_path,
                    attempt,
                    self.max_model_retries,
                    exc,
                )

            try:
                if response is None:
                    response = await self.model(messages=messages)
                raw_text = extract_agent_result_text(response)
                parsed = _search_schema_payload(raw_text, schema)
                if parsed is not None:
                    return parsed
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Fallback model call failed for %s on %s (attempt %s/%s): %s",
                    schema.__name__,
                    self.selected_file_path,
                    attempt,
                    self.max_model_retries,
                    exc,
                )

            repair_note = (
                f"Your previous response could not be parsed into {schema.__name__}. "
                "Return only valid JSON with every required field and no markdown fences."
            )
            messages.append({"role": "assistant", "content": _truncate_text(raw_text, 4000) or "{}"})
            messages.append({"role": "user", "content": repair_note})

        raise ValueError(
            f"Model did not return a valid {schema.__name__} payload for {self.selected_file_path}."
        ) from last_error

    async def _invoke_text(self, *, user_prompt: str) -> str:
        response = await self.model(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
        text = extract_agent_result_text(response).strip()
        if text:
            return text
        raise ValueError(f"Model returned an empty answer for {self.selected_file_path}.")

    async def _iterate_stream_output(self, response: Any) -> AsyncGenerator[Any, None]:
        if hasattr(response, "__aiter__"):
            async for item in response:
                yield item
            return
        yield response

    async def _invoke_text_streaming(
        self,
        *,
        stage: str,
        user_prompt: str,
        formatter,
    ) -> str:
        response = await self.streaming_text_model(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )

        latest_text = ""
        async for item in self._iterate_stream_output(response):
            current_text = extract_agent_result_text(item).strip()
            if not current_text or current_text == latest_text:
                continue
            latest_text = current_text
            await self._publish_stream_event(stage, formatter(latest_text))

        if latest_text:
            return latest_text
        raise ValueError(f"Model returned an empty streaming answer for {self.selected_file_path}.")

    def _build_analyze_prompt(self, parsed: ParsedFileContent) -> str:
        file_context = json.dumps(self._build_prompt_file_context(parsed), ensure_ascii=False, indent=2)
        user_request = self.query or "Please analyze the selected file comprehensively."
        return f"""
Analyze the selected file before any execution happens.

User request:
{user_request}

Selected file context:
{file_context}

Return JSON with this exact shape:
{{
  "analyze_summary": "A concise but substantive initial analysis of the file and the user's goal.",
  "key_observations": ["2-5 evidence-backed observations that are already visible from the parsed context"],
  "next_focus": "What needs to be verified next before the final answer is trustworthy"
}}
""".strip()

    def _build_understand_prompt(
        self,
        *,
        parsed: ParsedFileContent,
        analyze_response: AnalyzeResponse,
        execution_results: list[ExecutionResult],
        round_index: int,
    ) -> str:
        file_context = json.dumps(self._build_prompt_file_context(parsed), ensure_ascii=False, indent=2)
        execution_payload = json.dumps(
            [item.to_model_payload() for item in execution_results],
            ensure_ascii=False,
            indent=2,
        )
        user_request = self.query or "Please analyze the selected file comprehensively."
        return f"""
We are in the Understand stage of a closed-loop file analysis workflow.

User request:
{user_request}

Selected file context:
{file_context}

Initial Analyze stage result:
{json.dumps(analyze_response.model_dump(), ensure_ascii=False, indent=2)}

Previous execution results:
{execution_payload if execution_results else "[]"}

Current round: {round_index} of {self.max_rounds}

Decide whether more code execution is needed.

Rules:
- If the current evidence is already sufficient, set "ready_to_answer" to true, set "needs_code_execution" to false, and leave "python_code" empty.
- If more evidence is needed, set "ready_to_answer" to false, set "needs_code_execution" to true, and provide complete Python 3.11 code in "python_code".
- Any Python must be read-only, self-contained, and only analyze the selected file.
- The execution environment will provide:
  - TARGET_FILE_PATH: a pathlib.Path to the selected file
  - WORKSPACE_DIR: a pathlib.Path to the runtime workspace
  - emit(value): helper that prints dict/list values as JSON
- Prefer printing concise JSON or concise text, not huge tables.
- Do not use network access, shell commands, subprocesses, or file writes.
- For CSV/XLSX/XLSM/JSON/JSONL files, pandas is acceptable.

Return JSON with this exact shape:
{{
  "understanding": "What we know now, what still needs verification, and why",
  "evidence_gaps": ["Optional list of specific missing checks or uncertainties"],
  "ready_to_answer": false,
  "needs_code_execution": true,
  "code_purpose": "What the code is trying to verify",
  "python_code": "Full Python code when execution is needed, otherwise empty"
}}
""".strip()

    def _build_answer_prompt(
        self,
        *,
        parsed: ParsedFileContent,
        analyze_response: AnalyzeResponse,
        execution_results: list[ExecutionResult],
    ) -> str:
        user_request = self.query or "Please analyze the selected file comprehensively."
        execution_payload = json.dumps(
            [item.to_model_payload() for item in execution_results],
            ensure_ascii=False,
            indent=2,
        )
        return f"""
We are in the Answer stage of a closed-loop file analysis workflow.

User request:
{user_request}

Selected file:
- Relative path: {self.selected_file_path}
- File name: {parsed.file_name}
- File type: {parsed.file_type}

Analyze stage:
{json.dumps(analyze_response.model_dump(), ensure_ascii=False, indent=2)}

Execution results:
{execution_payload if execution_results else "[]"}

Write the final answer for the user.

Requirements:
- Use only verified evidence from the Analyze stage and execution results.
- If a check failed or could not be executed, mention that limitation explicitly.
- Answer the user's request directly.
- Keep it structured and practical.
- Match the user's language.
""".strip()

    def _validate_generated_code(self, code: str) -> str:
        normalized = _clean_text(code)
        if not normalized:
            return "No Python code was provided."

        lowered = normalized.lower()
        for token, reason in _BANNED_CODE_PATTERNS:
            if token in lowered:
                return f"Blocked by execution policy: {reason}."
        return ""

    async def _execute_python(self, *, code: str, purpose: str, round_index: int) -> ExecutionResult:
        rejection_reason = self._validate_generated_code(code)
        if rejection_reason:
            return ExecutionResult(
                round_index=round_index,
                purpose=purpose,
                code=code,
                returncode=1,
                stdout="",
                stderr="",
                rejected=True,
                rejection_reason=rejection_reason,
            )

        script_dir = self.workspace_dir / ".agent_iterations"
        script_dir.mkdir(parents=True, exist_ok=True)
        script_path = script_dir / f"analysis_round_{round_index}_{uuid.uuid4().hex[:8]}.py"
        script_body = "\n".join(
            [
                "from __future__ import annotations",
                "import json",
                "from pathlib import Path",
                "",
                f"TARGET_FILE_PATH = Path(r'''{self.file_path}''')",
                f"WORKSPACE_DIR = Path(r'''{self.workspace_dir}''')",
                "",
                "def emit(value):",
                "    if isinstance(value, (dict, list)):",
                "        print(json.dumps(value, ensure_ascii=False, indent=2))",
                "    else:",
                "        print(value)",
                "",
                "# Generated analysis code starts here.",
                code.strip(),
                "",
            ]
        )
        script_path.write_text(script_body, encoding="utf-8")

        process = await asyncio.create_subprocess_exec(
            self.python_executable,
            str(script_path),
            cwd=str(self.workspace_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=self.execution_timeout_seconds,
            )
            stdout = _truncate_text(stdout_bytes.decode("utf-8", errors="ignore"), self.max_execution_output_chars)
            stderr = _truncate_text(stderr_bytes.decode("utf-8", errors="ignore"), self.max_execution_output_chars)
            return ExecutionResult(
                round_index=round_index,
                purpose=purpose,
                code=code,
                returncode=int(process.returncode or 0),
                stdout=stdout,
                stderr=stderr,
                script_path=script_path.as_posix(),
            )
        except asyncio.TimeoutError:
            with suppress(ProcessLookupError):
                process.kill()
            stdout_bytes, stderr_bytes = await process.communicate()
            stdout = _truncate_text(stdout_bytes.decode("utf-8", errors="ignore"), self.max_execution_output_chars)
            stderr = _truncate_text(stderr_bytes.decode("utf-8", errors="ignore"), self.max_execution_output_chars)
            return ExecutionResult(
                round_index=round_index,
                purpose=purpose,
                code=code,
                returncode=124,
                stdout=stdout,
                stderr=stderr,
                timed_out=True,
                script_path=script_path.as_posix(),
            )

    def _format_analyze_markdown(self, parsed: ParsedFileContent, response: AnalyzeResponse) -> str:
        return "\n".join(
            [
                f"### {self.file_heading}",
                f"- Relative path: `{self.selected_file_path}`",
                f"- File type: `{parsed.file_type}`",
                f"- Parser: `{parsed.parser}`",
                f"- Size: `{parsed.size_bytes}` bytes",
                "",
                "**Initial analysis**",
                response.analyze_summary.strip(),
                "",
                "**Visible observations**",
                _format_markdown_list(response.key_observations),
                "",
                "**Next focus**",
                response.next_focus.strip(),
            ]
        ).strip()

    def _format_understand_markdown(self, response: UnderstandResponse, round_index: int) -> str:
        next_action = (
            "Move to Answer because the evidence is sufficient."
            if response.ready_to_answer
            else _first_non_empty(
                [
                    response.code_purpose or "",
                    "Generate Python to verify the remaining evidence gaps.",
                ]
            )
        )
        return "\n".join(
            [
                f"### {self.file_heading}",
                response.understanding.strip(),
                "",
                "**Evidence gaps**",
                _format_markdown_list(response.evidence_gaps),
                "",
                "**Next action**",
                next_action,
            ]
        ).strip()

    def _format_code_markdown(self, purpose: str, code: str, round_index: int) -> str:
        return "\n".join(
            [
                f"### {self.file_heading}",
                "",
                "**Purpose**",
                _clean_text(purpose) or "Verify the next analytical hypothesis.",
                "",
                "```python",
                code.strip(),
                "```",
            ]
        ).strip()

    def _format_execute_markdown(self, result: ExecutionResult) -> str:
        header = [f"### {self.file_heading}"]
        if result.rejected:
            header.extend(
                [
                    "",
                    "**Execution blocked**",
                    result.rejection_reason,
                ]
            )
            return "\n".join(header).strip()

        status_lines = [
            f"- Return code: `{result.returncode}`",
            f"- Timed out: `{'Yes' if result.timed_out else 'No'}`",
        ]
        if result.script_path:
            status_lines.append(f"- Script: `{result.script_path}`")

        parts = header + [
            "",
            "**Status**",
            "\n".join(status_lines),
        ]
        if result.stdout:
            parts.extend(["", "**Stdout**", "```text", result.stdout, "```"])
        else:
            parts.extend(["", "**Stdout**", "```text", "(empty)", "```"])

        if result.stderr:
            parts.extend(["", "**Stderr**", "```text", result.stderr, "```"])

        return "\n".join(parts).strip()

    def _format_execute_live_markdown(
        self,
        *,
        round_index: int,
        purpose: str,
        stdout: str,
        stderr: str,
    ) -> str:
        parts = [
            f"### {self.file_heading}",
            "",
            "**Running**",
            f"- Purpose: `{_clean_text(purpose) or 'Executing generated Python'}`",
        ]
        if stdout:
            parts.extend(["", "**Stdout (live)**", "```text", stdout, "```"])
        if stderr:
            parts.extend(["", "**Stderr (live)**", "```text", stderr, "```"])
        if not stdout and not stderr:
            parts.extend(["", "**Live output**", "```text", "(waiting for output)", "```"])
        return "\n".join(parts).strip()

    def _format_answer_markdown(self, answer: str) -> str:
        return "\n".join([f"### {self.file_heading}", "", answer.strip()]).strip()

    async def _execute_python_streaming(self, *, code: str, purpose: str, round_index: int) -> ExecutionResult:
        rejection_reason = self._validate_generated_code(code)
        if rejection_reason:
            return ExecutionResult(
                round_index=round_index,
                purpose=purpose,
                code=code,
                returncode=1,
                stdout="",
                stderr="",
                rejected=True,
                rejection_reason=rejection_reason,
            )

        script_dir = self.workspace_dir / ".agent_iterations"
        script_dir.mkdir(parents=True, exist_ok=True)
        script_path = script_dir / f"analysis_round_{round_index}_{uuid.uuid4().hex[:8]}.py"
        script_body = "\n".join(
            [
                "from __future__ import annotations",
                "import json",
                "from pathlib import Path",
                "",
                f"TARGET_FILE_PATH = Path(r'''{self.file_path}''')",
                f"WORKSPACE_DIR = Path(r'''{self.workspace_dir}''')",
                "",
                "def emit(value):",
                "    if isinstance(value, (dict, list)):",
                "        print(json.dumps(value, ensure_ascii=False, indent=2))",
                "    else:",
                "        print(value)",
                "",
                "# Generated analysis code starts here.",
                code.strip(),
                "",
            ]
        )
        script_path.write_text(script_body, encoding="utf-8")

        process = await asyncio.create_subprocess_exec(
            self.python_executable,
            str(script_path),
            cwd=str(self.workspace_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout_parts: list[str] = []
        stderr_parts: list[str] = []

        async def pump_stream(stream, collector: list[str]) -> None:
            if stream is None:
                return
            while True:
                chunk = await stream.read(self.execution_live_read_bytes)
                if not chunk:
                    break
                collector.append(chunk.decode("utf-8", errors="ignore"))
                await self._publish_stream_event(
                    EXECUTE_STAGE,
                    self._format_execute_live_markdown(
                        round_index=round_index,
                        purpose=purpose,
                        stdout=_truncate_text("".join(stdout_parts), self.max_execution_output_chars),
                        stderr=_truncate_text("".join(stderr_parts), self.max_execution_output_chars),
                    ),
                )

        stdout_task = asyncio.create_task(pump_stream(process.stdout, stdout_parts))
        stderr_task = asyncio.create_task(pump_stream(process.stderr, stderr_parts))

        try:
            await asyncio.wait_for(process.wait(), timeout=self.execution_timeout_seconds)
            await asyncio.gather(stdout_task, stderr_task)
            return ExecutionResult(
                round_index=round_index,
                purpose=purpose,
                code=code,
                returncode=int(process.returncode or 0),
                stdout=_truncate_text("".join(stdout_parts), self.max_execution_output_chars),
                stderr=_truncate_text("".join(stderr_parts), self.max_execution_output_chars),
                script_path=script_path.as_posix(),
            )
        except asyncio.TimeoutError:
            with suppress(ProcessLookupError):
                process.kill()
            await process.communicate()
            with suppress(asyncio.CancelledError):
                await asyncio.gather(stdout_task, stderr_task)
            return ExecutionResult(
                round_index=round_index,
                purpose=purpose,
                code=code,
                returncode=124,
                stdout=_truncate_text("".join(stdout_parts), self.max_execution_output_chars),
                stderr=_truncate_text("".join(stderr_parts), self.max_execution_output_chars),
                timed_out=True,
                script_path=script_path.as_posix(),
            )

    async def run(self) -> str:
        execution_results: list[ExecutionResult] = []
        parsed: ParsedFileContent | None = None
        analyze_response: AnalyzeResponse | None = None

        try:
            self.current_stage = ANALYZE_STAGE
            await self._publish_start(
                ANALYZE_STAGE,
                f"Reviewing {self.file_heading} and aligning the analysis with the user's request.",
            )

            raw = self.file_path.read_bytes()
            parsed = self.parser.parse(file_name=self.file_name, content_type=None, raw=raw)
            analyze_response = AnalyzeResponse.model_validate(
                await self._invoke_structured(
                    schema=AnalyzeResponse,
                    user_prompt=self._build_analyze_prompt(parsed),
                )
            )
            await self._publish_streamed_finish(ANALYZE_STAGE, self._format_analyze_markdown(parsed, analyze_response))

            for round_index in range(1, self.max_rounds + 1):
                self.current_stage = UNDERSTAND_STAGE
                await self._publish_start(
                    UNDERSTAND_STAGE,
                    f"Continuing verification for {self.file_heading}.",
                )
                understand_response = UnderstandResponse.model_validate(
                    await self._invoke_structured(
                        schema=UnderstandResponse,
                        user_prompt=self._build_understand_prompt(
                            parsed=parsed,
                            analyze_response=analyze_response,
                            execution_results=execution_results,
                            round_index=round_index,
                        ),
                    )
                )
                await self._publish_streamed_finish(
                    UNDERSTAND_STAGE,
                    self._format_understand_markdown(understand_response, round_index),
                )

                if understand_response.ready_to_answer:
                    break

                code = _clean_text(understand_response.python_code)
                purpose = _clean_text(understand_response.code_purpose) or "Verify the next analytical hypothesis."
                if not understand_response.needs_code_execution or not code:
                    self.current_stage = EXECUTE_STAGE
                    await self._publish_start(
                        EXECUTE_STAGE,
                        f"Execution was skipped because the model did not provide valid Python for {self.file_heading}.",
                    )
                    blocked_result = ExecutionResult(
                        round_index=round_index,
                        purpose=purpose,
                        code=code,
                        returncode=1,
                        stdout="",
                        stderr="Model requested more analysis but did not provide executable Python.",
                        rejected=True,
                        rejection_reason="Model requested another loop without valid code.",
                    )
                    execution_results.append(blocked_result)
                    await self._publish_streamed_finish(EXECUTE_STAGE, self._format_execute_markdown(blocked_result))
                    continue

                self.current_stage = CODE_STAGE
                await self._publish_start(CODE_STAGE, f"Generating Python for {purpose}")
                await self._publish_streamed_finish(CODE_STAGE, self._format_code_markdown(purpose, code, round_index))

                self.current_stage = EXECUTE_STAGE
                await self._publish_start(
                    EXECUTE_STAGE,
                    f"Running generated Python in the runtime environment for {self.file_heading}.",
                )
                execution_result = await self._execute_python_streaming(
                    code=code,
                    purpose=purpose,
                    round_index=round_index,
                )
                execution_results.append(execution_result)
                await self._publish_finish(EXECUTE_STAGE, self._format_execute_markdown(execution_result))

            self.current_stage = ANSWER_STAGE
            await self._publish_start(ANSWER_STAGE, f"Synthesizing the verified findings for {self.file_heading}.")
            answer_text = await self._invoke_text_streaming(
                stage=ANSWER_STAGE,
                user_prompt=self._build_answer_prompt(
                    parsed=parsed,
                    analyze_response=analyze_response,
                    execution_results=execution_results,
                ),
                formatter=self._format_answer_markdown,
            )
            answer_markdown = self._format_answer_markdown(answer_text)
            await self._publish_finish(ANSWER_STAGE, answer_markdown)
            return answer_markdown
        except Exception as exc:
            logger.error(
                "Iterative file analysis failed for %s at stage %s: %s",
                self.selected_file_path,
                self.current_stage,
                exc,
                exc_info=True,
            )
            await self._publish_error(self.current_stage, exc if isinstance(exc, Exception) else Exception(str(exc)))
            raise


async def run_iterative_file_analysis(
    query: str,
    selected_file_path: str,
    workspace_dir: Path,
    *,
    file_index: int = 1,
    total_files: int = 1,
) -> str:
    analyzer = IterativeFileAnalyzer(
        query=query,
        selected_file_path=selected_file_path,
        workspace_dir=workspace_dir,
        file_index=file_index,
        total_files=total_files,
    )
    return await analyzer.run()
