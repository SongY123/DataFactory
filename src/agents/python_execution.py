from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from agentscope.message import TextBlock
from agentscope.tool import ToolResponse

from agents.context import PROJECT_ROOT, get_python_interpreter, get_workspace


def _resolve_runtime_dir() -> Path:
    workspace = str(get_workspace() or "").strip()
    if workspace:
        workspace_path = Path(workspace).expanduser()
        try:
            resolved = workspace_path.resolve()
            if resolved.exists() and resolved.is_dir():
                runtime_dir = resolved / ".agent_tool_runtime"
                runtime_dir.mkdir(parents=True, exist_ok=True)
                return runtime_dir
        except Exception:
            pass

    runtime_dir = PROJECT_ROOT / "runtime" / "agent_tool_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return runtime_dir


def _resolve_python_executable() -> str:
    configured = str(get_python_interpreter() or "").strip()
    if not configured:
        return str(Path(sys.executable).resolve())
    return str(Path(configured).expanduser().resolve())


async def execute_python_code(
    code: str,
    timeout: float = 300,
    **kwargs: Any,
) -> ToolResponse:
    runtime_dir = _resolve_runtime_dir()
    script_path = runtime_dir / f"tmp_{uuid.uuid4().hex[:12]}.py"
    script_path.write_text(str(code or ""), encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    python_executable = _resolve_python_executable()

    try:
        proc = await asyncio.create_subprocess_exec(
            python_executable,
            "-u",
            str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout)
            stdout, stderr = await proc.communicate()
            stdout_str = stdout.decode("utf-8", errors="ignore")
            stderr_str = stderr.decode("utf-8", errors="ignore")
            returncode = int(proc.returncode or 0)
        except asyncio.TimeoutError:
            stderr_suffix = f"TimeoutError: The code execution exceeded the timeout of {timeout} seconds."
            returncode = -1
            try:
                proc.terminate()
                stdout, stderr = await proc.communicate()
                stdout_str = stdout.decode("utf-8", errors="ignore")
                stderr_str = stderr.decode("utf-8", errors="ignore")
                stderr_str = f"{stderr_str}\n{stderr_suffix}".strip() if stderr_str else stderr_suffix
            except ProcessLookupError:
                stdout_str = ""
                stderr_str = stderr_suffix
    finally:
        try:
            script_path.unlink(missing_ok=True)
        except Exception:
            pass

    return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text=(
                    f"<returncode>{returncode}</returncode>"
                    f"<stdout>{stdout_str}</stdout>"
                    f"<stderr>{stderr_str}</stderr>"
                ),
            ),
        ],
    )
