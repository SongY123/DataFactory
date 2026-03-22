from __future__ import annotations

from contextvars import ContextVar
from pathlib import Path
from typing import Callable, Optional

from agents.event_bus import EventBus, create_stream_event
from utils.config_loader import get_config

try:
    from agentscope.message import Msg
except ImportError:
    Msg = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]

event_bus_context: ContextVar[Optional[EventBus]] = ContextVar("event_bus_context", default=None)
workspace_context: ContextVar[str] = ContextVar("workspace_context", default="")
python_interpreter_context: ContextVar[str] = ContextVar("python_interpreter_context", default="")


def get_event_bus() -> Optional[EventBus]:
    return event_bus_context.get()


def set_event_bus(event_bus: Optional[EventBus]) -> None:
    event_bus_context.set(event_bus)


def get_workspace() -> str:
    return str(workspace_context.get() or "")


def set_workspace(workspace: str) -> None:
    workspace_context.set(str(workspace or ""))


def get_python_interpreter() -> str:
    return str(python_interpreter_context.get() or "")


def set_python_interpreter(python_interpreter: str) -> None:
    python_interpreter_context.set(str(python_interpreter or ""))


def _resolve_workspace_base() -> Path:
    configured = str(get_config("workspace.base_path", "workspace") or "workspace").strip()
    base = Path(configured)
    if base.is_absolute():
        return base
    return PROJECT_ROOT / base


def get_data_directory() -> str:
    workspace = get_workspace()
    if not workspace:
        return str(_resolve_workspace_base())

    workspace_path = Path(workspace)
    if workspace_path.is_absolute() or len(workspace_path.parts) > 1:
        return str(workspace_path)

    return str(_resolve_workspace_base() / workspace_path)


def create_streaming_hook(agent_name: str) -> Callable:
    async def streaming_output_hook(agent_self, kwargs_dict):
        event_bus = get_event_bus()
        msg = kwargs_dict.get("msg")
        if msg is None or not event_bus:
            return None

        if Msg is not None and isinstance(msg, Msg):
            chunk = msg.get_text_content()
        elif isinstance(msg, str):
            chunk = msg
        else:
            chunk = str(msg)

        if chunk:
            try:
                content = {"result": chunk} if isinstance(chunk, str) else chunk
                stream_event = await create_stream_event(agent_name=agent_name, chunk=content)
                await event_bus.publish(stream_event)
            except Exception as exc:
                if str(exc).strip() == "EventBus is closed":
                    return None
                print(f"[StreamingHook:{agent_name}] publish failed: {exc}")

        return None

    return streaming_output_hook


def register_streaming_hook(worker, agent_name: str) -> None:
    worker._instance_pre_print_hooks["streaming_output"] = create_streaming_hook(agent_name)
