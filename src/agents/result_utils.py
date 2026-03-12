from __future__ import annotations

from typing import Any


def _collect_text_parts(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []

    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            parts.extend(_collect_text_parts(item))
        return parts

    if isinstance(value, dict):
        block_type = str(value.get("type") or "").strip().lower()
        if block_type == "text":
            return _collect_text_parts(value.get("text"))
        if block_type == "thinking":
            return _collect_text_parts(value.get("thinking"))
        if block_type == "tool_result":
            return _collect_text_parts(value.get("output"))

        parts: list[str] = []
        for key in ("text", "result", "output", "content", "message", "task"):
            if key in value:
                parts.extend(_collect_text_parts(value.get(key)))
        return parts

    saw_structured_text_interface = False

    content = getattr(value, "content", None)
    if content is not None:
        saw_structured_text_interface = True
        parts = _collect_text_parts(content)
        if parts:
            return parts

    get_text_content = getattr(value, "get_text_content", None)
    if callable(get_text_content):
        saw_structured_text_interface = True
        try:
            text = get_text_content()
        except TypeError:
            text = get_text_content(chr(10))
        parts = _collect_text_parts(text)
        if parts:
            return parts

    text_attr = getattr(value, "text", None)
    if text_attr is not None:
        saw_structured_text_interface = True
        parts = _collect_text_parts(text_attr)
        if parts:
            return parts

    if saw_structured_text_interface:
        return []

    return _collect_text_parts(str(value))


def extract_agent_result_text(value: Any) -> str:
    parts = _collect_text_parts(value)
    if not parts:
        return ""

    deduped: list[str] = []
    for part in parts:
        if not deduped or deduped[-1] != part:
            deduped.append(part)
    return "\n\n".join(deduped).strip()
