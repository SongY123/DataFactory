from __future__ import annotations

import json
import os
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STORAGE_PATH = PROJECT_ROOT / "runtime" / "sandbox_environments.json"


class SandboxEnvironmentService:
    _file_lock = threading.RLock()

    def __init__(self, storage_path: Optional[Path] = None) -> None:
        self.storage_path = Path(storage_path or DEFAULT_STORAGE_PATH).resolve()

    def list_environments(self) -> Dict[str, Any]:
        payload = self._load_payload()
        return {
            "default_id": payload["default_id"],
            "items": [self._serialize_item(item, payload["default_id"]) for item in payload["items"]],
        }

    def get_environment(self, environment_id: Optional[str] = None) -> Dict[str, Any]:
        payload = self._load_payload()
        target_id = str(environment_id or "").strip() or str(payload["default_id"])
        items = payload["items"]
        item = next((entry for entry in items if str(entry["id"]) == target_id), None)
        if item is None:
            raise ValueError(f"sandbox environment not found: {environment_id}")
        return self._serialize_item(item, payload["default_id"])

    def create_environment(self, *, name: str, python_path: str) -> Dict[str, Any]:
        clean_name = str(name or "").strip()
        if not clean_name:
            raise ValueError("Environment name is required.")

        normalized_path = self._normalize_python_path(python_path)
        now = self._now_iso()

        with self._file_lock:
            payload = self._load_payload()
            items = list(payload["items"])

            if any(str(item["name"]).strip().lower() == clean_name.lower() for item in items):
                raise ValueError(f"Environment name already exists: {clean_name}")
            if any(str(item["python_path"]).strip() == normalized_path for item in items):
                raise ValueError("Python path is already registered.")

            item = {
                "id": f"env-{uuid.uuid4().hex[:10]}",
                "name": clean_name,
                "python_path": normalized_path,
                "created_at": now,
                "updated_at": now,
            }
            items.append(item)
            payload["items"] = items
            self._write_payload(payload)
            return self._serialize_item(item, payload["default_id"])

    def delete_environment(self, environment_id: str) -> Dict[str, Any]:
        target_id = str(environment_id or "").strip()
        if not target_id:
            raise ValueError("Environment id is required.")

        with self._file_lock:
            payload = self._load_payload()
            items = list(payload["items"])
            if len(items) <= 1:
                raise ValueError("At least one sandbox environment must be retained.")

            remaining = [item for item in items if str(item["id"]) != target_id]
            if len(remaining) == len(items):
                raise ValueError(f"sandbox environment not found: {environment_id}")

            default_id = str(payload["default_id"])
            if default_id == target_id:
                default_id = str(remaining[0]["id"])

            payload["items"] = remaining
            payload["default_id"] = default_id
            self._write_payload(payload)
            return {
                "deleted_id": target_id,
                "default_id": default_id,
                "items": [self._serialize_item(item, default_id) for item in remaining],
            }

    def resolve_python_executable(self, environment_id: Optional[str] = None) -> Dict[str, Any]:
        item = self.get_environment(environment_id)
        python_path = Path(str(item["python_path"])).resolve()
        if not python_path.exists() or not python_path.is_file():
            raise ValueError(f"Python executable not found: {python_path}")
        if not os.access(str(python_path), os.X_OK):
            raise ValueError(f"Python executable is not runnable: {python_path}")
        return item

    def _load_payload(self) -> Dict[str, Any]:
        with self._file_lock:
            if not self.storage_path.exists():
                payload = self._default_payload()
                self._write_payload(payload)
                return payload

            try:
                raw = json.loads(self.storage_path.read_text(encoding="utf-8"))
            except Exception:
                payload = self._default_payload()
                self._write_payload(payload)
                return payload

            items = raw.get("items") if isinstance(raw, dict) else None
            normalized_items = self._normalize_items(items if isinstance(items, list) else [])
            if not normalized_items:
                payload = self._default_payload()
                self._write_payload(payload)
                return payload

            default_id = str(raw.get("default_id") or normalized_items[0]["id"]).strip()
            if default_id not in {str(item["id"]) for item in normalized_items}:
                default_id = str(normalized_items[0]["id"])

            payload = {
                "default_id": default_id,
                "items": normalized_items,
            }
            return payload

    def _write_payload(self, payload: Dict[str, Any]) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = {
            "default_id": str(payload.get("default_id") or ""),
            "items": [
                {
                    "id": str(item["id"]),
                    "name": str(item["name"]),
                    "python_path": str(item["python_path"]),
                    "created_at": str(item["created_at"]),
                    "updated_at": str(item["updated_at"]),
                }
                for item in (payload.get("items") or [])
            ],
        }
        self.storage_path.write_text(json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8")

    def _default_payload(self) -> Dict[str, Any]:
        now = self._now_iso()
        default_python = str(Path(sys.executable).resolve())
        item = {
            "id": "env-default",
            "name": "Default Python",
            "python_path": default_python,
            "created_at": now,
            "updated_at": now,
        }
        return {
            "default_id": item["id"],
            "items": [item],
        }

    def _normalize_items(self, items: List[Any]) -> List[Dict[str, str]]:
        normalized: List[Dict[str, str]] = []
        seen_ids = set()
        for index, raw in enumerate(items):
            if not isinstance(raw, dict):
                continue
            item_id = str(raw.get("id") or f"env-{index + 1}").strip()
            if not item_id or item_id in seen_ids:
                continue
            name = str(raw.get("name") or item_id).strip() or item_id
            python_path = str(raw.get("python_path") or "").strip()
            if not python_path:
                continue
            created_at = str(raw.get("created_at") or self._now_iso()).strip() or self._now_iso()
            updated_at = str(raw.get("updated_at") or created_at).strip() or created_at
            normalized.append(
                {
                    "id": item_id,
                    "name": name,
                    "python_path": python_path,
                    "created_at": created_at,
                    "updated_at": updated_at,
                }
            )
            seen_ids.add(item_id)
        return normalized

    @staticmethod
    def _serialize_item(item: Dict[str, Any], default_id: str) -> Dict[str, Any]:
        return {
            "id": str(item["id"]),
            "name": str(item["name"]),
            "python_path": str(item["python_path"]),
            "created_at": str(item["created_at"]),
            "updated_at": str(item["updated_at"]),
            "is_default": str(item["id"]) == str(default_id),
        }

    @staticmethod
    def _normalize_python_path(path_value: str) -> str:
        raw = Path(str(path_value or "").strip())
        if not str(raw):
            raise ValueError("Python path is required.")
        resolved = raw if raw.is_absolute() else (PROJECT_ROOT / raw)
        resolved = resolved.resolve()
        if not resolved.exists() or not resolved.is_file():
            raise ValueError(f"Python executable not found: {resolved}")
        if not os.access(str(resolved), os.X_OK):
            raise ValueError(f"Python executable is not runnable: {resolved}")
        return str(resolved)

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
