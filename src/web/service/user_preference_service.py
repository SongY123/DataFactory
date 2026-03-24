from __future__ import annotations

import json
import re
from typing import Any

from ..dao import UserPreferenceDAO


ALLOWED_PREFERENCE_KEYS = {
    "trajectory_synthesis",
    "reasoning_synthesis",
    "interactive_testing",
}

_PREFERENCE_KEY_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


class UserPreferenceService:
    def __init__(self, dao: UserPreferenceDAO | None = None) -> None:
        self.dao = dao or UserPreferenceDAO()

    def get_preference(self, *, user_id: int, preference_key: str) -> dict[str, Any]:
        key = self._normalize_preference_key(preference_key)
        item = self.dao.get_preference(user_id=user_id, preference_key=key)
        if item is None:
            return {
                "preference_key": key,
                "value": None,
            }
        data = item.to_dict()
        return {
            "preference_key": key,
            "value": data.get("value"),
            "update_time": data.get("update_time"),
        }

    def save_preference(self, *, user_id: int, preference_key: str, value: Any) -> dict[str, Any]:
        key = self._normalize_preference_key(preference_key)
        try:
            payload = json.dumps(value, ensure_ascii=False)
        except TypeError as exc:
            raise ValueError(f"preference value is not JSON serializable: {exc}") from exc
        item = self.dao.upsert_preference(
            user_id=user_id,
            preference_key=key,
            preference_json=payload,
        )
        data = item.to_dict()
        return {
            "preference_key": key,
            "value": data.get("value"),
            "update_time": data.get("update_time"),
        }

    @staticmethod
    def _normalize_preference_key(value: str) -> str:
        key = str(value or "").strip().lower()
        if not key:
            raise ValueError("preference_key is required")
        if not _PREFERENCE_KEY_PATTERN.match(key):
            raise ValueError("preference_key is invalid")
        if key not in ALLOWED_PREFERENCE_KEYS:
            raise ValueError(f"unsupported preference_key: {key}")
        return key
