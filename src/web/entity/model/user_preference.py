from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, text
from sqlalchemy.orm import validates

from .base import Base


def _utc_now() -> datetime:
    return datetime.utcnow()


class UserPreference(Base):
    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    preference_key = Column(String(128), nullable=False, index=True)
    preference_json = Column(Text, nullable=True)
    insert_time = Column(DateTime, nullable=False, default=_utc_now, server_default=text("CURRENT_TIMESTAMP"))
    update_time = Column(DateTime, nullable=False, default=_utc_now, onupdate=_utc_now, server_default=text("CURRENT_TIMESTAMP"))

    @validates("preference_key")
    def _validate_preference_key(self, key, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("preference_key must not be empty")
        if len(normalized) > 128:
            raise ValueError("preference_key is too long")
        return normalized

    def to_dict(self) -> Dict[str, Any]:
        try:
            value = json.loads(self.preference_json) if self.preference_json else None
        except Exception:
            value = None

        return {
            "id": self.id,
            "user_id": self.user_id,
            "preference_key": self.preference_key,
            "value": value,
            "insert_time": self.insert_time.isoformat() if self.insert_time else None,
            "update_time": self.update_time.isoformat() if self.update_time else None,
        }
