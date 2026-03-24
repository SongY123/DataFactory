from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class UserPreferenceUpdateRequest(BaseModel):
    value: Any = Field(default=None)
