from __future__ import annotations

from typing import Optional

from sqlalchemy import select

from .base_dao import BaseDAO
from ..entity.model import UserPreference


class UserPreferenceDAO(BaseDAO):
    def get_preference(self, *, user_id: int, preference_key: str) -> Optional[UserPreference]:
        with self.session_scope() as session:
            stmt = select(UserPreference).where(
                UserPreference.user_id == int(user_id),
                UserPreference.preference_key == str(preference_key),
            )
            return session.execute(stmt).scalars().first()

    def upsert_preference(self, *, user_id: int, preference_key: str, preference_json: str | None) -> UserPreference:
        with self.session_scope() as session:
            stmt = select(UserPreference).where(
                UserPreference.user_id == int(user_id),
                UserPreference.preference_key == str(preference_key),
            )
            item = session.execute(stmt).scalars().first()
            if item is None:
                item = UserPreference(
                    user_id=int(user_id),
                    preference_key=str(preference_key),
                    preference_json=preference_json,
                )
            else:
                item.preference_json = preference_json
            session.add(item)
            session.flush()
            session.refresh(item)
            return item
