from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy import select

from .base_dao import BaseDAO
from ..entity.model import Dataset


class DatasetDAO(BaseDAO):
    def list_datasets(self, user_id: Optional[int] = None) -> List[Dataset]:
        with self.session_scope() as session:
            stmt = select(Dataset).order_by(Dataset.update_time.desc(), Dataset.id.desc())
            if user_id is not None:
                stmt = stmt.where(Dataset.user_id == int(user_id))
            return list(session.execute(stmt).scalars().all())

    def get_dataset_by_id(self, dataset_id: int, user_id: Optional[int] = None) -> Optional[Dataset]:
        with self.session_scope() as session:
            stmt = select(Dataset).where(Dataset.id == int(dataset_id))
            if user_id is not None:
                stmt = stmt.where(Dataset.user_id == int(user_id))
            return session.execute(stmt).scalars().first()

    def get_datasets_by_ids(self, dataset_ids: List[int], user_id: Optional[int] = None) -> List[Dataset]:
        clean_ids = [int(x) for x in dataset_ids or []]
        if not clean_ids:
            return []
        with self.session_scope() as session:
            stmt = select(Dataset).where(Dataset.id.in_(clean_ids))
            if user_id is not None:
                stmt = stmt.where(Dataset.user_id == int(user_id))
            return list(session.execute(stmt).scalars().all())

    def insert_dataset(self, payload: Dict[str, Any]) -> Dataset:
        with self.session_scope() as session:
            dataset = Dataset(**payload)
            session.add(dataset)
            session.flush()
            session.refresh(dataset)
            return dataset

    def update_dataset(self, dataset_id: int, payload: Dict[str, Any], user_id: Optional[int] = None) -> Optional[Dataset]:
        with self.session_scope() as session:
            stmt = select(Dataset).where(Dataset.id == int(dataset_id))
            if user_id is not None:
                stmt = stmt.where(Dataset.user_id == int(user_id))
            dataset = session.execute(stmt).scalars().first()
            if dataset is None:
                return None
            for key, value in payload.items():
                setattr(dataset, key, value)
            session.add(dataset)
            session.flush()
            session.refresh(dataset)
            return dataset

    def delete_dataset(self, dataset_id: int, user_id: Optional[int] = None) -> bool:
        with self.session_scope() as session:
            stmt = select(Dataset).where(Dataset.id == int(dataset_id))
            if user_id is not None:
                stmt = stmt.where(Dataset.user_id == int(user_id))
            dataset = session.execute(stmt).scalars().first()
            if dataset is None:
                return False
            session.delete(dataset)
            session.flush()
            return True
