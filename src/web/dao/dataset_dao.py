from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy import select

from .base_dao import BaseDAO
from ..entity.model import Dataset


class DatasetDAO(BaseDAO):
    def list_datasets(self) -> List[Dataset]:
        with self.session_scope() as session:
            stmt = select(Dataset).order_by(Dataset.update_time.desc(), Dataset.id.desc())
            return list(session.execute(stmt).scalars().all())

    def get_dataset_by_id(self, dataset_id: int) -> Optional[Dataset]:
        with self.session_scope() as session:
            return session.get(Dataset, int(dataset_id))

    def insert_dataset(self, payload: Dict[str, Any]) -> Dataset:
        with self.session_scope() as session:
            dataset = Dataset(**payload)
            session.add(dataset)
            session.flush()
            session.refresh(dataset)
            return dataset

    def update_dataset(self, dataset_id: int, payload: Dict[str, Any]) -> Optional[Dataset]:
        with self.session_scope() as session:
            dataset = session.get(Dataset, int(dataset_id))
            if dataset is None:
                return None
            for key, value in payload.items():
                setattr(dataset, key, value)
            session.add(dataset)
            session.flush()
            session.refresh(dataset)
            return dataset

    def delete_dataset(self, dataset_id: int) -> bool:
        with self.session_scope() as session:
            dataset = session.get(Dataset, int(dataset_id))
            if dataset is None:
                return False
            session.delete(dataset)
            session.flush()
            return True
