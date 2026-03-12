from __future__ import annotations

import csv
import io
import json
import shutil
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional

from fastapi import UploadFile

from ..dao import DatasetDAO


class DatasetService:
    def __init__(self, dataset_dao: Optional[DatasetDAO] = None, upload_dir: Optional[Path] = None) -> None:
        self.dataset_dao = dataset_dao or DatasetDAO()
        self.upload_dir = upload_dir or (Path(__file__).resolve().parents[3] / "uploads")
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _parse_samples(filename: str, raw: bytes) -> List[Dict]:
        suffix = Path(filename).suffix.lower()

        if suffix == ".csv":
            text = raw.decode("utf-8", errors="ignore")
            reader = csv.DictReader(io.StringIO(text))
            return [dict(row) for _, row in zip(range(5), reader)]

        if suffix == ".json":
            text = raw.decode("utf-8", errors="ignore").strip()
            if not text:
                return []
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [item for item in parsed[:5] if isinstance(item, dict)]
            if isinstance(parsed, dict):
                return [parsed]
            return [{"value": parsed}]

        if suffix == ".jsonl":
            rows: List[Dict] = []
            for line in raw.decode("utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                parsed = json.loads(line)
                rows.append(parsed if isinstance(parsed, dict) else {"value": parsed})
                if len(rows) >= 5:
                    break
            return rows

        preview = raw.decode("utf-8", errors="ignore")[:500]
        return [{"preview": preview}] if preview else []

    @staticmethod
    def _normalize_text(value) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _to_payload(self, row: Dict) -> Dict:
        sample_data = row.get("sample_data")
        if isinstance(sample_data, str) and sample_data.strip():
            try:
                sample_data = json.loads(sample_data)
            except json.JSONDecodeError:
                sample_data = [{"preview": sample_data}]
        elif not sample_data:
            sample_data = []

        dataset_id = row.get("id")
        cover_url = f"/api/datasets/{dataset_id}/cover" if row.get("cover_path") else None

        return {
            "id": dataset_id,
            "user_id": row.get("user_id"),
            "name": row.get("name"),
            "type": row.get("type"),
            "source": row.get("source"),
            "language": row.get("language"),
            "size": int(row.get("size") or 0),
            "status": row.get("status"),
            "note": row.get("note"),
            "cover_url": cover_url,
            "sample_data": sample_data,
            "insert_time": row.get("insert_time"),
            "update_time": row.get("update_time"),
        }

    def _save_upload_bytes(self, user_id: int, dataset_id: int, filename: str, content: bytes) -> str:
        relative_path = self._normalize_relative_upload_path(filename)
        saved_path = self.upload_dir / str(int(user_id)) / str(int(dataset_id)) / relative_path
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        saved_path.write_bytes(content)
        return str(saved_path.resolve())

    @staticmethod
    def _normalize_relative_upload_path(filename: str) -> Path:
        raw = str(filename or "").strip().replace("\\", "/")
        parsed = PurePosixPath(raw)
        parts = [part for part in parsed.parts if part not in {"", ".", ".."}]
        if not parts:
            parts = ["file"]
        return Path(*parts)

    async def _save_cover(self, user_id: int, dataset_id: int, cover: Optional[UploadFile]) -> Optional[str]:
        if cover is None or not cover.filename:
            return None
        content = await cover.read()
        if not content:
            return None
        return self._save_upload_bytes(user_id=user_id, dataset_id=dataset_id, filename=cover.filename, content=content)

    @staticmethod
    def _safe_unlink(path_value: Optional[str]) -> None:
        if not path_value:
            return
        try:
            path = Path(path_value)
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
                return
            path.unlink(missing_ok=True)
        except Exception:
            return

    def list_datasets(self, user_id: int) -> List[Dict]:
        rows = self.dataset_dao.list_datasets(user_id=user_id)
        return [self._to_payload(x.to_dict(include_internal=True)) for x in rows]

    def get_dataset(self, user_id: int, dataset_id: int) -> Optional[Dict]:
        row = self.dataset_dao.get_dataset_by_id(dataset_id=dataset_id, user_id=user_id)
        if row is None:
            return None
        return self._to_payload(row.to_dict(include_internal=True))

    def get_cover_path(self, user_id: int, dataset_id: int) -> Optional[str]:
        row = self.dataset_dao.get_dataset_by_id(dataset_id=dataset_id, user_id=user_id)
        if row is None:
            return None
        return row.to_dict(include_internal=True).get("cover_path")

    def create_dataset(self, user_id: int, payload: Dict) -> Dict:
        data = {
            "user_id": int(user_id),
            "name": str(payload.get("name") or "").strip(),
            "type": str(payload.get("type") or "instruction").strip().lower(),
            "source": self._normalize_text(payload.get("source")),
            "language": str(payload.get("language") or "multi").strip().lower(),
            "size": int(payload.get("size") or 0),
            "status": str(payload.get("status") or "uploaded").strip().lower() or "uploaded",
            "note": self._normalize_text(payload.get("note")),
            "sample_data": json.dumps(payload.get("sample_data") or [], ensure_ascii=False),
        }
        created = self.dataset_dao.insert_dataset(data)
        return self._to_payload(created.to_dict(include_internal=True))

    def update_dataset(self, user_id: int, dataset_id: int, payload: Dict) -> Optional[Dict]:
        updates: Dict[str, object] = {}
        for key in ("name", "type", "language", "status"):
            if key in payload and payload.get(key) is not None:
                updates[key] = str(payload.get(key)).strip().lower() if key in {"type", "language", "status"} else str(payload.get(key)).strip()

        if "source" in payload:
            updates["source"] = self._normalize_text(payload.get("source"))
        if "note" in payload:
            updates["note"] = self._normalize_text(payload.get("note"))

        if not updates:
            row = self.dataset_dao.get_dataset_by_id(dataset_id=dataset_id, user_id=user_id)
            return self._to_payload(row.to_dict(include_internal=True)) if row else None

        updated = self.dataset_dao.update_dataset(dataset_id=dataset_id, payload=updates, user_id=user_id)
        if updated is None:
            return None
        return self._to_payload(updated.to_dict(include_internal=True))

    async def update_cover(self, user_id: int, dataset_id: int, cover: UploadFile) -> Optional[Dict]:
        row = self.dataset_dao.get_dataset_by_id(dataset_id=dataset_id, user_id=user_id)
        if row is None:
            return None

        old = row.to_dict(include_internal=True).get("cover_path")
        cover_path = await self._save_cover(user_id=user_id, dataset_id=dataset_id, cover=cover)
        if not cover_path:
            return self._to_payload(row.to_dict(include_internal=True))

        updated = self.dataset_dao.update_dataset(dataset_id=dataset_id, payload={"cover_path": cover_path}, user_id=user_id)
        self._safe_unlink(old)
        return self._to_payload(updated.to_dict(include_internal=True)) if updated else None

    def delete_dataset(self, user_id: int, dataset_id: int) -> bool:
        row = self.dataset_dao.get_dataset_by_id(dataset_id=dataset_id, user_id=user_id)
        if row is None:
            return False

        raw = row.to_dict(include_internal=True)
        ok = self.dataset_dao.delete_dataset(dataset_id=dataset_id, user_id=user_id)
        if ok:
            self._safe_unlink(raw.get("file_path"))
            self._safe_unlink(raw.get("cover_path"))
        return ok

    async def upload_dataset(
        self,
        user_id: int,
        file: Optional[UploadFile],
        files: Optional[List[UploadFile]],
        cover: Optional[UploadFile],
        name: str,
        dataset_type: str,
        language: str,
        source: Optional[str] = None,
        note: Optional[str] = None,
    ) -> Dict:
        multi_files = [x for x in (files or []) if x is not None and x.filename]
        if multi_files:
            upload_list = multi_files
        elif file is not None and file.filename:
            upload_list = [file]
        else:
            upload_list = []
        if not upload_list:
            raise ValueError("file is required")

        first_name = upload_list[0].filename or "dataset"
        first_bytes = await upload_list[0].read()
        samples = self._parse_samples(first_name, first_bytes)

        total_size = 0
        contents: List[bytes] = []
        for idx, up in enumerate(upload_list):
            if idx == 0:
                content = first_bytes
            else:
                content = await up.read()
            contents.append(content)
            total_size += len(content)

        created = self.dataset_dao.insert_dataset(
            {
                "user_id": int(user_id),
                "name": str(name or "").strip() or Path(first_name).stem,
                "type": str(dataset_type or "instruction").strip().lower(),
                "source": self._normalize_text(source),
                "language": str(language or "multi").strip().lower(),
                "size": total_size,
                "status": "uploaded",
                "note": self._normalize_text(note),
                "file_name": Path(first_name).name,
                "file_path": None,
                "cover_path": None,
                "sample_data": json.dumps(samples, ensure_ascii=False),
            }
        )
        dataset_id = int(created.id)
        dataset_root = self.upload_dir / str(int(user_id)) / str(int(dataset_id))
        seen_relative_paths = set()
        for idx, up in enumerate(upload_list):
            normalized_relative = str(self._normalize_relative_upload_path(up.filename or f"file-{idx}")).replace("\\", "/")
            if normalized_relative in seen_relative_paths:
                continue
            seen_relative_paths.add(normalized_relative)

            saved = self._save_upload_bytes(
                user_id=user_id,
                dataset_id=dataset_id,
                filename=up.filename or f"file-{idx}",
                content=contents[idx],
            )

        cover_path = await self._save_cover(user_id=user_id, dataset_id=dataset_id, cover=cover)
        updated = self.dataset_dao.update_dataset(
            dataset_id=dataset_id,
            payload={
                "file_path": str(dataset_root.resolve()),
                "cover_path": cover_path,
            },
            user_id=user_id,
        )
        if updated is None:
            updated = created
        return self._to_payload(updated.to_dict(include_internal=True))
