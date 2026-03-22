from __future__ import annotations

import csv
import io
import json
import re
import shutil
import sqlite3
import threading
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote

import pandas as pd
from fastapi import UploadFile
from huggingface_hub import HfApi, hf_hub_download

from utils.config_loader import get_config
from utils.logger import logger
from ..dao import DatasetDAO


PROJECT_ROOT = Path(__file__).resolve().parents[3]
README_CANDIDATES = ("README.md", "readme.md", "README.txt", "readme.txt")
SKIP_FILE_NAMES = {".gitattributes"}
SKIP_DIR_NAMES = {".cache", "__pycache__"}
TABLE_PREVIEW_LIMIT = 200
SQL_SOURCE_LIMIT = 5000
SQL_RESULT_LIMIT = 200
MAX_README_CHARS = 200_000
MAX_TEXT_PREVIEW_LINES = 200
HF_MODALITY_TAGS = {"3d", "audio", "geospatial", "image", "tabular", "text", "timeseries", "video"}
HF_FORMAT_TAGS = {"arrow", "audiofolder", "csv", "excel", "imagefolder", "json", "jsonl", "parquet", "sqlite", "text", "tsv", "webdataset"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
GEOSPATIAL_EXTENSIONS = {".geojson", ".shp", ".gpkg"}
TIMESERIES_EXTENSIONS = {".ts", ".tsv"}
FORMAT_BY_SUFFIX = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".json": "json",
    ".jsonl": "jsonl",
    ".parquet": "parquet",
    ".xlsx": "excel",
    ".xls": "excel",
    ".sqlite": "sqlite",
    ".db": "sqlite",
    ".txt": "text",
    ".md": "text",
}
PREVIEWABLE_EXTENSIONS = set(FORMAT_BY_SUFFIX.keys())
SQL_ENABLED_EXTENSIONS = {".csv", ".tsv", ".json", ".jsonl", ".parquet", ".xlsx", ".xls", ".txt", ".md"}
PREVIEW_PRIORITY = {
    ".csv": 0,
    ".tsv": 1,
    ".parquet": 2,
    ".jsonl": 3,
    ".json": 4,
    ".xlsx": 5,
    ".xls": 6,
    ".txt": 7,
    ".md": 8,
}


class DatasetService:
    SAMPLE_PREVIEW_LIMIT = 20

    def __init__(self, dataset_dao: Optional[DatasetDAO] = None, upload_dir: Optional[Path] = None) -> None:
        self.dataset_dao = dataset_dao or DatasetDAO()
        self.upload_dir = upload_dir or self._resolve_upload_dir()
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.hf_api = HfApi()
        self._lock = threading.RLock()
        self._running_threads: Dict[int, threading.Thread] = {}

    @staticmethod
    def _resolve_upload_dir() -> Path:
        configured = str(get_config("dataset.base_path", "uploads") or "uploads").strip()
        base = Path(configured)
        if not base.is_absolute():
            base = PROJECT_ROOT / base
        return base.resolve()

    @staticmethod
    def _is_path_within_root(path: Path, root: Path) -> bool:
        try:
            path.resolve().relative_to(root.resolve())
            return True
        except Exception:
            return False

    @staticmethod
    def _placeholder_upload_prefixes() -> Tuple[str, ...]:
        return ("/abs/path/uploads/", "/abs/path/uploads")

    def _rewrite_placeholder_upload_path(self, value: Optional[str]) -> Optional[str]:
        text = self._normalize_text(value)
        if not text:
            return None
        normalized = text.replace("\\", "/")
        for prefix in self._placeholder_upload_prefixes():
            if normalized.startswith(prefix):
                relative = normalized[len(prefix):].lstrip("/")
                return str((self.upload_dir / relative).resolve())
        return text

    def _dataset_storage_root(self, user_id: int, dataset_id: int) -> Path:
        return (self.upload_dir / str(int(user_id)) / str(int(dataset_id))).resolve()

    def _ensure_dataset_storage_under_project(self, row: Dict[str, Any]) -> Dict[str, Any]:
        dataset_id = int(row.get("id") or 0)
        user_id = int(row.get("user_id") or 0)
        if dataset_id <= 0 or user_id <= 0:
            return row

        updates: Dict[str, Any] = {}
        file_path = self._rewrite_placeholder_upload_path(row.get("file_path"))
        cover_path = self._rewrite_placeholder_upload_path(row.get("cover_path"))
        target_root = self._dataset_storage_root(user_id=user_id, dataset_id=dataset_id)

        if file_path and file_path != row.get("file_path"):
            updates["file_path"] = file_path
            row["file_path"] = file_path
        if cover_path and cover_path != row.get("cover_path"):
            updates["cover_path"] = cover_path
            row["cover_path"] = cover_path

        current_file_path = self._normalize_text(row.get("file_path"))
        if current_file_path:
            source_path = Path(current_file_path)
            if source_path.exists() and not self._is_path_within_root(source_path, self.upload_dir):
                if source_path.is_dir():
                    if not target_root.exists():
                        target_root.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(source_path), str(target_root))
                    row["file_path"] = str(target_root.resolve())
                    updates["file_path"] = row["file_path"]
                elif source_path.is_file():
                    target_root.mkdir(parents=True, exist_ok=True)
                    target_file = target_root / source_path.name
                    if not target_file.exists():
                        shutil.move(str(source_path), str(target_file))
                    row["file_path"] = str(target_file.resolve())
                    updates["file_path"] = row["file_path"]

        current_cover_path = self._normalize_text(row.get("cover_path"))
        if current_cover_path:
            source_cover = Path(current_cover_path)
            if source_cover.exists() and not self._is_path_within_root(source_cover, self.upload_dir):
                target_root.mkdir(parents=True, exist_ok=True)
                target_cover = target_root / source_cover.name
                if not target_cover.exists():
                    shutil.move(str(source_cover), str(target_cover))
                row["cover_path"] = str(target_cover.resolve())
                updates["cover_path"] = row["cover_path"]

        if updates:
            updated = self.dataset_dao.update_dataset(dataset_id=dataset_id, user_id=user_id, payload=updates)
            if updated is not None:
                return updated.to_dict(include_internal=True)
        return row

    @staticmethod
    def _normalize_text(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _normalize_dataset_status(value: Any) -> str:
        normalized = str(value or "uploaded").strip().lower() or "uploaded"
        return "uploaded" if normalized == "ready" else normalized

    @staticmethod
    def _to_json_text(value: Any, default: Any) -> str:
        return json.dumps(value if value is not None else default, ensure_ascii=False)

    @staticmethod
    def _normalize_relative_upload_path(filename: str) -> Path:
        raw = str(filename or "").strip().replace("\\", "/")
        parsed = PurePosixPath(raw)
        parts = [part for part in parsed.parts if part not in {"", ".", ".."}]
        if not parts:
            parts = ["file"]
        return Path(*parts)

    @staticmethod
    def _decode_json_dict(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    @staticmethod
    def _decode_json_list(value: Any) -> List[str]:
        if isinstance(value, list):
            raw_values = value
        elif isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = [item.strip() for item in value.split(",")]
            raw_values = parsed if isinstance(parsed, list) else []
        else:
            raw_values = []

        result: List[str] = []
        seen = set()
        for item in raw_values:
            normalized = str(item or "").strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(normalized)
        return result

    @staticmethod
    def _normalize_tag_list(values: Any) -> List[str]:
        return DatasetService._decode_json_list(values)

    @staticmethod
    def _size_bucket(size: Any) -> str:
        value = int(size or 0)
        if value >= 1024 * 1024 * 1024:
            return "gb"
        if value >= 1024 * 1024:
            return "mb"
        return "kb"

    @staticmethod
    def _expand_format_aliases(values: Any) -> set[str]:
        aliases: set[str] = set()
        for item in DatasetService._normalize_tag_list(values):
            aliases.add(item)
            if item in {"excel", "xlsx", "xls"}:
                aliases.update({"excel", "xlsx", "xls"})
        return aliases

    @staticmethod
    def _is_generated_payload(payload: Dict[str, Any]) -> bool:
        dataset_type = str(payload.get("type") or "").strip().lower()
        source_kind = str(payload.get("source_kind") or "").strip().lower()
        return bool(
            payload.get("origin_stage")
            or payload.get("origin_task_id")
            or payload.get("origin_dataset_id")
            or source_kind == "generated"
            or dataset_type in {"trajectory", "reasoning"}
        )

    def _matches_dataset_query(self, payload: Dict[str, Any], filters: Optional[Dict[str, Any]] = None) -> bool:
        query = filters or {}

        name_keyword = str(query.get("name_keyword") or "").strip().lower()
        if name_keyword and name_keyword not in str(payload.get("name") or "").strip().lower():
            return False

        format_filters = self._expand_format_aliases(query.get("format_tags"))
        if format_filters:
            payload_formats = self._expand_format_aliases(payload.get("format_tags"))
            if payload_formats.isdisjoint(format_filters):
                return False

        language_filters = set(self._normalize_language_tags(query.get("language_tags"), None))
        if language_filters:
            payload_languages = set(self._normalize_language_tags(payload.get("language_tags"), payload.get("language")))
            if payload_languages.isdisjoint(language_filters):
                return False

        size_levels = set(self._normalize_tag_list(query.get("size_levels")))
        if size_levels and self._size_bucket(payload.get("size")) not in size_levels:
            return False

        status_filters = {self._normalize_dataset_status(item) for item in self._normalize_tag_list(query.get("statuses"))}
        if status_filters and self._normalize_dataset_status(payload.get("status")) not in status_filters:
            return False

        return True

    @staticmethod
    def _normalize_scalar(value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "item") and callable(getattr(value, "item")):
            try:
                value = value.item()
            except Exception:
                pass
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        if isinstance(value, (list, tuple, set)):
            return [DatasetService._normalize_scalar(item) for item in value]
        if isinstance(value, dict):
            return {str(key): DatasetService._normalize_scalar(item) for key, item in value.items()}
        return value

    @classmethod
    def _records_from_dataframe(cls, frame: pd.DataFrame) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for record in frame.to_dict(orient="records"):
            records.append({str(key): cls._normalize_scalar(value) for key, value in record.items()})
        return records

    @staticmethod
    def _relative_file_list(root: Path) -> List[str]:
        result: List[str] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            try:
                relative = path.relative_to(root)
            except ValueError:
                continue
            if any(part in SKIP_DIR_NAMES for part in relative.parts):
                continue
            if path.name in SKIP_FILE_NAMES:
                continue
            result.append(relative.as_posix())
        return result

    @staticmethod
    def _priority_sort_key(relative_path: str) -> Tuple[int, str]:
        suffix = Path(relative_path).suffix.lower()
        return (PREVIEW_PRIORITY.get(suffix, 99), relative_path.lower())

    @staticmethod
    def _is_data_preview_file(relative_path: str) -> bool:
        path = Path(relative_path)
        if path.name in README_CANDIDATES:
            return False
        return path.suffix.lower() in PREVIEWABLE_EXTENSIONS

    @staticmethod
    def _canonical_language_tag(value: str) -> str:
        normalized = str(value or "").strip().lower()
        if not normalized:
            return ""
        if normalized in {"zh-cn", "zh-hans", "chinese"}:
            return "zh"
        if normalized in {"english"}:
            return "en"
        return normalized

    @classmethod
    def _normalize_language_tags(cls, value: Any, fallback_language: Optional[str] = None) -> List[str]:
        tags = [cls._canonical_language_tag(item) for item in cls._normalize_tag_list(value)]
        tags = [item for item in tags if item]
        if tags:
            return tags
        fallback = cls._canonical_language_tag(str(fallback_language or "").strip().lower())
        return [fallback] if fallback else []

    @classmethod
    def _primary_language_from_tags(cls, tags: List[str], fallback: str = "multi") -> str:
        normalized = [cls._canonical_language_tag(item) for item in (tags or []) if cls._canonical_language_tag(item)]
        normalized = list(dict.fromkeys(normalized))
        if len(normalized) == 1 and normalized[0] in {"zh", "en"}:
            return normalized[0]
        return fallback if fallback in {"zh", "en", "multi"} else "multi"

    @staticmethod
    def _merge_tags(*tag_groups: Iterable[str]) -> List[str]:
        result: List[str] = []
        seen = set()
        for group in tag_groups:
            for item in group or []:
                normalized = str(item or "").strip().lower()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                result.append(normalized)
        return result

    @staticmethod
    def _dataset_source_url(repo_id: str) -> str:
        return f"https://huggingface.co/datasets/{quote(repo_id, safe='/')}"

    def _decode_sample_data(self, value: Any) -> List[Dict[str, Any]]:
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return [{"preview": value}]
        else:
            parsed = value

        if isinstance(parsed, list):
            result = []
            for item in parsed[: self.SAMPLE_PREVIEW_LIMIT]:
                if isinstance(item, dict):
                    result.append(item)
                else:
                    result.append({"value": item})
            return result
        if isinstance(parsed, dict):
            return [parsed]
        return []

    @classmethod
    def _parse_json_rows(cls, raw_text: str) -> List[Dict[str, Any]]:
        text = str(raw_text or "").strip()
        if not text:
            return []
        parsed = json.loads(text)
        if isinstance(parsed, list):
            rows = parsed[: cls.SAMPLE_PREVIEW_LIMIT]
            return [item if isinstance(item, dict) else {"value": item} for item in rows]
        if isinstance(parsed, dict):
            return [parsed]
        return [{"value": parsed}]

    @classmethod
    def _parse_samples(cls, filename: str, raw: bytes) -> List[Dict[str, Any]]:
        suffix = Path(filename).suffix.lower()
        if suffix in {".csv", ".tsv"}:
            text = raw.decode("utf-8", errors="ignore")
            reader = csv.DictReader(io.StringIO(text), delimiter="\t" if suffix == ".tsv" else ",")
            return [dict(row) for _, row in zip(range(cls.SAMPLE_PREVIEW_LIMIT), reader)]
        if suffix == ".json":
            return cls._parse_json_rows(raw.decode("utf-8", errors="ignore"))
        if suffix == ".jsonl":
            rows: List[Dict[str, Any]] = []
            for line in raw.decode("utf-8", errors="ignore").splitlines():
                item = line.strip()
                if not item:
                    continue
                parsed = json.loads(item)
                rows.append(parsed if isinstance(parsed, dict) else {"value": parsed})
                if len(rows) >= cls.SAMPLE_PREVIEW_LIMIT:
                    break
            return rows
        if suffix in {".xlsx", ".xls"}:
            frames = pd.read_excel(io.BytesIO(raw), sheet_name=None, nrows=cls.SAMPLE_PREVIEW_LIMIT)
            for _, frame in frames.items():
                if not frame.empty:
                    return cls._records_from_dataframe(frame.head(cls.SAMPLE_PREVIEW_LIMIT))
            return []
        preview_lines = raw.decode("utf-8", errors="ignore").splitlines()[: cls.SAMPLE_PREVIEW_LIMIT]
        if preview_lines:
            return [{"line_no": index + 1, "text": line} for index, line in enumerate(preview_lines)]
        return []

    def _safe_unlink(self, path_value: Optional[str]) -> None:
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

    def _save_upload_bytes(self, user_id: int, dataset_id: int, filename: str, content: bytes) -> str:
        relative_path = self._normalize_relative_upload_path(filename)
        saved_path = self.upload_dir / str(int(user_id)) / str(int(dataset_id)) / relative_path
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        saved_path.write_bytes(content)
        return str(saved_path.resolve())

    async def _save_cover(self, user_id: int, dataset_id: int, cover: Optional[UploadFile]) -> Optional[str]:
        if cover is None or not cover.filename:
            return None
        content = await cover.read()
        if not content:
            return None
        return self._save_upload_bytes(user_id=user_id, dataset_id=dataset_id, filename=cover.filename, content=content)

    def _resolve_dataset_root(self, row: Dict[str, Any]) -> Tuple[Optional[Path], Optional[str]]:
        raw_path = self._normalize_text(row.get("file_path"))
        if not raw_path:
            return None, None
        path = Path(raw_path)
        if path.is_file():
            return path.parent, path.name
        if path.is_dir():
            return path, None
        return None, None

    def _resolve_dataset_target(self, row: Dict[str, Any], relative_path: Optional[str] = None) -> Optional[Path]:
        root, primary_file_name = self._resolve_dataset_root(row)
        if root is None or not root.exists():
            return None

        if relative_path:
            normalized = self._normalize_relative_upload_path(relative_path)
            target = (root / normalized).resolve()
            root_resolved = root.resolve()
            if target != root_resolved and root_resolved not in target.parents:
                raise ValueError("dataset path escapes root")
            return target if target.exists() else None

        if primary_file_name:
            primary = root / primary_file_name
            if primary.exists():
                return primary
        return root

    def _detect_readme_path(self, root: Path) -> Optional[Path]:
        candidates: List[Path] = []
        for name in README_CANDIDATES:
            direct = root / name
            if direct.exists() and direct.is_file():
                return direct
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.name not in README_CANDIDATES:
                continue
            try:
                relative = path.relative_to(root)
            except ValueError:
                continue
            if any(part in SKIP_DIR_NAMES for part in relative.parts):
                continue
            candidates.append(path)
        candidates.sort(key=lambda item: (len(item.relative_to(root).parts), item.as_posix().lower()))
        return candidates[0] if candidates else None

    def _load_readme_text(self, row: Dict[str, Any]) -> Optional[str]:
        stored = self._normalize_text(row.get("readme_text"))
        if stored:
            return stored
        root, _ = self._resolve_dataset_root(row)
        if root is None or not root.exists():
            return None
        readme_path = self._detect_readme_path(root)
        if readme_path is None:
            return None
        try:
            return readme_path.read_text(encoding="utf-8", errors="ignore")[:MAX_README_CHARS]
        except Exception:
            return None

    def _infer_tags_from_paths(self, relative_paths: Iterable[str]) -> Tuple[List[str], List[str]]:
        modality_tags: List[str] = []
        format_tags: List[str] = []

        has_tabular = False
        has_text = False
        has_image = False
        has_audio = False
        has_video = False
        has_geo = False
        has_timeseries = False

        normalized_paths = [str(path or "") for path in relative_paths if str(path or "").strip()]
        for relative_path in normalized_paths:
            suffix = Path(relative_path).suffix.lower()
            format_tag = FORMAT_BY_SUFFIX.get(suffix)
            if format_tag:
                format_tags = self._merge_tags(format_tags, [format_tag])

            if suffix in {".csv", ".tsv", ".parquet", ".xlsx", ".xls", ".sqlite", ".db"}:
                has_tabular = True
            if suffix in {".json", ".jsonl", ".txt", ".md"}:
                has_text = True
            if suffix in IMAGE_EXTENSIONS:
                has_image = True
            if suffix in AUDIO_EXTENSIONS:
                has_audio = True
            if suffix in VIDEO_EXTENSIONS:
                has_video = True
            if suffix in GEOSPATIAL_EXTENSIONS:
                has_geo = True
            if suffix in TIMESERIES_EXTENSIONS:
                has_timeseries = True

        if has_tabular:
            modality_tags.append("tabular")
        if has_text:
            modality_tags.append("text")
        if has_image:
            modality_tags.append("image")
            if not format_tags:
                format_tags.append("imagefolder")
        if has_audio:
            modality_tags.append("audio")
            if not format_tags:
                format_tags.append("audiofolder")
        if has_video:
            modality_tags.append("video")
        if has_geo:
            modality_tags.append("geospatial")
        if has_timeseries:
            modality_tags.append("timeseries")

        return self._merge_tags(modality_tags), self._merge_tags(format_tags)

    def _extract_hf_metadata(self, info) -> Dict[str, Any]:
        info_tags = [str(tag or "").strip().lower() for tag in (getattr(info, "tags", None) or []) if str(tag or "").strip()]
        card_data = getattr(info, "card_data", None)
        card_dict = card_data.to_dict() if hasattr(card_data, "to_dict") else {}

        modality_tags = [tag.split(":", 1)[1] for tag in info_tags if tag.startswith("modality:")]
        modality_tags = [tag for tag in modality_tags if tag in HF_MODALITY_TAGS]

        format_tags = [tag.split(":", 1)[1] for tag in info_tags if tag.startswith("format:")]
        format_tags = [tag for tag in format_tags if tag in HF_FORMAT_TAGS]

        language_tags = [tag.split(":", 1)[1] for tag in info_tags if tag.startswith("language:")]
        if not language_tags:
            language_tags = card_dict.get("language") or []
        language_tags = self._normalize_language_tags(language_tags)

        license_tag = ""
        for tag in info_tags:
            if tag.startswith("license:"):
                license_tag = tag.split(":", 1)[1]
                break
        if not license_tag:
            license_value = card_dict.get("license")
            if isinstance(license_value, list):
                license_tag = str(license_value[0] or "").strip().lower() if license_value else ""
            else:
                license_tag = str(license_value or "").strip().lower()

        return {
            "pretty_name": self._normalize_text(card_dict.get("pretty_name")) or self._normalize_text(getattr(info, "id", None)),
            "description": self._normalize_text(getattr(info, "description", None)),
            "modality_tags": self._merge_tags(modality_tags),
            "format_tags": self._merge_tags(format_tags),
            "language_tags": language_tags,
            "license_tag": license_tag or None,
        }

    @staticmethod
    def _render_excerpt(value: Optional[str], limit: int = 180) -> Optional[str]:
        text = str(value or "").strip()
        if not text:
            return None
        compact = re.sub(r"\s+", " ", text)
        return compact[:limit] + "..." if len(compact) > limit else compact

    def _default_preview_path(self, row: Dict[str, Any]) -> Optional[str]:
        root, primary_file_name = self._resolve_dataset_root(row)
        if root is None or not root.exists():
            return None
        if primary_file_name and self._is_data_preview_file(primary_file_name):
            return primary_file_name
        file_list = self._relative_file_list(root)
        previewable = [item for item in file_list if self._is_data_preview_file(item)]
        if not previewable:
            previewable = [item for item in file_list if Path(item).suffix.lower() in PREVIEWABLE_EXTENSIONS]
        if not previewable:
            return None
        previewable.sort(key=self._priority_sort_key)
        return previewable[0]

    def _load_sample_data(self, row: Dict[str, Any], refresh_from_file: bool = False) -> List[Dict[str, Any]]:
        sample_data = self._decode_sample_data(row.get("sample_data"))
        if not refresh_from_file:
            return sample_data[: self.SAMPLE_PREVIEW_LIMIT]
        preview_path = self._default_preview_path(row)
        if not preview_path:
            return sample_data[: self.SAMPLE_PREVIEW_LIMIT]
        try:
            preview = self.get_dataset_preview(
                user_id=int(row.get("user_id") or 0),
                dataset_id=int(row.get("id") or 0),
                path=preview_path,
                limit=self.SAMPLE_PREVIEW_LIMIT,
                _row_override=row,
            )
            return list(preview.get("rows") or [])[: self.SAMPLE_PREVIEW_LIMIT]
        except Exception:
            return sample_data[: self.SAMPLE_PREVIEW_LIMIT]

    def _to_payload(self, row: Dict[str, Any], refresh_samples: bool = False) -> Dict[str, Any]:
        sample_data = self._load_sample_data(row, refresh_from_file=refresh_samples)
        generation_meta = self._decode_json_dict(row.get("generation_meta_json") or row.get("generation_meta"))
        modality_tags = self._normalize_tag_list(row.get("modality_tags_json") or row.get("modality_tags"))
        format_tags = self._normalize_tag_list(row.get("format_tags_json") or row.get("format_tags"))
        language_tags = self._normalize_language_tags(row.get("language_tags_json") or row.get("language_tags"), row.get("language"))
        license_tag = self._normalize_text(row.get("license_tag"))
        tag_groups = {
            "modality": modality_tags,
            "format": format_tags,
            "language": language_tags,
            "license": [license_tag] if license_tag else [],
        }
        all_tags = self._merge_tags(modality_tags, format_tags, language_tags, tag_groups["license"])
        dataset_id = row.get("id")
        status = self._normalize_dataset_status(row.get("status"))
        import_progress = int(row.get("import_progress") or (0 if status == "downloading" else 100))
        import_total_files = int(row.get("import_total_files") or 0)
        import_downloaded_files = int(row.get("import_downloaded_files") or 0)
        cover_url = f"/api/datasets/{dataset_id}/cover" if row.get("cover_path") else None
        readme_text = self._normalize_text(row.get("readme_text"))

        return {
            "id": dataset_id,
            "user_id": row.get("user_id"),
            "name": row.get("name"),
            "type": row.get("type"),
            "source": row.get("source"),
            "source_kind": row.get("source_kind") or "upload",
            "hf_repo_id": row.get("hf_repo_id"),
            "hf_revision": row.get("hf_revision"),
            "language": row.get("language"),
            "language_tags": language_tags,
            "size": int(row.get("size") or 0),
            "status": status,
            "note": row.get("note"),
            "cover_url": cover_url,
            "sample_data": sample_data,
            "modality_tags": modality_tags,
            "format_tags": format_tags,
            "license_tag": license_tag,
            "tags": all_tags,
            "tag_groups": tag_groups,
            "readme_available": bool(readme_text),
            "readme_excerpt": self._render_excerpt(readme_text),
            "import_progress": import_progress,
            "import_total_files": import_total_files,
            "import_downloaded_files": import_downloaded_files,
            "import_error_message": row.get("import_error_message"),
            "download_state": {
                "progress": import_progress,
                "total_files": import_total_files,
                "downloaded_files": import_downloaded_files,
                "is_importing": status == "downloading",
                "error_message": row.get("import_error_message"),
            },
            "origin_stage": row.get("origin_stage"),
            "origin_dataset_id": row.get("origin_dataset_id"),
            "origin_task_type": row.get("origin_task_type"),
            "origin_task_id": row.get("origin_task_id"),
            "generation_meta": generation_meta,
            "generated_output": generation_meta,
            "insert_time": row.get("insert_time"),
            "update_time": row.get("update_time"),
        }

    @staticmethod
    def _normalize_sql_table_name(value: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9_]+", "_", str(value or "").strip()).strip("_").lower()
        if not normalized:
            normalized = "dataset"
        if normalized[0].isdigit():
            normalized = f"t_{normalized}"
        return normalized

    @classmethod
    def _ensure_unique_table_name(cls, name: str, seen: set[str]) -> str:
        base = cls._normalize_sql_table_name(name)
        candidate = base
        index = 2
        while candidate in seen:
            candidate = f"{base}_{index}"
            index += 1
        seen.add(candidate)
        return candidate

    def _load_dataframe_bundle(self, path: Path, row_limit: int) -> Dict[str, Any]:
        suffix = path.suffix.lower()
        seen_names: set[str] = set()
        tables: List[Dict[str, Any]] = []

        if suffix in {".csv", ".tsv"}:
            frame = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",", nrows=row_limit)
            tables.append({"name": "dataset", "source_name": path.name, "frame": frame})
        elif suffix == ".json":
            parsed = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(parsed, list):
                rows = [item if isinstance(item, dict) else {"value": item} for item in parsed[:row_limit]]
                frame = pd.DataFrame(rows)
            elif isinstance(parsed, dict):
                frame = pd.json_normalize(parsed)
            else:
                frame = pd.DataFrame([{"value": parsed}])
            tables.append({"name": "dataset", "source_name": path.name, "frame": frame})
        elif suffix == ".jsonl":
            rows: List[Dict[str, Any]] = []
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                item = line.strip()
                if not item:
                    continue
                parsed = json.loads(item)
                rows.append(parsed if isinstance(parsed, dict) else {"value": parsed})
                if len(rows) >= row_limit:
                    break
            frame = pd.DataFrame(rows)
            tables.append({"name": "dataset", "source_name": path.name, "frame": frame})
        elif suffix in {".xlsx", ".xls"}:
            workbook = pd.read_excel(path, sheet_name=None, nrows=row_limit)
            for sheet_name, frame in workbook.items():
                table_name = self._ensure_unique_table_name(str(sheet_name or "sheet"), seen_names)
                tables.append({"name": table_name, "source_name": str(sheet_name), "frame": frame})
        elif suffix == ".parquet":
            frame = pd.read_parquet(path).head(row_limit)
            tables.append({"name": "dataset", "source_name": path.name, "frame": frame})
        elif suffix in {".txt", ".md"}:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()[: min(row_limit, MAX_TEXT_PREVIEW_LINES)]
            frame = pd.DataFrame([{"line_no": index + 1, "text": line} for index, line in enumerate(lines)])
            tables.append({"name": "dataset", "source_name": path.name, "frame": frame})
        else:
            raise ValueError(f"unsupported preview format: {suffix}")

        if not tables:
            raise ValueError("no previewable tables found")

        for table in tables:
            if table["name"] not in seen_names:
                seen_names.add(table["name"])
        selected = tables[0]
        return {
            "path": path,
            "format": FORMAT_BY_SUFFIX.get(suffix, suffix.lstrip(".")),
            "selected_table": selected["name"],
            "available_tables": [{"name": item["name"], "source_name": item["source_name"]} for item in tables],
            "tables": tables,
            "sql_supported": suffix in SQL_ENABLED_EXTENSIONS,
        }

    def list_datasets(self, user_id: int) -> List[Dict[str, Any]]:
        rows = self.dataset_dao.list_datasets(user_id=user_id)
        payloads: List[Dict[str, Any]] = []
        for item in rows:
            normalized_row = self._ensure_dataset_storage_under_project(item.to_dict(include_internal=True))
            payloads.append(self._to_payload(normalized_row))
        return payloads

    def search_datasets(self, user_id: int, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        rows = self.dataset_dao.list_datasets(user_id=user_id)
        payloads: List[Dict[str, Any]] = []
        for item in rows:
            normalized_row = self._ensure_dataset_storage_under_project(item.to_dict(include_internal=True))
            payloads.append(self._to_payload(normalized_row))

        filtered = [payload for payload in payloads if self._matches_dataset_query(payload, filters)]
        return {
            "items": filtered,
            "total_count": len(payloads),
            "filtered_count": len(filtered),
            "importing_count": sum(1 for payload in payloads if self._normalize_dataset_status(payload.get("status")) == "downloading"),
            "generated_count": sum(1 for payload in payloads if self._is_generated_payload(payload)),
        }

    def get_dataset(self, user_id: int, dataset_id: int) -> Optional[Dict[str, Any]]:
        row = self.dataset_dao.get_dataset_by_id(dataset_id=dataset_id, user_id=user_id)
        if row is None:
            return None
        raw = self._ensure_dataset_storage_under_project(row.to_dict(include_internal=True))
        payload = self._to_payload(raw, refresh_samples=True)
        payload["default_preview_path"] = self._default_preview_path(raw)
        payload["is_explorable"] = payload.get("status") != "downloading"
        return payload

    def get_cover_path(self, user_id: int, dataset_id: int) -> Optional[str]:
        row = self.dataset_dao.get_dataset_by_id(dataset_id=dataset_id, user_id=user_id)
        if row is None:
            return None
        raw = self._ensure_dataset_storage_under_project(row.to_dict(include_internal=True))
        return raw.get("cover_path")

    def create_dataset(self, user_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        language_tags = self._normalize_language_tags(payload.get("language_tags"), payload.get("language"))
        language = self._primary_language_from_tags(language_tags, str(payload.get("language") or "multi").strip().lower() or "multi")
        data = {
            "user_id": int(user_id),
            "name": str(payload.get("name") or "").strip(),
            "type": str(payload.get("type") or "instruction").strip().lower(),
            "source": self._normalize_text(payload.get("source")),
            "source_kind": str(payload.get("source_kind") or "upload").strip().lower() or "upload",
            "language": language,
            "language_tags_json": self._to_json_text(language_tags, []),
            "size": int(payload.get("size") or 0),
            "status": str(payload.get("status") or "uploaded").strip().lower() or "uploaded",
            "note": self._normalize_text(payload.get("note")),
            "sample_data": self._to_json_text(payload.get("sample_data") or [], []),
            "modality_tags_json": self._to_json_text(self._normalize_tag_list(payload.get("modality_tags")), []),
            "format_tags_json": self._to_json_text(self._normalize_tag_list(payload.get("format_tags")), []),
            "license_tag": self._normalize_text(payload.get("license_tag")),
            "readme_text": self._normalize_text(payload.get("readme_text")),
            "hf_repo_id": self._normalize_text(payload.get("hf_repo_id")),
            "hf_revision": self._normalize_text(payload.get("hf_revision")),
            "import_progress": int(payload.get("import_progress") or 100),
            "import_total_files": int(payload.get("import_total_files") or 0),
            "import_downloaded_files": int(payload.get("import_downloaded_files") or 0),
            "import_error_message": self._normalize_text(payload.get("import_error_message")),
            "origin_stage": self._normalize_text(payload.get("origin_stage")),
            "origin_dataset_id": int(payload.get("origin_dataset_id")) if payload.get("origin_dataset_id") else None,
            "origin_task_type": self._normalize_text(payload.get("origin_task_type")),
            "origin_task_id": int(payload.get("origin_task_id")) if payload.get("origin_task_id") else None,
            "generation_meta_json": self._to_json_text(payload.get("generation_meta") or {}, {}),
        }
        created = self.dataset_dao.insert_dataset(data)
        return self._to_payload(created.to_dict(include_internal=True))

    def register_generated_dataset(
        self,
        *,
        user_id: int,
        name: str,
        dataset_type: str,
        language: str,
        source: Optional[str],
        note: Optional[str],
        file_path: str,
        file_name: str,
        size: int,
        sample_data: Optional[List[Dict[str, Any]]] = None,
        origin_stage: str,
        origin_dataset_id: Optional[int],
        origin_task_type: Optional[str],
        origin_task_id: Optional[int],
        generation_meta: Optional[Dict[str, Any]] = None,
        status: str = "ready",
    ) -> Dict[str, Any]:
        relative_paths = [Path(file_path).name]
        modality_tags, format_tags = self._infer_tags_from_paths(relative_paths)
        language_tags = self._normalize_language_tags([language] if language else [], language)
        payload = {
            "user_id": int(user_id),
            "name": str(name or "").strip(),
            "type": str(dataset_type or "instruction").strip().lower(),
            "source": self._normalize_text(source),
            "source_kind": "generated",
            "language": self._primary_language_from_tags(language_tags, str(language or "multi").strip().lower() or "multi"),
            "language_tags_json": self._to_json_text(language_tags, []),
            "size": int(size or 0),
            "status": str(status or "ready").strip().lower() or "ready",
            "note": self._normalize_text(note),
            "file_name": str(file_name or "").strip() or Path(file_path).name,
            "file_path": str(file_path),
            "cover_path": None,
            "sample_data": self._to_json_text(sample_data or [], []),
            "modality_tags_json": self._to_json_text(modality_tags, []),
            "format_tags_json": self._to_json_text(format_tags, []),
            "license_tag": None,
            "readme_text": None,
            "import_progress": 100,
            "import_total_files": 1,
            "import_downloaded_files": 1,
            "import_error_message": None,
            "origin_stage": self._normalize_text(origin_stage),
            "origin_dataset_id": int(origin_dataset_id) if origin_dataset_id else None,
            "origin_task_type": self._normalize_text(origin_task_type),
            "origin_task_id": int(origin_task_id) if origin_task_id else None,
            "generation_meta_json": self._to_json_text(generation_meta or {}, {}),
        }
        created = self.dataset_dao.insert_dataset(payload)
        return self._to_payload(created.to_dict(include_internal=True))

    def update_dataset(self, user_id: int, dataset_id: int, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        updates: Dict[str, Any] = {}
        if "name" in payload and payload.get("name") is not None:
            updates["name"] = str(payload.get("name")).strip()
        if "type" in payload and payload.get("type") is not None:
            updates["type"] = str(payload.get("type")).strip().lower()
        if "status" in payload and payload.get("status") is not None:
            updates["status"] = str(payload.get("status")).strip().lower()
        if "source" in payload:
            updates["source"] = self._normalize_text(payload.get("source"))
        if "note" in payload:
            updates["note"] = self._normalize_text(payload.get("note"))
        if "modality_tags" in payload:
            updates["modality_tags_json"] = self._to_json_text(self._normalize_tag_list(payload.get("modality_tags")), [])
        if "format_tags" in payload:
            updates["format_tags_json"] = self._to_json_text(self._normalize_tag_list(payload.get("format_tags")), [])
        if "license_tag" in payload:
            updates["license_tag"] = self._normalize_text(payload.get("license_tag"))

        if "language_tags" in payload and payload.get("language_tags") is not None:
            language_tags = self._normalize_language_tags(payload.get("language_tags"), payload.get("language"))
            updates["language_tags_json"] = self._to_json_text(language_tags, [])
            updates["language"] = self._primary_language_from_tags(language_tags, str(payload.get("language") or "multi").strip().lower() or "multi")
        elif "language" in payload and payload.get("language") is not None:
            language = str(payload.get("language") or "multi").strip().lower() or "multi"
            updates["language"] = language
            updates["language_tags_json"] = self._to_json_text(self._normalize_language_tags([language], language), [])

        if not updates:
            row = self.dataset_dao.get_dataset_by_id(dataset_id=dataset_id, user_id=user_id)
            return self._to_payload(row.to_dict(include_internal=True)) if row else None

        updated = self.dataset_dao.update_dataset(dataset_id=dataset_id, payload=updates, user_id=user_id)
        if updated is None:
            return None
        return self._to_payload(updated.to_dict(include_internal=True))

    async def update_cover(self, user_id: int, dataset_id: int, cover: UploadFile) -> Optional[Dict[str, Any]]:
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
    ) -> Dict[str, Any]:
        multi_files = [item for item in (files or []) if item is not None and item.filename]
        upload_list = multi_files if multi_files else ([file] if file is not None and file.filename else [])
        if not upload_list:
            raise ValueError("file is required")

        upload_entries: List[Tuple[str, bytes]] = []
        total_size = 0
        samples: List[Dict[str, Any]] = []
        relative_names: List[str] = []
        readme_text = None
        first_name = upload_list[0].filename or "dataset"

        for index, upload in enumerate(upload_list):
            raw_name = upload.filename or f"file-{index}"
            content = await upload.read()
            upload_entries.append((raw_name, content))
            total_size += len(content)
            normalized_relative = str(self._normalize_relative_upload_path(raw_name)).replace("\\", "/")
            relative_names.append(normalized_relative)

            if not samples:
                try:
                    parsed_samples = self._parse_samples(raw_name, content)
                except Exception:
                    parsed_samples = []
                if parsed_samples:
                    samples = parsed_samples

            if readme_text is None and Path(raw_name).name in README_CANDIDATES:
                readme_text = content.decode("utf-8", errors="ignore")[:MAX_README_CHARS]

        modality_tags, format_tags = self._infer_tags_from_paths(relative_names)
        language_tags = self._normalize_language_tags([language], language)

        created = self.dataset_dao.insert_dataset(
            {
                "user_id": int(user_id),
                "name": str(name or "").strip() or Path(first_name).stem,
                "type": str(dataset_type or "instruction").strip().lower(),
                "source": self._normalize_text(source),
                "source_kind": "upload",
                "language": self._primary_language_from_tags(language_tags, str(language or "multi").strip().lower() or "multi"),
                "language_tags_json": self._to_json_text(language_tags, []),
                "size": total_size,
                "status": "uploaded",
                "note": self._normalize_text(note),
                "file_name": Path(first_name).name,
                "file_path": None,
                "cover_path": None,
                "sample_data": self._to_json_text(samples, []),
                "readme_text": self._normalize_text(readme_text),
                "modality_tags_json": self._to_json_text(modality_tags, []),
                "format_tags_json": self._to_json_text(format_tags, []),
                "license_tag": None,
                "import_progress": 100,
                "import_total_files": len(upload_entries),
                "import_downloaded_files": len(upload_entries),
                "import_error_message": None,
            }
        )
        dataset_id = int(created.id)
        dataset_root = self.upload_dir / str(int(user_id)) / str(int(dataset_id))
        seen_relative_paths = set()

        try:
            for index, (raw_name, content) in enumerate(upload_entries):
                normalized_relative = str(self._normalize_relative_upload_path(raw_name)).replace("\\", "/")
                if normalized_relative in seen_relative_paths:
                    continue
                seen_relative_paths.add(normalized_relative)
                self._save_upload_bytes(
                    user_id=user_id,
                    dataset_id=dataset_id,
                    filename=raw_name or f"file-{index}",
                    content=content,
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
        except Exception:
            self._safe_unlink(str(dataset_root.resolve()) if dataset_root.exists() else None)
            self.dataset_dao.delete_dataset(dataset_id=dataset_id, user_id=user_id)
            raise

    def import_huggingface_dataset(
        self,
        *,
        user_id: int,
        repo_id: str,
        revision: Optional[str] = None,
        name: Optional[str] = None,
        note: Optional[str] = None,
    ) -> Dict[str, Any]:
        clean_repo_id = str(repo_id or "").strip()
        if not clean_repo_id:
            raise ValueError("repo_id is required")

        try:
            info = self.hf_api.dataset_info(clean_repo_id, revision=revision)
        except Exception as exc:
            raise ValueError(f"failed to fetch HuggingFace dataset info: {exc}") from exc

        hf_meta = self._extract_hf_metadata(info)
        dataset_name = self._normalize_text(name) or self._normalize_text(hf_meta.get("pretty_name")) or clean_repo_id.split("/")[-1]
        description = self._normalize_text(note) or self._normalize_text(hf_meta.get("description"))
        language_tags = self._normalize_language_tags(hf_meta.get("language_tags"), None)
        language = self._primary_language_from_tags(language_tags, "multi")

        created = self.dataset_dao.insert_dataset(
            {
                "user_id": int(user_id),
                "name": dataset_name,
                "type": "instruction",
                "source": self._dataset_source_url(clean_repo_id),
                "source_kind": "huggingface",
                "hf_repo_id": clean_repo_id,
                "hf_revision": self._normalize_text(revision),
                "language": language,
                "language_tags_json": self._to_json_text(language_tags, []),
                "size": 0,
                "status": "downloading",
                "note": description,
                "file_name": dataset_name,
                "file_path": None,
                "cover_path": None,
                "sample_data": self._to_json_text([], []),
                "readme_text": None,
                "modality_tags_json": self._to_json_text(hf_meta.get("modality_tags") or [], []),
                "format_tags_json": self._to_json_text(hf_meta.get("format_tags") or [], []),
                "license_tag": self._normalize_text(hf_meta.get("license_tag")),
                "import_progress": 0,
                "import_total_files": 0,
                "import_downloaded_files": 0,
                "import_error_message": None,
            }
        )

        dataset_id = int(created.id)
        dataset_root = self.upload_dir / str(int(user_id)) / str(dataset_id)
        dataset_root.mkdir(parents=True, exist_ok=True)

        readme_text = None
        try:
            readme_local_path = hf_hub_download(
                repo_id=clean_repo_id,
                repo_type="dataset",
                revision=revision,
                filename="README.md",
                local_dir=dataset_root,
                local_dir_use_symlinks=False,
            )
            readme_text = Path(readme_local_path).read_text(encoding="utf-8", errors="ignore")[:MAX_README_CHARS]
        except Exception:
            readme_text = self._normalize_text(hf_meta.get("description"))

        updated = self.dataset_dao.update_dataset(
            dataset_id=dataset_id,
            user_id=user_id,
            payload={
                "file_path": str(dataset_root.resolve()),
                "readme_text": self._normalize_text(readme_text),
            },
        )
        if updated is None:
            raise ValueError("failed to initialize dataset import record")

        thread = threading.Thread(
            target=self._run_huggingface_import,
            args=(int(user_id), dataset_id, clean_repo_id, self._normalize_text(revision)),
            daemon=True,
            name=f"dataset-hf-import-{dataset_id}",
        )
        with self._lock:
            self._running_threads[dataset_id] = thread
        thread.start()
        return self._to_payload(updated.to_dict(include_internal=True))

    def _run_huggingface_import(self, user_id: int, dataset_id: int, repo_id: str, revision: Optional[str]) -> None:
        dataset_root = self.upload_dir / str(int(user_id)) / str(int(dataset_id))
        downloaded_files = 0
        total_files = 0

        try:
            repo_files = self.hf_api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
            ordered_files = sorted(repo_files, key=lambda item: (0 if item == "README.md" else 1, item.lower()))
            total_files = len(ordered_files)
            self.dataset_dao.update_dataset(
                dataset_id=dataset_id,
                user_id=user_id,
                payload={"import_total_files": total_files, "import_progress": 0, "status": "downloading"},
            )

            readme_text = None
            for file_name in ordered_files:
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    revision=revision,
                    filename=file_name,
                    local_dir=dataset_root,
                    local_dir_use_symlinks=False,
                )
                downloaded_files += 1

                if file_name == "README.md" and readme_text is None:
                    try:
                        readme_text = (dataset_root / "README.md").read_text(encoding="utf-8", errors="ignore")[:MAX_README_CHARS]
                    except Exception:
                        readme_text = None

                progress = int(downloaded_files * 100 / total_files) if total_files else 100
                update_payload = {
                    "import_downloaded_files": downloaded_files,
                    "import_progress": progress,
                    "status": "downloading",
                }
                if readme_text is not None:
                    update_payload["readme_text"] = readme_text
                self.dataset_dao.update_dataset(dataset_id=dataset_id, user_id=user_id, payload=update_payload)

            row = self.dataset_dao.get_dataset_by_id(dataset_id=dataset_id, user_id=user_id)
            if row is None:
                return
            raw = row.to_dict(include_internal=True)
            root, _ = self._resolve_dataset_root(raw)
            relative_paths = self._relative_file_list(root) if root else []
            modality_tags, format_tags = self._infer_tags_from_paths(relative_paths)
            existing_modality = self._normalize_tag_list(raw.get("modality_tags_json"))
            existing_format = self._normalize_tag_list(raw.get("format_tags_json"))
            preview_path = self._default_preview_path(raw)
            samples: List[Dict[str, Any]] = []
            if preview_path:
                try:
                    preview_payload = self.get_dataset_preview(
                        user_id=user_id,
                        dataset_id=dataset_id,
                        path=preview_path,
                        limit=self.SAMPLE_PREVIEW_LIMIT,
                        _row_override=raw,
                    )
                    samples = list(preview_payload.get("rows") or [])[: self.SAMPLE_PREVIEW_LIMIT]
                except Exception as exc:
                    logger.warning("failed to build HF sample preview: dataset_id=%s err=%s", dataset_id, exc)

            total_size = 0
            if root and root.exists():
                for path in root.rglob("*"):
                    if path.is_file():
                        try:
                            relative = path.relative_to(root)
                        except ValueError:
                            continue
                        if any(part in SKIP_DIR_NAMES for part in relative.parts) or path.name in SKIP_FILE_NAMES:
                            continue
                        total_size += path.stat().st_size

            self.dataset_dao.update_dataset(
                dataset_id=dataset_id,
                user_id=user_id,
                payload={
                    "size": total_size,
                    "sample_data": self._to_json_text(samples, []),
                    "modality_tags_json": self._to_json_text(self._merge_tags(existing_modality, modality_tags), []),
                    "format_tags_json": self._to_json_text(self._merge_tags(existing_format, format_tags), []),
                    "status": "ready",
                    "import_progress": 100,
                    "import_downloaded_files": total_files,
                    "import_total_files": total_files,
                    "import_error_message": None,
                },
            )
        except Exception as exc:
            logger.exception("HuggingFace dataset import failed. dataset_id=%s repo_id=%s", dataset_id, repo_id)
            self.dataset_dao.update_dataset(
                dataset_id=dataset_id,
                user_id=user_id,
                payload={
                    "status": "failed",
                    "import_progress": int(downloaded_files * 100 / total_files) if total_files else 0,
                    "import_downloaded_files": downloaded_files,
                    "import_total_files": total_files,
                    "import_error_message": str(exc),
                },
            )
        finally:
            with self._lock:
                self._running_threads.pop(dataset_id, None)

    def get_dataset_readme(self, user_id: int, dataset_id: int) -> Optional[Dict[str, Any]]:
        row = self.dataset_dao.get_dataset_by_id(dataset_id=dataset_id, user_id=user_id)
        if row is None:
            return None
        raw = self._ensure_dataset_storage_under_project(row.to_dict(include_internal=True))
        content = self._load_readme_text(raw) or ""
        if content and not self._normalize_text(raw.get("readme_text")):
            self.dataset_dao.update_dataset(dataset_id=dataset_id, user_id=user_id, payload={"readme_text": content})
        return {
            "dataset_id": dataset_id,
            "available": bool(content),
            "content": content,
        }

    def _build_file_tree_node(self, root: Path, current: Path) -> Optional[Dict[str, Any]]:
        try:
            relative = current.relative_to(root)
        except ValueError:
            return None

        if any(part in SKIP_DIR_NAMES for part in relative.parts):
            return None
        if current.name in SKIP_FILE_NAMES:
            return None

        relative_path = relative.as_posix() if relative.parts else ""
        if current.is_dir():
            children: List[Dict[str, Any]] = []
            for child in sorted(current.iterdir(), key=lambda item: (0 if item.is_dir() else 1, item.name.lower())):
                child_node = self._build_file_tree_node(root, child)
                if child_node is not None:
                    children.append(child_node)
            return {
                "id": relative_path or "root",
                "name": current.name if relative_path else "Root",
                "type": "folder",
                "path": relative_path,
                "children": children,
            }

        return {
            "id": relative_path,
            "name": current.name,
            "type": "file",
            "path": relative_path,
            "extension": current.suffix.lower(),
            "size": current.stat().st_size,
        }

    def get_dataset_files(self, user_id: int, dataset_id: int) -> Optional[Dict[str, Any]]:
        row = self.dataset_dao.get_dataset_by_id(dataset_id=dataset_id, user_id=user_id)
        if row is None:
            return None
        raw = self._ensure_dataset_storage_under_project(row.to_dict(include_internal=True))
        payload = self._to_payload(raw)
        root, _ = self._resolve_dataset_root(raw)
        if root is None or not root.exists():
            return {
                "dataset_id": dataset_id,
                "ready": False,
                "tree": None,
                "data_files": [],
                "default_preview_path": None,
            }
        if payload.get("status") == "downloading":
            return {
                "dataset_id": dataset_id,
                "ready": False,
                "tree": None,
                "data_files": [],
                "default_preview_path": None,
            }

        tree = self._build_file_tree_node(root, root)
        data_files = [item for item in self._relative_file_list(root) if self._is_data_preview_file(item)]
        data_files.sort(key=self._priority_sort_key)
        return {
            "dataset_id": dataset_id,
            "ready": True,
            "tree": tree,
            "data_files": [{"path": item, "name": Path(item).name, "extension": Path(item).suffix.lower()} for item in data_files],
            "default_preview_path": self._default_preview_path(raw),
        }

    def get_dataset_preview(
        self,
        user_id: int,
        dataset_id: int,
        path: Optional[str] = None,
        limit: int = TABLE_PREVIEW_LIMIT,
        _row_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        row = _row_override
        if row is None:
            entity = self.dataset_dao.get_dataset_by_id(dataset_id=dataset_id, user_id=user_id)
            if entity is None:
                raise ValueError("dataset not found")
            row = self._ensure_dataset_storage_under_project(entity.to_dict(include_internal=True))

        payload = self._to_payload(row)
        if payload.get("status") == "downloading":
            raise ValueError("dataset files are still downloading")

        effective_path = path or self._default_preview_path(row)
        if not effective_path:
            raise ValueError("dataset has no previewable data file")
        target = self._resolve_dataset_target(row, effective_path)
        if target is None or not target.is_file():
            raise ValueError("dataset preview file not found")

        bundle = self._load_dataframe_bundle(target, max(1, min(int(limit or TABLE_PREVIEW_LIMIT), TABLE_PREVIEW_LIMIT)))
        selected_table = next(item for item in bundle["tables"] if item["name"] == bundle["selected_table"])
        frame = selected_table["frame"].head(limit)
        root, _ = self._resolve_dataset_root(row)
        relative_path = target.relative_to(root).as_posix() if root else target.name

        return {
            "dataset_id": dataset_id,
            "path": relative_path,
            "file_name": target.name,
            "format": bundle["format"],
            "columns": [str(column) for column in list(frame.columns)],
            "rows": self._records_from_dataframe(frame),
            "row_count": int(len(frame.index)),
            "selected_table": bundle["selected_table"],
            "available_tables": bundle["available_tables"],
            "sql_supported": bundle["sql_supported"],
            "default_sql": f"SELECT * FROM {bundle['selected_table']} LIMIT 100",
        }

    def query_dataset_sql(self, user_id: int, dataset_id: int, path: Optional[str], sql: str, limit: int = SQL_RESULT_LIMIT) -> Dict[str, Any]:
        entity = self.dataset_dao.get_dataset_by_id(dataset_id=dataset_id, user_id=user_id)
        if entity is None:
            raise ValueError("dataset not found")
        row = self._ensure_dataset_storage_under_project(entity.to_dict(include_internal=True))
        payload = self._to_payload(row)
        if payload.get("status") == "downloading":
            raise ValueError("dataset files are still downloading")

        effective_path = path or self._default_preview_path(row)
        if not effective_path:
            raise ValueError("dataset has no queryable data file")
        target = self._resolve_dataset_target(row, effective_path)
        if target is None or not target.is_file():
            raise ValueError("dataset query file not found")

        bundle = self._load_dataframe_bundle(target, SQL_SOURCE_LIMIT)
        if not bundle["sql_supported"]:
            raise ValueError("SQL query is not supported for this file format")

        normalized_sql = str(sql or "").strip()
        lowered = normalized_sql.lower()
        if not lowered.startswith(("select", "with")):
            raise ValueError("only SELECT queries are supported")
        if ";" in normalized_sql.rstrip().rstrip(";"):
            raise ValueError("only a single SQL statement is supported")
        if re.search(r"\b(insert|update|delete|drop|alter|create|attach|pragma|vacuum|replace)\b", lowered):
            raise ValueError("mutating SQL is not supported")

        connection = sqlite3.connect(":memory:")
        try:
            for table in bundle["tables"]:
                table["frame"].to_sql(table["name"], connection, index=False, if_exists="replace")

            result_frame = pd.read_sql_query(normalized_sql, connection)
            result_limit = max(1, min(int(limit or SQL_RESULT_LIMIT), SQL_RESULT_LIMIT))
            truncated = len(result_frame.index) > result_limit
            if truncated:
                result_frame = result_frame.head(result_limit)

            root, _ = self._resolve_dataset_root(row)
            relative_path = target.relative_to(root).as_posix() if root else target.name
            return {
                "dataset_id": dataset_id,
                "path": relative_path,
                "columns": [str(column) for column in list(result_frame.columns)],
                "rows": self._records_from_dataframe(result_frame),
                "row_count": int(len(result_frame.index)),
                "truncated": truncated,
                "available_tables": bundle["available_tables"],
            }
        except Exception as exc:
            raise ValueError(f"SQL query failed: {exc}") from exc
        finally:
            connection.close()
