from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Dict

from fastapi import UploadFile

from utils.config_loader import get_config


PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".xlsm", ".json", ".jsonl"}
_INVALID_PATH_CHARS = set('<>:"|?*\0')
_WORKSPACE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class AgentAssetService:
    def _workspace_base(self) -> Path:
        configured = str(get_config("workspace.base_path", "workspace") or "workspace").strip()
        base = Path(configured)
        if not base.is_absolute():
            base = PROJECT_ROOT / base
        base.mkdir(parents=True, exist_ok=True)
        return base.resolve()

    @staticmethod
    def _normalize_relative_path(value: str | None, *, allow_empty: bool = True) -> str:
        raw = str(value or "").strip().replace("\\", "/").strip("/")
        if not raw:
            if allow_empty:
                return ""
            raise ValueError("Path is required.")

        parts = PurePosixPath(raw).parts
        normalized = []
        for part in parts:
            if part in {"", ".", ".."}:
                raise ValueError("Path contains invalid segments.")
            if any(ch in _INVALID_PATH_CHARS for ch in part):
                raise ValueError(f"Path contains invalid character: {part}")
            normalized.append(part)
        return "/".join(normalized)

    @classmethod
    def _normalize_folder_name(cls, value: str | None) -> str:
        name = cls._normalize_relative_path(value, allow_empty=False)
        if "/" in name:
            raise ValueError("Folder name must be a single path segment.")
        return name

    @staticmethod
    def _validate_workspace_name(value: str | None) -> str:
        name = str(value or "").strip()
        if not name:
            raise ValueError("Workspace name is required.")
        if not _WORKSPACE_NAME_RE.fullmatch(name):
            raise ValueError("Invalid workspace name.")
        return name

    def _user_root(self, user_id: int) -> Path:
        root = self._workspace_base() / "_agent_users" / f"user_{int(user_id)}"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def asset_root(self, user_id: int) -> Path:
        root = self._user_root(user_id) / "assets"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def runtime_root(self, user_id: int) -> Path:
        root = self._user_root(user_id) / "runtime"
        root.mkdir(parents=True, exist_ok=True)
        return root

    @staticmethod
    def _resolve_under(base: Path, relative_path: str) -> Path:
        base_resolved = base.resolve()
        target = (base_resolved / relative_path).resolve() if relative_path else base_resolved
        if target != base_resolved and base_resolved not in target.parents:
            raise ValueError("Resolved path escapes asset root.")
        return target

    @staticmethod
    def _iso_from_stat(path: Path) -> str:
        return datetime.utcfromtimestamp(path.stat().st_mtime).isoformat()

    @staticmethod
    def _sort_key(path: Path):
        return (0 if path.is_dir() else 1, path.name.lower())

    def _build_tree(self, root: Path, current: Path) -> Dict[str, Any]:
        relative = current.relative_to(root).as_posix() if current != root else ""
        if current.is_dir():
            children = [self._build_tree(root, item) for item in sorted(current.iterdir(), key=self._sort_key)]
            folder_count = sum(child.get("folder_count", 0) + (1 if child["type"] == "folder" else 0) for child in children)
            file_count = sum(child.get("file_count", 1 if child["type"] == "file" else 0) for child in children)
            total_size = sum(child.get("total_size", child.get("size", 0)) for child in children)
            return {
                "id": relative or "root",
                "name": current.name if relative else "Root",
                "type": "folder",
                "path": relative,
                "is_root": relative == "",
                "updated_at": self._iso_from_stat(current),
                "children": children,
                "folder_count": folder_count,
                "file_count": file_count,
                "total_size": total_size,
            }

        return {
            "id": relative,
            "name": current.name,
            "type": "file",
            "path": relative,
            "extension": current.suffix.lower(),
            "size": current.stat().st_size,
            "updated_at": self._iso_from_stat(current),
        }

    def list_asset_tree(self, user_id: int) -> Dict[str, Any]:
        root = self.asset_root(user_id)
        tree = self._build_tree(root, root)
        return {
            "root": tree,
            "items": tree.get("children", []),
            "summary": {
                "folder_count": tree.get("folder_count", 0),
                "file_count": tree.get("file_count", 0),
                "total_size": tree.get("total_size", 0),
            },
            "asset_root": str(root),
        }

    def create_folder(self, user_id: int, name: str, parent_path: str | None = "") -> Dict[str, Any]:
        root = self.asset_root(user_id)
        normalized_parent = self._normalize_relative_path(parent_path, allow_empty=True)
        folder_name = self._normalize_folder_name(name)

        parent_dir = self._resolve_under(root, normalized_parent)
        if not parent_dir.exists():
            raise ValueError("Parent folder does not exist.")
        if not parent_dir.is_dir():
            raise ValueError("Parent path is not a folder.")

        target = parent_dir / folder_name
        if target.exists():
            raise ValueError("Folder already exists.")

        target.mkdir(parents=False, exist_ok=False)
        relative = target.relative_to(root).as_posix()
        return {
            "name": folder_name,
            "path": relative,
            "parent_path": normalized_parent,
            "updated_at": self._iso_from_stat(target),
        }

    async def upload_file(self, user_id: int, file: UploadFile, folder_path: str | None = "") -> Dict[str, Any]:
        root = self.asset_root(user_id)
        normalized_folder = self._normalize_relative_path(folder_path, allow_empty=True)
        target_dir = self._resolve_under(root, normalized_folder)
        if not target_dir.exists():
            raise ValueError("Target folder does not exist.")
        if not target_dir.is_dir():
            raise ValueError("Target path is not a folder.")

        file_name = Path(str(file.filename or "")).name
        if not file_name:
            raise ValueError("filename is required")
        self._normalize_folder_name(file_name)

        suffix = Path(file_name).suffix.lower()
        if suffix not in _ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {suffix}")

        content = await file.read()
        if not content:
            raise ValueError("uploaded file is empty")

        target = target_dir / file_name
        replaced = target.exists()
        target.write_bytes(content)
        relative = target.relative_to(root).as_posix()
        return {
            "filename": file_name,
            "path": relative,
            "folder_path": normalized_folder,
            "size": len(content),
            "replaced": replaced,
            "updated_at": self._iso_from_stat(target),
        }

    def delete_file(self, user_id: int, path: str) -> bool:
        root = self.asset_root(user_id)
        normalized = self._normalize_relative_path(path, allow_empty=False)
        target = self._resolve_under(root, normalized)
        if not target.exists():
            raise ValueError("File does not exist.")
        if not target.is_file():
            raise ValueError("Target path is not a file.")
        target.unlink()
        return True

    def delete_folder(self, user_id: int, path: str, *, force: bool = False) -> bool:
        root = self.asset_root(user_id)
        normalized = self._normalize_relative_path(path, allow_empty=False)
        target = self._resolve_under(root, normalized)
        if not target.exists():
            raise ValueError("Folder does not exist.")
        if not target.is_dir():
            raise ValueError("Target path is not a folder.")
        if any(target.iterdir()) and not force:
            raise ValueError("Folder is not empty.")
        if force:
            shutil.rmtree(target)
        else:
            target.rmdir()
        return True

    def has_files(self, user_id: int) -> bool:
        root = self.asset_root(user_id)
        return any(item.is_file() for item in root.rglob("*"))

    def normalize_asset_path(self, path: str | None, *, allow_empty: bool = False) -> str:
        return self._normalize_relative_path(path, allow_empty=allow_empty)

    def resolve_file_under(self, base_dir: Path, path: str | None) -> Path:
        normalized = self._normalize_relative_path(path, allow_empty=False)
        target = self._resolve_under(base_dir, normalized)
        if not target.exists():
            raise ValueError("Selected file does not exist.")
        if not target.is_file():
            raise ValueError("Selected path is not a file.")
        return target

    def prepare_runtime_workspace(self, user_id: int, workspace_name: str | None) -> Dict[str, Any]:
        name = self._validate_workspace_name(workspace_name)
        runtime_root = self.runtime_root(user_id)
        runtime_dir = runtime_root / name
        runtime_dir.mkdir(parents=True, exist_ok=True)

        source_root = self.asset_root(user_id)
        for item in source_root.iterdir():
            target = runtime_dir / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)

        return {
            "workspace_name": name,
            "workspace_dir": runtime_dir,
            "runtime_root": runtime_root,
        }

