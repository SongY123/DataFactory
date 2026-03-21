from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Dict

from fastapi import UploadFile

from utils.config_loader import get_config
from web.dao import DatasetDAO, ReasoningDistillationTaskDAO
from web.dao.agentic_synthesis_task_dao import AgenticSynthesisTaskDAO


PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".xlsm", ".json", ".jsonl"}
_ALLOWED_ARTIFACT_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}
_INVALID_PATH_CHARS = set('<>:"|?*\0')
_WORKSPACE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class AgentAssetService:
    def __init__(
        self,
        dataset_dao: DatasetDAO | None = None,
        agentic_task_dao: AgenticSynthesisTaskDAO | None = None,
        distillation_task_dao: ReasoningDistillationTaskDAO | None = None,
    ) -> None:
        self.dataset_dao = dataset_dao or DatasetDAO()
        self.agentic_task_dao = agentic_task_dao or AgenticSynthesisTaskDAO()
        self.distillation_task_dao = distillation_task_dao or ReasoningDistillationTaskDAO()

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

    def normalize_workspace_name(self, value: str | None) -> str:
        return self._validate_workspace_name(value)

    def normalize_artifact_path(self, path: str | None) -> str:
        raw = str(path or '').strip().replace('\\', '/').strip()
        if not raw:
            raise ValueError('Artifact path is required.')

        while raw.startswith('./'):
            raw = raw[2:]

        if raw.startswith('../output/'):
            raw = raw[3:]
        elif raw.startswith('../charts/'):
            raw = f"output/charts/{raw[len('../charts/') :]}"
        elif raw.startswith('charts/'):
            raw = f"output/charts/{raw[len('charts/') :]}"
        elif raw.startswith('./charts/'):
            raw = f"output/charts/{raw[len('./charts/') :]}"

        normalized = self._normalize_relative_path(raw.lstrip('/'), allow_empty=False)
        suffix = Path(normalized).suffix.lower()
        if suffix not in _ALLOWED_ARTIFACT_EXTENSIONS:
            raise ValueError(f'Unsupported artifact type: {suffix or normalized}')
        return normalized

    def resolve_runtime_artifact(self, user_id: int, workspace_name: str | None, artifact_path: str | None) -> Path:
        workspace = self._validate_workspace_name(workspace_name)
        normalized = self.normalize_artifact_path(artifact_path)
        runtime_dir = self.runtime_root(user_id) / workspace
        target = self._resolve_under(runtime_dir, normalized)
        if not target.exists() or not target.is_file():
            raise ValueError('Artifact file does not exist.')
        return target

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

    @staticmethod
    def _sanitize_copy_name(value: str | None, fallback: str) -> str:
        raw = str(value or "").strip().replace("\\", "/")
        candidate = Path(raw).name or fallback
        candidate = re.sub(r"[^A-Za-z0-9._-]+", "_", candidate).strip("._")
        return candidate or fallback

    @staticmethod
    def _copy_path(source: Path, target_dir: Path, target_name: str | None = None) -> Path:
        target_dir.mkdir(parents=True, exist_ok=True)
        name = target_name or source.name
        target = target_dir / name
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        return target

    def _resolve_dataset_source(self, user_id: int, dataset_id: int) -> Dict[str, Any]:
        dataset = self.dataset_dao.get_dataset_by_id(dataset_id=dataset_id, user_id=user_id)
        if dataset is None:
            raise ValueError("Dataset not found.")
        path = Path(str(dataset.file_path or "").strip())
        if not path.exists():
            raise ValueError("Dataset file path does not exist.")
        return {
            "source_type": "dataset",
            "source_id": int(dataset.id),
            "label": str(dataset.name),
            "path": path,
        }

    def _resolve_agentic_task_source(self, user_id: int, task_id: int) -> Dict[str, Any]:
        task = self.agentic_task_dao.get_task_by_id(task_id=task_id, user_id=user_id)
        if task is None:
            raise ValueError("Trajectory task not found.")
        if task.generated_dataset_id:
            dataset_payload = self._resolve_dataset_source(user_id=user_id, dataset_id=int(task.generated_dataset_id))
            dataset_payload["source_type"] = "trajectory_task"
            dataset_payload["source_id"] = int(task.id)
            dataset_payload["label"] = f"trajectory_task_{task.id}"
            return dataset_payload
        path = Path(str(task.output_file_path or "").strip())
        if not path.exists():
            raise ValueError("Trajectory task output does not exist.")
        resolved = path.parent if path.is_file() else path
        return {
            "source_type": "trajectory_task",
            "source_id": int(task.id),
            "label": f"trajectory_task_{task.id}",
            "path": resolved,
        }

    def _resolve_distillation_task_source(self, user_id: int, task_id: int) -> Dict[str, Any]:
        task = self.distillation_task_dao.get_task_by_id(task_id=task_id, user_id=user_id)
        if task is None:
            raise ValueError("Distillation task not found.")
        if task.generated_dataset_id:
            dataset_payload = self._resolve_dataset_source(user_id=user_id, dataset_id=int(task.generated_dataset_id))
            dataset_payload["source_type"] = "distillation_task"
            dataset_payload["source_id"] = int(task.id)
            dataset_payload["label"] = f"distillation_task_{task.id}"
            return dataset_payload
        path = Path(str(task.output_file_path or "").strip())
        if not path.exists():
            raise ValueError("Distillation task output does not exist.")
        resolved = path.parent if path.is_file() else path
        return {
            "source_type": "distillation_task",
            "source_id": int(task.id),
            "label": f"distillation_task_{task.id}",
            "path": resolved,
        }

    def resolve_platform_source(self, user_id: int, source_type: str, source_id: int) -> Dict[str, Any]:
        normalized = str(source_type or "").strip().lower()
        if normalized == "dataset":
            return self._resolve_dataset_source(user_id=user_id, dataset_id=int(source_id))
        if normalized == "trajectory_task":
            return self._resolve_agentic_task_source(user_id=user_id, task_id=int(source_id))
        if normalized == "distillation_task":
            return self._resolve_distillation_task_source(user_id=user_id, task_id=int(source_id))
        raise ValueError(f"Unsupported source_type: {source_type}")

    def import_platform_object(
        self,
        user_id: int,
        source_type: str,
        source_id: int,
        target_folder_path: str | None = "",
    ) -> Dict[str, Any]:
        source = self.resolve_platform_source(user_id=user_id, source_type=source_type, source_id=source_id)
        root = self.asset_root(user_id)
        normalized_folder = self._normalize_relative_path(target_folder_path, allow_empty=True)
        target_dir = self._resolve_under(root, normalized_folder)
        if not target_dir.exists():
            raise ValueError("Target folder does not exist.")
        if not target_dir.is_dir():
            raise ValueError("Target path is not a folder.")

        source_path = source["path"]
        target_name = self._sanitize_copy_name(source.get("label") or source_path.name, source_path.name)
        copied = self._copy_path(source_path, target_dir, target_name=target_name)
        relative = copied.relative_to(root).as_posix()
        return {
            "source_type": source["source_type"],
            "source_id": source["source_id"],
            "label": source["label"],
            "path": relative,
            "type": "folder" if copied.is_dir() else "file",
            "updated_at": self._iso_from_stat(copied),
        }

    def stage_context_items(self, user_id: int, runtime_dir: Path, context_items: list[Dict[str, Any]] | None) -> list[Dict[str, Any]]:
        staged: list[Dict[str, Any]] = []
        if not context_items:
            return staged

        attachments_root = runtime_dir / "_attached_context"
        attachments_root.mkdir(parents=True, exist_ok=True)
        asset_root = self.asset_root(user_id)

        for index, item in enumerate(context_items, start=1):
            item_type = str((item or {}).get("type") or "").strip().lower()
            if item_type == "asset_file":
                asset_path = self._normalize_relative_path((item or {}).get("path"), allow_empty=False)
                source = self._resolve_under(asset_root, asset_path)
                if not source.exists() or not source.is_file():
                    raise ValueError(f"Asset file does not exist: {asset_path}")
                target = self._copy_path(source, attachments_root / "asset_files", target_name=source.name)
                staged.append({"type": item_type, "path": str(target.relative_to(runtime_dir).as_posix()), "name": target.name})
                continue

            if item_type in {"dataset", "trajectory_task", "distillation_task"}:
                ref_id = int((item or {}).get("ref_id") or 0)
                if ref_id <= 0:
                    raise ValueError(f"ref_id is required for context type: {item_type}")
                source = self.resolve_platform_source(user_id=user_id, source_type=item_type, source_id=ref_id)
                source_path = source["path"]
                target_name = self._sanitize_copy_name(f"{item_type}_{ref_id}_{source_path.name}", f"{item_type}_{ref_id}")
                target = self._copy_path(source_path, attachments_root / item_type, target_name=target_name)
                staged.append({"type": item_type, "ref_id": ref_id, "path": str(target.relative_to(runtime_dir).as_posix()), "name": target.name})
                continue

            raise ValueError(f"Unsupported context item type: {item_type}")

        return staged
