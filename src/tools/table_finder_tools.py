"""Recursive table finder tools used by TableFinderWorker."""

from __future__ import annotations

import os
import json
from pathlib import Path

import pandas as pd
from agents.context import get_workspace
from agentscope.message import TextBlock
from agentscope.tool import ToolResponse

_SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".xlsm", ".json", ".jsonl"}


_TEXT_ENCODINGS = ["utf-8", "utf-8-sig", "gbk", "gb2312"]


def _dataframe_from_json_obj(data) -> pd.DataFrame:
    if isinstance(data, list):
        if not data:
            return pd.DataFrame()
        if all(isinstance(item, dict) for item in data):
            return pd.json_normalize(data)
        return pd.DataFrame({"value": data})

    if isinstance(data, dict):
        list_candidates = [(key, value) for key, value in data.items() if isinstance(value, list)]
        if len(list_candidates) == 1:
            key, records = list_candidates[0]
            if not records:
                return pd.DataFrame(columns=[key])
            if all(isinstance(item, dict) for item in records):
                df = pd.json_normalize(records)
            else:
                df = pd.DataFrame({key: records})
            for meta_key, meta_value in data.items():
                if meta_key == key or isinstance(meta_value, (list, dict)):
                    continue
                df[meta_key] = meta_value
            return df

        if all(not isinstance(value, (list, dict)) for value in data.values()):
            return pd.DataFrame([data])

        return pd.json_normalize(data)

    return pd.DataFrame({"value": [data]})


def _read_json_file(file_path: str) -> pd.DataFrame:
    last_error: Exception | None = None

    for encoding in _TEXT_ENCODINGS:
        try:
            raw_text = Path(file_path).read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue

        stripped = raw_text.strip()
        if not stripped:
            return pd.DataFrame()

        try:
            return _dataframe_from_json_obj(json.loads(stripped))
        except json.JSONDecodeError as exc:
            last_error = exc
            try:
                return pd.read_json(file_path, lines=True, encoding=encoding)
            except ValueError as line_exc:
                last_error = line_exc
                continue

    if last_error:
        raise last_error
    raise ValueError(f"Unable to read JSON file: {file_path}")


def _resolve_directory_path(directory: str) -> str:
    directory = str(directory or "").strip()
    if not directory:
        return str(get_workspace() or "")

    if os.path.isabs(directory):
        return directory

    if directory.startswith("workspace/") or directory.startswith("workspace\\"):
        return directory

    workspace = get_workspace()
    return os.path.join(workspace, directory) if workspace else directory


def _resolve_file_path(file_path: str) -> str:
    file_path = str(file_path or "").strip()
    if not file_path:
        return file_path

    if os.path.isabs(file_path):
        return file_path

    if file_path.startswith("workspace/") or file_path.startswith("workspace\\"):
        return file_path

    workspace = get_workspace()
    return os.path.join(workspace, file_path) if workspace else file_path


def _read_data_file(file_path: str) -> pd.DataFrame:
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext in {".xlsx", ".xls", ".xlsm"}:
        return pd.read_excel(file_path)
    if file_ext == ".csv":
        for encoding in _TEXT_ENCODINGS:
            try:
                return pd.read_csv(file_path, encoding=encoding)
            except (UnicodeDecodeError, UnicodeError):
                continue
        return pd.read_csv(file_path)
    if file_ext in {".json", ".jsonl"}:
        return _read_json_file(file_path)

    raise Exception(f"Unsupported file format: {file_ext}")


def _iter_data_files(directory: str) -> list[Path]:
    root = Path(directory)
    if not root.exists() or not root.is_dir():
        return []

    files = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in _SUPPORTED_EXTENSIONS]
    files.sort(key=lambda item: (str(item.parent).lower(), item.name.lower()))
    return files


def _relative_to_root(root: str, path: Path) -> str:
    try:
        return path.relative_to(Path(root)).as_posix()
    except ValueError:
        return path.as_posix()


def list_data_files(directory: str) -> ToolResponse:
    try:
        directory = _resolve_directory_path(directory)
        if not os.path.exists(directory):
            return ToolResponse(content=[TextBlock(type="text", text=f"Directory does not exist: {directory}")])

        files = _iter_data_files(directory)
        if not files:
            return ToolResponse(content=[TextBlock(type="text", text=f"No data files found in directory: {directory}")])

        result = f"Data files under directory: {directory}\nFound {len(files)} file(s):\n\n"
        for index, file_path in enumerate(files, 1):
            size = os.path.getsize(file_path)
            size_str = f"{size / 1024:.2f} KB" if size < 1024 * 1024 else f"{size / (1024 * 1024):.2f} MB"
            relative_path = _relative_to_root(directory, file_path)
            result += f"{index}. {file_path.name}\n"
            result += f"   - relative path: {relative_path}\n"
            result += f"   - absolute path: {file_path}\n"
            result += f"   - type: {file_path.suffix.lower()}\n"
            result += f"   - size: {size_str}\n\n"

        return ToolResponse(content=[TextBlock(type="text", text=result)])
    except Exception as e:
        return ToolResponse(content=[TextBlock(type="text", text=f"List files failed: {str(e)}")])


def inspect_table_structure(file_path: str) -> ToolResponse:
    try:
        file_path = _resolve_file_path(file_path)
        df = _read_data_file(file_path)

        result = f"Table structure: {file_path}\n{'=' * 60}\n\n"
        result += f"Rows: {len(df):,}\n"
        result += f"Columns: {len(df.columns)}\n\n"
        result += "Column details:\n\n"

        for index, col in enumerate(df.columns, 1):
            result += f"{index}. {col}\n"
            result += f"   - dtype: {df[col].dtype}\n"
            result += f"   - non-null: {df[col].count():,}\n"

            sample_values = df[col].dropna().head(3).tolist()
            if sample_values:
                sample_str = ", ".join(str(v) for v in sample_values)
                if len(sample_str) > 50:
                    sample_str = sample_str[:47] + "..."
                result += f"   - sample: {sample_str}\n"

            if pd.api.types.is_numeric_dtype(df[col]):
                result += f"   - range: {df[col].min()} ~ {df[col].max()}\n"
                result += f"   - mean: {df[col].mean():.2f}\n"
            elif pd.api.types.is_object_dtype(df[col]):
                unique_count = df[col].nunique()
                result += f"   - unique values: {unique_count}\n"
                if unique_count <= 10:
                    result += f"   - values: {', '.join(str(v) for v in df[col].dropna().unique()[:10])}\n"
            result += "\n"

        result += f"Preview (first 3 rows):\n{'-' * 60}\n{df.head(3).to_string(index=False)}\n{'-' * 60}\n"
        return ToolResponse(content=[TextBlock(type="text", text=result)])
    except Exception as e:
        return ToolResponse(content=[TextBlock(type="text", text=f"Inspect table failed: {str(e)}")])


def search_tables_by_keywords(keywords: str, directory: str) -> ToolResponse:
    try:
        directory = _resolve_directory_path(directory)
        if not os.path.exists(directory):
            return ToolResponse(content=[TextBlock(type="text", text=f"Directory does not exist: {directory}")])

        keyword_list = [item.strip().lower() for item in str(keywords or "").split(",") if item.strip()]
        files = _iter_data_files(directory)
        matches: list[dict] = []

        for file_path in files:
            score = 0
            matched_keywords: list[str] = []
            matched_columns: list[str] = []
            filename_lower = file_path.name.lower()

            for keyword in keyword_list:
                if keyword in filename_lower:
                    score += 10
                    matched_keywords.append(keyword)

            try:
                df = _read_data_file(str(file_path))
            except Exception:
                continue

            for keyword in keyword_list:
                for col in df.columns:
                    col_lower = str(col).lower()
                    if keyword in col_lower:
                        score += 5
                        matched_columns.append(str(col))
                        if keyword not in matched_keywords:
                            matched_keywords.append(keyword)

            if score > 0:
                matches.append(
                    {
                        "filename": file_path.name,
                        "path": str(file_path),
                        "relative_path": _relative_to_root(directory, file_path),
                        "score": score,
                        "matched_keywords": matched_keywords,
                        "matched_columns": matched_columns,
                        "total_columns": len(df.columns),
                        "total_rows": len(df),
                        "all_columns": [str(col) for col in df.columns],
                    }
                )

        matches.sort(key=lambda item: item["score"], reverse=True)

        if not matches:
            result = (
                f"Keyword search: {keywords}\n"
                f"No matching data files were found under directory: {directory}\n\n"
                "Suggestions:\n"
                "1. Use list_data_files to inspect the available files.\n"
                "2. Use inspect_table_structure on the likely files.\n"
            )
            return ToolResponse(content=[TextBlock(type="text", text=result)])

        result = f"Keyword search: {keywords}\nFound {len(matches)} matching data file(s):\n\n"
        for index, match in enumerate(matches, 1):
            result += f"{index}. {match['filename']} (score: {match['score']})\n"
            result += f"   - relative path: {match['relative_path']}\n"
            result += f"   - absolute path: {match['path']}\n"
            result += f"   - rows x columns: {match['total_rows']:,} x {match['total_columns']}\n"
            result += f"   - matched keywords: {', '.join(match['matched_keywords'])}\n"
            if match['matched_columns']:
                result += f"   - matched columns: {', '.join(match['matched_columns'])}\n"
            result += f"   - columns: {', '.join(match['all_columns'][:5])}"
            if len(match['all_columns']) > 5:
                result += f" ... (total {match['total_columns']} columns)"
            result += "\n\n"

        result += f"Recommended file: {matches[0]['path']}\n"
        return ToolResponse(content=[TextBlock(type="text", text=result)])
    except Exception as e:
        return ToolResponse(content=[TextBlock(type="text", text=f"Search tables failed: {str(e)}")])
