"""
数据分析工具函数
这些函数将被注册到 AgentScope 的 Toolkit 中供 ReAct Agent 调用
"""
import pandas as pd
import os
import json
from typing import Dict, Any, List
from pathlib import Path
from agentscope.tool import ToolResponse
from agentscope.message import TextBlock
from agents.context import get_workspace

_TEXT_ENCODINGS = ["utf-8", "utf-8-sig", "gbk", "gb2312"]


def _dataframe_from_json_obj(data: Any) -> pd.DataFrame:
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
    """Resolve a directory relative to the active workspace when needed."""
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
    """Resolve a file path relative to the active workspace when needed."""
    file_path = str(file_path or "").strip()
    if not file_path:
        return file_path

    if os.path.isabs(file_path):
        return file_path

    if file_path.startswith("workspace/") or file_path.startswith("workspace\\"):
        return file_path

    workspace = get_workspace()
    return os.path.join(workspace, file_path) if workspace else file_path


def _resolve_file_from_directory(file_path: str, directory: str) -> str:
    """Resolve a file path relative to a caller-provided directory."""
    file_path = str(file_path or "").strip()
    if not file_path:
        return file_path

    if os.path.isabs(file_path):
        return file_path

    if file_path.startswith("workspace/") or file_path.startswith("workspace\\"):
        return file_path

    return os.path.join(_resolve_directory_path(directory), file_path)


def _read_data_file(file_path: str) -> pd.DataFrame:
    """Read supported tabular files into a DataFrame."""
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext in [".xlsx", ".xls", ".xlsm"]:
        return pd.read_excel(file_path)
    if file_ext == ".csv":
        for encoding in _TEXT_ENCODINGS:
            try:
                return pd.read_csv(file_path, encoding=encoding)
            except (UnicodeDecodeError, UnicodeError):
                continue
        return pd.read_csv(file_path)
    if file_ext in [".json", ".jsonl"]:
        return _read_json_file(file_path)

    raise Exception(f"Unsupported file format: {file_ext}. Supported formats: .xlsx, .xls, .xlsm, .csv, .json, .jsonl")


def read_excel_file(file_path: str) -> ToolResponse:
    """
    读取数据文件并返回基本信息
    
    支持格式：Excel (.xlsx, .xls, .xlsm) 和 CSV (.csv)
    
    Args:
        file_path: 数据文件路径（可以是文件名或完整路径）
    
    Returns:
        包含数据基本信息的字符串
    """
    try:
        # 解析文件路径
        file_path = _resolve_file_path(file_path)
        
        if not os.path.exists(file_path):
            return ToolResponse(content=[TextBlock(type="text", text=f"❌ 错误：文件 {file_path} 不存在")])
        
        df = _read_data_file(file_path)
        
        info = f"""
📊 数据文件读取成功：{file_path}
- 数据行数：{len(df)}
- 列名：{', '.join(df.columns.tolist())}
- 数据类型：{df.dtypes.to_dict()}

前 5 行数据预览：
{df.head().to_string()}
"""
        return ToolResponse(content=[TextBlock(type="text", text=info)])
    except Exception as e:
        return ToolResponse(content=[TextBlock(type="text", text=f"❌ 读取文件失败：{str(e)}")])


def get_column_stats(column_name: str, file_path: str) -> ToolResponse:
    """
    获取指定列的统计信息
    
    支持格式：Excel (.xlsx, .xls, .xlsm) 和 CSV (.csv)
    
    Args:
        column_name: 列名
        file_path: 数据文件路径（可以是文件名或完整路径）
    
    Returns:
        该列的统计信息
    """
    try:
        # 解析文件路径
        file_path = _resolve_file_path(file_path)
        
        df = _read_data_file(file_path)
        
        if column_name not in df.columns:
            return ToolResponse(content=[TextBlock(type="text", text=f"❌ 错误：列 '{column_name}' 不存在。可用列：{', '.join(df.columns)}")])
        
        col_data = df[column_name]
        
        # 判断是数值型还是文本型
        if pd.api.types.is_numeric_dtype(col_data):
            stats = f"""
📈 列 '{column_name}' 的统计信息（数值型）：
- 总计：{col_data.sum():,.2f}
- 平均值：{col_data.mean():,.2f}
- 中位数：{col_data.median():,.2f}
- 最小值：{col_data.min():,.2f}
- 最大值：{col_data.max():,.2f}
- 标准差：{col_data.std():,.2f}
"""
        else:
            value_counts = col_data.value_counts().head(10)
            stats = f"""
📋 列 '{column_name}' 的统计信息（分类型）：
- 唯一值数量：{col_data.nunique()}
- 前 10 个高频值：
{value_counts.to_string()}
"""
        return ToolResponse(content=[TextBlock(type="text", text=stats)])
    except Exception as e:
        return ToolResponse(content=[TextBlock(type="text", text=f"❌ 分析失败：{str(e)}")])


def filter_data(
    column_name: str,
    condition: str,
    value: Any,
    file_path: str
) -> ToolResponse:
    """
    根据条件筛选数据
    
    支持格式：Excel (.xlsx, .xls, .xlsm) 和 CSV (.csv)
    
    Args:
        column_name: 要筛选的列名
        condition: 条件 (等于、大于、小于、包含)
        value: 筛选值
        file_path: 数据文件路径
    
    Returns:
        筛选后的数据信息
    """
    try:
        # 解析文件路径
        file_path = _resolve_file_path(file_path)
        
        df = _read_data_file(file_path)
        
        if column_name not in df.columns:
            return ToolResponse(content=[TextBlock(type="text", text=f"❌ 错误：列 '{column_name}' 不存在")])
        
        # 根据条件筛选
        if condition == "等于":
            filtered_df = df[df[column_name] == value]
        elif condition == "大于":
            filtered_df = df[df[column_name] > float(value)]
        elif condition == "小于":
            filtered_df = df[df[column_name] < float(value)]
        elif condition == "包含":
            filtered_df = df[df[column_name].astype(str).str.contains(str(value), na=False)]
        else:
            return ToolResponse(content=[TextBlock(type="text", text=f"❌ 不支持的条件：{condition}。支持的条件：等于、大于、小于、包含")])
        
        result = f"""
🔍 筛选结果：{column_name} {condition} {value}
- 符合条件的记录数：{len(filtered_df)}
- 占总数据的比例：{len(filtered_df)/len(df)*100:.2f}%

前 10 条记录：
{filtered_df.head(10).to_string()}
"""
        return ToolResponse(content=[TextBlock(type="text", text=result)])
    except Exception as e:
        return ToolResponse(content=[TextBlock(type="text", text=f"❌ 筛选失败：{str(e)}")])


def group_analysis(
    group_by_column: str,
    agg_column: str,
    agg_func: str = "sum",
    file_path: str = None
) -> ToolResponse:
    """
    按指定列分组并进行聚合分析
    
    支持格式：Excel (.xlsx, .xls, .xlsm) 和 CSV (.csv)
    
    Args:
        group_by_column: 分组依据的列名
        agg_column: 要聚合的列名
        agg_func: 聚合函数 (sum/mean/count/max/min)
        file_path: 数据文件路径
    
    Returns:
        分组聚合结果
    """
    try:
        # 解析文件路径
        file_path = _resolve_file_path(file_path)
        
        df = _read_data_file(file_path)
        
        if group_by_column not in df.columns:
            return ToolResponse(content=[TextBlock(type="text", text=f"❌ 错误：分组列 '{group_by_column}' 不存在")])
        
        if agg_column not in df.columns:
            return ToolResponse(content=[TextBlock(type="text", text=f"❌ 错误：聚合列 '{agg_column}' 不存在")])
        
        # 执行分组聚合
        agg_funcs = {
            "sum": "求和",
            "mean": "平均值",
            "count": "计数",
            "max": "最大值",
            "min": "最小值"
        }
        
        if agg_func not in agg_funcs:
            return ToolResponse(content=[TextBlock(type="text", text=f"❌ 不支持的聚合函数：{agg_func}。支持：{', '.join(agg_funcs.keys())}")])
        
        grouped = df.groupby(group_by_column)[agg_column].agg(agg_func).sort_values(ascending=False)
        
        result = f"""
📊 分组分析结果：
- 分组依据：{group_by_column}
- 聚合列：{agg_column}
- 聚合方式：{agg_funcs[agg_func]}

结果：
{grouped.to_string()}
"""
        return ToolResponse(content=[TextBlock(type="text", text=result)])
    except Exception as e:
        return ToolResponse(content=[TextBlock(type="text", text=f"❌ 分组分析失败：{str(e)}")])


def compare_data(
    column_to_compare: str,
    group_column: str,
    file_path: str
) -> ToolResponse:
    """Compare values across groups in one data file."""
    try:
        file_path = _resolve_file_path(file_path)
        df = _read_data_file(file_path)

        if column_to_compare not in df.columns or group_column not in df.columns:
            return ToolResponse(content=[TextBlock(type="text", text="Column does not exist in the selected file.")])

        comparison = df.groupby(group_column)[column_to_compare].agg(["sum", "mean", "count"])
        comparison = comparison.sort_values("sum", ascending=False)

        result = f"""
Comparison analysis: {group_column} vs {column_to_compare}

{comparison.to_string()}

Summary:
- Highest group: {comparison.index[0]} (sum: {comparison.iloc[0]['sum']:,.2f})
- Lowest group: {comparison.index[-1]} (sum: {comparison.iloc[-1]['sum']:,.2f})
- Difference multiple: {comparison.iloc[0]['sum'] / comparison.iloc[-1]['sum']:.2f}x
"""
        return ToolResponse(content=[TextBlock(type="text", text=result)])
    except Exception as e:
        return ToolResponse(content=[TextBlock(type="text", text=f"Compare analysis failed: {str(e)}")])


# ============= 多文件分析工具 =============

def list_excel_files(directory: str) -> ToolResponse:
    """List supported data files under a directory."""
    try:
        directory = _resolve_directory_path(directory)
        if not os.path.exists(directory):
            return ToolResponse(content=[TextBlock(type="text", text=f"Directory does not exist: {directory}")])

        data_files = []
        for ext in ["*.xlsx", "*.xls", "*.xlsm", "*.csv", "*.json", "*.jsonl"]:
            data_files.extend(Path(directory).glob(ext))

        if not data_files:
            return ToolResponse(content=[TextBlock(type="text", text=f"No data files found in directory: {directory}")])

        excel_files = [f for f in data_files if f.suffix.lower() in [".xlsx", ".xls", ".xlsm"]]
        csv_files = [f for f in data_files if f.suffix.lower() == ".csv"]
        json_files = [f for f in data_files if f.suffix.lower() in [".json", ".jsonl"]]

        result = f"Data files in directory: {directory}\n\n"

        if excel_files:
            result += "Excel files:\n"
            for i, file in enumerate(excel_files, 1):
                file_size = os.path.getsize(file) / 1024
                result += f"{i}. {file.name} (size: {file_size:.2f} KB)\n"
                result += f"   path: {file}\n"
            result += "\n"

        if csv_files:
            result += "CSV files:\n"
            for i, file in enumerate(csv_files, 1):
                file_size = os.path.getsize(file) / 1024
                result += f"{i}. {file.name} (size: {file_size:.2f} KB)\n"
                result += f"   path: {file}\n"
            result += "\n"

        if json_files:
            result += "JSON files:\n"
            for i, file in enumerate(json_files, 1):
                file_size = os.path.getsize(file) / 1024
                result += f"{i}. {file.name} (size: {file_size:.2f} KB)\n"
                result += f"   path: {file}\n"

        return ToolResponse(content=[TextBlock(type="text", text=result)])
    except Exception as e:
        return ToolResponse(content=[TextBlock(type="text", text=f"List files failed: {str(e)}")])


def read_multiple_excel_files(file_paths: str, directory: str) -> ToolResponse:
    """Read multiple files and return summary information."""
    try:
        directory = _resolve_directory_path(directory)
        file_list = [f.strip() for f in file_paths.split(",") if f.strip()]

        result = f"Reading multiple data files:\n{'=' * 60}\n\n"

        for idx, filename in enumerate(file_list, 1):
            file_path = _resolve_file_from_directory(filename, directory)

            if not os.path.exists(file_path):
                result += f"{idx}. Missing file: {filename}\n\n"
                continue

            try:
                df = _read_data_file(file_path)
                file_ext = os.path.splitext(filename)[1].upper()
                result += f"{idx}. {filename} ({file_ext[1:]})\n"
                result += f"   - rows: {len(df)}\n"
                result += f"   - columns: {len(df.columns)}\n"
                result += f"   - column names: {', '.join(df.columns.tolist()[:5])}"
                if len(df.columns) > 5:
                    result += f" ... (total {len(df.columns)} columns)"
                result += f"\n   - path: {file_path}\n\n"
            except Exception as e:
                result += f"{idx}. Read failed: {filename} - {str(e)}\n\n"

        return ToolResponse(content=[TextBlock(type="text", text=result)])
    except Exception as e:
        return ToolResponse(content=[TextBlock(type="text", text=f"Read multiple files failed: {str(e)}")])


def merge_excel_files(
    file_paths: str,
    directory: str,
    merge_column: str = None
) -> ToolResponse:
    """Merge multiple files for combined analysis."""
    try:
        directory = _resolve_directory_path(directory)
        file_list = [f.strip() for f in file_paths.split(",") if f.strip()]
        dfs = []

        for filename in file_list:
            file_path = _resolve_file_from_directory(filename, directory)

            if not os.path.exists(file_path):
                return ToolResponse(content=[TextBlock(type="text", text=f"Missing file: {filename}")])

            df = _read_data_file(file_path)
            df["_source_file"] = filename
            dfs.append(df)

        if merge_column and merge_column in dfs[0].columns:
            merged_df = dfs[0]
            for df in dfs[1:]:
                if merge_column in df.columns:
                    merged_df = pd.merge(merged_df, df, on=merge_column, how="outer")
            merge_type = f"join on column '{merge_column}'"
        else:
            merged_df = pd.concat(dfs, ignore_index=True)
            merge_type = "vertical concat"

        result = f"""
Merged multiple data files:
- file count: {len(file_list)}
- merge type: {merge_type}
- merged rows: {len(merged_df)}
- merged columns: {len(merged_df.columns)}

Source row counts:
{merged_df['_source_file'].value_counts().to_string()}

Preview:
{merged_df.head().to_string()}
"""
        return ToolResponse(content=[TextBlock(type="text", text=result)])
    except Exception as e:
        return ToolResponse(content=[TextBlock(type="text", text=f"Merge files failed: {str(e)}")])


def compare_excel_files(
    file_paths: str,
    compare_column: str,
    directory: str
) -> ToolResponse:
    """Compare one column across multiple files."""
    try:
        directory = _resolve_directory_path(directory)
        file_list = [f.strip() for f in file_paths.split(",") if f.strip()]
        comparison_data = {}

        for filename in file_list:
            file_path = _resolve_file_from_directory(filename, directory)

            if not os.path.exists(file_path):
                return ToolResponse(content=[TextBlock(type="text", text=f"Missing file: {filename}")])

            df = _read_data_file(file_path)

            if compare_column not in df.columns:
                return ToolResponse(content=[TextBlock(type="text", text=f"Column '{compare_column}' does not exist in file {filename}")])

            col_data = df[compare_column]

            if pd.api.types.is_numeric_dtype(col_data):
                comparison_data[filename] = {
                    "rows": len(df),
                    "sum": col_data.sum(),
                    "mean": col_data.mean(),
                    "max": col_data.max(),
                    "min": col_data.min(),
                }
            else:
                comparison_data[filename] = {
                    "rows": len(df),
                    "unique_values": col_data.nunique(),
                    "mode": col_data.mode()[0] if len(col_data.mode()) > 0 else "N/A",
                }

        comparison_df = pd.DataFrame(comparison_data).T

        result = f"""
Multi-file comparison for column '{compare_column}'
{'=' * 60}

{comparison_df.to_string()}

Summary:
- compared files: {len(file_list)}
"""

        if "sum" in comparison_df.columns:
            max_file = comparison_df["sum"].idxmax()
            min_file = comparison_df["sum"].idxmin()
            result += f"- highest sum: {max_file} ({comparison_df.loc[max_file, 'sum']:,.2f})\n"
            result += f"- lowest sum: {min_file} ({comparison_df.loc[min_file, 'sum']:,.2f})\n"

        return ToolResponse(content=[TextBlock(type="text", text=result)])
    except Exception as e:
        return ToolResponse(content=[TextBlock(type="text", text=f"Compare files failed: {str(e)}")])
