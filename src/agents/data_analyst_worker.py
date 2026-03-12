"""
DataAnalystWorker - 数据分析师 Agent
====================================

职责：
    - 读取和探索数据文件
    - 执行统计分析
    - 数据筛选和分组分析
    - 数据对比分析
    - 复杂的数据处理（使用 Python 代码）

工具集：
    - 9 个数据分析工具（来自 data_tools.py）
    - execute_python_code（用于复杂数据处理和计算）
"""

from typing import Optional, List
from utils.config_loader import get_config
from utils.model_factory import create_model, get_formatter
from agentscope.agent import ReActAgent
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.tool import Toolkit, ToolResponse, execute_python_code

# 导入 prompt
from agents.prompts import DATA_ANALYST_PROMPT
# 导入数据分析工具
from tools.data_tools import (
    read_excel_file,
    get_column_stats,
    filter_data,
    group_analysis,
    compare_data,
    list_excel_files,
    read_multiple_excel_files,
    merge_excel_files,
    compare_excel_files,
)
# 导入上下文和事件总线
from agents.context import get_event_bus, register_streaming_hook
from agents.result_utils import extract_agent_result_text
from agents.event_bus import (
    create_agent_start_event,
    create_agent_finish_event,
    create_agent_error_event,
)


async def create_data_analyst_worker(
    task_description: str,
    file_paths: Optional[List[str]] = None,
    display_task_description: Optional[str] = None,
    agent_name: str = "DataAnalyst",
) -> ToolResponse:
    """
    数据分析工具,执行智能而专业的数据分析任务
    
    这是一个具有 ReAct 能力的数据分析 Agent，负责执行数据探索、统计分析和数据计算。
    它的职责是产出结构化的分析结果，而非完整报告（报告由 ReportWriter 负责）。
    
    核心能力：
    1. 数据探索和质量检查
       - 读取 Excel/CSV 文件（支持 .xlsx, .xls, .csv）
       - 查看数据结构（行列数、数据类型、缺失值）
       - 数据预览和样本检查
    
    2. 统计分析和计算
       - 描述性统计（均值、中位数、分位数、标准差）
       - 分组聚合（sum, mean, count, max, min）
       - 数据筛选和对比
       - 时间序列分析
    
    3. 复杂数据处理
       - 执行自定义 Python 代码（pandas/numpy）
       - 多文件合并和关联分析
       - 大规模数据处理（保存结果到文件）
    
    工作方式：
    - 自动探索数据结构，选择合适的分析方法
    - 自主决策分析维度（时间、地域、分类等）
    - 如果统计结果较大，保存到文件并返回路径
    - 产出结构化、客观、简洁的分析结果
    
    Args:
        task_description (str): 数据分析任务的详细描述。
            应明确说明：
            - 分析目标（如：趋势分析、对比分析、排名分析）
            - 关注维度（如：时间、地域、类别）
            - 特殊要求（如：仅分析2018年数据）
            
            示例：
            - "分析订单数据的时间趋势，重点关注月度变化"
            - "对比不同州的订单量和金额，找出前5名"
            - "分析产品类别的销售情况，计算每个类别的订单数和平均价格"
            - "找出复购率最高的客户群体"
        
        file_paths (Optional[List[str]]): 待分析的数据文件路径列表。
            - 如果提供，agent 将直接分析这些文件（推荐，避免重复搜索）
            - 如果为 None，agent 会自主搜索相关文件
            
            示例：
            - ["数据工作台/olist_orders_dataset.csv"]
            - ["数据工作台/olist_orders_dataset.csv", "数据工作台/olist_customers_dataset.csv"]
    
    Returns:
        ToolResponse: 包含结构化的数据分析结果，具体包括：
        
        1. **数据概况** (Data Overview)
           - 数据来源：分析了哪些文件
           - 数据规模：总行数、列数
           - 时间范围：数据覆盖的时间段（如果有时间字段）
           - 数据质量：缺失值情况、异常值提示
        
        2. **关键发现** (Key Findings)
           - 3-5个核心数据洞察
           - 每个发现必须基于具体数据，客观陈述
           - 简洁明了，每个发现不超过2行
           - 示例：
             * "SP州订单量最多，占总订单的41.97%（41,746单）"
             * "订单完成率达97.02%，取消率仅0.63%"
             * "2018年订单量较2017年增长80%"
        
        3. **统计数据** (Statistical Results)
           - 核心指标的具体数值, 可供其他agent使用，做进一步分析
           - 如果是分组统计（如前10名、各类别统计），有两种处理方式：
             * 数据量小（<20行）：直接列出关键数值
             * 数据量大（≥20行）：保存为CSV文件，返回文件路径
           - 文件保存路径示例：`results/analysis_group_stats_20260126.csv`
        
        4. **客观分析** (Objective Analysis)
           - 基于数据的客观解读（非主观建议）
           - 数据之间的关联和模式
           - 异常点或特殊情况的说明
           - 注意：这里只做客观分析，不提供业务建议
    """
    
    # 从上下文获取事件总线
    event_bus = get_event_bus()
    
    # 发送开始事件
    if event_bus:
        await event_bus.publish(await create_agent_start_event(
            agent_name,
            task_description=display_task_description or task_description
        ))
    
    try:
        # ====== 参数类型处理 ======
        # 如果 file_paths 是字符串（LLM 可能传入 JSON 字符串），尝试解析
        import json
        if isinstance(file_paths, str):
            try:
                file_paths = json.loads(file_paths)
            except (json.JSONDecodeError, ValueError):
                # 如果不是有效的 JSON，当作单个文件路径处理
                file_paths = [file_paths] if file_paths.strip() else None
        
        # 创建专门的数据分析工具集
        toolkit = Toolkit()
        
        # 注册数据分析工具
        toolkit.register_tool_function(read_excel_file)
        toolkit.register_tool_function(get_column_stats)
        toolkit.register_tool_function(filter_data)
        toolkit.register_tool_function(group_analysis)
        toolkit.register_tool_function(compare_data)
        toolkit.register_tool_function(list_excel_files)
        toolkit.register_tool_function(read_multiple_excel_files)
        toolkit.register_tool_function(merge_excel_files)
        toolkit.register_tool_function(compare_excel_files)
        
        # 注册 Python 代码执行工具（用于复杂数据处理和分析结果生成）
        toolkit.register_tool_function(execute_python_code)
        
        # 创建数据分析师 Agent
        worker = ReActAgent(
        name=f"{agent_name}Worker",
        sys_prompt=DATA_ANALYST_PROMPT,
        model=create_model(),
        formatter=get_formatter(),
        toolkit=toolkit,
        memory=InMemoryMemory(),
    )
        
        # ====== 注册流式输出钩子 ======
        register_streaming_hook(worker, agent_name)
        
        # 构建任务消息
        enhanced_task = task_description
        if file_paths:
            file_list = "\n".join(f"  - {path}" for path in file_paths)
            enhanced_task = f"""{task_description}

你需要分析以下数据文件：
{file_list}

请直接使用这些文件路径进行分析，不需要搜索其他文件。"""
    
        # 执行任务
        print(f"\n🔬 [DataAnalystWorker] 开始分析任务：{task_description}")
        if file_paths:
            print(f"   📂 指定文件：{', '.join(file_paths)}")
        result = await worker(Msg("user", enhanced_task, "user"))
        
        # 提取文本内容
        content = extract_agent_result_text(result)
        
        print(f"✅ [DataAnalystWorker] 任务完成\n")
        
        # 返回ToolResponse，content必须是TextBlock列表
        if event_bus:
            await event_bus.publish(await create_agent_finish_event(
                agent_name,
                result=content
            ))

        from agentscope.message import TextBlock
        return ToolResponse(content=[TextBlock(type="text", text=content)])
    except Exception as e:
        # 发送错误事件
        if event_bus:
            await event_bus.publish(await create_agent_error_event(
                agent_name,
                e
            ))
        raise
