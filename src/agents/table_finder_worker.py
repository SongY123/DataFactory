"""
TableFinder Worker Agent
职责：从 数据工作台 目录中找到与用户请求相关的数据表
"""

import os
from utils.model_factory import create_model, get_formatter
from agentscope.agent import ReActAgent
from agentscope.tool import Toolkit, ToolResponse
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg

# 导入 prompt
from agents.prompts import TABLE_FINDER_PROMPT
# 导入数据探索工具
from tools.table_finder_tools import (
    list_data_files,
    inspect_table_structure,
    search_tables_by_keywords
)
# 导入上下文和事件总线
from agents.context import get_event_bus, register_streaming_hook, get_data_directory
from agents.event_bus import (
    create_agent_start_event,
    create_agent_finish_event,
    create_agent_error_event,
)
from agents.result_utils import extract_agent_result_text


async def create_table_finder_worker(
        user_request: str,
) -> ToolResponse:
    """
    创建表查找专家 Worker
    
    核心理念：
        - 接收用户的数据分析需求
        - 从 数据工作台 目录中智能找到相关的数据表
        - 分析表的结构和内容相关性
        - 返回推荐的表路径和理由
    
    工具集：
        - list_data_files: 列出所有数据文件
        - inspect_table_structure: 检查表结构
        - search_tables_by_keywords: 关键词搜索表
    
    Args:
        user_request: 用户的数据分析需求描述

    Returns:
        ToolResponse: 包含推荐表路径和分析理由
    """

    # 从上下文获取事件总线（关键步骤：通过 contextvars 获取）
    event_bus = get_event_bus()

    data_directory = get_data_directory()
    # 发送开始事件
    if event_bus:
        await event_bus.publish(await create_agent_start_event(
            "TableFinder",
            task_description=user_request
        ))

    try:
        # 注册工具
        toolkit = Toolkit()
        toolkit.register_tool_function(list_data_files)
        toolkit.register_tool_function(inspect_table_structure)
        toolkit.register_tool_function(search_tables_by_keywords)

        # 确保输出目录存在
        os.makedirs(data_directory, exist_ok=True)

        # 构建任务描述
        full_task = f"""🔍 【你是一位数据表查找专家】

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 用户的数据分析需求：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{user_request}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 你的任务：
从 {data_directory} 目录中找到与上述需求最相关的数据表

[警告] 工作要求：
1. 理解用户需求，提取关键词（如：销售、用户、订单、时间等）
2. 使用工具搜索和分析数据表
3. 评估每个表的相关性
4. 返回推荐的表路径和选择理由

💡 注意事项：
- 可能有多个表，请选择最相关的
- 如果有多个相关表，可以推荐多个
- 如果都不太相关，说明原因并建议用户提供更多信息
- 务必说明选择理由（基于文件名、列名、数据内容等）
"""
        # 创建 Worker
        worker = ReActAgent(
            name="TableFinderWorker",
            sys_prompt=TABLE_FINDER_PROMPT,
            model=create_model(),
            formatter=get_formatter(),
            toolkit=toolkit,
            memory=InMemoryMemory(),
        )

        # ====== 注册流式输出钩子 ======
        register_streaming_hook(worker, "TableFinder")

        print(f"\n🔍 [TableFinderWorker] 开始查找相关数据表...")
        result = await worker(Msg("user", full_task, "user"))
        content = extract_agent_result_text(result)

        print(f"✅ [TableFinderWorker] 数据表查找完成\n")
        if event_bus:
            await event_bus.publish(await create_agent_finish_event(
                "TableFinder",
                result=content
            ))

        from agentscope.message import TextBlock
        return ToolResponse(content=[TextBlock(type="text", text=content)])
    except Exception as e:
        # 发送错误事件
        if event_bus:
            await event_bus.publish(await create_agent_error_event(
                "TableFinder",
                e
            ))
        raise
