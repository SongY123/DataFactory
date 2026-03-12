"""
VisualizationWorker - 可视化专家 Agent
========================================

职责：
    - 自主分析数据特征
    - 自主决定最佳可视化方案（图表类型、数量、布局等）
    - 生成专业的数据可视化图表
    - 可以生成单个或多个图表
    - 可以创建组合图表（subplot）

工具集：
    - execute_python_code（生成和保存图表）

核心理念：
    - Orchestrator 只负责提供数据分析结果
    - VisualizationWorker 自主分析数据特征
    - 自主决定最佳可视化方案
"""

import os
from typing import Optional
from utils.config_loader import get_config
from utils.model_factory import create_model, get_formatter
from agentscope.agent import ReActAgent
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg, TextBlock
from agentscope.tool import Toolkit, ToolResponse, execute_python_code

# 导入 prompt
from agents.prompts import VISUALIZATION_PROMPT
# 导入上下文和事件总线
from agents.context import get_event_bus, register_streaming_hook
from agents.result_utils import extract_agent_result_text
from agents.event_bus import (
    create_agent_start_event,
    create_agent_finish_event,
    create_agent_error_event,
)


async def create_visualization_worker(
        analysis_data: str,
        custom_requirements: Optional[str] = None
) -> ToolResponse:
    """
    创建可视化专家 Worker（优化版 - 具有完全自主决策能力）
    
    Args:
        analysis_data: DataAnalystWorker 的输出数据（必需，第一参数）
        custom_requirements: 用户的特定要求（可选，如"重点展示趋势"、"需要对比图"）
        
    Returns:
        ToolResponse: 包含图表路径和可视化说明
    """

    # 从上下文获取事件总线
    event_bus = get_event_bus()

    # 发送开始事件
    if event_bus:
        await event_bus.publish(await create_agent_start_event(
            "Visualization",
            task_description=custom_requirements or "生成数据可视化图表"
        ))

    try:
        toolkit = Toolkit()

        # 注册 Python 代码执行工具
        toolkit.register_tool_function(execute_python_code)

        # 确保有数据传入
        if not analysis_data:
            error_response = ToolResponse(
                content=[TextBlock(
                    type="text",
                    text="❌ 错误：缺少数据分析结果，无法生成图表。VisualizationWorker 需要接收 DataAnalystWorker 的输出数据。"
                )]
            )
            if event_bus:
                await event_bus.publish(await create_agent_error_event(
                    "Visualization",
                    Exception("缺少数据分析结果")
                ))
            return error_response

        # 构建完整的任务描述，赋予完全自主权
        requirements_text = f"\n\n💡 用户的特定要求：\n{custom_requirements}" if custom_requirements else "\n\n💡 用户没有特定要求，你拥有完全自主决策权"

        full_task = f"""🎨 【你是一位具有完全自主决策权的数据可视化专家】🎨

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 提供给你的数据分析结果：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{analysis_data}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{requirements_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 你的任务：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【自主分析数据】→【自主设计可视化方案】→【生成专业图表】

请你：
1. 仔细分析上述数据的特征和模式
2. 自主决定最佳的可视化方案
3. 生成能够清晰传达数据洞察的图表
4. 为每个图表提供专业的说明

注意：不要询问任何问题，直接根据数据特征做出最佳决策。
"""
        # 确保输出目录存在
        os.makedirs("output/charts", exist_ok=True)

        worker = ReActAgent(
            name="VisualizationWorker",
            sys_prompt=VISUALIZATION_PROMPT,
            model=create_model(),
            formatter=get_formatter(),
            toolkit=toolkit,
            memory=InMemoryMemory(),
        )

        # ====== 注册流式输出钩子 ======
        register_streaming_hook(worker, "Visualization")

        print(f"\n📊 [VisualizationWorker] 开始自主设计可视化方案...")
        result = await worker(Msg("user", full_task, "user"))

        content = extract_agent_result_text(result)

        print(f"✅ [VisualizationWorker] 图表生成完成\n")
        if event_bus:
            await event_bus.publish(await create_agent_finish_event(
                "Visualization",
                result=content
            ))

        from agentscope.message import TextBlock
        return ToolResponse(content=[TextBlock(type="text", text=content)])
    except Exception as e:
        # 发送错误事件
        if event_bus:
            await event_bus.publish(await create_agent_error_event(
                "Visualization",
                e
            ))
        raise
