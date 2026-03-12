"""
ReportWriterWorker - 报告写作者 Agent
======================================

职责：
    - 整合所有分析结果
    - 撰写结构化的 Markdown 分析报告
    - 在报告中引用可视化图表
    - 保存报告到文件

工具集：
    - write_text_file（创建/覆盖 Markdown 文件）
    - insert_text_file（在指定位置插入内容）
    - view_text_file（查看文件内容）
"""

import os
from typing import Optional
from utils.config_loader import get_config
from utils.model_factory import create_model, get_formatter
from agentscope.agent import ReActAgent
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.tool import (
    Toolkit,
    ToolResponse,
    write_text_file,
    insert_text_file,
    view_text_file,
)

# 导入 prompt
from agents.prompts import REPORT_WRITER_PROMPT
# 导入上下文和事件总线
from agents.context import get_event_bus, register_streaming_hook
from agents.result_utils import extract_agent_result_text
from agents.event_bus import (
    create_agent_start_event,
    create_agent_finish_event,
    create_agent_error_event,
)


async def create_report_writer_worker(
    task_description: str,
    all_results: Optional[str] = None
) -> ToolResponse:
    """
    创建报告写作者 Worker
    
    Args:
        task_description: 任务描述
        all_results: 所有 Worker 的结果汇总
        
    Returns:
        ToolResponse: 包含报告路径
    """
    
    # 从上下文获取事件总线
    event_bus = get_event_bus()
    
    # 发送开始事件
    if event_bus:
        await event_bus.publish(await create_agent_start_event(
            "ReportWriter",
            task_description=task_description
        ))
    
    try:
        toolkit = Toolkit()
        
        # 注册文本文件编辑工具
        toolkit.register_tool_function(write_text_file)
        toolkit.register_tool_function(insert_text_file)
        toolkit.register_tool_function(view_text_file)
        
        # 构建完整的任务描述
        full_task = task_description
        if all_results:
            full_task = f"""整合以下所有分析结果，撰写一份完整的数据分析报告：

{all_results}

报告要求：
{task_description}

重要提示：
- 使用 write_text_file 工具创建 Markdown 报告
- 报告保存路径：output/reports/data_analysis_report.md
- 如果有图表路径，使用 Markdown 语法引用图片：![图表说明](../charts/图表文件名.png)
"""
        # 确保输出目录存在
        os.makedirs("output/reports", exist_ok=True)
        
        worker = ReActAgent(
        name="ReportWriterWorker",
        sys_prompt=REPORT_WRITER_PROMPT,
        model=create_model(),
        formatter=get_formatter(),
        toolkit=toolkit,            memory=InMemoryMemory(),
        )
        
        # ====== 注册流式输出钩子 ======
        register_streaming_hook(worker, "ReportWriter")
        
        print(f"\n📝 [ReportWriterWorker] 开始撰写报告：{task_description}")
        result = await worker(Msg("user", full_task, "user"))
        
        content = extract_agent_result_text(result)
        
        print(f"✅ [ReportWriterWorker] 报告完成\n")
        if event_bus:
            await event_bus.publish(await create_agent_finish_event(
                "ReportWriter",
                result=content
            ))

        from agentscope.message import TextBlock
        return ToolResponse(content=[TextBlock(type="text", text=content)])
    except Exception as e:
        # 发送错误事件
        if event_bus:
            await event_bus.publish(await create_agent_error_event(
                "ReportWriter",
                e
            ))
        raise
