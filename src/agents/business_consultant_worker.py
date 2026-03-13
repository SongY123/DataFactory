"""
BusinessConsultantWorker - 业务顾问 Agent
==========================================

职责：
    - 解读数据分析结果
    - 提供业务洞察
    - 识别机会和风险
    - 给出业务建议

工具集：
    - 无（纯 LLM 推理，不需要工具）
"""

from typing import Optional
from utils.model_factory import create_model, get_formatter
from agentscope.agent import ReActAgent
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.tool import Toolkit, ToolResponse

# 导入 prompt
from agents.prompts import BUSINESS_CONSULTANT_PROMPT
# 导入上下文和事件总线
from agents.context import get_event_bus, register_streaming_hook
from agents.result_utils import extract_agent_result_text
from agents.event_bus import (
    create_agent_start_event,
    create_agent_finish_event,
    create_agent_error_event,
)


async def create_business_consultant_worker(
    task_description: str,
    analysis_data: Optional[str] = None,
    display_task_description: Optional[str] = None,
    agent_name: str = "BusinessConsultant",
) -> ToolResponse:
    """
    创建业务顾问 Worker
    
    Args:
        task_description: 任务描述
        analysis_data: 数据分析结果（可选，如果有则基于此提供洞察）
        
    Returns:
        ToolResponse: 包含业务洞察
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
        # 业务顾问不需要任何工具，完全依靠 LLM 的推理能力
        toolkit = Toolkit()
        
        # 构建完整的任务描述
        full_task = task_description
        if analysis_data:
            full_task = f"""基于以下数据分析结果，提供业务洞察：

数据分析结果：
{analysis_data}

任务要求：
{task_description}
"""
        
        worker = ReActAgent(
        name=f"{agent_name}Worker",
        sys_prompt=BUSINESS_CONSULTANT_PROMPT,
        model=create_model(),
        formatter=get_formatter(),
        toolkit=toolkit,            memory=InMemoryMemory(),
        )
        
        # ====== 注册流式输出钩子 ======
        register_streaming_hook(worker, agent_name)
        
        print(f"\n💼 [BusinessConsultantWorker] 开始解读数据：{task_description}")
        result = await worker(Msg("user", full_task, "user"))
        
        content = extract_agent_result_text(result)
        
        print(f"✅ [BusinessConsultantWorker] 洞察完成\n")
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
