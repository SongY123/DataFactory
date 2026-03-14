"""
BusinessConsultantWorker - Business consultant agent
===================================================

Responsibilities:
    - Interpret data analysis results
    - Provide business insights
    - Identify opportunities and risks
    - Recommend business actions

Toolkit:
    - None (pure LLM reasoning, no tools required)
"""

from typing import Optional
from utils.model_factory import create_model, get_formatter
from agentscope.agent import ReActAgent
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.tool import Toolkit, ToolResponse

# Import prompt
from agents.prompts import BUSINESS_CONSULTANT_PROMPT
# Import context and event bus
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
    Create the business consultant worker.

    Args:
        task_description: Task description.
        analysis_data: Data analysis results. If provided, insights should be based on it.

    Returns:
        ToolResponse: Business insights.
    """
    
    # Get the event bus from the shared context.
    event_bus = get_event_bus()
    
    # Publish the start event.
    if event_bus:
        await event_bus.publish(await create_agent_start_event(
            agent_name,
            task_description=display_task_description or task_description
        ))
    
    try:
        # This worker relies entirely on LLM reasoning and does not need tools.
        toolkit = Toolkit()
        
        # Build the full task description.
        full_task = task_description
        if analysis_data:
            full_task = f"""Provide business insights based on the following data analysis results:

Data analysis results:
{analysis_data}

Task requirements:
{task_description}
"""
        
        worker = ReActAgent(
        name=f"{agent_name}Worker",
        sys_prompt=BUSINESS_CONSULTANT_PROMPT,
        model=create_model(),
        formatter=get_formatter(),
        toolkit=toolkit,            memory=InMemoryMemory(),
        )
        
        # Register the streaming output hook.
        register_streaming_hook(worker, agent_name)
        
        print(f"\n💼 [BusinessConsultantWorker] Starting insight generation: {task_description}")
        result = await worker(Msg("user", full_task, "user"))
        
        content = extract_agent_result_text(result)
        
        print(f"✅ [BusinessConsultantWorker] Insight generation completed\n")
        if event_bus:
            await event_bus.publish(await create_agent_finish_event(
                agent_name,
                result=content
            ))

        from agentscope.message import TextBlock
        return ToolResponse(content=[TextBlock(type="text", text=content)])
    except Exception as e:
        # Publish the error event.
        if event_bus:
            await event_bus.publish(await create_agent_error_event(
                agent_name,
                e
            ))
        raise
