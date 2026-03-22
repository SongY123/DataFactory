"""
VisualizationWorker - Visualization specialist agent
===================================================

Responsibilities:
    - Analyze data characteristics independently
    - Decide the best visualization approach, including chart type and layout
    - Generate professional data visualizations
    - Produce one or multiple charts when needed
    - Build combined subplot layouts when appropriate

Toolkit:
    - `execute_python_code` for chart generation and persistence

Core idea:
    - The orchestrator provides analysis results
    - VisualizationWorker interprets the data on its own
    - VisualizationWorker chooses the most suitable visualization strategy
"""

import os
from typing import Optional
from utils.model_factory import create_model, get_formatter
from agentscope.agent import ReActAgent
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg, TextBlock
from agentscope.tool import Toolkit, ToolResponse

from agents.python_execution import execute_python_code
# Import prompt
from agents.prompts import VISUALIZATION_PROMPT
# Import context and event bus
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
    Create the visualization worker with full decision-making autonomy.

    Args:
        analysis_data: Output from `DataAnalystWorker`. This is required.
        custom_requirements: Optional user requirements, such as focusing on trends
            or requesting comparison charts.

    Returns:
        ToolResponse: Chart paths and visualization notes.
    """

    # Get the event bus from the shared context.
    event_bus = get_event_bus()

    # Publish the start event.
    if event_bus:
        await event_bus.publish(await create_agent_start_event(
            "Visualization",
            task_description=custom_requirements or "Generate data visualization charts"
        ))

    try:
        toolkit = Toolkit()

        # Register the Python execution tool.
        toolkit.register_tool_function(execute_python_code)

        # Ensure input data is present.
        if not analysis_data:
            error_response = ToolResponse(
                content=[TextBlock(
                    type="text",
                    text="❌ Error: missing data analysis results, so charts cannot be generated. VisualizationWorker requires the output from DataAnalystWorker."
                )]
            )
            if event_bus:
                await event_bus.publish(await create_agent_error_event(
                    "Visualization",
                    Exception("Missing data analysis results")
                ))
            return error_response

        # Build the full task description and grant full autonomy.
        requirements_text = (
            f"\n\n💡 User-specific requirements:\n{custom_requirements}"
            if custom_requirements
            else "\n\n💡 No specific user requirements were provided. You have full freedom to decide the best visualization approach."
        )

        full_task = f"""🎨 You are a data visualization specialist with full decision-making autonomy. 🎨

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Data analysis results provided to you:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{analysis_data}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{requirements_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 Your task:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Analyze the data independently -> Design the visualization plan -> Generate professional charts

Please:
1. Carefully analyze the characteristics and patterns in the data.
2. Decide on the best visualization approach independently.
3. Generate charts that clearly communicate the insights.
4. Provide professional explanations for each chart.

Do not ask follow-up questions. Make the best decision directly from the data characteristics.
"""
        # Ensure the output directory exists.
        os.makedirs("output/charts", exist_ok=True)

        worker = ReActAgent(
            name="VisualizationWorker",
            sys_prompt=VISUALIZATION_PROMPT,
            model=create_model(),
            formatter=get_formatter(),
            toolkit=toolkit,
            memory=InMemoryMemory(),
        )

        # Register the streaming output hook.
        register_streaming_hook(worker, "Visualization")

        print(f"\n📊 [VisualizationWorker] Starting visualization design...")
        result = await worker(Msg("user", full_task, "user"))

        content = extract_agent_result_text(result)

        print(f"✅ [VisualizationWorker] Chart generation completed\n")
        if event_bus:
            await event_bus.publish(await create_agent_finish_event(
                "Visualization",
                result=content
            ))

        from agentscope.message import TextBlock
        return ToolResponse(content=[TextBlock(type="text", text=content)])
    except Exception as e:
        # Publish the error event.
        if event_bus:
            await event_bus.publish(await create_agent_error_event(
                "Visualization",
                e
            ))
        raise
