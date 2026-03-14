"""
ReportWriterWorker - Report writer agent
========================================

Responsibilities:
    - Consolidate all analysis results
    - Write a structured Markdown analysis report
    - Reference visualization charts in the report
    - Save the final report to a file

Toolkit:
    - `write_text_file` to create or overwrite Markdown files
    - `insert_text_file` to insert content at a specific position
    - `view_text_file` to inspect file contents
"""

import os
from typing import Optional
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

# Import prompt
from agents.prompts import REPORT_WRITER_PROMPT
# Import context and event bus
from agents.context import get_event_bus, register_streaming_hook
from agents.result_utils import extract_agent_result_text
from agents.event_bus import (
    create_agent_start_event,
    create_agent_finish_event,
    create_agent_error_event,
)


async def create_report_writer_worker(
    task_description: str,
    all_results: Optional[str] = None,
    display_task_description: Optional[str] = None,
    agent_name: str = "ReportWriter",
) -> ToolResponse:
    """
    Create the report writer worker.

    Args:
        task_description: Task description.
        all_results: Aggregated results from all workers.

    Returns:
        ToolResponse: The generated report content or report path.
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
        toolkit = Toolkit()
        
        # Register text-editing tools.
        toolkit.register_tool_function(write_text_file)
        toolkit.register_tool_function(insert_text_file)
        toolkit.register_tool_function(view_text_file)
        
        # Build the full task description.
        full_task = task_description
        if all_results:
            full_task = f"""Consolidate the following analysis results and write a complete data analysis report:

{all_results}

Report requirements:
{task_description}

Important notes:
- Use the `write_text_file` tool to create the Markdown report
- Save the report to: output/reports/data_analysis_report.md
- If chart paths are available, embed them with Markdown syntax such as ![Chart description](../charts/chart_file_name.png)
"""
        # Ensure the output directory exists.
        os.makedirs("output/reports", exist_ok=True)
        
        worker = ReActAgent(
        name=f"{agent_name}Worker",
        sys_prompt=REPORT_WRITER_PROMPT,
        model=create_model(),
        formatter=get_formatter(),
        toolkit=toolkit,            memory=InMemoryMemory(),
        )
        
        # Register the streaming output hook.
        register_streaming_hook(worker, agent_name)
        
        print(f"\n📝 [ReportWriterWorker] Starting report writing: {task_description}")
        result = await worker(Msg("user", full_task, "user"))
        
        content = extract_agent_result_text(result).strip()
        report_path = os.path.join("output", "reports", "data_analysis_report.md")
        if not content and os.path.exists(report_path):
            try:
                with open(report_path, "r", encoding="utf-8") as report_file:
                    content = report_file.read().strip()
            except OSError:
                pass

        if not content and all_results:
            content = "# Data Analysis Report" + chr(10) + chr(10) + all_results.strip()

        print("[ReportWriterWorker] report completed")
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
