"""
TableFinder worker agent.

Responsibility: find the data tables in the data workspace that are most relevant
to the user's request.
"""

import os
from utils.model_factory import create_model, get_formatter
from agentscope.agent import ReActAgent
from agentscope.tool import Toolkit, ToolResponse
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg

# Import prompt
from agents.prompts import TABLE_FINDER_PROMPT
# Import data exploration tools
from tools.table_finder_tools import (
    list_data_files,
    inspect_table_structure,
    search_tables_by_keywords
)
# Import context and event bus
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
    Create the table-finder worker.

    Core idea:
        - Accept the user's data-analysis request
        - Find relevant data tables in the data workspace
        - Analyze table structure and content relevance
        - Return recommended table paths with rationale

    Toolkit:
        - `list_data_files`: list all available data files
        - `inspect_table_structure`: inspect table schemas
        - `search_tables_by_keywords`: search tables by keywords

    Args:
        user_request: Description of the user's data-analysis need.

    Returns:
        ToolResponse: Recommended table paths and reasons.
    """

    # Get the event bus from the shared context.
    event_bus = get_event_bus()

    data_directory = get_data_directory()
    # Publish the start event.
    if event_bus:
        await event_bus.publish(await create_agent_start_event(
            "TableFinder",
            task_description=user_request
        ))

    try:
        # Register tools.
        toolkit = Toolkit()
        toolkit.register_tool_function(list_data_files)
        toolkit.register_tool_function(inspect_table_structure)
        toolkit.register_tool_function(search_tables_by_keywords)

        # Ensure the data directory exists.
        os.makedirs(data_directory, exist_ok=True)

        # Build the task description.
        full_task = f"""🔍 You are a data table discovery specialist.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 User request:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{user_request}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 Your task:
Find the data table or tables in `{data_directory}` that are most relevant to this request.

Work requirements:
1. Understand the user's request and extract keywords such as sales, users, orders, or time.
2. Use tools to search for and analyze data tables.
3. Evaluate the relevance of each candidate table.
4. Return the recommended table paths and explain why they were chosen.

Notes:
- There may be multiple relevant tables; prioritize the most relevant ones.
- If several tables are useful, you may recommend more than one.
- If none are sufficiently relevant, explain why and suggest what additional information would help.
- Always explain your reasoning based on file names, column names, or data content.
"""
        # Create the worker.
        worker = ReActAgent(
            name="TableFinderWorker",
            sys_prompt=TABLE_FINDER_PROMPT,
            model=create_model(),
            formatter=get_formatter(),
            toolkit=toolkit,
            memory=InMemoryMemory(),
        )

        # Register the streaming output hook.
        register_streaming_hook(worker, "TableFinder")

        print(f"\n🔍 [TableFinderWorker] Starting table discovery...")
        result = await worker(Msg("user", full_task, "user"))
        content = extract_agent_result_text(result)

        print(f"✅ [TableFinderWorker] Table discovery completed\n")
        if event_bus:
            await event_bus.publish(await create_agent_finish_event(
                "TableFinder",
                result=content
            ))

        from agentscope.message import TextBlock
        return ToolResponse(content=[TextBlock(type="text", text=content)])
    except Exception as e:
        # Publish the error event.
        if event_bus:
            await event_bus.publish(await create_agent_error_event(
                "TableFinder",
                e
            ))
        raise
