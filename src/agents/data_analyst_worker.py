"""
DataAnalystWorker - Data analyst agent
======================================

Responsibilities:
    - Read and explore data files
    - Run statistical analysis
    - Perform filtering and grouped analysis
    - Compare datasets
    - Handle advanced processing with Python code

Toolkit:
    - Nine data analysis tools from `data_tools.py`
    - `execute_python_code` for advanced processing and calculations
"""

from typing import Optional, List
from utils.model_factory import create_model, get_formatter
from agentscope.agent import ReActAgent
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.tool import Toolkit, ToolResponse

from agents.python_execution import execute_python_code
# Import prompt
from agents.prompts import DATA_ANALYST_PROMPT
# Import data analysis tools
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
# Import context and event bus
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
    Execute intelligent and professional data analysis tasks.

    This ReAct-style agent is responsible for data exploration, statistical
    analysis, and calculations. It produces structured analytical output
    rather than a finished report, which is handled by `ReportWriter`.

    Core capabilities:
    1. Data exploration and quality checks
       - Read Excel and CSV files (`.xlsx`, `.xls`, `.csv`)
       - Inspect structure, data types, and missing values
       - Preview data and sample records

    2. Statistical analysis and calculations
       - Descriptive statistics such as mean, median, quantiles, and standard deviation
       - Grouped aggregations such as sum, mean, count, max, and min
       - Filtering and comparative analysis
       - Time series analysis

    3. Advanced data processing
       - Execute custom Python code with pandas or numpy
       - Merge multiple files and analyze relationships
       - Handle large-scale processing and save results to files

    Working style:
    - Explore the data structure automatically and choose suitable methods
    - Decide analysis dimensions independently, such as time, region, or category
    - Save large statistical outputs to files and return their paths
    - Produce structured, objective, and concise findings

    Args:
        task_description: Detailed description of the analysis task.
            It should clearly specify:
            - The analysis goal, such as trend, comparison, or ranking analysis
            - The focus dimensions, such as time, geography, or category
            - Any special constraints, such as limiting the scope to 2018

            Example:
            - "Analyze order trends over time with a focus on monthly changes"
            - "Compare order volume and revenue across states and identify the top five"
            - "Analyze product-category sales and calculate order counts and average price by category"
            - "Identify the customer segment with the highest repeat-purchase rate"

        file_paths: Optional list of data files to analyze.
            - If provided, the agent should analyze these files directly
            - If `None`, the agent may search for relevant files independently

            Example:
            - ["path/to/orders.csv"]
            - ["path/to/orders.csv", "path/to/customers.csv"]

    Returns:
        ToolResponse: Structured analysis results including:

        1. Data Overview
           - Data sources used in the analysis
           - Data volume such as row count and column count
           - Time range covered by the data when time fields exist
           - Data quality notes such as missing values or anomalies

        2. Key Findings
           - Three to five core insights
           - Each finding must be grounded in concrete data
           - Keep each finding concise, ideally within two lines

        3. Statistical Results
           - Concrete values for core metrics that other agents can reuse
           - Small grouped outputs can be listed directly
           - Large grouped outputs should be saved to CSV and returned by path

        4. Objective Analysis
           - Objective interpretation of the data, not business advice
           - Relationships and patterns across the data
           - Notes on anomalies or special cases
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
        # Normalize argument types.
        # If `file_paths` arrives as a JSON string, parse it.
        import json
        if isinstance(file_paths, str):
            try:
                file_paths = json.loads(file_paths)
            except (json.JSONDecodeError, ValueError):
                # Fall back to treating it as a single file path.
                file_paths = [file_paths] if file_paths.strip() else None
        
        # Create the dedicated data-analysis toolkit.
        toolkit = Toolkit()
        
        # Register data-analysis tools.
        toolkit.register_tool_function(read_excel_file)
        toolkit.register_tool_function(get_column_stats)
        toolkit.register_tool_function(filter_data)
        toolkit.register_tool_function(group_analysis)
        toolkit.register_tool_function(compare_data)
        toolkit.register_tool_function(list_excel_files)
        toolkit.register_tool_function(read_multiple_excel_files)
        toolkit.register_tool_function(merge_excel_files)
        toolkit.register_tool_function(compare_excel_files)
        
        # Register Python execution for advanced processing and result generation.
        toolkit.register_tool_function(execute_python_code)
        
        # Create the data analyst agent.
        worker = ReActAgent(
        name=f"{agent_name}Worker",
        sys_prompt=DATA_ANALYST_PROMPT,
        model=create_model(),
        formatter=get_formatter(),
        toolkit=toolkit,
        memory=InMemoryMemory(),
    )
        
        # Register the streaming output hook.
        register_streaming_hook(worker, agent_name)
        
        # Build the task message.
        enhanced_task = task_description
        if file_paths:
            file_list = "\n".join(f"  - {path}" for path in file_paths)
            enhanced_task = f"""{task_description}

Analyze the following data files:
{file_list}

Use these file paths directly and do not search for other files."""
    
        # Execute the task.
        print(f"\n🔬 [DataAnalystWorker] Starting analysis task: {task_description}")
        if file_paths:
            print(f"   📂 Specified files: {', '.join(file_paths)}")
        result = await worker(Msg("user", enhanced_task, "user"))
        
        # Extract text content from the result.
        content = extract_agent_result_text(result)
        
        print(f"✅ [DataAnalystWorker] Task completed\n")
        
        # Return a ToolResponse whose content is a list of TextBlock objects.
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
