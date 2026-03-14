from agentscope.agent import ReActAgent
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit

from agents.business_consultant_worker import create_business_consultant_worker
from agents.context import register_streaming_hook
from agents.data_analyst_worker import create_data_analyst_worker
from agents.prompts import ORCHESTRATOR_PROMPT
from agents.report_writer_worker import create_report_writer_worker
from agents.table_finder_worker import create_table_finder_worker
from agents.visualization_worker import create_visualization_worker
from utils.model_factory import create_model, get_formatter


def create_orchestrator() -> ReActAgent:
    toolkit = Toolkit()
    toolkit.register_tool_function(create_table_finder_worker)
    toolkit.register_tool_function(create_data_analyst_worker)
    toolkit.register_tool_function(create_business_consultant_worker)
    toolkit.register_tool_function(create_visualization_worker)
    toolkit.register_tool_function(create_report_writer_worker)

    orchestrator = ReActAgent(
        name="DataAnalysisOrchestrator",
        sys_prompt=ORCHESTRATOR_PROMPT,
        model=create_model(),
        formatter=get_formatter(),
        toolkit=toolkit,
        memory=InMemoryMemory(),
    )
    register_streaming_hook(orchestrator, "Orchestrator")
    return orchestrator
