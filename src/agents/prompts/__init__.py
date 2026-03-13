"""Central registry for agent system prompts."""

from agents.prompts.business_consultant_prompt import BUSINESS_CONSULTANT_PROMPT
from agents.prompts.data_analyst_prompt import DATA_ANALYST_PROMPT
from agents.prompts.orchestrator_prompt import ORCHESTRATOR_PROMPT
from agents.prompts.report_writer_prompt import REPORT_WRITER_PROMPT
from agents.prompts.table_finder_prompt import TABLE_FINDER_PROMPT
from agents.prompts.visualization_prompt import VISUALIZATION_PROMPT

__all__ = [
    "ORCHESTRATOR_PROMPT",
    "BUSINESS_CONSULTANT_PROMPT",
    "DATA_ANALYST_PROMPT",
    "TABLE_FINDER_PROMPT",
    "REPORT_WRITER_PROMPT",
    "VISUALIZATION_PROMPT",
]
