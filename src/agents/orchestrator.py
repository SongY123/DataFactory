"""
DataAnalysisOrchestrator - 数据分析编排者
===========================================

职责：
    - 接收用户的数据分析需求
    - 将任务分解为子任务
    - 动态创建合适的 Worker Agents
    - 协调 Workers 的执行
    - 汇总结果并返回给用户
"""

from typing import Optional
from utils.config_loader import get_config
from utils.model_factory import create_model, get_formatter
from agentscope.agent import ReActAgent
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit

# 导入 prompt
from agents.prompts import ORCHESTRATOR_PROMPT
# 导入所有 Worker 创建函数
from agents.data_analyst_worker import create_data_analyst_worker
from agents.business_consultant_worker import create_business_consultant_worker
from agents.visualization_worker import create_visualization_worker
from agents.report_writer_worker import create_report_writer_worker
from agents.table_finder_worker import create_table_finder_worker
# 导入事件总线
from agents.event_bus import EventBus
# 导入上下文和流式输出钩子
from agents.context import get_event_bus, register_streaming_hook


def create_orchestrator(event_bus: Optional[EventBus] = None) -> ReActAgent:
    """
    创建数据分析编排者 (Orchestrator)
    
    Args:
        event_bus: 可选的事件总线，用于聚合多个 Worker 的流式输出
    
    Returns:
        ReActAgent: Orchestrator Agent
    """
    # 注册所有 Worker 创建函数
    toolkit = Toolkit()
    toolkit.register_tool_function(create_table_finder_worker)
    toolkit.register_tool_function(create_data_analyst_worker)
    toolkit.register_tool_function(create_business_consultant_worker)
    toolkit.register_tool_function(create_visualization_worker)
    toolkit.register_tool_function(create_report_writer_worker)
    
    orchestrator = ReActAgent(
        name="DataAnalysisOrchestrator",
        sys_prompt=ORCHESTRATOR_PROMPT,
        model=create_model(),  # 使用模型工厂创建模型
        formatter=get_formatter(),  # 使用对应的 formatter
        toolkit=toolkit,
        memory=InMemoryMemory(),
    )
    # ====== 注册流式输出钩子 ======
    register_streaming_hook(orchestrator, "Orchestrator")
    
    return orchestrator
