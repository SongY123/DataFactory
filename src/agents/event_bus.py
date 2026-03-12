"""
事件总线 (Event Bus) - 用于聚合多个 Agent 的流式输出
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Set
from datetime import datetime
import uuid


class EventType(Enum):
    """事件类型枚举"""
    STREAM_CHUNK = "stream_chunk"       # 流式输出的文本块
    AGENT_START = "agent_start"         # Agent 开始执行
    AGENT_FINISH = "agent_finish"       # Agent 执行完成
    AGENT_ERROR = "agent_error"         # Agent 执行错误
    TOOL_CALL = "tool_call"             # 工具调用
    SYSTEM_MESSAGE = "system_message"   # 系统消息


@dataclass
class StreamEvent:
    """
    流式事件数据结构
    
    Attributes:
        event_id: 事件唯一ID
        event_type: 事件类型
        agent_name: 产生事件的 Agent 名称
        timestamp: 事件产生时间戳
        data: 事件数据（可以是文本块、错误信息等）
        metadata: 额外的元数据
        is_final: 是否是该 Agent 的最后一个事件
    """
    event_type: EventType
    agent_name: str
    data: Any
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_final: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，便于 JSON 序列化"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "agent_name": self.agent_name,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "metadata": self.metadata,
            "is_final": self.is_final,
        }


class EventBus:
    """
    异步事件总线
    
    核心特性：
        1. 基于 asyncio.Queue 实现，无需外部依赖
        2. 支持多生产者（多个 Agent 可同时发送事件）
        3. 支持多消费者（可以有多个订阅者）
        4. 自动处理背压（队列满时会阻塞）
        5. 支持过滤器（可以只订阅特定类型的事件）
    """
    
    def __init__(self, maxsize: int = 0):
        """
        初始化事件总线
        
        Args:
            maxsize: 队列最大长度，0 表示无限制
                    设置合理的大小可以避免内存溢出
        """
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._subscribers: Set[str] = set()
        self._closed = False
        
    async def publish(self, event: StreamEvent) -> None:
        """
        发布事件到总线
        
        这是一个异步操作，如果队列满了会阻塞等待
        类似于 Go 的 channel <- 数据工作台
        
        Args:
            event: 要发布的事件
            
        Raises:
            RuntimeError: 如果总线已关闭
        """
        if self._closed:
            raise RuntimeError("EventBus is closed")
            
        await self._queue.put(event)
        
    def publish_nowait(self, event: StreamEvent) -> None:
        """
        非阻塞方式发布事件
        
        如果队列满了会立即抛出异常而不是等待
        
        Args:
            event: 要发布的事件
            
        Raises:
            asyncio.QueueFull: 如果队列已满
            RuntimeError: 如果总线已关闭
        """
        if self._closed:
            raise RuntimeError("EventBus is closed")
            
        self._queue.put_nowait(event)
        
    async def subscribe(
        self,
        filter_agent: Optional[str] = None,
        filter_types: Optional[List[EventType]] = None,
        enable_aggregation: bool = True,
    ) -> AsyncIterator[StreamEvent]:
        """
        订阅事件流
        
        返回一个异步迭代器，可以用 async for 来消费事件
        类似于 Go 的 for event := range channel
        
        Args:
            filter_agent: 只接收特定 Agent 的事件
            filter_types: 只接收特定类型的事件
            enable_aggregation: 是否启用智能聚合（默认 True）
                              聚合逻辑：
                              1. 使用数组存储同一 Agent 的多次模型调用结果
                              2. 如果当前 chunk 以数组最后一个元素为前缀 → 替换最后一个元素（同一次模型调用的增量更新）
                              3. 如果不是前缀 → 添加新元素到数组（新的模型调用）
                              4. 输出时把数组所有元素拼接
                              5. **Agent 切换时自动 flush**：检测到 agent 切换时清空历史记录，避免跨对话轮次聚合
            
        Yields:
            StreamEvent: 事件对象
            
        Example:
            async for event in bus.subscribe(filter_agent="DataAnalyst"):
                print(f"收到事件: {event.数据工作台}")
        """
        subscriber_id = str(uuid.uuid4())
        self._subscribers.add(subscriber_id)
        
        # 聚合状态：按 agent_name 存储多次模型调用的结果数组
        # 数组中每个元素代表一次完整的模型调用结果
        chunks_history: Dict[str, List[str]] = {}
        
        # 记录上一个事件的 agent，用于检测 agent 切换
        last_agent: Optional[str] = None
        
        try:
            while not self._closed or not self._queue.empty():
                try:
                    # 使用超时避免永久阻塞
                    event = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=0.1
                    )
                    
                    # 应用过滤器
                    if filter_agent and event.agent_name != filter_agent:
                        continue
                        
                    if filter_types and event.event_type not in filter_types:
                        continue
                    
                    # 检测 agent 切换，清空历史记录（flush）
                    if last_agent is not None and event.agent_name != last_agent:
                        chunks_history.clear()
                        # 可选：添加调试日志
                        # print(f"[EventBus] Agent 切换: {last_agent} -> {event.agent_name}, 清空历史记录")
                    
                    # 更新 last_agent
                    last_agent = event.agent_name
                    
                    # 智能聚合逻辑（仅对 stream_chunk 生效）
                    if enable_aggregation and event.event_type == EventType.STREAM_CHUNK:
                        event = self._aggregate_chunk(event, chunks_history)
                        
                    yield event
                    
                except asyncio.TimeoutError:
                    # 超时后继续检查是否关闭
                    continue
                    
        finally:
            self._subscribers.discard(subscriber_id)
    
    def _aggregate_chunk(
        self, 
        event: StreamEvent, 
        chunks_history: Dict[str, List[str]]
    ) -> StreamEvent:
        """
        智能聚合 stream_chunk 事件
        
        聚合策略：
        1. 使用数组存储每次模型调用的结果（单次调用内 chunk 已累加）
        2. 如果当前内容以数组最后一个元素为前缀：
           → 替换数组最后一个元素（同一次模型调用的增量更新）
        3. 如果不是前缀：
           → 添加新元素到数组（新的模型调用）
        4. 最终输出：数组所有元素拼接
        
        Args:
            event: 当前事件
            chunks_history: 按 agent_name 存储的多次模型调用结果数组（会被修改）
            
        Returns:
            聚合后的事件
        """
        agent_name = event.agent_name
        
        # 提取当前 chunk 的文本内容
        current_content = self._extract_content(event.data)
        
        # 如果是新 Agent，初始化数组
        if agent_name not in chunks_history:
            chunks_history[agent_name] = [current_content]
            return event
        
        # 获取该 Agent 的历史数组
        history = chunks_history[agent_name]
        
        # 获取数组最后一个元素（最近一次模型调用的内容）
        last_chunk = history[-1]
        
        # 判断是否是前缀关系
        if current_content.startswith(last_chunk):
            # 情况1: 同一次模型调用的增量更新，替换数组最后一个元素
            history[-1] = current_content
            
            # 计算聚合后的完整内容（所有模型调用的拼接）
            aggregated_content = "".join(history)
            
            # 创建聚合事件
            aggregated_event = StreamEvent(
                event_type=event.event_type,
                agent_name=event.agent_name,
                data=self._build_data(event.data, aggregated_content),
                event_id=event.event_id,
                timestamp=event.timestamp,
                metadata={
                    **event.metadata,
                    "aggregated": True,
                    "aggregation_type": "incremental_update",
                    "chunks_count": len(history),
                },
                is_final=event.is_final,
            )
            
            return aggregated_event
        else:
            # 情况2: 新的模型调用，添加到数组
            history.append(current_content)
            

            aggregated_content = "".join(history)

            # 创建聚合事件
            aggregated_event = StreamEvent(
                event_type=event.event_type,
                agent_name=event.agent_name,
                data=self._build_data(event.data, aggregated_content),
                event_id=event.event_id,
                timestamp=event.timestamp,
                metadata={
                    **event.metadata,
                    "aggregated": True,
                    "aggregation_type": "new_model_call",
                    "chunks_count": len(history),
                },
                is_final=event.is_final,
            )
            
            return aggregated_event
    
    def _extract_content(self, data: Any) -> str:
        """
        从 event.data 中提取文本内容
        
        支持两种格式：
        1. dict 格式: {"result": "文本内容"}
        2. str 格式: "文本内容"
        
        Args:
            data: event.data
            
        Returns:
            提取的文本内容
        """
        if isinstance(data, dict):
            return data.get("result", "")
        elif isinstance(data, str):
            return data
        else:
            return str(data)
    
    def _build_data(self, original_data: Any, content: str) -> Any:
        """
        构建聚合后的 data 对象，保持原始格式
        
        Args:
            original_data: 原始的 event.data
            content: 聚合后的文本内容
            
        Returns:
            保持原始格式的 data 对象
        """
        if isinstance(original_data, dict):
            return {**original_data, "result": content}
        else:
            return content
            
    async def close(self) -> None:
        """
        关闭事件总线
        
        关闭后不能再发布新事件，但已有的事件仍可被消费
        """
        self._closed = True
        
    def is_closed(self) -> bool:
        """检查总线是否已关闭"""
        return self._closed
        
    def qsize(self) -> int:
        """返回当前队列中的事件数量"""
        return self._queue.qsize()
        
    @property
    def subscriber_count(self) -> int:
        """返回当前订阅者数量"""
        return len(self._subscribers)


class EventBusContext:
    """
    事件总线上下文管理器
    
    使用场景：在请求级别创建独立的事件总线
    
    Example:
        async with EventBusContext() as bus:
            # 使用 bus
            await bus.publish(event)
    """
    
    def __init__(self, maxsize: int = 0):
        self.maxsize = maxsize
        self.bus: Optional[EventBus] = None
        
    async def __aenter__(self) -> EventBus:
        self.bus = EventBus(maxsize=self.maxsize)
        return self.bus
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.bus:
            await self.bus.close()


# ============================================================================
# 辅助函数：简化常见操作
# ============================================================================

async def create_stream_event(
    agent_name: str,
    chunk: str,
    is_final: bool = False,
    **metadata
) -> StreamEvent:
    """
    创建流式输出事件的辅助函数
    
    Args:
        agent_name: Agent 名称
        chunk: 文本块
        is_final: 是否是最后一个块
        **metadata: 额外的元数据
        
    Returns:
        StreamEvent 对象
    """
    return StreamEvent(
        event_type=EventType.STREAM_CHUNK,
        agent_name=agent_name,
        data=chunk,
        is_final=is_final,
        metadata=metadata,
    )


async def create_agent_start_event(
    agent_name: str,
    task_description: str = "",
    **metadata
) -> StreamEvent:
    """创建 Agent 开始事件"""
    return StreamEvent(
        event_type=EventType.AGENT_START,
        agent_name=agent_name,
        data={"task": task_description},
        metadata=metadata,
    )


async def create_agent_finish_event(
    agent_name: str,
    result: Any = None,
    **metadata
) -> StreamEvent:
    """创建 Agent 完成事件"""
    return StreamEvent(
        event_type=EventType.AGENT_FINISH,
        agent_name=agent_name,
        data={"result": result},
        is_final=True,
        metadata=metadata,
    )


async def create_agent_error_event(
    agent_name: str,
    error: Exception,
    **metadata
) -> StreamEvent:
    """创建 Agent 错误事件"""
    return StreamEvent(
        event_type=EventType.AGENT_ERROR,
        agent_name=agent_name,
        data={
            "error_type": type(error).__name__,
            "error_message": str(error),
        },
        is_final=True,
        metadata=metadata,
    )


# ============================================================================
# 全局单例（可选）
# ============================================================================

_global_bus: Optional[EventBus] = None


def get_global_event_bus() -> EventBus:
    """
    获取全局事件总线单例
    
    注意：全局单例适用于简单场景，对于多请求并发的场景，
    建议使用请求级别的独立事件总线
    """
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus()
    return _global_bus


def reset_global_event_bus() -> None:
    """重置全局事件总线"""
    global _global_bus
    _global_bus = None
