import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Set


class EventType(Enum):
    STREAM_CHUNK = "stream_chunk"
    AGENT_START = "agent_start"
    AGENT_FINISH = "agent_finish"
    AGENT_ERROR = "agent_error"


@dataclass
class StreamEvent:
    event_type: EventType
    agent_name: str
    data: Any
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_final: bool = False

    def to_dict(self) -> Dict[str, Any]:
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
    def __init__(self, maxsize: int = 0):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._subscribers: Set[str] = set()
        self._closed = False

    async def publish(self, event: StreamEvent) -> None:
        if self._closed:
            raise RuntimeError("EventBus is closed")
        await self._queue.put(event)

    async def subscribe(
        self,
        filter_agent: Optional[str] = None,
        filter_types: Optional[List[EventType]] = None,
        enable_aggregation: bool = True,
    ) -> AsyncIterator[StreamEvent]:
        subscriber_id = str(uuid.uuid4())
        self._subscribers.add(subscriber_id)

        chunks_history: Dict[str, List[str]] = {}
        last_agent: Optional[str] = None

        try:
            while not self._closed or not self._queue.empty():
                try:
                    event = await asyncio.wait_for(self._queue.get(), timeout=0.1)

                    if filter_agent and event.agent_name != filter_agent:
                        continue
                    if filter_types and event.event_type not in filter_types:
                        continue

                    if last_agent is not None and event.agent_name != last_agent:
                        chunks_history.clear()
                    last_agent = event.agent_name

                    if enable_aggregation and event.event_type == EventType.STREAM_CHUNK:
                        event = self._aggregate_chunk(event, chunks_history)

                    yield event
                except asyncio.TimeoutError:
                    continue
        finally:
            self._subscribers.discard(subscriber_id)

    def _aggregate_chunk(self, event: StreamEvent, chunks_history: Dict[str, List[str]]) -> StreamEvent:
        agent_name = event.agent_name
        current_content = self._extract_content(event.data)

        if agent_name not in chunks_history:
            chunks_history[agent_name] = [current_content]
            return event

        history = chunks_history[agent_name]
        last_chunk = history[-1]

        if current_content.startswith(last_chunk):
            history[-1] = current_content
            aggregation_type = "incremental_update"
        else:
            history.append(current_content)
            aggregation_type = "new_model_call"

        aggregated_content = "".join(history)
        return StreamEvent(
            event_type=event.event_type,
            agent_name=event.agent_name,
            data=self._build_data(event.data, aggregated_content),
            event_id=event.event_id,
            timestamp=event.timestamp,
            metadata={
                **event.metadata,
                "aggregated": True,
                "aggregation_type": aggregation_type,
                "chunks_count": len(history),
            },
            is_final=event.is_final,
        )

    @staticmethod
    def _extract_content(data: Any) -> str:
        if isinstance(data, dict):
            return str(data.get("result", ""))
        if isinstance(data, str):
            return data
        return str(data)

    @staticmethod
    def _build_data(original_data: Any, content: str) -> Any:
        if isinstance(original_data, dict):
            return {**original_data, "result": content}
        return content

    async def close(self) -> None:
        self._closed = True


async def create_stream_event(
    agent_name: str,
    chunk: Any,
    is_final: bool = False,
    **metadata,
) -> StreamEvent:
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
    **metadata,
) -> StreamEvent:
    return StreamEvent(
        event_type=EventType.AGENT_START,
        agent_name=agent_name,
        data={"task": task_description},
        metadata=metadata,
    )


async def create_agent_finish_event(
    agent_name: str,
    result: Any = None,
    **metadata,
) -> StreamEvent:
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
    **metadata,
) -> StreamEvent:
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
