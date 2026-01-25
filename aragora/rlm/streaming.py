"""
Streaming RLM queries for real-time context drill-down.

Enables progressive loading of context at different abstraction levels,
allowing agents to start with summaries and drill down as needed.

Usage:
    from aragora.rlm.streaming import StreamingRLMQuery, stream_context

    async for chunk in stream_context(rlm_context, query="consensus points"):
        process(chunk)

    # Or with the query object
    query = StreamingRLMQuery(rlm_context)
    async for level, content in query.drill_down("agent positions"):
        print(f"Level {level}: {content[:100]}...")
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Optional,
    TypeVar,
)

if TYPE_CHECKING:
    from aragora.rlm.types import RLMContext

logger = logging.getLogger(__name__)

T = TypeVar("T")


class StreamMode(Enum):
    """Mode for streaming context."""

    TOP_DOWN = "top_down"  # Start abstract, drill down
    BOTTOM_UP = "bottom_up"  # Start detailed, roll up
    TARGETED = "targeted"  # Jump to specific level
    PROGRESSIVE = "progressive"  # Load progressively with delays


@dataclass
class StreamChunk:
    """A chunk of streamed context."""

    level: str
    content: str
    token_count: int
    is_final: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class StreamConfig:
    """Configuration for streaming queries."""

    mode: StreamMode = StreamMode.TOP_DOWN
    """Streaming mode."""

    chunk_size: int = 500
    """Approximate tokens per chunk."""

    delay_between_levels: float = 0.0
    """Delay between abstraction levels (seconds)."""

    include_metadata: bool = True
    """Whether to include metadata in chunks."""

    timeout: float = 30.0
    """Timeout for the entire stream."""

    levels: Optional[list[str]] = None
    """Specific levels to include (None = all)."""


class StreamingRLMQuery:
    """
    Streaming query interface for RLM context.

    Provides progressive access to hierarchical context,
    allowing callers to process content as it becomes available.
    """

    def __init__(
        self,
        rlm_context: "RLMContext",
        config: Optional[StreamConfig] = None,
    ):
        """
        Initialize the streaming query.

        Args:
            rlm_context: The RLM context to query
            config: Streaming configuration
        """
        self.context = rlm_context
        self.config = config or StreamConfig()
        self._started = False
        self._completed = False

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        return len(text) // 4

    def _get_level_order(self) -> list[str]:
        """Get the order of levels based on mode."""
        all_levels = ["ABSTRACT", "SUMMARY", "DETAILED", "FULL"]

        if self.config.levels:
            return [level for level in all_levels if level in self.config.levels]

        if self.config.mode == StreamMode.BOTTOM_UP:
            return list(reversed(all_levels))

        return all_levels

    async def _get_level_content(self, level: str) -> Optional[str]:
        """Get content at a specific level."""
        try:
            from aragora.rlm.types import AbstractionLevel

            level_enum = AbstractionLevel[level]
            return self.context.get_at_level(level_enum)
        except (KeyError, AttributeError) as e:
            logger.debug(f"Could not get level {level}: {e}")
            return None

    async def stream_all(self) -> AsyncGenerator[StreamChunk, None]:
        """
        Stream all levels of context with timeout enforcement.

        Yields:
            StreamChunk objects for each level

        Raises:
            asyncio.TimeoutError: If streaming exceeds configured timeout
        """
        self._started = True
        levels = self._get_level_order()
        start_time = time.time()

        for i, level in enumerate(levels):
            # Production hardening: Check timeout before each level
            elapsed = time.time() - start_time
            if elapsed > self.config.timeout:
                logger.warning(
                    f"Streaming timeout after {elapsed:.1f}s "
                    f"(limit: {self.config.timeout}s), yielded {i}/{len(levels)} levels"
                )
                # Yield a final partial chunk to indicate timeout
                yield StreamChunk(
                    level=level,
                    content="[TIMEOUT - partial results returned]",
                    token_count=0,
                    is_final=True,
                    metadata={
                        "level_index": i,
                        "total_levels": len(levels),
                        "timeout": True,
                        "elapsed_seconds": elapsed,
                    },
                )
                self._completed = False
                return

            is_final = i == len(levels) - 1

            content = await self._get_level_content(level)
            if content:
                yield StreamChunk(
                    level=level,
                    content=content,
                    token_count=self._estimate_tokens(content),
                    is_final=is_final,
                    metadata={"level_index": i, "total_levels": len(levels)},
                )

            if not is_final and self.config.delay_between_levels > 0:
                await asyncio.sleep(self.config.delay_between_levels)

        self._completed = True

    async def drill_down(
        self,
        query: Optional[str] = None,
        start_level: str = "ABSTRACT",
    ) -> AsyncGenerator[tuple[str, str], None]:
        """
        Drill down through context levels with timeout enforcement.

        Args:
            query: Optional query to filter content
            start_level: Level to start from

        Yields:
            Tuples of (level, content)
        """
        levels = self._get_level_order()
        start_time = time.time()

        try:
            start_idx = levels.index(start_level)
        except ValueError:
            start_idx = 0

        for i, level in enumerate(levels[start_idx:]):
            # Production hardening: Check timeout before each level
            elapsed = time.time() - start_time
            if elapsed > self.config.timeout:
                logger.warning(
                    f"Drill-down timeout after {elapsed:.1f}s (limit: {self.config.timeout}s)"
                )
                return

            content = await self._get_level_content(level)
            if content:
                # Filter by query if provided
                if query:
                    content = self._filter_by_query(content, query)

                if content:
                    yield level, content

            if self.config.delay_between_levels > 0:
                await asyncio.sleep(self.config.delay_between_levels)

    def _filter_by_query(self, content: str, query: str) -> str:
        """Filter content by query relevance."""
        # Simple keyword-based filtering
        query_lower = query.lower()
        paragraphs = content.split("\n\n")

        relevant = [p for p in paragraphs if any(word in p.lower() for word in query_lower.split())]

        return "\n\n".join(relevant) if relevant else content

    async def get_progressive(
        self,
        callback: Callable[[StreamChunk], None],
    ) -> None:
        """
        Get context progressively with callback.

        Args:
            callback: Function to call with each chunk
        """
        async for chunk in self.stream_all():
            callback(chunk)

    async def search(
        self,
        query: str,
        level: str = "DETAILED",
    ) -> list[str]:
        """
        Search for relevant content.

        Args:
            query: Search query
            level: Level to search in

        Returns:
            List of relevant content snippets
        """
        content = await self._get_level_content(level)
        if not content:
            return []

        # Split into paragraphs and score by relevance
        query_words = set(query.lower().split())
        paragraphs = content.split("\n\n")

        scored = []
        for p in paragraphs:
            p_words = set(p.lower().split())
            overlap = len(query_words & p_words)
            if overlap > 0:
                scored.append((overlap, p))

        # Sort by relevance
        scored.sort(reverse=True, key=lambda x: x[0])

        return [p for _, p in scored[:10]]


async def stream_context(
    rlm_context: "RLMContext",
    query: Optional[str] = None,
    mode: StreamMode = StreamMode.TOP_DOWN,
    chunk_callback: Optional[Callable[[StreamChunk], None]] = None,
) -> AsyncGenerator[StreamChunk, None]:
    """
    Stream context from an RLM context object.

    Args:
        rlm_context: The RLM context to stream
        query: Optional query to filter content
        mode: Streaming mode
        chunk_callback: Optional callback for each chunk

    Yields:
        StreamChunk objects
    """
    config = StreamConfig(mode=mode)
    streamer = StreamingRLMQuery(rlm_context, config)

    async for chunk in streamer.stream_all():
        if chunk_callback:
            chunk_callback(chunk)
        yield chunk


async def quick_summary(rlm_context: "RLMContext") -> str:
    """
    Get a quick summary from RLM context.

    Args:
        rlm_context: The RLM context

    Returns:
        Summary text or empty string
    """
    try:
        from aragora.rlm.types import AbstractionLevel

        return rlm_context.get_at_level(AbstractionLevel.SUMMARY) or ""
    except (ImportError, AttributeError):
        return ""


async def progressive_load(
    rlm_context: "RLMContext",
    on_level: Callable[[str, str], None],
    delay: float = 0.5,
) -> None:
    """
    Progressively load context levels with delays.

    Useful for UI that wants to show content as it becomes available.

    Args:
        rlm_context: The RLM context
        on_level: Callback for each level (level_name, content)
        delay: Delay between levels in seconds
    """
    config = StreamConfig(
        mode=StreamMode.TOP_DOWN,
        delay_between_levels=delay,
    )
    streamer = StreamingRLMQuery(rlm_context, config)

    async for level, content in streamer.drill_down():
        on_level(level, content)


class StreamingContextIterator:
    """
    Async iterator for streaming context with timeout support.

    Usage:
        async with StreamingContextIterator(context, timeout=30) as stream:
            async for chunk in stream:
                process(chunk)
    """

    def __init__(
        self,
        rlm_context: "RLMContext",
        config: Optional[StreamConfig] = None,
        timeout: float = 30.0,
    ):
        self.query = StreamingRLMQuery(rlm_context, config)
        self.timeout = timeout
        self._iterator: Optional[AsyncGenerator[StreamChunk, None]] = None

    async def __aenter__(self) -> "StreamingContextIterator":
        self._iterator = self.query.stream_all()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._iterator:
            await self._iterator.aclose()

    def __aiter__(self) -> "StreamingContextIterator":
        return self

    async def __anext__(self) -> StreamChunk:
        if not self._iterator:
            raise StopAsyncIteration

        try:
            return await asyncio.wait_for(
                self._iterator.__anext__(),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            raise StopAsyncIteration
        except StopAsyncIteration:
            raise
