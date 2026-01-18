"""
Extended debate rounds with RLM context management.

Enables debates with 50+ rounds by using hierarchical context compression
to maintain coherent context without exceeding token limits.

Usage:
    from aragora.debate.extended_rounds import ExtendedDebateConfig, RLMContextManager

    config = ExtendedDebateConfig(
        max_rounds=100,
        compression_threshold=10000,
        context_window_rounds=10,
    )

    context_manager = RLMContextManager(config)
    compressed = await context_manager.prepare_round_context(debate_context, round_num)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from aragora.debate.context import DebateContext
    from aragora.rlm.types import RLMContext

logger = logging.getLogger(__name__)


class ContextStrategy(Enum):
    """Strategy for managing context in extended debates."""

    FULL = "full"  # Use full context (small debates)
    SLIDING_WINDOW = "sliding_window"  # Keep last N rounds full, compress rest
    HIERARCHICAL = "hierarchical"  # Multi-level compression
    ADAPTIVE = "adaptive"  # Auto-select based on context size


@dataclass
class ExtendedDebateConfig:
    """Configuration for extended debate rounds."""

    # Round limits
    max_rounds: int = 100
    """Maximum number of rounds allowed."""

    soft_limit_rounds: int = 50
    """Rounds at which to start aggressive compression."""

    # Context management
    compression_threshold: int = 10000
    """Token count at which to trigger compression."""

    context_window_rounds: int = 10
    """Number of recent rounds to keep at full detail."""

    min_context_ratio: float = 0.3
    """Minimum context ratio to maintain (compressed/original)."""

    # RLM settings
    enable_rlm: bool = True
    """Whether to use RLM for context compression."""

    rlm_max_levels: int = 4
    """Maximum RLM abstraction levels."""

    rlm_cache_enabled: bool = True
    """Whether to cache compressed contexts."""

    # Performance
    compression_timeout: float = 30.0
    """Timeout for compression operations."""

    parallel_compression: bool = True
    """Whether to compress rounds in parallel."""

    # Strategy
    context_strategy: ContextStrategy = ContextStrategy.ADAPTIVE
    """Strategy for managing context."""


@dataclass
class RoundSummary:
    """Summary of a debate round for context compression."""

    round_num: int
    proposals: dict[str, str]
    critiques: list[str]
    key_points: list[str]
    consensus_progress: float
    token_count: int
    compressed_at: Optional[float] = None


@dataclass
class ExtendedContextState:
    """State for extended debate context management."""

    # Round tracking
    current_round: int = 0
    total_tokens: int = 0
    compressed_tokens: int = 0

    # Summaries
    round_summaries: dict[int, RoundSummary] = field(default_factory=dict)
    compressed_history: str = ""

    # RLM context
    rlm_context: Optional[Any] = None

    # Statistics
    compressions_performed: int = 0
    tokens_saved: int = 0
    compression_time_total: float = 0.0


class RLMContextManager:
    """
    Manages context for extended debates using RLM compression.

    Automatically compresses older rounds while maintaining recent
    rounds at full detail for coherent debate continuation.
    """

    def __init__(self, config: Optional[ExtendedDebateConfig] = None):
        """Initialize the context manager."""
        self.config = config or ExtendedDebateConfig()
        self._state = ExtendedContextState()
        self._compressor = None
        self._lock = asyncio.Lock()

    async def _get_compressor(self):
        """Lazy-load the RLM compressor."""
        if self._compressor is None and self.config.enable_rlm:
            try:
                from aragora.rlm import HierarchicalCompressor

                self._compressor = HierarchicalCompressor()
            except ImportError:
                logger.warning("RLM module not available, falling back to simple compression")
        return self._compressor

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (approx 4 chars per token)."""
        return len(text) // 4

    def _select_strategy(self, total_tokens: int, round_num: int) -> ContextStrategy:
        """Select the best context strategy based on current state."""
        if self.config.context_strategy != ContextStrategy.ADAPTIVE:
            return self.config.context_strategy

        # Small context: use full
        if total_tokens < self.config.compression_threshold:
            return ContextStrategy.FULL

        # Medium context: use sliding window
        if round_num < self.config.soft_limit_rounds:
            return ContextStrategy.SLIDING_WINDOW

        # Large context: use hierarchical
        return ContextStrategy.HIERARCHICAL

    async def prepare_round_context(
        self,
        debate_context: "DebateContext",
        round_num: int,
    ) -> str:
        """
        Prepare context for a debate round.

        Compresses older rounds if needed while maintaining recent
        rounds at full detail.

        Args:
            debate_context: The current debate context
            round_num: The round number being prepared

        Returns:
            Formatted context string for the round
        """
        async with self._lock:
            self._state.current_round = round_num

            # Calculate current context size
            full_context = self._build_full_context(debate_context)
            total_tokens = self._estimate_tokens(full_context)
            self._state.total_tokens = total_tokens

            # Select strategy
            strategy = self._select_strategy(total_tokens, round_num)

            if strategy == ContextStrategy.FULL:
                return full_context

            if strategy == ContextStrategy.SLIDING_WINDOW:
                return await self._apply_sliding_window(debate_context, round_num)

            # Hierarchical compression
            return await self._apply_hierarchical_compression(debate_context, round_num)

    def _build_full_context(self, debate_context: "DebateContext") -> str:
        """Build full context from all messages."""
        parts = []

        # Add task
        if debate_context.env and debate_context.env.task:
            parts.append(f"## Task\n{debate_context.env.task}")

        # Add historical context if available
        if debate_context.historical_context_cache:
            parts.append(f"## Historical Context\n{debate_context.historical_context_cache[:2000]}")

        # Add all messages
        for msg in debate_context.context_messages:
            role = getattr(msg, "role", "unknown")
            agent = getattr(msg, "agent", "unknown")
            content = getattr(msg, "content", "")
            round_num = getattr(msg, "round", 0)
            parts.append(f"### Round {round_num} - {agent} ({role})\n{content}")

        return "\n\n".join(parts)

    async def _apply_sliding_window(
        self,
        debate_context: "DebateContext",
        round_num: int,
    ) -> str:
        """Apply sliding window compression."""
        parts = []
        window_start = max(0, round_num - self.config.context_window_rounds)

        # Add task
        if debate_context.env and debate_context.env.task:
            parts.append(f"## Task\n{debate_context.env.task}")

        # Add compressed summary of older rounds
        if window_start > 0 and self._state.compressed_history:
            parts.append(f"## Previous Rounds Summary (1-{window_start})\n{self._state.compressed_history}")

        # Add recent rounds at full detail
        for msg in debate_context.context_messages:
            msg_round = getattr(msg, "round", 0)
            if msg_round >= window_start:
                role = getattr(msg, "role", "unknown")
                agent = getattr(msg, "agent", "unknown")
                content = getattr(msg, "content", "")
                parts.append(f"### Round {msg_round} - {agent} ({role})\n{content}")

        return "\n\n".join(parts)

    async def _apply_hierarchical_compression(
        self,
        debate_context: "DebateContext",
        round_num: int,
    ) -> str:
        """Apply hierarchical RLM compression."""
        compressor = await self._get_compressor()

        if not compressor:
            # Fallback to sliding window if RLM not available
            return await self._apply_sliding_window(debate_context, round_num)

        # Build content to compress (older rounds)
        window_start = max(0, round_num - self.config.context_window_rounds)
        old_content = self._build_old_rounds_content(debate_context, window_start)

        if old_content:
            start_time = time.perf_counter()
            try:
                result = await asyncio.wait_for(
                    compressor.compress(
                        old_content,
                        source_type="debate",
                        max_levels=self.config.rlm_max_levels,
                    ),
                    timeout=self.config.compression_timeout,
                )

                self._state.compressions_performed += 1
                self._state.compression_time_total += time.perf_counter() - start_time

                if result and result.context:
                    # Get summary level for compressed history
                    from aragora.rlm.types import AbstractionLevel

                    self._state.compressed_history = result.context.get_at_level(
                        AbstractionLevel.SUMMARY
                    ) or ""
                    self._state.rlm_context = result.context

                    # Track token savings
                    original_tokens = self._estimate_tokens(old_content)
                    compressed_tokens = self._estimate_tokens(self._state.compressed_history)
                    self._state.tokens_saved += original_tokens - compressed_tokens
                    self._state.compressed_tokens = compressed_tokens

            except asyncio.TimeoutError:
                logger.warning(f"RLM compression timed out for round {round_num}")
            except Exception as e:
                logger.error(f"RLM compression failed: {e}")

        # Build final context
        return await self._apply_sliding_window(debate_context, round_num)

    def _build_old_rounds_content(
        self,
        debate_context: "DebateContext",
        window_start: int,
    ) -> str:
        """Build content from older rounds for compression."""
        parts = []

        for msg in debate_context.context_messages:
            msg_round = getattr(msg, "round", 0)
            if msg_round < window_start:
                role = getattr(msg, "role", "unknown")
                agent = getattr(msg, "agent", "unknown")
                content = getattr(msg, "content", "")
                parts.append(f"Round {msg_round} - {agent} ({role}): {content}")

        return "\n\n".join(parts)

    async def get_drill_down_context(
        self,
        query: str,
        level: str = "DETAILED",
    ) -> str:
        """
        Get detailed context for a specific query.

        Allows agents to drill down into compressed context when needed.

        Args:
            query: The query to search for
            level: Abstraction level (ABSTRACT, SUMMARY, DETAILED, FULL)

        Returns:
            Relevant context at the specified detail level
        """
        if not self._state.rlm_context:
            return ""

        try:
            from aragora.rlm.types import AbstractionLevel

            level_enum = AbstractionLevel[level.upper()]
            return self._state.rlm_context.get_at_level(level_enum) or ""
        except (KeyError, AttributeError):
            return ""

    def get_statistics(self) -> dict[str, Any]:
        """Get compression statistics."""
        return {
            "current_round": self._state.current_round,
            "total_tokens": self._state.total_tokens,
            "compressed_tokens": self._state.compressed_tokens,
            "compressions_performed": self._state.compressions_performed,
            "tokens_saved": self._state.tokens_saved,
            "compression_time_total": self._state.compression_time_total,
            "compression_ratio": (
                self._state.compressed_tokens / self._state.total_tokens
                if self._state.total_tokens > 0
                else 0
            ),
        }

    def reset(self) -> None:
        """Reset the context manager state."""
        self._state = ExtendedContextState()


def create_extended_config(
    max_rounds: int = 100,
    aggressive: bool = False,
) -> ExtendedDebateConfig:
    """
    Create an extended debate configuration.

    Args:
        max_rounds: Maximum rounds to allow
        aggressive: Use aggressive compression settings

    Returns:
        Configuration for extended debates
    """
    if aggressive:
        return ExtendedDebateConfig(
            max_rounds=max_rounds,
            soft_limit_rounds=max_rounds // 2,
            compression_threshold=5000,
            context_window_rounds=5,
            min_context_ratio=0.2,
            rlm_max_levels=5,
            context_strategy=ContextStrategy.HIERARCHICAL,
        )

    return ExtendedDebateConfig(
        max_rounds=max_rounds,
        soft_limit_rounds=50,
        compression_threshold=10000,
        context_window_rounds=10,
        min_context_ratio=0.3,
        rlm_max_levels=4,
        context_strategy=ContextStrategy.ADAPTIVE,
    )
