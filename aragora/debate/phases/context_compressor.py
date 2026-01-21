"""
Context compression module for debate rounds.

Handles compressing debate context using RLM cognitive load limiter
for long debates. This module is extracted from debate_rounds.py
for better modularity.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

if TYPE_CHECKING:
    from aragora.core import Critique
    from aragora.debate.context import DebateContext

logger = logging.getLogger(__name__)

# Timeout for async callbacks
DEFAULT_CALLBACK_TIMEOUT = 30.0

# Minimum messages before compression is triggered
MIN_MESSAGES_FOR_COMPRESSION = 10


async def _with_callback_timeout(
    coro,
    timeout: float = DEFAULT_CALLBACK_TIMEOUT,
    default=None,
):
    """Execute coroutine with timeout, returning default on timeout."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Callback timed out after {timeout}s, using default")
        return default


class ContextCompressor:
    """
    Compresses debate context using RLM cognitive load limiter.

    Called at the start of each round after the threshold to keep
    context manageable for long debates. Old messages are summarized
    while recent messages are kept at full detail.

    Usage:
        compressor = ContextCompressor(
            compress_callback=arena._compress_context,
            hooks=arena.hooks,
            notify_spectator=arena._notify_spectator,
            heartbeat_callback=arena._emit_heartbeat,
        )
        await compressor.compress_context(ctx, round_num, partial_critiques)
    """

    def __init__(
        self,
        compress_callback: Optional[Callable] = None,
        hooks: Optional[dict] = None,
        notify_spectator: Optional[Callable] = None,
        heartbeat_callback: Optional[Callable] = None,
        timeout: float = DEFAULT_CALLBACK_TIMEOUT,
        min_messages: int = MIN_MESSAGES_FOR_COMPRESSION,
    ):
        """
        Initialize the context compressor.

        Args:
            compress_callback: Async callback for compressing context.
                              Signature: (messages, critiques) -> (compressed_msgs, compressed_crits)
            hooks: Dictionary of event hooks
            notify_spectator: Callback for spectator notifications
            heartbeat_callback: Callback for emitting heartbeats
            timeout: Timeout in seconds for compression operations
            min_messages: Minimum messages before compression is triggered
        """
        self._compress_context = compress_callback
        self.hooks = hooks or {}
        self._notify_spectator = notify_spectator
        self._emit_heartbeat = heartbeat_callback
        self._timeout = timeout
        self._min_messages = min_messages

    async def compress_context(
        self,
        ctx: "DebateContext",
        round_num: int,
        partial_critiques: List["Critique"],
    ) -> Tuple[int, int]:
        """
        Compress debate context using RLM cognitive load limiter.

        Args:
            ctx: The DebateContext with messages to compress
            round_num: Current round number
            partial_critiques: List of critiques to potentially compress

        Returns:
            Tuple of (original_count, compressed_count) for messages.
            Returns (0, 0) if compression was skipped.
        """
        if not self._compress_context:
            return (0, 0)

        # Only compress if there are enough messages to warrant it
        if len(ctx.context_messages) < self._min_messages:
            return (0, 0)

        original_count = len(ctx.context_messages)

        try:
            # Emit heartbeat to signal compression is happening
            if self._emit_heartbeat:
                self._emit_heartbeat(f"round_{round_num}", "compressing_context")

            # Call Arena's compress_debate_messages method
            compressed_msgs, compressed_crits = await _with_callback_timeout(
                self._compress_context(
                    messages=ctx.context_messages,
                    critiques=partial_critiques,
                ),
                timeout=self._timeout,
                default=(ctx.context_messages, partial_critiques),
            )

            # Update context with compressed messages
            if compressed_msgs is not ctx.context_messages:
                ctx.context_messages = list(compressed_msgs)
                logger.info(
                    f"[rlm] Compressed context: {original_count} → {len(ctx.context_messages)} messages"
                )

                # Notify spectator about compression
                if self._notify_spectator:
                    self._notify_spectator(
                        "context_compression",
                        details=f"Compressed {original_count} → {len(ctx.context_messages)} messages",
                        agent="system",
                    )

                # Emit hook for WebSocket clients
                if "on_context_compression" in self.hooks:
                    self.hooks["on_context_compression"](
                        round_num=round_num,
                        original_count=original_count,
                        compressed_count=len(ctx.context_messages),
                    )

                return (original_count, len(ctx.context_messages))

            return (original_count, original_count)

        except Exception as e:
            logger.warning(f"[rlm] Context compression failed: {e}")
            # Continue without compression - don't break the debate
            return (original_count, original_count)

    def should_compress(self, ctx: "DebateContext") -> bool:
        """
        Check if context should be compressed.

        Args:
            ctx: The DebateContext to check

        Returns:
            True if compression should be attempted
        """
        return (
            self._compress_context is not None and len(ctx.context_messages) >= self._min_messages
        )
