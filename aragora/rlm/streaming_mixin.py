"""
RLM Streaming Mixin.

Provides streaming query and compression methods for AragoraRLM.
Extracted from bridge.py for better modularity.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Protocol
from collections.abc import AsyncIterator, Callable

from .types import (
    AbstractionLevel,
    RLMContext,
    RLMResult,
    RLMStreamEvent,
    RLMStreamEventType,
)

if TYPE_CHECKING:
    from .compressor import HierarchicalCompressor

logger = logging.getLogger(__name__)


class _RLMHostProtocol(Protocol):
    """
    Protocol defining the interface expected from the host class.

    This Protocol enables proper type checking for the RLMStreamingMixin
    by defining the methods and attributes it expects from AragoraRLM.
    """

    _compressor: HierarchicalCompressor

    async def query(
        self,
        query: str,
        context: RLMContext,
        strategy: str = "auto",
    ) -> RLMResult:
        """Query using RLM over hierarchical context."""
        ...

    async def _query_iteration(
        self,
        query: str,
        context: RLMContext,
        strategy: str,
        iteration: int,
        feedback: str | None,
    ) -> RLMResult:
        """Execute a single query iteration with optional feedback."""
        ...

    def _default_feedback(self, result: RLMResult, original_query: str) -> str:
        """Generate default feedback for incomplete answers."""
        ...


class RLMStreamingMixin:
    """
    Mixin providing streaming capabilities for RLM queries.

    Adds streaming versions of query, refinement, and compression
    methods that yield RLMStreamEvent instances for real-time
    progress tracking.

    This mixin expects to be combined with a class implementing
    the _RLMHostProtocol interface (i.e., AragoraRLM).
    """

    async def query_stream(
        self: _RLMHostProtocol,
        query: str,
        context: RLMContext,
        strategy: str = "auto",
    ) -> AsyncIterator[RLMStreamEvent]:
        """
        Stream RLM query execution with progress events.

        Yields RLMStreamEvents as the query progresses through
        different abstraction levels and nodes.

        Args:
            query: The query to answer
            context: Pre-compressed hierarchical context
            strategy: Decomposition strategy

        Yields:
            RLMStreamEvent instances representing query progress
        """
        time.perf_counter()

        # Emit query start
        yield RLMStreamEvent(
            event_type=RLMStreamEventType.QUERY_START,
            query=query,
        )

        try:
            # Track levels we're entering
            for level in [
                AbstractionLevel.ABSTRACT,
                AbstractionLevel.SUMMARY,
                AbstractionLevel.DETAILED,
            ]:
                if level in context.levels:
                    yield RLMStreamEvent(
                        event_type=RLMStreamEventType.LEVEL_ENTERED,
                        query=query,
                        level=level,
                        tokens_processed=context.total_tokens_at_level(level),
                    )

                    # Emit node examination events
                    for node in context.levels[level][:5]:  # Limit to first 5 nodes
                        yield RLMStreamEvent(
                            event_type=RLMStreamEventType.NODE_EXAMINED,
                            query=query,
                            level=level,
                            node_id=node.id,
                            content=(
                                node.content[:200] + "..."
                                if len(node.content) > 200
                                else node.content
                            ),
                        )

            # Execute the actual query (provided by base class)
            result = await self.query(query, context, strategy)

            # Emit completion
            yield RLMStreamEvent(
                event_type=RLMStreamEventType.QUERY_COMPLETE,
                query=query,
                tokens_processed=result.tokens_processed,
                sub_calls_made=result.sub_calls_made,
                confidence=result.confidence,
                result=result,
            )

        except (RuntimeError, ValueError, TimeoutError, ConnectionError, OSError) as e:
            logger.error(f"Streaming query failed: {e}")
            yield RLMStreamEvent(
                event_type=RLMStreamEventType.ERROR,
                query=query,
                error=str(e),
            )
            raise

    async def query_with_refinement_stream(
        self: _RLMHostProtocol,
        query: str,
        context: RLMContext,
        strategy: str = "auto",
        max_iterations: int = 3,
        feedback_generator: Callable[[RLMResult], str] | None = None,
    ) -> AsyncIterator[RLMStreamEvent]:
        """
        Stream iterative refinement with events for each iteration.

        Implements Prime Intellect's iterative refinement protocol with
        streaming visibility into each iteration's progress.

        Args:
            query: The query to answer
            context: Pre-compressed hierarchical context
            strategy: Decomposition strategy
            max_iterations: Maximum refinement iterations
            feedback_generator: Optional function to generate feedback

        Yields:
            RLMStreamEvent instances for refinement progress
        """
        refinement_history: list[str] = []
        iteration = 0
        result: RLMResult | None = None

        # Emit query start
        yield RLMStreamEvent(
            event_type=RLMStreamEventType.QUERY_START,
            query=query,
        )

        while iteration < max_iterations:
            # Emit iteration start
            yield RLMStreamEvent(
                event_type=RLMStreamEventType.ITERATION_START,
                query=query,
                iteration=iteration,
            )

            # Generate feedback for iterations > 0
            feedback: str | None = None
            if iteration > 0 and result:
                if feedback_generator:
                    feedback = feedback_generator(result)
                else:
                    feedback = self._default_feedback(result, query)

                yield RLMStreamEvent(
                    event_type=RLMStreamEventType.FEEDBACK_GENERATED,
                    query=query,
                    iteration=iteration,
                    content=feedback,
                )

            # Execute query iteration
            try:
                result = await self._query_iteration(
                    query=query,
                    context=context,
                    strategy=strategy,
                    iteration=iteration,
                    feedback=feedback,
                )
            except (RuntimeError, ValueError, TimeoutError, ConnectionError, OSError) as e:
                logger.error(f"Iteration {iteration} failed: {e}")
                yield RLMStreamEvent(
                    event_type=RLMStreamEventType.ERROR,
                    query=query,
                    iteration=iteration,
                    error=str(e),
                )
                raise

            # Track iteration
            result.iteration = iteration
            if iteration > 0:
                refinement_history.append(result.answer)

            # Emit partial answer (for non-final iterations)
            if not result.ready and iteration < max_iterations - 1:
                yield RLMStreamEvent(
                    event_type=RLMStreamEventType.PARTIAL_ANSWER,
                    query=query,
                    iteration=iteration,
                    partial_answer=result.answer,
                    confidence=result.confidence,
                )

            # Emit confidence update
            yield RLMStreamEvent(
                event_type=RLMStreamEventType.CONFIDENCE_UPDATE,
                query=query,
                iteration=iteration,
                confidence=result.confidence,
            )

            # Emit iteration complete
            yield RLMStreamEvent(
                event_type=RLMStreamEventType.ITERATION_COMPLETE,
                query=query,
                iteration=iteration,
                tokens_processed=result.tokens_processed,
                sub_calls_made=result.sub_calls_made,
                confidence=result.confidence,
                result=result,
            )

            logger.info(
                f"RLM refinement iteration={iteration} ready={result.ready} "
                f"confidence={result.confidence:.2f}"
            )

            # Check if answer is ready
            if result.ready:
                break

            iteration += 1

        # Finalize result
        if result:
            result.refinement_history = refinement_history
            result.iteration = iteration

        final_result = result or RLMResult(
            answer="[Failed to generate answer after max iterations]",
            ready=True,
            iteration=max_iterations,
            refinement_history=refinement_history,
        )

        # Emit final answer
        yield RLMStreamEvent(
            event_type=RLMStreamEventType.FINAL_ANSWER,
            query=query,
            iteration=iteration,
            confidence=final_result.confidence,
            result=final_result,
        )

        # Emit query complete
        yield RLMStreamEvent(
            event_type=RLMStreamEventType.QUERY_COMPLETE,
            query=query,
            iteration=iteration,
            tokens_processed=final_result.tokens_processed,
            sub_calls_made=final_result.sub_calls_made,
            confidence=final_result.confidence,
            result=final_result,
        )

    async def compress_stream(
        self: _RLMHostProtocol,
        content: str,
        source_type: str = "text",
    ) -> AsyncIterator[RLMStreamEvent]:
        """
        Stream compression with progress events.

        Args:
            content: Content to compress
            source_type: Type of content (text, debate, code)

        Yields:
            RLMStreamEvent instances for compression progress
        """
        start_time = time.perf_counter()

        yield RLMStreamEvent(
            event_type=RLMStreamEventType.QUERY_START,
            query=f"compress:{source_type}",
            content=f"Compressing {len(content)} characters",
        )

        try:
            compression = await self._compressor.compress(content, source_type)

            # Emit events for each level created
            for level in [
                AbstractionLevel.DETAILED,
                AbstractionLevel.SUMMARY,
                AbstractionLevel.ABSTRACT,
            ]:
                if level in compression.context.levels:
                    nodes = compression.context.levels[level]
                    yield RLMStreamEvent(
                        event_type=RLMStreamEventType.LEVEL_ENTERED,
                        query=f"compress:{source_type}",
                        level=level,
                        tokens_processed=sum(n.token_count for n in nodes),
                        content=f"Created {len(nodes)} nodes at {level.name}",
                    )

            elapsed = time.perf_counter() - start_time

            yield RLMStreamEvent(
                event_type=RLMStreamEventType.QUERY_COMPLETE,
                query=f"compress:{source_type}",
                tokens_processed=compression.original_tokens,
                sub_calls_made=compression.sub_calls_made,
                confidence=compression.estimated_fidelity,
                content=f"Compression complete in {elapsed:.2f}s",
            )

        except (RuntimeError, ValueError, TimeoutError, ConnectionError, OSError) as e:
            logger.error(f"Streaming compression failed: {e}")
            yield RLMStreamEvent(
                event_type=RLMStreamEventType.ERROR,
                query=f"compress:{source_type}",
                error=str(e),
            )
            raise


__all__ = ["RLMStreamingMixin"]
