"""
Bridge layer between official RLM library and Aragora.

This module provides Aragora-specific adapters that work with
the official RLM library (github.com/alexzhang13/rlm).

## TRUE RLM vs COMPRESSION

This module prioritizes TRUE RLM (REPL-based recursive decomposition) over
compression-based approaches. Per the official RLM methodology:

**True RLM** (primary, when `rlm` package is installed):
- Model recursively calls itself via REPL
- Context stored as Python variables (NOT stuffed in prompt)
- Model WRITES CODE to query/grep/partition context
- Model has ACTIVE AGENCY in context management

**Compression** (fallback only, when `rlm` package unavailable):
- Pre-processing hierarchical summarization
- HierarchicalCompressor creates 5-level summaries
- Used ONLY when official RLM is not installed

The official library handles:
- REPL environment isolation (Docker, Modal, local)
- Backend abstraction (OpenAI, Anthropic, vLLM, etc.)
- Trajectory logging and visualization

Aragora adapters handle:
- Debate history formatting for programmatic access
- Knowledge Mound integration
- Aragora agent wrapping

Usage:
    from aragora.rlm.bridge import AragoraRLM, DebateContextAdapter

    # Create RLM with Aragora integration
    rlm = AragoraRLM(backend="openai", model="gpt-4")

    # Process long debate history
    adapter = DebateContextAdapter()
    context = adapter.format_for_rlm(debate_result)

    # Query with RLM (uses TRUE RLM if available, compression as fallback)
    answer = await rlm.query("What consensus was reached?", context)
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Optional

logger = logging.getLogger(__name__)

# Check if official RLM is available
try:
    from rlm import RLM as OfficialRLM
    HAS_OFFICIAL_RLM = True
except ImportError:
    HAS_OFFICIAL_RLM = False
    OfficialRLM = None

from .types import (
    AbstractionLevel,
    RLMConfig,
    RLMContext,
    RLMResult,
    RLMStreamEvent,
    RLMStreamEventType,
)
from .compressor import HierarchicalCompressor


@dataclass
class RLMBackendConfig:
    """Configuration for RLM backend."""

    backend: str = "openai"  # openai, anthropic, openrouter, litellm
    model_name: str = "gpt-4o"
    sub_model_name: str = "gpt-4o-mini"

    # REPL configuration
    repl_type: str = "local"  # local, docker, modal
    repl_timeout: int = 120

    # Official RLM kwargs
    verbose: bool = False
    log_dir: Optional[str] = None


class AragoraRLM:
    """
    Aragora-integrated RLM interface.

    Prioritizes TRUE RLM (REPL-based recursive decomposition) over compression:

    1. TRUE RLM (primary): Model writes code to query context via REPL
       - Used when official `rlm` package is installed
       - Model has agency in deciding how to process context
       - Context stored as variables, not stuffed in prompt

    2. COMPRESSION (fallback only): HierarchicalCompressor summarization
       - Used ONLY when official `rlm` package is NOT installed
       - Pre-processing that creates 5-level summaries

    Also provides:
    - Debate history formatting
    - Knowledge Mound integration
    - Aragora agent wrapping
    """

    def __init__(
        self,
        backend_config: Optional[RLMBackendConfig] = None,
        aragora_config: Optional[RLMConfig] = None,
        agent_registry: Optional[Any] = None,
    ):
        """
        Initialize Aragora RLM.

        Args:
            backend_config: Configuration for RLM backend
            aragora_config: Aragora-specific RLM configuration
            agent_registry: Aragora agent registry for fallback
        """
        self.backend_config = backend_config or RLMBackendConfig()
        self.aragora_config = aragora_config or RLMConfig()
        self.agent_registry = agent_registry

        self._official_rlm: Optional[Any] = None
        # Compressor is ONLY used as fallback when official RLM unavailable
        self._compressor = HierarchicalCompressor(
            config=self.aragora_config,
            agent_call=self._agent_call,
        )

        # Track which approach was used (for debugging/telemetry)
        self._last_query_used_true_rlm: bool = False
        self._last_query_used_compression_fallback: bool = False

        if HAS_OFFICIAL_RLM:
            self._init_official_rlm()
        else:
            logger.warning(
                "[AragoraRLM] Official RLM library not installed. "
                "Will use compression-based FALLBACK for all queries. "
                "For TRUE RLM (REPL-based), install with: pip install rlm"
            )

    def _init_official_rlm(self) -> None:
        """Initialize the official RLM library."""
        try:
            self._official_rlm = OfficialRLM(
                backend=self.backend_config.backend,
                backend_kwargs={
                    "model_name": self.backend_config.model_name,
                },
                repl=self.backend_config.repl_type,
                verbose=self.backend_config.verbose,
            )
            logger.info(
                f"Initialized official RLM with backend={self.backend_config.backend}, "
                f"model={self.backend_config.model_name}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize official RLM: {e}")
            self._official_rlm = None

    def _agent_call(self, prompt: str, model: str) -> str:
        """Call agent for compression/summarization."""
        if self._official_rlm:
            # Use official RLM for simple completions
            try:
                return self._official_rlm.completion(prompt)
            except Exception as e:
                logger.warning(f"Official RLM call failed: {e}")

        # Fallback to Aragora agent registry
        if self.agent_registry:
            try:
                agent = self.agent_registry.get_agent(model)
                return agent.complete(prompt)
            except Exception as e:
                logger.warning(f"Aragora agent call failed: {e}")

        raise RuntimeError("No backend available for agent calls")

    async def query(
        self,
        query: str,
        context: RLMContext,
        strategy: str = "auto",
    ) -> RLMResult:
        """
        Query using RLM over hierarchical context.

        Prioritizes TRUE RLM (REPL-based) when official library is available.
        Falls back to compression-based approach only when unavailable.

        Args:
            query: The query to answer
            context: Pre-compressed hierarchical context
            strategy: Decomposition strategy (auto, peek, grep, partition_map, etc.)

        Returns:
            RLMResult with answer and provenance. Check `used_true_rlm` and
            `used_compression_fallback` fields to see which approach was used.
        """
        # Reset tracking flags
        self._last_query_used_true_rlm = False
        self._last_query_used_compression_fallback = False

        if self._official_rlm:
            # PRIMARY: Use TRUE RLM (REPL-based recursive decomposition)
            logger.info(
                "[AragoraRLM] Using TRUE RLM (REPL-based) for query - "
                "model will write code to examine context"
            )
            result = await self._true_rlm_query(query, context, strategy)
            result.used_true_rlm = self._last_query_used_true_rlm
            result.used_compression_fallback = self._last_query_used_compression_fallback
            return result
        else:
            # FALLBACK: Use compression-based approach
            logger.warning(
                "[AragoraRLM] Using COMPRESSION FALLBACK (official RLM not available) - "
                "context will be pre-summarized rather than model-driven"
            )
            self._last_query_used_compression_fallback = True
            result = await self._compression_fallback(query, context, strategy)
            result.used_true_rlm = False
            result.used_compression_fallback = True
            return result

    async def _true_rlm_query(
        self,
        query: str,
        context: RLMContext,
        strategy: str,
    ) -> RLMResult:
        """
        Query using TRUE RLM (REPL-based recursive decomposition).

        This is the CORRECT approach per official RLM methodology:
        - Model has access to context as Python variables in REPL
        - Model WRITES CODE to query/grep/partition context
        - Model can recursively call itself on subsets
        - Model has ACTIVE AGENCY in deciding how to process context

        Falls back to compression ONLY if TRUE RLM fails.
        """
        # Format context for RLM REPL
        formatted = self._format_context_for_repl(context)

        # Build RLM prompt - note we're giving model REPL access, not stuffing context
        rlm_prompt = f"""You have access to a hierarchical context with multiple abstraction levels.

## Context Structure
{formatted['structure']}

## Available Variables (in your REPL environment)
- CONTEXT_FULL: Full original content ({context.original_tokens} tokens)
- CONTEXT_SUMMARY: Summary level content
- CONTEXT_ABSTRACT: High-level abstract
- get_node(id): Get specific node by ID
- drill_down(id): Get children of a node

## Instructions
Use Python code to examine the context programmatically.
You can grep, filter, partition, and recursively call yourself on subsets.
Start with abstracts, drill down only as needed.

## Task
Answer this question: {query}

Use FINAL(answer) when done.
"""

        try:
            # Run RLM completion (handles REPL internally)
            # The model will write code to examine context
            result = self._official_rlm.completion(
                rlm_prompt,
                # Inject context as REPL variables (NOT in prompt)
                context_vars={
                    "CONTEXT_FULL": context.original_content,
                    "CONTEXT_SUMMARY": context.get_at_level(AbstractionLevel.SUMMARY),
                    "CONTEXT_ABSTRACT": context.get_at_level(AbstractionLevel.ABSTRACT),
                    "get_node": lambda nid: self._get_node_dict(context, nid),
                    "drill_down": lambda nid: self._drill_down_dicts(context, nid),
                },
            )

            # TRUE RLM succeeded
            self._last_query_used_true_rlm = True
            logger.info("[AragoraRLM] TRUE RLM query completed successfully")

            return RLMResult(
                answer=result,
                nodes_examined=[],  # Would need trajectory parsing
                levels_traversed=[],
                citations=[],
                tokens_processed=context.original_tokens,
                sub_calls_made=0,  # Tracked by official RLM
                time_seconds=0.0,
                confidence=0.8,
                uncertainty_sources=[],
            )

        except Exception as e:
            logger.error(f"[AragoraRLM] TRUE RLM query failed: {e}")
            logger.warning(
                "[AragoraRLM] Falling back to COMPRESSION approach "
                "(this is suboptimal - TRUE RLM gives model agency)"
            )
            # Fall back to compression-based approach
            self._last_query_used_compression_fallback = True
            return await self._compression_fallback(query, context, strategy)

    async def _compression_fallback(
        self,
        query: str,
        context: RLMContext,
        strategy: str,
    ) -> RLMResult:
        """
        FALLBACK: Query using compression-based approach.

        This is NOT true RLM - it pre-processes context via HierarchicalCompressor
        rather than giving the model agency to examine context programmatically.

        Used ONLY when:
        1. Official RLM library is not installed
        2. TRUE RLM query fails for some reason

        For true RLM behavior, install: pip install rlm
        """
        logger.debug(
            "[AragoraRLM] Executing COMPRESSION FALLBACK - "
            "context is pre-summarized, model doesn't write code to examine it"
        )

        from .types import DecompositionStrategy, RLMQuery
        from .strategies import get_strategy

        # Parse strategy
        try:
            strategy_enum = DecompositionStrategy(strategy)
        except ValueError:
            strategy_enum = DecompositionStrategy.AUTO

        # Create query
        rlm_query = RLMQuery(
            query=query,
            preferred_strategy=strategy_enum,
        )

        # Get and execute strategy (compression-based)
        strategy_impl = get_strategy(
            strategy_enum,
            self.aragora_config,
            self._agent_call_async,
        )

        result = await strategy_impl.execute(rlm_query, context)

        return RLMResult(
            answer=result.answer,
            nodes_examined=result.nodes_used,
            levels_traversed=[],
            citations=[],
            tokens_processed=result.tokens_examined,
            sub_calls_made=result.sub_calls,
            time_seconds=0.0,
            confidence=result.confidence,
            uncertainty_sources=[],
        )

    def _agent_call_async(self, prompt: str, model: str, context: str) -> str:
        """Async-compatible agent call."""
        full_prompt = f"{prompt}\n\nContext:\n{context}" if context else prompt
        return self._agent_call(full_prompt, model)

    def _format_context_for_repl(self, context: RLMContext) -> dict[str, str]:
        """Format context structure for REPL documentation."""
        structure_parts = []

        for level in [
            AbstractionLevel.ABSTRACT,
            AbstractionLevel.SUMMARY,
            AbstractionLevel.DETAILED,
            AbstractionLevel.FULL,
        ]:
            if level in context.levels:
                nodes = context.levels[level]
                structure_parts.append(
                    f"- {level.name}: {len(nodes)} nodes, "
                    f"~{sum(n.token_count for n in nodes)} tokens"
                )

        return {
            "structure": "\n".join(structure_parts) if structure_parts else "Flat content only",
        }

    def _get_node_dict(self, context: RLMContext, node_id: str) -> Optional[dict]:
        """Get node as dictionary for REPL access."""
        node = context.get_node(node_id)
        if not node:
            return None
        return {
            "id": node.id,
            "level": node.level.name,
            "content": node.content,
            "token_count": node.token_count,
            "key_topics": node.key_topics,
            "child_ids": node.child_ids,
        }

    def _drill_down_dicts(self, context: RLMContext, node_id: str) -> list[dict]:
        """Drill down and return children as dictionaries."""
        children = context.drill_down(node_id)
        return [
            {
                "id": c.id,
                "level": c.level.name,
                "content": c.content[:500] + "..." if len(c.content) > 500 else c.content,
                "token_count": c.token_count,
            }
            for c in children
        ]

    async def compress_and_query(
        self,
        query: str,
        content: str,
        source_type: str = "text",
    ) -> RLMResult:
        """
        Convenience method: compress content and query in one step.

        Args:
            query: The query to answer
            content: Raw content to compress
            source_type: Type of content (text, debate, code)

        Returns:
            RLMResult with answer
        """
        # Compress first
        compression = await self._compressor.compress(content, source_type)

        # Then query
        return await self.query(query, compression.context)

    async def query_with_refinement(
        self,
        query: str,
        context: RLMContext,
        strategy: str = "auto",
        max_iterations: int = 3,
        feedback_generator: Optional[Callable[[RLMResult], str]] = None,
    ) -> RLMResult:
        """
        Query with iterative refinement (Prime Intellect alignment).

        Implements the iterative refinement protocol where the LLM can
        signal incomplete answers via ready=False, triggering additional
        refinement iterations with feedback.

        Args:
            query: The query to answer
            context: Pre-compressed hierarchical context
            strategy: Decomposition strategy (auto, peek, grep, partition_map, etc.)
            max_iterations: Maximum refinement iterations
            feedback_generator: Optional function to generate feedback from
                              incomplete result. If None, uses default feedback.

        Returns:
            RLMResult with final answer and refinement history
        """
        refinement_history: list[str] = []
        iteration = 0
        result: Optional[RLMResult] = None

        while iteration < max_iterations:
            # Generate feedback for iterations > 0
            feedback: Optional[str] = None
            if iteration > 0 and result:
                if feedback_generator:
                    feedback = feedback_generator(result)
                else:
                    feedback = self._default_feedback(result, query)

            # Execute query iteration
            result = await self._query_iteration(
                query=query,
                context=context,
                strategy=strategy,
                iteration=iteration,
                feedback=feedback,
            )

            # Track iteration
            result.iteration = iteration
            if iteration > 0:
                refinement_history.append(result.answer)

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

        return result or RLMResult(
            answer="[Failed to generate answer after max iterations]",
            ready=True,
            iteration=max_iterations,
            refinement_history=refinement_history,
        )

    async def _query_iteration(
        self,
        query: str,
        context: RLMContext,
        strategy: str,
        iteration: int,
        feedback: Optional[str],
    ) -> RLMResult:
        """Execute a single query iteration with optional feedback."""
        # Modify query to include feedback context
        effective_query = query
        if feedback and iteration > 0:
            effective_query = f"""Previous answer was incomplete. Feedback:
{feedback}

Original question: {query}

Please provide an improved answer based on the feedback."""

        # Execute query
        result = await self.query(effective_query, context, strategy)

        # If using built-in REPL, set iteration context
        # (The official RLM would handle this internally)
        result.iteration = iteration

        return result

    def _default_feedback(self, result: RLMResult, original_query: str) -> str:
        """Generate default feedback for incomplete answers."""
        feedback_parts = ["Your previous answer was marked as incomplete."]

        if result.uncertainty_sources:
            feedback_parts.append(
                f"Uncertainty sources identified: {', '.join(result.uncertainty_sources)}"
            )

        if result.confidence < 0.5:
            feedback_parts.append(
                "Confidence was low. Try drilling down into more specific context sections."
            )

        if result.sub_calls_made == 0:
            feedback_parts.append(
                "Consider using RLM_M() to delegate complex sub-queries."
            )

        feedback_parts.append(
            f"Focus on answering: {original_query[:200]}"
        )

        return "\n".join(feedback_parts)

    async def query_stream(
        self,
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
        start_time = time.perf_counter()

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
                            content=node.content[:200] + "..." if len(node.content) > 200 else node.content,
                        )

            # Execute the actual query
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

        except Exception as e:
            logger.error(f"Streaming query failed: {e}")
            yield RLMStreamEvent(
                event_type=RLMStreamEventType.ERROR,
                query=query,
                error=str(e),
            )
            raise

    async def query_with_refinement_stream(
        self,
        query: str,
        context: RLMContext,
        strategy: str = "auto",
        max_iterations: int = 3,
        feedback_generator: Optional[Callable[[RLMResult], str]] = None,
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
        result: Optional[RLMResult] = None

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
            feedback: Optional[str] = None
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
            except Exception as e:
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
        self,
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

        except Exception as e:
            logger.error(f"Streaming compression failed: {e}")
            yield RLMStreamEvent(
                event_type=RLMStreamEventType.ERROR,
                query=f"compress:{source_type}",
                error=str(e),
            )
            raise


class DebateContextAdapter:
    """
    Adapter for formatting debate history for RLM processing.

    Transforms Aragora debate structures into RLM-compatible format
    with programmatic access to rounds, proposals, critiques, and votes.

    Enhanced with query capabilities for drill-down access to specific
    aspects of debate history.
    """

    def __init__(self, aragora_rlm: Optional[AragoraRLM] = None):
        """
        Initialize the adapter.

        Args:
            aragora_rlm: Optional AragoraRLM instance for queries
        """
        self._rlm = aragora_rlm
        self._cached_context: Optional[RLMContext] = None
        self._compressor = HierarchicalCompressor()

    async def compress_debate(
        self,
        debate_result: Any,
    ) -> RLMContext:
        """
        Compress debate history into hierarchical RLM context.

        Args:
            debate_result: DebateResult from aragora.core

        Returns:
            RLMContext with hierarchical representation
        """
        text = self.to_text(debate_result)
        result = await self._compressor.compress(text, source_type="debate")
        self._cached_context = result.context
        return result.context

    async def query_debate(
        self,
        query: str,
        debate_result: Optional[Any] = None,
        strategy: str = "auto",
    ) -> str:
        """
        Query debate history using RLM.

        Enables natural language queries about debate content:
        - "What were the main disagreements?"
        - "What did Alice argue about security?"
        - "Was consensus reached on pricing?"

        Args:
            query: Natural language query
            debate_result: Optional DebateResult (uses cached if None)
            strategy: Decomposition strategy (auto, grep, partition_map)

        Returns:
            Answer extracted from debate history
        """
        # Get or build context
        if debate_result:
            context = await self.compress_debate(debate_result)
        elif self._cached_context:
            context = self._cached_context
        else:
            return "No debate context available. Provide a debate_result."

        # Use RLM for query
        if self._rlm:
            result = await self._rlm.query(query, context, strategy)
            return result.answer
        else:
            # Fallback: search in context
            return self._simple_query(query, context)

    def _simple_query(self, query: str, context: RLMContext) -> str:
        """Simple keyword-based query fallback."""
        query_terms = query.lower().split()

        # Search in all levels, starting from summary
        for level in [AbstractionLevel.SUMMARY, AbstractionLevel.DETAILED, AbstractionLevel.FULL]:
            if level in context.levels:
                for node in context.levels[level]:
                    content_lower = node.content.lower()
                    matches = sum(1 for term in query_terms if term in content_lower)
                    if matches >= len(query_terms) // 2:
                        return f"From {level.name}:\n{node.content[:1000]}"

        return "No relevant information found in debate history."

    async def get_agent_positions(
        self,
        debate_result: Any,
        agent_name: str,
    ) -> str:
        """
        Get all positions/proposals from a specific agent.

        Args:
            debate_result: DebateResult
            agent_name: Name of agent to query

        Returns:
            Summary of agent's positions across rounds
        """
        data = self.format_for_rlm(debate_result)
        proposals = data["get_proposals_by"](agent_name)

        if not proposals:
            return f"No proposals found from agent '{agent_name}'"

        return f"## {agent_name}'s Positions ({len(proposals)} proposals)\n\n" + "\n\n---\n\n".join(
            f"**Proposal {i+1}:**\n{p[:500]}{'...' if len(p) > 500 else ''}"
            for i, p in enumerate(proposals)
        )

    async def get_critiques_summary(
        self,
        debate_result: Any,
        target_agent: Optional[str] = None,
    ) -> str:
        """
        Get summary of critiques, optionally filtered by target.

        Args:
            debate_result: DebateResult
            target_agent: Optional agent name to filter critiques for

        Returns:
            Summary of critiques
        """
        data = self.format_for_rlm(debate_result)

        if target_agent:
            critiques = data["get_critiques_for"](target_agent)
        else:
            critiques = data["CRITIQUES"]

        if not critiques:
            return "No critiques found."

        summary_parts = [f"## Critique Summary ({len(critiques)} total)\n"]

        for c in critiques[:10]:  # Limit to 10
            summary_parts.append(
                f"**{c['critic']} → {c['target']}**: {c['content'][:200]}..."
            )

        return "\n\n".join(summary_parts)

    async def find_consensus_points(
        self,
        debate_result: Any,
    ) -> str:
        """
        Identify points of agreement across agents.

        Args:
            debate_result: DebateResult

        Returns:
            Summary of consensus points
        """
        # Use RLM to find consensus
        context = await self.compress_debate(debate_result)

        query = (
            "What points did all agents agree on? "
            "List specific areas of consensus or shared conclusions."
        )

        if self._rlm:
            result = await self._rlm.query(query, context, strategy="grep")
            return result.answer

        # Fallback: look for agreement keywords
        data = self.format_for_rlm(debate_result)
        agreement_indicators = []

        for r in data["ROUNDS"]:
            for p in r.get("proposals", []):
                content = p.get("content", "").lower()
                if any(w in content for w in ["agree", "consensus", "shared", "common ground"]):
                    agreement_indicators.append(p.get("content", "")[:200])

        if agreement_indicators:
            return "## Potential Consensus Points\n\n" + "\n---\n".join(agreement_indicators[:5])

        return "No explicit consensus points identified."

    async def find_disagreements(
        self,
        debate_result: Any,
    ) -> str:
        """
        Identify key points of disagreement.

        Args:
            debate_result: DebateResult

        Returns:
            Summary of disagreements
        """
        context = await self.compress_debate(debate_result)

        query = (
            "What were the main disagreements or conflicts between agents? "
            "List specific points where agents held opposing views."
        )

        if self._rlm:
            result = await self._rlm.query(query, context, strategy="grep")
            return result.answer

        # Fallback
        data = self.format_for_rlm(debate_result)
        disagreements = []

        for c in data["CRITIQUES"]:
            content = c.get("content", "").lower()
            if any(w in content for w in ["disagree", "incorrect", "wrong", "however", "but"]):
                disagreements.append(
                    f"{c['critic']} → {c['target']}: {c.get('content', '')[:150]}"
                )

        if disagreements:
            return "## Key Disagreements\n\n" + "\n\n".join(disagreements[:5])

        return "No explicit disagreements identified in critiques."

    def format_for_rlm(
        self,
        debate_result: Any,  # DebateResult from aragora.core
    ) -> dict[str, Any]:
        """
        Format debate result for RLM REPL access.

        Returns a dictionary that can be injected into REPL as variables:
        - ROUNDS: List of round data
        - PROPOSALS: Dict of agent -> proposal
        - CRITIQUES: Dict of (critic, target) -> critique
        - CONSENSUS: Final consensus if reached
        - get_round(n): Function to get specific round
        - get_critiques_for(agent): Function to get critiques targeting agent
        """
        rounds: list[dict[str, Any]] = []
        proposals: dict[str, list[str]] = {}
        critiques: list[dict[str, str]] = []

        if hasattr(debate_result, "rounds"):
            for i, r in enumerate(debate_result.rounds):
                round_proposals: list[dict[str, str]] = []
                round_critiques: list[dict[str, str]] = []

                # Extract proposals
                if hasattr(r, "proposals"):
                    for p in r.proposals:
                        agent = getattr(p, "agent", "unknown")
                        content = getattr(p, "content", str(p))
                        round_proposals.append({
                            "agent": agent,
                            "content": content,
                        })
                        proposals.setdefault(agent, []).append(content)

                # Extract critiques
                if hasattr(r, "critiques"):
                    for c in r.critiques:
                        critic = getattr(c, "critic", "unknown")
                        target = getattr(c, "target", "unknown")
                        content = getattr(c, "content", str(c))
                        critique_data = {
                            "critic": critic,
                            "target": target,
                            "content": content,
                        }
                        round_critiques.append(critique_data)
                        critiques.append(critique_data)

                round_data: dict[str, Any] = {
                    "number": i + 1,
                    "proposals": round_proposals,
                    "critiques": round_critiques,
                }
                rounds.append(round_data)

        # Extract consensus
        consensus = None
        if hasattr(debate_result, "consensus"):
            consensus = debate_result.consensus
        elif hasattr(debate_result, "final_answer"):
            consensus = debate_result.final_answer

        # Build helper functions
        def get_round(n: int) -> Optional[dict]:
            if 0 < n <= len(rounds):
                return rounds[n - 1]
            return None

        def get_critiques_for(agent: str) -> list[dict]:
            return [c for c in critiques if c["target"] == agent]

        def get_proposals_by(agent: str) -> list[str]:
            return proposals.get(agent, [])

        return {
            "ROUNDS": rounds,
            "PROPOSALS": proposals,
            "CRITIQUES": critiques,
            "CONSENSUS": consensus,
            "ROUND_COUNT": len(rounds),
            "AGENTS": list(proposals.keys()),
            "get_round": get_round,
            "get_critiques_for": get_critiques_for,
            "get_proposals_by": get_proposals_by,
        }

    def to_text(self, debate_result: Any) -> str:
        """Convert debate result to text for compression."""
        data = self.format_for_rlm(debate_result)
        parts = []

        for r in data["ROUNDS"]:
            parts.append(f"## Round {r['number']}")

            for p in r["proposals"]:
                parts.append(f"### {p['agent']}'s Proposal")
                parts.append(p["content"])
                parts.append("")

            if r["critiques"]:
                parts.append("### Critiques")
                for c in r["critiques"]:
                    parts.append(f"**{c['critic']} → {c['target']}**: {c['content']}")
                parts.append("")

        if data["CONSENSUS"]:
            parts.append("## Consensus")
            parts.append(str(data["CONSENSUS"]))

        return "\n".join(parts)


class KnowledgeMoundAdapter:
    """
    Adapter for integrating Knowledge Mound with RLM.

    Provides hierarchical access to knowledge nodes through RLM REPL.
    """

    def __init__(self, mound: Any):
        """
        Initialize with Knowledge Mound instance.

        Args:
            mound: KnowledgeMound instance from aragora.knowledge.mound
        """
        self.mound = mound

    async def to_rlm_context(
        self,
        workspace_id: str,
        query: Optional[str] = None,
        max_nodes: int = 100,
    ) -> RLMContext:
        """
        Convert Knowledge Mound contents to RLM context.

        Args:
            workspace_id: Workspace to query
            query: Optional query to filter relevant nodes
            max_nodes: Maximum nodes to include

        Returns:
            RLMContext with hierarchical representation
        """
        from .types import AbstractionNode, RLMContext

        # Query relevant nodes
        if query:
            nodes = await self.mound.query_semantic(
                text=query,
                limit=max_nodes,
                workspace_id=workspace_id,
            )
        else:
            nodes = await self.mound.get_recent_nodes(
                workspace_id=workspace_id,
                limit=max_nodes,
            )

        # Build content from nodes
        content_parts = []
        for node in nodes:
            content_parts.append(f"[{node.id}] {node.content}")

        full_content = "\n\n".join(content_parts)

        # Create basic context
        context = RLMContext(
            original_content=full_content,
            original_tokens=len(full_content) // 4,
            source_type="knowledge",
        )

        # Group nodes by type for hierarchical representation
        nodes_by_type: dict[str, list] = {}
        for node in nodes:
            node_type = getattr(node, "node_type", "unknown")
            nodes_by_type.setdefault(node_type, []).append(node)

        # Create abstraction nodes per type
        abstraction_nodes = []
        for node_type, type_nodes in nodes_by_type.items():
            summary_content = f"**{node_type.upper()}** ({len(type_nodes)} items):\n"
            summary_content += "\n".join(
                f"- {n.content[:100]}..." if len(n.content) > 100 else f"- {n.content}"
                for n in type_nodes[:10]
            )

            abstraction_nodes.append(AbstractionNode(
                id=f"type_{node_type}",
                level=AbstractionLevel.SUMMARY,
                content=summary_content,
                token_count=len(summary_content) // 4,
                child_ids=[n.id for n in type_nodes],
            ))

        context.levels[AbstractionLevel.SUMMARY] = abstraction_nodes
        for node in abstraction_nodes:
            context.nodes_by_id[node.id] = node

        return context

    def get_repl_helpers(self) -> dict[str, Callable]:
        """
        Get helper functions for REPL access to Knowledge Mound.

        Returns dict of functions that can be injected into REPL namespace.
        """
        async def search_mound(query: str, limit: int = 10) -> list[dict]:
            nodes = await self.mound.query_semantic(query=query, limit=limit)
            return [
                {
                    "id": n.id,
                    "type": getattr(n, "node_type", "unknown"),
                    "content": n.content[:200],
                    "confidence": getattr(n, "confidence", 1.0),
                }
                for n in nodes
            ]

        async def get_mound_node(node_id: str) -> Optional[dict]:
            node = await self.mound.get_node(node_id)
            if not node:
                return None
            return {
                "id": node.id,
                "type": getattr(node, "node_type", "unknown"),
                "content": node.content,
                "confidence": getattr(node, "confidence", 1.0),
                "relationships": getattr(node, "relationships", {}),
            }

        return {
            "search_mound": search_mound,
            "get_mound_node": get_mound_node,
        }


# Convenience function
def create_aragora_rlm(
    backend: str = "openai",
    model: str = "gpt-4o",
    verbose: bool = False,
) -> AragoraRLM:
    """
    Create an AragoraRLM instance with sensible defaults.

    Args:
        backend: LLM backend (openai, anthropic, openrouter)
        model: Model name
        verbose: Enable verbose logging

    Returns:
        Configured AragoraRLM instance
    """
    return AragoraRLM(
        backend_config=RLMBackendConfig(
            backend=backend,
            model_name=model,
            verbose=verbose,
        ),
    )
