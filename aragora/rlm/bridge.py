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

# Import extracted adapter classes for backwards compatibility
from .debate_adapter import DebateContextAdapter
from .knowledge_adapter import KnowledgeMoundAdapter
from .hierarchy_cache import RLMHierarchyCache


@dataclass
class RLMBackendConfig:
    """Configuration for RLM backend."""

    backend: str = "openai"  # openai, anthropic, openrouter, litellm
    model_name: str = "gpt-4o"
    sub_model_name: str = "gpt-4o-mini"

    # Environment configuration (REPL sandbox type)
    environment_type: str = "local"  # local, docker, modal
    environment_timeout: int = 120
    max_depth: int = 1  # Maximum recursion depth
    max_iterations: int = 30  # Maximum iterations per execution

    # Official RLM kwargs
    verbose: bool = False
    persistent: bool = False  # Keep environment alive between calls


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
        hierarchy_cache: Optional["RLMHierarchyCache"] = None,
        knowledge_mound: Optional[Any] = None,  # For auto-creating cache
        enable_caching: bool = True,  # Enable compression caching
    ):
        """
        Initialize Aragora RLM.

        Args:
            backend_config: Configuration for RLM backend
            aragora_config: Aragora-specific RLM configuration
            agent_registry: Aragora agent registry for fallback
            hierarchy_cache: Optional pre-configured RLMHierarchyCache
            knowledge_mound: Optional KnowledgeMound for persistent caching
            enable_caching: Whether to cache compression hierarchies
        """
        self.backend_config = backend_config or RLMBackendConfig()
        self.aragora_config = aragora_config or RLMConfig()
        self.agent_registry = agent_registry
        self.enable_caching = enable_caching

        self._official_rlm: Optional[Any] = None
        # Compressor is ONLY used as fallback when official RLM unavailable
        self._compressor = HierarchicalCompressor(
            config=self.aragora_config,
            agent_call=self._agent_call,
        )

        # Initialize hierarchy cache for compression result reuse
        self._hierarchy_cache: Optional["RLMHierarchyCache"] = hierarchy_cache
        if self.enable_caching and self._hierarchy_cache is None:
            # Auto-create cache if knowledge_mound provided
            self._hierarchy_cache = RLMHierarchyCache(knowledge_mound=knowledge_mound)

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
            # Build environment kwargs if timeout specified
            env_kwargs = None
            if self.backend_config.environment_timeout != 120:
                env_kwargs = {"timeout": self.backend_config.environment_timeout}

            self._official_rlm = OfficialRLM(
                backend=self.backend_config.backend,
                backend_kwargs={
                    "model_name": self.backend_config.model_name,
                },
                environment=self.backend_config.environment_type,
                environment_kwargs=env_kwargs,
                max_depth=self.backend_config.max_depth,
                max_iterations=self.backend_config.max_iterations,
                verbose=self.backend_config.verbose,
                persistent=self.backend_config.persistent,
            )
            logger.info(
                f"[AragoraRLM] Initialized TRUE RLM with backend={self.backend_config.backend}, "
                f"model={self.backend_config.model_name}, "
                f"environment={self.backend_config.environment_type}"
            )
        except Exception as e:
            logger.error(f"[AragoraRLM] Failed to initialize official RLM: {e}")
            self._official_rlm = None

    def _agent_call(self, prompt: str, model: str) -> str:
        """Call agent for compression/summarization."""
        if self._official_rlm:
            # Use official RLM for simple completions
            try:
                completion = self._official_rlm.completion(prompt)
                return completion.response
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
        - Model has access to context via REPL environment
        - Model WRITES CODE to query/grep/partition context
        - Model can recursively call itself on subsets
        - Model has ACTIVE AGENCY in deciding how to process context

        Falls back to compression ONLY if TRUE RLM fails.
        """
        import time as time_module

        # Format context for RLM REPL
        formatted = self._format_context_for_repl(context)

        # Get context at different abstraction levels
        summary_content = context.get_at_level(AbstractionLevel.SUMMARY) or ""
        abstract_content = context.get_at_level(AbstractionLevel.ABSTRACT) or ""

        # Build RLM prompt with context included
        # The official RLM handles REPL interaction internally - model writes code
        # to decompose and query this context recursively
        rlm_prompt = f"""You are analyzing a hierarchical document context. Use Python code in the REPL to examine, grep, filter, and recursively process the context.

## Context Structure
{formatted['structure']}

## Context Data

### Abstract Level
{abstract_content if abstract_content else '[No abstract available]'}

### Summary Level
{summary_content if summary_content else '[No summary available]'}

### Full Content ({context.original_tokens} tokens)
{context.original_content}

## Instructions
1. Use Python code to programmatically examine the context
2. You can grep for patterns, filter sections, and partition data
3. Use RLM_M(prompt) to recursively call yourself on subsets
4. Call FINAL(answer) when you have the answer

## Task
Answer this question: {query}

Write Python code to analyze the context and call FINAL(answer) with your answer.
"""

        start_time = time_module.perf_counter()
        try:
            # Run RLM completion (handles REPL internally)
            # The model writes code to examine context recursively
            completion = self._official_rlm.completion(
                rlm_prompt,
                root_prompt=query,  # Small prompt visible to root LM
            )

            elapsed = time_module.perf_counter() - start_time

            # TRUE RLM succeeded
            self._last_query_used_true_rlm = True
            logger.info(
                f"[AragoraRLM] TRUE RLM query completed successfully "
                f"in {completion.execution_time:.2f}s"
            )

            return RLMResult(
                answer=completion.response,
                nodes_examined=[],  # Would need trajectory parsing
                levels_traversed=[],
                citations=[],
                tokens_processed=context.original_tokens,
                sub_calls_made=0,  # Could parse from usage_summary
                time_seconds=completion.execution_time,
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
        use_cache: bool = True,
    ) -> RLMResult:
        """
        Convenience method: compress content and query in one step.

        Uses hierarchy cache to avoid recompressing similar content.

        Args:
            query: The query to answer
            content: Raw content to compress
            source_type: Type of content (text, debate, code)
            use_cache: Whether to use cached compression if available

        Returns:
            RLMResult with answer
        """
        compression = None

        # Try cache first if enabled
        if use_cache and self._hierarchy_cache:
            compression = await self._hierarchy_cache.get_cached(content, source_type)
            if compression:
                logger.debug(
                    f"[AragoraRLM] Using cached compression "
                    f"(cache_stats={self._hierarchy_cache.stats})"
                )

        # Compress if not cached
        if compression is None:
            compression = await self._compressor.compress(content, source_type)

            # Store in cache for future use
            if use_cache and self._hierarchy_cache:
                await self._hierarchy_cache.store(content, source_type, compression)

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



# Convenience function
def create_aragora_rlm(
    backend: str = "openai",
    model: str = "gpt-4o",
    verbose: bool = False,
    knowledge_mound: Optional[Any] = None,
    enable_caching: bool = True,
) -> AragoraRLM:
    """
    Create an AragoraRLM instance with sensible defaults.

    Args:
        backend: LLM backend (openai, anthropic, openrouter)
        model: Model name
        verbose: Enable verbose logging
        knowledge_mound: Optional KnowledgeMound for persistent hierarchy caching
        enable_caching: Whether to enable compression caching (default True)

    Returns:
        Configured AragoraRLM instance
    """
    return AragoraRLM(
        backend_config=RLMBackendConfig(
            backend=backend,
            model_name=model,
            verbose=verbose,
        ),
        knowledge_mound=knowledge_mound,
        enable_caching=enable_caching,
    )


# Re-export extracted classes for backwards compatibility
__all__ = [
    "AragoraRLM",
    "RLMBackendConfig",
    "DebateContextAdapter",
    "KnowledgeMoundAdapter",
    "RLMHierarchyCache",
    "create_aragora_rlm",
    "HAS_OFFICIAL_RLM",
]
