"""
RLM Context Adapter - programmatic access to context as external environment.

Based on Prime Intellect's RLM paper (arXiv:2512.24601): instead of stuffing
everything into the prompt, treat context as an external environment that
the LLM can query programmatically.

Key RLM principles:
1. **External Environment**: Data lives outside the prompt, accessed via code
2. **Query-based Access**: LLM requests specific data when needed
3. **Drill-down**: Start with summaries, drill into details on demand
4. **Compression as Fallback**: Only compress when programmatic access unavailable

Usage:
    from aragora.rlm.adapter import RLMContextAdapter
    from aragora.rlm.types import AbstractionLevel

    # Create adapter with external context store
    adapter = RLMContextAdapter()

    # Register content in external store (not in prompt)
    context_id = adapter.register_content("evidence_123", long_evidence_text)

    # Get summary for prompt (minimal context)
    summary = adapter.get_summary(context_id, max_chars=200)

    # LLM can request more detail via query
    detail = await adapter.query(context_id, "What does the evidence say about X?")

    # Drill down to specific section
    section = adapter.drill_down(context_id, section="conclusion")
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

from .exceptions import (
    RLMCircuitOpenError,
    RLMContentNotFoundError,
    RLMProviderError,
    RLMTimeoutError,
)
from .types import RLMResult

if TYPE_CHECKING:
    from aragora.resilience import CircuitBreaker

    from .compressor import HierarchicalCompressor

logger = logging.getLogger(__name__)

# Default timeout for async operations (seconds)
DEFAULT_TIMEOUT_SECONDS = 30.0

# Circuit breaker configuration
DEFAULT_FAILURE_THRESHOLD = 5
DEFAULT_COOLDOWN_SECONDS = 60.0


@dataclass
class RegisteredContent:
    """Content registered in the external environment."""

    id: str
    """Unique identifier for this content."""

    full_content: str
    """The complete original content."""

    content_type: str
    """Type: evidence, dissent, debate, historical, etc."""

    summary: str = ""
    """Pre-computed summary for prompt inclusion."""

    sections: dict[str, str] = field(default_factory=dict)
    """Named sections for drill-down access."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata (source, timestamp, etc.)."""


class RLMContextAdapter:
    """
    RLM-style adapter: context as external environment, not prompt stuffing.

    Based on Prime Intellect's RLM paper (arXiv:2512.24601), this adapter
    implements the external environment pattern where content is stored
    externally and accessed programmatically rather than stuffed into prompts.

    Key principles:
    1. **External Environment**: Full content stored in registry, not prompts
    2. **Query-based Access**: LLM requests specific data when needed
    3. **Drill-down**: Start with summaries, drill into details on demand

    **Priority Order** (highest to lowest):
    1. TRUE RLM: Use LLM to query/summarize content programmatically
    2. COMPRESSION: Use hierarchical compressor if available
    3. TRUNCATION: Heuristic extraction as LAST RESORT only

    Async methods (generate_summary_async, format_for_prompt_async) use
    the full priority chain. Sync methods fall back to compression/truncation.
    """

    def __init__(
        self,
        compressor: Optional["HierarchicalCompressor"] = None,
        agent_call: Optional[Callable[[str, str], Any]] = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        enable_circuit_breaker: bool = True,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
    ):
        """
        Initialize the adapter.

        Args:
            compressor: Optional compressor for fallback compression
            agent_call: Optional LLM call function for queries
            timeout_seconds: Timeout for async operations (default: 30s)
            enable_circuit_breaker: Whether to use circuit breaker (default: True)
            failure_threshold: Failures before opening circuit (default: 5)
            cooldown_seconds: Seconds before circuit recovery attempt (default: 60)
        """
        self._compressor = compressor
        self._agent_call = agent_call
        self._registry: dict[str, RegisteredContent] = {}
        self._timeout_seconds = timeout_seconds

        # Circuit breaker for resilience
        self._circuit_breaker: Optional["CircuitBreaker"] = None
        if enable_circuit_breaker:
            try:
                from aragora.resilience import CircuitBreaker

                self._circuit_breaker = CircuitBreaker(
                    name="rlm_adapter",
                    failure_threshold=failure_threshold,
                    cooldown_seconds=cooldown_seconds,
                )
            except ImportError:
                logger.debug("CircuitBreaker not available, proceeding without")

    # =========================================================================
    # External Environment: Register and Access Content
    # =========================================================================

    def register_content(
        self,
        content_id: str,
        content: str,
        content_type: str = "text",
        summary: Optional[str] = None,
        sections: Optional[dict[str, str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Register content in the external environment.

        The content is stored externally rather than stuffed into prompts.
        LLMs can query it programmatically when needed.

        Args:
            content_id: Unique identifier (or auto-generate)
            content: Full content to register
            content_type: Type for context-aware handling
            summary: Pre-computed summary (auto-generates if None)
            sections: Named sections for drill-down
            metadata: Additional metadata

        Returns:
            The content_id for later access
        """
        if not content_id:
            content_id = self._generate_id(content)

        # Auto-generate summary if not provided
        if summary is None:
            summary = self._extract_summary(content, content_type)

        # Auto-extract sections if not provided
        if sections is None:
            sections = self._extract_sections(content, content_type)

        self._registry[content_id] = RegisteredContent(
            id=content_id,
            full_content=content,
            content_type=content_type,
            summary=summary,
            sections=sections,
            metadata=metadata or {},
        )

        logger.debug(
            f"adapter_register id={content_id} type={content_type} "
            f"size={len(content)} sections={list(sections.keys())}"
        )

        return content_id

    def get_summary(
        self,
        content_id: str,
        max_chars: Optional[int] = None,
    ) -> str:
        """
        Get a minimal summary for prompt inclusion.

        This is the RLM pattern: include only a summary in the prompt,
        not the full content. The LLM can query for details if needed.

        Args:
            content_id: ID of registered content
            max_chars: Optional character limit

        Returns:
            Summary text (or empty string if not found)
        """
        registered = self._registry.get(content_id)
        if not registered:
            logger.warning(f"adapter_get_summary not_found id={content_id}")
            return ""

        summary = registered.summary
        if max_chars and len(summary) > max_chars:
            summary = self._smart_truncate(summary, max_chars)

        return summary

    def get_full_content(self, content_id: str) -> str:
        """
        Get the full registered content (for queries, not prompts).

        Args:
            content_id: ID of registered content

        Returns:
            Full content text
        """
        registered = self._registry.get(content_id)
        return registered.full_content if registered else ""

    def drill_down(
        self,
        content_id: str,
        section: Optional[str] = None,
        query: Optional[str] = None,
        max_chars: Optional[int] = None,
    ) -> str:
        """
        Drill down to specific content section.

        RLM pattern: start with summary, drill into details on demand.

        Args:
            content_id: ID of registered content
            section: Named section to retrieve (e.g., "conclusion", "evidence")
            query: Search query to find relevant portion
            max_chars: Optional character limit

        Returns:
            Section content or query-matched portion
        """
        registered = self._registry.get(content_id)
        if not registered:
            return ""

        # Get specific section
        if section and section in registered.sections:
            content = registered.sections[section]
            if max_chars and len(content) > max_chars:
                content = self._smart_truncate(content, max_chars)
            return content

        # Search for query match
        if query:
            return self._search_content(registered.full_content, query, max_chars)

        # Return full content (truncated if needed)
        content = registered.full_content
        if max_chars and len(content) > max_chars:
            content = self._smart_truncate(content, max_chars)
        return content

    async def query(
        self,
        content_id: str,
        question: str,
        max_response_chars: int = 500,
        timeout_seconds: float | None = None,
    ) -> RLMResult:
        """
        Query registered content with a question.

        Uses LLM to answer questions about the content without
        including the full content in the main conversation.

        Args:
            content_id: ID of registered content
            question: Question to answer about the content
            max_response_chars: Maximum response length
            timeout_seconds: Override default timeout (optional)

        Returns:
            RLMResult with answer and provenance

        Raises:
            RLMContentNotFoundError: If content_id not in registry
            RLMCircuitOpenError: If circuit breaker is open
            RLMTimeoutError: If operation times out
            RLMProviderError: If LLM provider fails
        """
        registered = self._registry.get(content_id)
        if not registered:
            raise RLMContentNotFoundError(
                f"Content not found: {content_id}",
                operation="query",
                content_id=content_id,
            )

        if not self._agent_call:
            # No LLM available - return relevant snippet
            snippet = self._search_content(registered.full_content, question, max_response_chars)
            return RLMResult(
                answer=snippet,
                ready=True,
                confidence=0.5,
                nodes_examined=[content_id],
            )

        # Check circuit breaker
        if self._circuit_breaker and not self._circuit_breaker.can_proceed():
            raise RLMCircuitOpenError(
                "Circuit breaker is open due to repeated failures",
                cooldown_remaining=self._circuit_breaker.cooldown_seconds,
                operation="query",
                content_id=content_id,
            )

        # Build query prompt
        prompt = f"""Based on the following content, answer the question concisely.

Content ({registered.content_type}):
{registered.full_content[:4000]}

Question: {question}

Answer (be specific and cite relevant parts):"""

        timeout = timeout_seconds or self._timeout_seconds
        start_time = time.monotonic()

        try:
            async with asyncio.timeout(timeout):
                response = await self._agent_call(prompt, "sub_model")

            # Record success
            if self._circuit_breaker:
                self._circuit_breaker.record_success()

            return RLMResult(
                answer=str(response)[:max_response_chars],
                ready=True,
                confidence=0.8,
                nodes_examined=[content_id],
            )

        except asyncio.TimeoutError:
            if self._circuit_breaker:
                self._circuit_breaker.record_failure()
            elapsed = time.monotonic() - start_time
            logger.warning(f"adapter_query_timeout elapsed={elapsed:.2f}s timeout={timeout}s")
            raise RLMTimeoutError(
                f"Query timed out after {elapsed:.2f}s",
                timeout_seconds=timeout,
                operation="query",
                content_id=content_id,
            )

        except ConnectionError as e:
            if self._circuit_breaker:
                self._circuit_breaker.record_failure()
            logger.warning(f"adapter_query_connection_error error={e}")
            raise RLMProviderError(
                f"Provider connection failed: {e}",
                is_transient=True,
                operation="query",
                content_id=content_id,
            ) from e

        except Exception as e:
            if self._circuit_breaker:
                self._circuit_breaker.record_failure()
            logger.warning(f"adapter_query_failed error={e}")
            # Return degraded result instead of raising
            return RLMResult(
                answer=self._search_content(registered.full_content, question, max_response_chars),
                ready=True,
                confidence=0.3,
            )

    def list_registered(self) -> list[str]:
        """List all registered content IDs."""
        return list(self._registry.keys())

    def unregister(self, content_id: str) -> bool:
        """Remove content from the registry."""
        if content_id in self._registry:
            del self._registry[content_id]
            return True
        return False

    # =========================================================================
    # Fallback: Truncation (when RLM access not available)
    # =========================================================================

    def smart_truncate(
        self,
        content: str,
        max_chars: int,
        content_type: str = "text",
    ) -> str:
        """
        Smart truncation as fallback when RLM access not available.

        Unlike naive [:max_chars], this method:
        - Respects sentence boundaries when possible
        - Adds ellipsis indicator when truncated
        - Preserves key structural elements

        IMPORTANT: Use register_content + get_summary instead when possible.
        This is a fallback for cases where content must fit in a prompt.

        Args:
            content: Content to truncate
            max_chars: Maximum characters
            content_type: Type for context-aware truncation

        Returns:
            Truncated content with ellipsis if needed
        """
        return self._smart_truncate(content, max_chars, content_type)

    def format_for_prompt(
        self,
        content: str,
        max_chars: int,
        content_type: str = "text",
        include_hint: bool = True,
    ) -> str:
        """
        Format content for prompt inclusion with RLM hint (sync version).

        Uses heuristic summary. For TRUE RLM with LLM summarization,
        use format_for_prompt_async() instead.

        Args:
            content: Full content
            max_chars: Max chars for prompt
            content_type: Type of content
            include_hint: Include drill-down hint

        Returns:
            Formatted summary for prompt
        """
        if not content:
            return ""

        # Register content for later access
        content_id = self.register_content(
            content_id="",  # Auto-generate
            content=content,
            content_type=content_type,
        )

        summary = self.get_summary(content_id, max_chars - 50 if include_hint else max_chars)

        if include_hint and len(content) > max_chars:
            summary += f"\n[Full {content_type} available: {content_id}]"

        return summary

    async def format_for_prompt_async(
        self,
        content: str,
        max_chars: int,
        content_type: str = "text",
        include_hint: bool = True,
    ) -> str:
        """
        Format content for prompt inclusion - TRUE RLM with LLM summarization.

        This is the preferred method: registers content externally and uses
        LLM to generate a meaningful summary. Falls back to heuristics only
        when no LLM is available.

        Args:
            content: Full content
            max_chars: Max chars for prompt
            content_type: Type of content
            include_hint: Include drill-down hint

        Returns:
            LLM-generated summary with drill-down hint
        """
        if not content:
            return ""

        # Register content for later access (external environment)
        content_id = self.register_content(
            content_id="",  # Auto-generate
            content=content,
            content_type=content_type,
        )

        # TRUE RLM: Use LLM to generate summary
        summary = await self.generate_summary_async(
            content_id, max_chars - 50 if include_hint else max_chars
        )

        if include_hint and len(content) > max_chars:
            summary += f"\n[Full {content_type} available: {content_id}]"

        return summary

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _generate_id(self, content: str) -> str:
        """Generate a unique ID for content."""
        return hashlib.sha256(f"{content[:100]}:{len(content)}".encode()).hexdigest()[:12]

    async def generate_summary_async(
        self,
        content_id: str,
        max_chars: Optional[int] = None,
        timeout_seconds: float | None = None,
    ) -> str:
        """
        Generate summary using TRUE RLM pattern.

        Priority order (highest to lowest):
        1. TRUE RLM: Use LLM to query external content
        2. COMPRESSION: Use hierarchical compressor if available
        3. TRUNCATION: Heuristic extraction as last resort

        Args:
            content_id: ID of registered content
            max_chars: Optional character limit
            timeout_seconds: Override default timeout (optional)

        Returns:
            Summary (LLM > compression > truncation fallback)

        Note:
            This method gracefully degrades on failures - it will always
            return a summary (even if just truncation) unless content not found.
        """
        registered = self._registry.get(content_id)
        if not registered:
            return ""

        # Short content doesn't need summarization
        if len(registered.full_content) <= 200:
            return registered.summary or registered.full_content

        timeout = timeout_seconds or self._timeout_seconds

        # PRIORITY 1: TRUE RLM - Use LLM to generate summary
        if self._agent_call:
            # Check circuit breaker (but don't fail - just skip to fallback)
            circuit_open = self._circuit_breaker and not self._circuit_breaker.can_proceed()
            if circuit_open:
                logger.debug("adapter_summary circuit_open, skipping LLM")
            else:
                try:
                    prompt = f"""Summarize this {registered.content_type} in 2-3 concise sentences.
Focus on the key points and conclusions.

Content:
{registered.full_content[:4000]}

Summary:"""
                    async with asyncio.timeout(timeout):
                        response = await self._agent_call(prompt, "sub_model")

                    # Record success
                    if self._circuit_breaker:
                        self._circuit_breaker.record_success()

                    summary = str(response).strip()

                    # Cache the LLM-generated summary
                    registered.summary = summary

                    if max_chars and len(summary) > max_chars:
                        summary = self._smart_truncate(summary, max_chars)
                    return summary

                except asyncio.TimeoutError:
                    if self._circuit_breaker:
                        self._circuit_breaker.record_failure()
                    logger.warning(
                        f"adapter_llm_summary_timeout timeout={timeout}s, trying compression"
                    )

                except ConnectionError as e:
                    if self._circuit_breaker:
                        self._circuit_breaker.record_failure()
                    logger.warning(
                        f"adapter_llm_summary_connection_error error={e}, trying compression"
                    )

                except Exception as e:
                    if self._circuit_breaker:
                        self._circuit_breaker.record_failure()
                    logger.warning(f"adapter_llm_summary_failed error={e}, trying compression")

        # PRIORITY 2: COMPRESSION - Use compressor if available
        if self._compressor:
            try:
                async with asyncio.timeout(timeout):
                    result = await self._compressor.compress(
                        registered.full_content,
                        source_type=registered.content_type,
                    )
                summary = (
                    result.context.get_at_level(  # type: ignore[attr-defined,union-attr]
                        result.context.abstraction_levels.get("summary", "SUMMARY")  # type: ignore[attr-defined]
                    )
                    if hasattr(result, "context")
                    else str(result)
                )
                registered.summary = summary

                if max_chars and len(summary) > max_chars:
                    summary = self._smart_truncate(summary, max_chars)
                return summary

            except asyncio.TimeoutError:
                logger.warning(f"adapter_compress_timeout timeout={timeout}s, using truncation")

            except Exception as e:
                logger.warning(f"adapter_compress_failed error={e}, using truncation")

        # PRIORITY 3: TRUNCATION - Heuristic extraction as last resort
        summary = registered.summary or self._heuristic_summary(
            registered.full_content, registered.content_type
        )
        if max_chars and len(summary) > max_chars:
            summary = self._smart_truncate(summary, max_chars)
        return summary

    def _heuristic_summary(self, content: str, content_type: str) -> str:
        """
        LAST RESORT: Heuristic summary extraction via truncation.

        Priority order:
        1. TRUE RLM (LLM queries) - PREFERRED
        2. COMPRESSION (hierarchical) - FALLBACK
        3. TRUNCATION (this method) - LAST RESORT ONLY

        This method should only be used when both LLM and compression
        are unavailable. Use generate_summary_async() for true RLM behavior.
        """
        if not content:
            return ""

        # For short content, return as-is
        if len(content) <= 200:
            return content

        # Try to extract first paragraph or sentence
        paragraphs = content.split("\n\n")
        if paragraphs:
            first_para = paragraphs[0].strip()
            if len(first_para) <= 300:
                return first_para

        # Extract first 2-3 sentences
        sentences = re.split(r"(?<=[.!?])\s+", content[:500])
        if sentences:
            summary_sentences = []
            total_len = 0
            for s in sentences[:3]:
                if total_len + len(s) > 250:
                    break
                summary_sentences.append(s)
                total_len += len(s)
            if summary_sentences:
                return " ".join(summary_sentences)

        # Fallback to truncation
        return self._smart_truncate(content, 200)

    def _extract_summary(self, content: str, content_type: str) -> str:
        """Alias for _heuristic_summary for backwards compatibility."""
        return self._heuristic_summary(content, content_type)

    def _extract_sections(self, content: str, content_type: str) -> dict[str, str]:
        """Extract named sections from content."""
        sections: dict[str, str] = {}

        # Look for markdown headers
        header_pattern = r"^##?\s+(.+)$"
        lines = content.split("\n")
        current_section = "intro"
        current_content: list[str] = []

        for line in lines:
            match = re.match(header_pattern, line.strip())
            if match:
                # Save previous section
                if current_content:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = match.group(1).lower().replace(" ", "_")
                current_content = []
            else:
                current_content.append(line)

        # Save final section
        if current_content:
            sections[current_section] = "\n".join(current_content).strip()

        # For specific content types, extract key sections
        if content_type == "evidence":
            # Look for conclusion/finding
            for keyword in ["conclusion", "finding", "result"]:
                for line in lines:
                    if keyword in line.lower() and ":" in line:
                        sections["conclusion"] = line.split(":", 1)[-1].strip()
                        break

        elif content_type == "dissent":
            # Look for core disagreement
            for keyword in ["disagree", "concern", "issue", "however"]:
                for i, line in enumerate(lines):
                    if keyword in line.lower():
                        # Extract surrounding context
                        start = max(0, i - 1)
                        end = min(len(lines), i + 3)
                        sections["core"] = "\n".join(lines[start:end]).strip()
                        break

        return sections

    def _search_content(self, content: str, query: str, max_chars: Optional[int] = None) -> str:
        """Search content for query-relevant portion."""
        if not query or not content:
            return content[:max_chars] if max_chars else content

        # Simple keyword search
        query_words = set(query.lower().split())
        lines = content.split("\n")

        # Score each line by keyword matches
        scored_lines: list[tuple[int, int, str]] = []
        for i, line in enumerate(lines):
            line_lower = line.lower()
            score = sum(1 for word in query_words if word in line_lower)
            if score > 0:
                scored_lines.append((score, i, line))

        if not scored_lines:
            # No matches - return beginning
            return content[:max_chars] if max_chars else content

        # Sort by score and get best matches with context
        scored_lines.sort(reverse=True)
        best_line_idx = scored_lines[0][1]

        # Include surrounding context
        start = max(0, best_line_idx - 2)
        end = min(len(lines), best_line_idx + 3)
        result = "\n".join(lines[start:end])

        if max_chars and len(result) > max_chars:
            result = self._smart_truncate(result, max_chars)

        return result

    def _smart_truncate(
        self,
        content: str,
        max_chars: int,
        content_type: str = "text",
    ) -> str:
        """Truncate preserving sentence boundaries."""
        if not content or len(content) <= max_chars:
            return content

        truncated = content[:max_chars]

        # Try to break at sentence boundary
        for i in range(len(truncated) - 1, int(max_chars * 0.5), -1):
            if truncated[i] in ".!?" and (i + 1 >= len(truncated) or truncated[i + 1] in " \n"):
                return content[: i + 1]

        # Break at word boundary
        if " " in truncated:
            last_space = truncated.rfind(" ")
            if last_space > max_chars * 0.7:
                return content[:last_space] + "..."

        return truncated + "..."


# =============================================================================
# REPLContextAdapter - TRUE RLM with REPL environment
# =============================================================================


class REPLContextAdapter(RLMContextAdapter):
    """
    TRUE RLM adapter using REPL environments for programmatic context access.

    Based on arXiv:2512.24601 "Recursive Language Models":
    This adapter creates REPL environments where content is stored as Python
    variables and the LLM writes code to examine/query the context.

    Key difference from RLMContextAdapter:
    - RLMContextAdapter: Content registered, LLM queries via API calls
    - REPLContextAdapter: Content in REPL, LLM writes code to navigate

    Usage:
        from aragora.rlm.adapter import REPLContextAdapter

        adapter = REPLContextAdapter()

        # Register debate content - creates REPL environment
        env = adapter.create_repl_for_debate(debate_result)

        # LLM writes code like:
        # >>> round1 = get_round(debate, 1)
        # >>> disagreements = search_debate(debate, r"disagree|however")
        # >>> FINAL(f"Key disagreements: {summarize(disagreements)}")

    Requires: pip install aragora[rlm] for TRUE RLM functionality
    """

    def __init__(
        self,
        compressor: Optional["HierarchicalCompressor"] = None,
        agent_call: Optional[Callable[[str, str], Any]] = None,
        rlm_agent_call: Optional[Callable[[str, str, str], str]] = None,
    ):
        """
        Initialize the REPL context adapter.

        Args:
            compressor: Optional compressor for fallback compression
            agent_call: Optional LLM call function for queries (2-arg: prompt, model)
            rlm_agent_call: Optional 3-arg LLM call for RLM sub-calls (model, query, context)
        """
        super().__init__(compressor=compressor, agent_call=agent_call)
        self._repl_environments: dict[str, Any] = {}
        self._has_true_rlm = self._check_true_rlm()
        self._rlm_agent_call = rlm_agent_call

    def _check_true_rlm(self) -> bool:
        """Check if TRUE RLM (official library) is available."""
        try:
            from .bridge import HAS_OFFICIAL_RLM

            return HAS_OFFICIAL_RLM
        except ImportError:
            return False

    @property
    def has_true_rlm(self) -> bool:
        """Whether TRUE RLM is available."""
        return self._has_true_rlm

    def create_repl_for_debate(
        self,
        debate_result: Any,
        content_id: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Create a TRUE RLM REPL environment for a debate.

        The debate content is loaded into a REPL environment where the LLM
        can write code to navigate and query the debate history.

        Args:
            debate_result: DebateResult from a completed debate
            content_id: Optional ID (auto-generated if not provided)

        Returns:
            RLM environment (or None if TRUE RLM not available)

        Example:
            >>> env = adapter.create_repl_for_debate(result)
            >>> # LLM can now write code like:
            >>> # debate = load_debate_context(result)
            >>> # proposals = get_proposals_by_agent(debate, "claude")
        """
        from .debate_helpers import get_debate_helpers, load_debate_context
        from .types import RLMConfig, RLMContext, AbstractionLevel, AbstractionNode

        # Generate content ID if not provided
        if not content_id:
            debate_id = getattr(debate_result, "debate_id", "")
            content_id = f"debate_{debate_id or self._generate_id(str(debate_result))}"

        try:
            # Load debate into structured context
            debate_context = load_debate_context(debate_result)

            # Convert debate to string content for RLMContext
            debate_content = self._debate_to_string(debate_context)

            # Create a summary node
            summary_text = f"Debate with {len(debate_context.agent_names)} agents over {debate_context.total_rounds} rounds. Task: {debate_context.task}"
            summary_node = AbstractionNode(
                id="debate_summary",
                level=AbstractionLevel.SUMMARY,
                content=summary_text,
                token_count=len(summary_text) // 4,
            )

            # Create RLMConfig
            config = RLMConfig()

            # Create RLMContext with debate content and summary
            rlm_context = RLMContext(
                original_content=debate_content,
                original_tokens=len(debate_content) // 4,  # Approximate
                levels={AbstractionLevel.SUMMARY: [summary_node]},
                nodes_by_id={summary_node.id: summary_node},
            )

            # Create REPL environment with proper initialization
            from .repl import RLMEnvironment

            env = RLMEnvironment(
                config=config,
                context=rlm_context,
                agent_call=self._rlm_agent_call,
            )

            # Inject debate context and helpers into the namespace
            # Note: RLMEnvironment already provides RLM_M and FINAL via _rlm_call and _final
            env.state.namespace["debate"] = debate_context
            env.state.namespace["debate_result"] = debate_result

            # Inject debate-specific navigation helpers (NOT RLM_M/FINAL - those come from env)
            for name, helper in get_debate_helpers(include_rlm_primitives=False).items():
                env.state.namespace[name] = helper

            # Store environment
            self._repl_environments[content_id] = env

            logger.info(
                f"repl_env_created type=debate id={content_id} "
                f"rounds={debate_context.total_rounds} agents={len(debate_context.agent_names)}"
            )

            return env

        except Exception as e:
            logger.error(f"Failed to create debate REPL environment: {e}")
            return None

    def _debate_to_string(self, debate_context: Any) -> str:
        """Convert DebateREPLContext to string representation for RLMContext."""
        lines = [
            f"# Debate: {debate_context.task}",
            f"Agents: {', '.join(debate_context.agent_names)}",
            f"Rounds: {debate_context.total_rounds}",
            f"Consensus: {debate_context.consensus_reached}",
            "",
        ]

        for round_num in sorted(debate_context.rounds.keys()):
            lines.append(f"## Round {round_num}")
            for msg in debate_context.rounds[round_num]:
                agent = msg.get("agent", "unknown")
                content = msg.get("content", "")[:500]
                lines.append(f"### {agent}")
                lines.append(content)
                lines.append("")

        if debate_context.final_answer:
            lines.append("## Final Answer")
            lines.append(debate_context.final_answer)

        return "\n".join(lines)

    def create_repl_for_knowledge(
        self,
        mound: Any,
        workspace_id: str,
        content_id: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Create a TRUE RLM REPL environment for Knowledge Mound queries.

        The knowledge content is loaded into a REPL environment where the LLM
        can write code to query facts, claims, and evidence.

        Args:
            mound: KnowledgeMound instance
            workspace_id: Workspace to load knowledge from
            content_id: Optional ID (auto-generated if not provided)

        Returns:
            RLM environment (or None if TRUE RLM not available)

        Example:
            >>> env = adapter.create_repl_for_knowledge(mound, "ws-123")
            >>> # LLM can now write code like:
            >>> # facts = get_facts(km, "rate limiting", min_confidence=0.8)
            >>> # related = get_related(km, facts[0].id)
        """
        from .knowledge_helpers import get_knowledge_helpers, load_knowledge_context
        from .types import RLMConfig, RLMContext, AbstractionLevel, AbstractionNode

        # Generate content ID if not provided
        if not content_id:
            content_id = f"knowledge_{workspace_id}"

        try:
            # Load knowledge into structured context
            km_context = load_knowledge_context(mound, workspace_id)

            # Convert knowledge to string content for RLMContext
            knowledge_content = self._knowledge_to_string(km_context)

            # Create a summary node
            summary_text = f"Knowledge Mound with {km_context.total_items} items. Avg confidence: {km_context.avg_confidence:.2f}"
            summary_node = AbstractionNode(
                id="knowledge_summary",
                level=AbstractionLevel.SUMMARY,
                content=summary_text,
                token_count=len(summary_text) // 4,
            )

            # Create RLMConfig
            config = RLMConfig()

            # Create RLMContext with knowledge content and summary
            rlm_context = RLMContext(
                original_content=knowledge_content,
                original_tokens=len(knowledge_content) // 4,  # Approximate
                levels={AbstractionLevel.SUMMARY: [summary_node]},
                nodes_by_id={summary_node.id: summary_node},
            )

            # Create REPL environment with proper initialization
            from .repl import RLMEnvironment

            env = RLMEnvironment(
                config=config,
                context=rlm_context,
                agent_call=self._rlm_agent_call,
            )

            # Inject knowledge context and helpers into the namespace
            # Note: RLMEnvironment already provides RLM_M and FINAL via _rlm_call and _final
            env.state.namespace["km"] = km_context
            env.state.namespace["mound"] = mound
            env.state.namespace["workspace_id"] = workspace_id

            # Inject knowledge-specific navigation helpers (NOT RLM_M/FINAL - those come from env)
            for name, helper in get_knowledge_helpers(mound, include_rlm_primitives=False).items():
                env.state.namespace[name] = helper

            # Store environment
            self._repl_environments[content_id] = env

            logger.info(
                f"repl_env_created type=knowledge id={content_id} "
                f"items={km_context.total_items} confidence={km_context.avg_confidence:.2f}"
            )

            return env

        except Exception as e:
            logger.error(f"Failed to create knowledge REPL environment: {e}")
            return None

    def _knowledge_to_string(self, km_context: Any) -> str:
        """Convert KnowledgeREPLContext to string representation for RLMContext."""
        lines = [
            "# Knowledge Mound",
            f"Total items: {km_context.total_items}",
            f"Average confidence: {km_context.avg_confidence:.2f}",
            "",
        ]

        # Add facts
        if hasattr(km_context, "facts") and km_context.facts:
            lines.append("## Facts")
            for fact in km_context.facts[:20]:  # Limit to first 20
                content = getattr(fact, "content", str(fact))[:200]
                confidence = getattr(fact, "confidence", 0.0)
                lines.append(f"- [{confidence:.2f}] {content}")
            lines.append("")

        # Add claims
        if hasattr(km_context, "claims") and km_context.claims:
            lines.append("## Claims")
            for claim in km_context.claims[:20]:
                content = getattr(claim, "content", str(claim))[:200]
                confidence = getattr(claim, "confidence", 0.0)
                lines.append(f"- [{confidence:.2f}] {content}")
            lines.append("")

        return "\n".join(lines)

    def get_repl_environment(self, content_id: str) -> Optional[Any]:
        """
        Get an existing REPL environment by content ID.

        Args:
            content_id: The content ID used when creating the environment

        Returns:
            RLM environment or None if not found
        """
        return self._repl_environments.get(content_id)

    def get_repl_prompt(self, content_id: str, content_type: str = "debate") -> str:
        """
        Get REPL system prompt for agent instructions.

        This prompt tells the agent how to use the REPL environment
        to query context programmatically.

        Args:
            content_id: The content ID for reference
            content_type: Type of content ("debate" or "knowledge")

        Returns:
            System prompt with REPL usage instructions
        """
        if content_type == "debate":
            return f"""You have access to a REPL environment containing debate context.
The debate is stored in the variable `debate` (DebateREPLContext).

Available functions:
- get_round(debate, round_num) → list of messages in that round
- get_proposals_by_agent(debate, agent_name) → all messages from an agent
- search_debate(debate, pattern) → messages matching regex pattern
- get_critiques(debate, target_agent) → critique messages
- partition_debate(debate, "round" or "agent") → partitioned messages

For recursive queries on subsets:
- RLM_M(query, subset=messages) → answer about subset

When you have your final answer:
- FINAL(answer) → return the final answer

Content ID: {content_id}
"""
        elif content_type == "knowledge":
            return f"""You have access to a REPL environment containing knowledge context.
The knowledge is stored in the variable `km` (KnowledgeREPLContext).

Available functions:
- get_facts(km, query, min_confidence) → filtered facts
- get_claims(km, query, validated_only) → filtered claims
- get_evidence(km, query, source_type) → filtered evidence
- get_related(km, item_id, depth) → related items via graph
- filter_by_confidence(items, min, max) → confidence filter
- search_knowledge(km, pattern) → regex search

For recursive queries on subsets:
- RLM_M(query, subset=items) → answer about subset

When you have your final answer:
- FINAL(answer) → return the final answer

Content ID: {content_id}
"""
        else:
            return f"REPL environment available for {content_type}. Content ID: {content_id}"

    def list_repl_environments(self) -> list[str]:
        """List all active REPL environment content IDs."""
        return list(self._repl_environments.keys())

    def close_repl_environment(self, content_id: str) -> bool:
        """Close and remove a REPL environment."""
        if content_id in self._repl_environments:
            del self._repl_environments[content_id]
            return True
        return False


# Global adapter instances
_global_adapter: Optional[RLMContextAdapter] = None
_global_repl_adapter: Optional[REPLContextAdapter] = None


def get_adapter() -> RLMContextAdapter:
    """Get the global RLMContextAdapter instance."""
    global _global_adapter
    if _global_adapter is None:
        _global_adapter = RLMContextAdapter()
    return _global_adapter


def get_repl_adapter() -> REPLContextAdapter:
    """
    Get the global REPLContextAdapter instance for TRUE RLM.

    This adapter creates REPL environments where LLMs write code
    to query context programmatically.

    Returns:
        REPLContextAdapter (prefers TRUE RLM when available)
    """
    global _global_repl_adapter
    if _global_repl_adapter is None:
        _global_repl_adapter = REPLContextAdapter()
    return _global_repl_adapter


__all__ = [
    "RLMContextAdapter",
    "REPLContextAdapter",
    "RegisteredContent",
    "get_adapter",
    "get_repl_adapter",
]
