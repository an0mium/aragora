"""
RLM-Enhanced Cognitive Load Limiter.

Extends the base CognitiveLoadLimiter with Recursive Language Model (RLM)
hierarchical compression. Instead of truncating content, older context is
compressed into summaries at multiple abstraction levels while maintaining
full semantic access through the RLM REPL.

Based on arXiv:2512.24601 - "Recursive Language Models"
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, List, Optional

from aragora.debate.cognitive_limiter import (
    STRESS_BUDGETS,
    CognitiveBudget,
    CognitiveLoadLimiter,
)

if TYPE_CHECKING:
    from aragora.rlm import HierarchicalCompressor
    from aragora.rlm.types import CompressionResult

logger = logging.getLogger(__name__)


@dataclass
class RLMCognitiveBudget(CognitiveBudget):
    """Extended budget with RLM compression parameters."""

    # RLM-specific settings
    enable_rlm_compression: bool = True
    compression_threshold: int = 3000  # Chars above which to use RLM
    max_recent_full_messages: int = 5  # Keep N most recent at full detail
    summary_level: str = "SUMMARY"  # Default abstraction level for older content
    preserve_first_message: bool = True  # Always keep task description full


@dataclass
class CompressedContext:
    """Context with hierarchical compression applied."""

    # Original content (may be compressed)
    messages: List[Any] = field(default_factory=list)
    critiques: List[Any] = field(default_factory=list)
    patterns: str = ""
    extra_context: str = ""

    # RLM metadata
    compression_applied: bool = False
    abstraction_levels: List[str] = field(default_factory=list)
    full_content_hash: str = ""  # For cache lookup of original
    rlm_environment_id: Optional[str] = None  # For REPL access

    # Stats
    original_chars: int = 0
    compressed_chars: int = 0

    @property
    def compression_ratio(self) -> float:
        """Compression ratio achieved."""
        if self.original_chars == 0:
            return 1.0
        return self.compressed_chars / self.original_chars


class RLMCognitiveLoadLimiter(CognitiveLoadLimiter):
    """
    RLM-enhanced cognitive load limiter.

    Uses hierarchical compression from RLM to preserve semantic content
    instead of discarding it through truncation.

    Key differences from base limiter:
    1. Older messages are compressed into summaries, not discarded
    2. Full content remains accessible through RLM REPL
    3. Abstraction levels allow retrieving detail on demand

    Usage:
        limiter = RLMCognitiveLoadLimiter()

        # Async compression (uses LLM for summarization)
        compressed = await limiter.compress_context_async(
            messages=debate_history,
            critiques=all_critiques,
        )

        # Sync fallback (uses rule-based compression)
        compressed = limiter.compress_context(
            messages=debate_history,
            critiques=all_critiques,
        )
    """

    def __init__(
        self,
        budget: Optional[RLMCognitiveBudget] = None,
        compressor: Optional["HierarchicalCompressor"] = None,
        summarize_fn: Optional[Callable[[str, str], str]] = None,
    ):
        """
        Initialize the RLM-enhanced limiter.

        Args:
            budget: RLM-aware cognitive budget
            compressor: HierarchicalCompressor instance (lazy-loaded if None)
            summarize_fn: Optional sync function for rule-based summarization
        """
        rlm_budget = budget or RLMCognitiveBudget()
        super().__init__(budget=rlm_budget)
        self._compressor = compressor
        self._summarize_fn = summarize_fn
        self._compression_cache: dict[str, CompressionResult] = {}

        # Extended stats
        self.stats.update({
            "rlm_compressions": 0,
            "compression_ratio_avg": 1.0,
            "abstraction_levels_used": {},
        })

    @property
    def compressor(self) -> Optional["HierarchicalCompressor"]:
        """Lazy-load the hierarchical compressor."""
        if self._compressor is None:
            try:
                from aragora.rlm import HierarchicalCompressor
                self._compressor = HierarchicalCompressor()
            except ImportError:
                logger.warning("RLM module not available, falling back to base limiter")
        return self._compressor

    @classmethod
    def for_stress_level(cls, level: str) -> "RLMCognitiveLoadLimiter":
        """Create RLM limiter for stress level."""
        base_budget = STRESS_BUDGETS.get(level, STRESS_BUDGETS["elevated"])

        # Convert to RLM budget with level-appropriate settings
        rlm_budget = RLMCognitiveBudget(
            max_context_tokens=base_budget.max_context_tokens,
            max_history_messages=base_budget.max_history_messages,
            max_critique_chars=base_budget.max_critique_chars,
            max_proposal_chars=base_budget.max_proposal_chars,
            max_patterns_chars=base_budget.max_patterns_chars,
            reserve_for_response=base_budget.reserve_for_response,
            # RLM settings scale with stress
            enable_rlm_compression=True,
            compression_threshold=base_budget.max_context_tokens * 2,  # 2x budget triggers compression
            max_recent_full_messages=max(2, base_budget.max_history_messages // 3),
            summary_level="ABSTRACT" if level == "critical" else "SUMMARY",
        )

        return cls(budget=rlm_budget)

    async def compress_context_async(
        self,
        messages: Optional[List[Any]] = None,
        critiques: Optional[List[Any]] = None,
        patterns: Optional[str] = None,
        extra_context: Optional[str] = None,
    ) -> CompressedContext:
        """
        Compress context using RLM hierarchical compression (async).

        Uses LLM-based summarization for high-quality compression.
        Falls back to rule-based compression if RLM unavailable.

        Args:
            messages: Message history
            critiques: Critiques received
            patterns: Pattern string
            extra_context: Additional context

        Returns:
            CompressedContext with hierarchically compressed content
        """
        budget = self.budget
        if not isinstance(budget, RLMCognitiveBudget):
            budget = RLMCognitiveBudget()

        # Calculate total content size
        total_chars = self._calculate_total_chars(messages, critiques, patterns, extra_context)

        # If under threshold, use base limiter
        if total_chars <= budget.compression_threshold or not budget.enable_rlm_compression:
            base_result = self.limit_context(messages, critiques, patterns, extra_context)
            return CompressedContext(
                messages=base_result.get("messages", []),
                critiques=base_result.get("critiques", []),
                patterns=base_result.get("patterns", ""),
                extra_context=base_result.get("extra_context", ""),
                compression_applied=False,
                original_chars=total_chars,
                compressed_chars=total_chars,
            )

        # Apply RLM hierarchical compression
        result = CompressedContext(
            compression_applied=True,
            original_chars=total_chars,
        )

        # Compress messages hierarchically
        if messages:
            compressed_messages, levels = await self._compress_messages_async(
                messages, budget
            )
            result.messages = compressed_messages
            result.abstraction_levels.extend(levels)

        # Compress critiques
        if critiques:
            compressed_critiques = await self._compress_critiques_async(
                critiques, budget
            )
            result.critiques = compressed_critiques

        # Patterns and extra context use rule-based compression
        if patterns:
            result.patterns = self._compress_text(patterns, budget.max_patterns_chars)

        if extra_context:
            other_budget = int(budget.max_context_chars * 0.15)
            result.extra_context = self._compress_text(extra_context, other_budget)

        result.compressed_chars = self._calculate_total_chars(
            result.messages, result.critiques, result.patterns, result.extra_context
        )

        # Update stats
        self.stats["rlm_compressions"] += 1
        ratio = result.compression_ratio
        self.stats["compression_ratio_avg"] = (
            self.stats["compression_ratio_avg"] * 0.9 + ratio * 0.1
        )

        for level in result.abstraction_levels:
            self.stats["abstraction_levels_used"][level] = (
                self.stats["abstraction_levels_used"].get(level, 0) + 1
            )

        logger.info(
            f"[rlm_compress] {result.original_chars} -> {result.compressed_chars} chars "
            f"(ratio={ratio:.2f}, levels={result.abstraction_levels})"
        )

        return result

    async def _compress_messages_async(
        self,
        messages: List[Any],
        budget: RLMCognitiveBudget,
    ) -> tuple[List[Any], List[str]]:
        """Compress message history hierarchically."""
        if not messages:
            return [], []

        levels_used = []
        result = []

        # Split into sections
        n_recent = budget.max_recent_full_messages
        first_msg = messages[0] if budget.preserve_first_message else None
        recent = messages[-n_recent:] if len(messages) > n_recent else messages
        middle = messages[1:-n_recent] if len(messages) > n_recent + 1 else []

        # Keep first message (task) at full detail
        if first_msg and first_msg not in recent:
            result.append(first_msg)
            levels_used.append("FULL")

        # Compress middle section if exists
        if middle:
            compressed_middle = await self._compress_message_section(
                middle, budget.summary_level
            )
            result.append(compressed_middle)
            levels_used.append(budget.summary_level)

        # Keep recent messages at full detail
        for msg in recent:
            if msg != first_msg:  # Avoid duplicate
                result.append(msg)
                levels_used.append("FULL")

        return result, levels_used

    async def _compress_message_section(
        self,
        messages: List[Any],
        target_level: str,
    ) -> Any:
        """Compress a section of messages to target abstraction level."""
        from aragora.core import Message

        # Format messages for compression
        content_parts = []
        for msg in messages:
            agent = getattr(msg, "agent", "unknown")
            role = getattr(msg, "role", "")
            content = getattr(msg, "content", str(msg))
            round_num = getattr(msg, "round", 0)
            content_parts.append(f"[Round {round_num}] {agent} ({role}): {content}")

        full_content = "\n\n".join(content_parts)

        # Try RLM compression
        if self.compressor:
            try:
                # HierarchicalCompressor.compress() is async
                compression_result = await self.compressor.compress(
                    content=full_content,
                    source_type="debate",
                    max_levels=3,
                )

                # Get the target level content from the context
                from aragora.rlm.types import AbstractionLevel
                level_enum = AbstractionLevel[target_level] if isinstance(target_level, str) else target_level
                target_content = compression_result.context.get_at_level(level_enum)
                if target_content:
                    return Message(
                        agent="system",
                        role="context",
                        content=f"## Compressed Debate History ({len(messages)} messages)\n\n{target_content}",
                        round=-1,
                    )
            except Exception as e:
                logger.warning(f"RLM compression failed, using fallback: {e}")

        # Fallback: rule-based summarization
        summary = self._rule_based_summarize(full_content, len(messages))
        return Message(
            agent="system",
            role="context",
            content=f"## Compressed Debate History ({len(messages)} messages)\n\n{summary}",
            round=-1,
        )

    def _rule_based_summarize(self, content: str, message_count: int) -> str:
        """Rule-based summarization when RLM not available."""
        # Extract key patterns
        lines = content.split("\n")

        # Find consensus/agreement indicators
        agreements = [line for line in lines if "agree" in line.lower() or "consensus" in line.lower()]
        disagreements = [line for line in lines if "disagree" in line.lower() or "however" in line.lower()]

        # Build summary
        summary_parts = [f"Summary of {message_count} messages:"]

        if agreements:
            summary_parts.append(f"- Key agreements: {len(agreements)} found")
            summary_parts.append(f"  Example: {agreements[0][:200]}...")

        if disagreements:
            summary_parts.append(f"- Key disagreements: {len(disagreements)} found")
            summary_parts.append(f"  Example: {disagreements[0][:200]}...")

        # Add first and last messages as bookends
        if len(lines) > 2:
            summary_parts.append(f"- Started with: {lines[0][:150]}...")
            summary_parts.append(f"- Ended with: {lines[-1][:150]}...")

        return "\n".join(summary_parts)

    async def _compress_critiques_async(
        self,
        critiques: List[Any],
        budget: RLMCognitiveBudget,
    ) -> List[Any]:
        """Compress critiques, grouping by severity."""
        if not critiques:
            return []

        # Group by severity
        high_severity = [c for c in critiques if getattr(c, "severity", 0.5) >= 0.7]
        medium_severity = [c for c in critiques if 0.4 <= getattr(c, "severity", 0.5) < 0.7]
        low_severity = [c for c in critiques if getattr(c, "severity", 0.5) < 0.4]

        result = []

        # Keep high-severity critiques mostly intact
        for c in high_severity[:3]:
            result.append(c)

        # Summarize medium-severity
        if medium_severity:
            summarized = self._summarize_critique_group(medium_severity, "medium")
            result.append(summarized)

        # Heavily compress low-severity
        if low_severity:
            summarized = self._summarize_critique_group(low_severity, "low")
            result.append(summarized)

        return result

    def _summarize_critique_group(self, critiques: List[Any], severity: str) -> dict:
        """Summarize a group of critiques into a single entry."""
        all_issues = []
        all_suggestions = []

        for c in critiques:
            all_issues.extend(getattr(c, "issues", []))
            all_suggestions.extend(getattr(c, "suggestions", []))

        return {
            "severity": severity,
            "count": len(critiques),
            "issues": list(set(all_issues))[:5],
            "suggestions": list(set(all_suggestions))[:3],
            "reasoning": f"Summary of {len(critiques)} {severity}-severity critiques",
        }

    def _compress_text(self, text: str, max_chars: int) -> str:
        """Compress text to max chars with smart truncation."""
        if len(text) <= max_chars:
            return text

        # Try to find natural break points
        sentences = text.split(". ")
        result = []
        current_len = 0

        for sentence in sentences:
            if current_len + len(sentence) + 2 <= max_chars - 50:
                result.append(sentence)
                current_len += len(sentence) + 2
            else:
                break

        if result:
            return ". ".join(result) + f"... [{len(text) - current_len} chars omitted]"

        # Fallback to hard truncation
        return text[:max_chars - 30] + "... [truncated]"

    def _calculate_total_chars(
        self,
        messages: Optional[List[Any]],
        critiques: Optional[List[Any]],
        patterns: Optional[str],
        extra_context: Optional[str],
    ) -> int:
        """Calculate total character count across all content."""
        total = 0

        if messages:
            for m in messages:
                total += len(getattr(m, "content", str(m)))

        if critiques:
            for c in critiques:
                total += len(getattr(c, "reasoning", str(c)))

        if patterns:
            total += len(patterns)

        if extra_context:
            total += len(extra_context)

        return total

    def compress_context(
        self,
        messages: Optional[List[Any]] = None,
        critiques: Optional[List[Any]] = None,
        patterns: Optional[str] = None,
        extra_context: Optional[str] = None,
    ) -> CompressedContext:
        """
        Synchronous wrapper for compress_context_async.

        Uses asyncio.run() if not in async context, otherwise
        falls back to rule-based compression.
        """
        try:
            asyncio.get_running_loop()
            # We're in an async context, use rule-based for sync call
            return self._compress_context_sync(messages, critiques, patterns, extra_context)
        except RuntimeError:
            # No event loop, safe to use asyncio.run()
            return asyncio.run(
                self.compress_context_async(messages, critiques, patterns, extra_context)
            )

    def _compress_context_sync(
        self,
        messages: Optional[List[Any]] = None,
        critiques: Optional[List[Any]] = None,
        patterns: Optional[str] = None,
        extra_context: Optional[str] = None,
    ) -> CompressedContext:
        """Synchronous compression using rule-based methods only."""
        budget = self.budget
        if not isinstance(budget, RLMCognitiveBudget):
            budget = RLMCognitiveBudget()

        total_chars = self._calculate_total_chars(messages, critiques, patterns, extra_context)

        if total_chars <= budget.compression_threshold:
            base_result = self.limit_context(messages, critiques, patterns, extra_context)
            return CompressedContext(
                messages=base_result.get("messages", []),
                critiques=base_result.get("critiques", []),
                patterns=base_result.get("patterns", ""),
                extra_context=base_result.get("extra_context", ""),
                compression_applied=False,
                original_chars=total_chars,
                compressed_chars=total_chars,
            )

        # Rule-based compression
        from aragora.core import Message

        result = CompressedContext(
            compression_applied=True,
            original_chars=total_chars,
        )

        if messages:
            n_recent = budget.max_recent_full_messages
            if len(messages) > n_recent + 1:
                # Compress older messages
                first = messages[0] if budget.preserve_first_message else None
                middle = messages[1:-n_recent]
                recent = messages[-n_recent:]

                compressed = []
                if first:
                    compressed.append(first)

                if middle:
                    summary = self._rule_based_summarize(
                        "\n".join(str(getattr(m, "content", m)) for m in middle),
                        len(middle)
                    )
                    compressed.append(Message(
                        agent="system",
                        role="context",
                        content=f"## Compressed History\n\n{summary}",
                        round=-1,
                    ))

                compressed.extend(recent)
                result.messages = compressed
                result.abstraction_levels = ["FULL", "SUMMARY", "FULL"]
            else:
                result.messages = list(messages)
                result.abstraction_levels = ["FULL"] * len(messages)

        if critiques:
            result.critiques = self.limit_critiques(critiques)

        if patterns:
            result.patterns = self._compress_text(patterns, budget.max_patterns_chars)

        if extra_context:
            other_budget = int(budget.max_context_chars * 0.15)
            result.extra_context = self._compress_text(extra_context, other_budget)

        result.compressed_chars = self._calculate_total_chars(
            result.messages, result.critiques, result.patterns, result.extra_context
        )

        return result


    async def query_compressed_context(
        self,
        query: str,
        compressed_context: CompressedContext,
        detail_level: str = "SUMMARY",
    ) -> str:
        """
        Query compressed context for specific information.

        This enables agents to drill down into compressed history
        without loading the full content. Uses RLM strategies to
        navigate the hierarchical context efficiently.

        Args:
            query: Natural language query about the context
            compressed_context: Previously compressed context
            detail_level: Target detail level (ABSTRACT, SUMMARY, DETAILED, FULL)

        Returns:
            Relevant information extracted from context

        Example:
            >>> result = await limiter.query_compressed_context(
            ...     "What were the main disagreements about privacy?",
            ...     compressed_history,
            ...     detail_level="DETAILED"
            ... )
        """
        if not self.compressor:
            # Fallback: search in messages
            return self._search_compressed_fallback(query, compressed_context)

        try:
            from aragora.rlm.bridge import AragoraRLM
            from aragora.rlm.types import RLMContext

            # Build RLM context from compressed messages
            all_content = []
            for msg in compressed_context.messages:
                content = getattr(msg, "content", str(msg))
                all_content.append(content)

            full_content = "\n\n".join(all_content)

            # Create RLM context
            context = RLMContext(
                original_content=full_content,
                original_tokens=len(full_content) // 4,
                source_type="debate",
            )

            # Use AragoraRLM for query
            rlm = AragoraRLM()
            result = await rlm.query(query, context, strategy="auto")

            return result.answer

        except Exception as e:
            logger.warning(f"RLM query failed, using fallback: {e}")
            return self._search_compressed_fallback(query, compressed_context)

    def _search_compressed_fallback(
        self,
        query: str,
        compressed_context: CompressedContext,
    ) -> str:
        """Fallback search when RLM not available."""
        # Simple keyword-based search
        query_terms = query.lower().split()
        relevant_parts = []

        for msg in compressed_context.messages:
            content = getattr(msg, "content", str(msg)).lower()
            # Count matching terms
            matches = sum(1 for term in query_terms if term in content)
            if matches >= len(query_terms) // 2 + 1:  # At least half match
                relevant_parts.append(getattr(msg, "content", str(msg)))

        if relevant_parts:
            return f"Found {len(relevant_parts)} relevant sections:\n\n" + "\n---\n".join(
                p[:500] + "..." if len(p) > 500 else p for p in relevant_parts[:3]
            )

        return "No directly relevant information found. Try a more specific query."


def create_rlm_limiter(
    stress_level: str = "elevated",
    compressor: Optional["HierarchicalCompressor"] = None,
) -> RLMCognitiveLoadLimiter:
    """
    Factory function to create an RLM-enhanced limiter.

    Args:
        stress_level: Current system stress level
        compressor: Optional pre-configured compressor

    Returns:
        RLMCognitiveLoadLimiter instance
    """
    limiter = RLMCognitiveLoadLimiter.for_stress_level(stress_level)
    if compressor:
        limiter._compressor = compressor
    return limiter
