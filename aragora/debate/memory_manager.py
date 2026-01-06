"""
Memory management for debates.

Extracted from Arena to improve code organization and testability.
Handles storage and retrieval of debate outcomes, evidence, and patterns
across ContinuumMemory, CritiqueStore, and DebateEmbeddings systems.
"""

import hashlib
import logging
import time
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from aragora.core import DebateResult
    from aragora.memory.continuum import ContinuumMemory
    from aragora.memory.critique_store import CritiqueStore

from aragora.memory.continuum import MemoryTier

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages debate memory operations across multiple memory systems.

    Handles:
    - ContinuumMemory: Cross-debate learning with tiered storage
    - CritiqueStore: Pattern-based learning from critiques
    - DebateEmbeddings: Similarity search for historical context
    """

    def __init__(
        self,
        continuum_memory: Optional["ContinuumMemory"] = None,
        critique_store: Optional["CritiqueStore"] = None,
        debate_embeddings=None,
        domain_extractor: Optional[Callable[[], str]] = None,
        event_emitter=None,
        spectator=None,
        loop_id: str = "",
    ):
        """Initialize memory manager with memory systems.

        Args:
            continuum_memory: ContinuumMemory instance for tiered cross-debate learning
            critique_store: CritiqueStore instance for critique patterns
            debate_embeddings: DebateEmbeddingsDatabase for similarity search
            domain_extractor: Callable that returns the current debate domain
            event_emitter: Optional event emitter for stream events
            spectator: Optional spectator stream for notifications
            loop_id: Loop ID for event scoping
        """
        self.continuum_memory = continuum_memory
        self.critique_store = critique_store
        self.debate_embeddings = debate_embeddings
        self._domain_extractor = domain_extractor
        self.event_emitter = event_emitter
        self.spectator = spectator
        self.loop_id = loop_id

        # Track retrieved memory IDs for outcome updates
        self._retrieved_ids: list = []

        # Pattern cache: (timestamp, formatted_patterns) - TTL 5 minutes
        self._patterns_cache: tuple[float, str] | None = None
        self._patterns_cache_ttl: float = 300.0  # 5 minutes

    def _get_domain(self) -> str:
        """Get current debate domain from extractor or default."""
        if self._domain_extractor:
            return self._domain_extractor()
        return "general"

    def store_debate_outcome(self, result: "DebateResult", task: str) -> None:
        """Store debate outcome in ContinuumMemory for future retrieval.

        Creates a memory entry from the winning approach to inform future debates.

        Args:
            result: The debate result to store
            task: The original debate task
        """
        if not self.continuum_memory or not result.final_answer:
            return

        try:
            # Calculate importance based on confidence and consensus
            importance = min(0.95, (result.confidence + 0.5) / 1.5)
            if result.consensus_reached:
                importance = min(1.0, importance + 0.1)

            # Determine tier based on debate quality
            # Multi-round debates with high confidence go to faster tiers
            if result.rounds_used >= 2 and result.confidence > 0.7:
                tier = MemoryTier.FAST
            elif result.rounds_used >= 1 and result.confidence > 0.5:
                tier = MemoryTier.MEDIUM
            else:
                tier = MemoryTier.SLOW

            # Store the winning approach with domain context
            domain = self._get_domain()
            memory_content = (
                f"[{domain}] Debate outcome: {result.final_answer[:300]}... "
                f"(Confidence: {result.confidence:.0%}, Rounds: {result.rounds_used})"
            )

            self.continuum_memory.add(
                id=f"debate_outcome_{result.id[:8]}",
                content=memory_content,
                tier=tier,
                importance=importance,
                metadata={
                    "debate_id": result.id,
                    "task": task[:100],
                    "domain": domain,
                    "winner": result.winner,
                    "confidence": result.confidence,
                    "consensus": result.consensus_reached,
                }
            )
            logger.info(f"  [continuum] Stored outcome as {tier}-tier memory (importance: {importance:.2f})")

        except Exception as e:
            logger.warning(f"  [continuum] Failed to store outcome: {e}")

    def store_evidence(self, evidence_snippets: list, task: str) -> None:
        """Store collected evidence snippets in ContinuumMemory for future retrieval.

        Evidence from web research and local docs is valuable for future debates
        on similar topics. This stores each unique snippet with moderate importance.

        Args:
            evidence_snippets: List of evidence snippets to store
            task: The debate task these snippets relate to
        """
        if not self.continuum_memory or not evidence_snippets:
            return

        try:
            domain = self._get_domain()
            stored_count = 0

            for snippet in evidence_snippets[:10]:  # Limit to top 10 snippets
                # Get content from snippet (handle different formats)
                content = getattr(snippet, 'content', str(snippet))[:500]
                source = getattr(snippet, 'source', 'unknown')
                relevance = getattr(snippet, 'relevance', 0.5)

                if len(content) < 50:  # Skip too-short snippets
                    continue

                # Store as medium-tier memory with moderate importance
                try:
                    self.continuum_memory.add(
                        id=f"evidence_{hashlib.sha256(content.encode()).hexdigest()[:10]}",
                        content=f"[Evidence:{domain}] {content} (Source: {source})",
                        tier="medium",
                        importance=min(0.7, relevance + 0.2),
                        metadata={
                            "task": task[:100],
                            "domain": domain,
                            "source": source,
                            "type": "evidence",
                        }
                    )
                    stored_count += 1
                except Exception as e:
                    logger.debug(f"Continuum storage error (non-fatal): {e}")

            if stored_count > 0:
                logger.info(f"  [continuum] Stored {stored_count} evidence snippets for future retrieval")

        except Exception as e:
            logger.warning(f"  [continuum] Failed to store evidence: {e}")

    def update_memory_outcomes(self, result: "DebateResult") -> None:
        """Update retrieved memories based on debate outcome.

        Implements surprise-based learning: memories that led to successful
        debates get reinforced, those that didn't get demoted.

        Args:
            result: The debate result to use for updates
        """
        if not self.continuum_memory or not self._retrieved_ids:
            return

        try:
            success = result.consensus_reached and result.confidence > 0.6
            updated_count = 0

            for mem_id in self._retrieved_ids:
                try:
                    # Update outcome with prediction error based on debate confidence
                    prediction_error = 1.0 - result.confidence if success else result.confidence
                    self.continuum_memory.update_outcome(
                        id=mem_id,
                        success=success,
                        agent_prediction_error=prediction_error,
                    )
                    updated_count += 1
                except Exception as e:
                    logger.debug(f"  [continuum] Failed to update memory {mem_id}: {e}")

            if updated_count > 0:
                logger.info(f"  [continuum] Updated {updated_count} memories with outcome (success={success})")

            # Clear tracked IDs after update
            self._retrieved_ids = []

        except Exception as e:
            logger.warning(f"  [continuum] Failed to update memory outcomes: {e}")

    async def fetch_historical_context(self, task: str, limit: int = 3) -> str:
        """Fetch similar past debates for historical context.

        This enables agents to learn from what worked (or didn't) in similar debates.

        Args:
            task: The debate task to find similar debates for
            limit: Maximum number of similar debates to retrieve

        Returns:
            Formatted string with historical context, or empty string
        """
        if not self.debate_embeddings:
            return ""

        try:
            results = await self.debate_embeddings.find_similar_debates(
                task, limit=limit, min_similarity=0.6
            )
            if not results:
                return ""

            # Emit memory_recall event for dashboard visualization ("Brain Flash")
            top_similarity = results[0][2] if results else 0
            if self.spectator:
                self._notify_spectator(
                    "memory_recall",
                    details=f"Retrieved {len(results)} similar debates (top: {top_similarity:.0%})",
                    metric=top_similarity
                )

            # Also emit to WebSocket stream for live dashboard
            if self.event_emitter:
                from aragora.server.stream import StreamEvent, StreamEventType
                self.event_emitter.emit(StreamEvent(
                    type=StreamEventType.MEMORY_RECALL,
                    loop_id=self.loop_id,
                    data={
                        "query": task,
                        "hits": [{"topic": excerpt, "similarity": round(sim, 2)} for _, excerpt, sim in results[:3]],
                        "count": len(results)
                    }
                ))

            lines = ["## HISTORICAL CONTEXT (Similar Past Debates)"]
            lines.append("Learn from these previous debates on similar topics:\n")

            for debate_id, excerpt, similarity in results:
                lines.append(f"**[{similarity:.0%} similar]** {excerpt}")
                lines.append("")  # blank line between entries

            return "\n".join(lines)
        except Exception as e:
            logger.debug(f"Historical context formatting error: {e}")
            return ""

    def get_successful_patterns(self, limit: int = 5) -> str:
        """Retrieve successful patterns from CritiqueStore memory.

        Patterns are historical argument patterns that led to consensus.
        Injecting them into debate context helps agents avoid past mistakes
        and reuse successful approaches.

        Uses a 5-minute TTL cache to avoid repeated database queries for the
        same patterns across multiple debates in a short time window.

        Args:
            limit: Maximum number of patterns to retrieve

        Returns:
            Formatted string to inject into debate context, or empty string
        """
        if not self.critique_store:
            return ""

        # Check cache first
        now = time.time()
        if self._patterns_cache is not None:
            cache_time, cached_patterns = self._patterns_cache
            if now - cache_time < self._patterns_cache_ttl:
                return cached_patterns

        try:
            # CritiqueStore.retrieve_patterns returns Pattern objects
            patterns = self.critique_store.retrieve_patterns(min_success=1, limit=limit)
            if not patterns:
                self._patterns_cache = (now, "")
                return ""

            # Convert Pattern objects to dict format and format for prompt
            result = self._format_patterns_for_prompt([
                {
                    "category": p.issue_type,
                    "pattern": f"{p.issue_text} → {p.suggestion_text}" if p.suggestion_text else p.issue_text,
                    "occurrences": p.success_count,
                    "avg_severity": p.avg_severity,
                }
                for p in patterns
            ])

            # Cache the result
            self._patterns_cache = (now, result)
            return result
        except Exception as e:
            logger.debug(f"Failed to retrieve patterns: {e}")
            return ""

    def _format_patterns_for_prompt(self, patterns: list[dict]) -> str:
        """Format learned patterns as prompt context for agents.

        Args:
            patterns: List of pattern dicts with 'category', 'pattern', 'occurrences'

        Returns:
            Formatted string to inject into debate context
        """
        if not patterns:
            return ""

        lines = ["## LEARNED PATTERNS (From Previous Debates)"]
        lines.append("Be especially careful about these recurring issues:\n")

        for p in patterns[:5]:  # Limit to top 5 patterns
            category = p.get("category", "general")
            pattern = p.get("pattern", "")
            occurrences = p.get("occurrences", 0)
            severity = p.get("avg_severity", 0)

            severity_label = ""
            if severity >= 0.7:
                severity_label = " [HIGH SEVERITY]"
            elif severity >= 0.4:
                severity_label = " [MEDIUM]"

            lines.append(f"- **{category.upper()}**{severity_label}: {pattern}")
            lines.append(f"  (Occurred in {occurrences} past debates)")

        lines.append("\nAddress these proactively to improve debate quality.")
        return "\n".join(lines)

    def _notify_spectator(self, event_type: str, details: str, metric: float = 0.0) -> None:
        """Notify spectator stream of an event."""
        if self.spectator:
            try:
                self.spectator.emit(event_type, details=details, metric=metric)
            except Exception as e:
                logger.debug(f"Spectator notification error: {e}")

    def track_retrieved_ids(self, ids: list) -> None:
        """Track retrieved memory IDs for later outcome updates.

        Args:
            ids: List of memory IDs that were retrieved
        """
        self._retrieved_ids = [i for i in ids if i]

    def clear_retrieved_ids(self) -> None:
        """Clear tracked retrieved IDs."""
        self._retrieved_ids = []
