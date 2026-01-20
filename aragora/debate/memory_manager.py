"""
Memory management for debates.

Extracted from Arena to improve code organization and testability.
Handles storage and retrieval of debate outcomes, evidence, and patterns
across ContinuumMemory, CritiqueStore, and DebateEmbeddings systems.
"""

import hashlib
import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Optional

from aragora.types.protocols import EventEmitterProtocol

from aragora.agents.errors import _build_error_action

if TYPE_CHECKING:
    from aragora.core import DebateResult
    from aragora.memory.consensus import ConsensusMemory, ConsensusStrength
    from aragora.memory.continuum import ContinuumMemory
    from aragora.memory.store import CritiqueStore
    from aragora.debate.embeddings import DebateEmbeddingsDatabase
    from aragora.spectate.stream import SpectatorStream

from aragora.memory.continuum import MemoryTier
from aragora.memory.tier_analytics import TierAnalyticsTracker

logger = logging.getLogger(__name__)


# Event types emitted by MemoryManager (for documentation and consistency)
class MemoryEventType:
    """Constants for memory-related event types."""

    MEMORY_STORED = "memory:stored"
    MEMORY_RETRIEVED = "memory:retrieved"
    MEMORY_PROMOTED = "memory:promoted"
    PATTERN_CACHED = "pattern:cached"
    PATTERN_RETRIEVED = "pattern:retrieved"


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
        consensus_memory: Optional["ConsensusMemory"] = None,
        debate_embeddings: Optional["DebateEmbeddingsDatabase"] = None,
        domain_extractor: Optional[Callable[[], str]] = None,
        event_emitter: Optional[EventEmitterProtocol] = None,
        spectator: Optional["SpectatorStream"] = None,
        loop_id: str = "",
        tier_analytics_tracker: Optional[TierAnalyticsTracker] = None,
    ) -> None:
        """Initialize memory manager with memory systems.

        Args:
            continuum_memory: ContinuumMemory instance for tiered cross-debate learning
            critique_store: CritiqueStore instance for critique patterns
            consensus_memory: ConsensusMemory instance for consensus/dissent records
            debate_embeddings: DebateEmbeddingsDatabase for similarity search
            domain_extractor: Callable that returns the current debate domain
            event_emitter: Optional event emitter for stream events
            spectator: Optional spectator stream for notifications
            loop_id: Loop ID for event scoping
            tier_analytics_tracker: Optional TierAnalyticsTracker for ROI tracking
        """
        self.continuum_memory = continuum_memory
        self.critique_store = critique_store
        self.consensus_memory = consensus_memory
        self.debate_embeddings = debate_embeddings
        self._domain_extractor = domain_extractor
        self.event_emitter = event_emitter
        self.spectator = spectator
        self.loop_id = loop_id
        self.tier_analytics_tracker = tier_analytics_tracker

        # Track retrieved memory IDs for outcome updates
        self._retrieved_ids: list[str] = []
        # Track tier info for analytics
        self._retrieved_tiers: dict[str, MemoryTier] = {}

        # Pattern cache: (timestamp, formatted_patterns) - TTL 5 minutes
        self._patterns_cache: tuple[float, str] | None = None
        self._patterns_cache_ttl: float = 300.0  # 5 minutes

    def _emit_event(self, event_type: str, **data: Any) -> None:
        """Emit a memory event if event_emitter is configured.

        Args:
            event_type: The event type (see MemoryEventType constants)
            **data: Event data to include
        """
        if self.event_emitter is None:
            return
        try:
            # Use emit_sync if available (SyncEventEmitter), otherwise emit
            emit_fn = getattr(self.event_emitter, "emit_sync", None)
            if emit_fn is None:
                emit_fn = getattr(self.event_emitter, "emit", None)
            if emit_fn is not None:
                emit_fn(event_type, loop_id=self.loop_id, **data)
        except (AttributeError, TypeError) as e:
            # Expected: emitter missing method or wrong signature
            logger.debug(f"Failed to emit memory event {event_type}: {e}")
        except Exception as e:
            # Unexpected error in event emission
            logger.warning(f"Unexpected error emitting memory event {event_type}: {e}")

    def _get_domain(self) -> str:
        """Get current debate domain from extractor or default."""
        if self._domain_extractor:
            return self._domain_extractor()
        return "general"

    def store_debate_outcome(
        self,
        result: "DebateResult",
        task: str,
        belief_cruxes: Optional[list[str]] = None,
    ) -> None:
        """Store debate outcome in ContinuumMemory for future retrieval.

        Creates a memory entry from the winning approach to inform future debates.

        Args:
            result: The debate result to store
            task: The original debate task
            belief_cruxes: Optional list of identified belief cruxes to store in metadata
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

            # Build metadata with optional crux claims
            metadata = {
                "debate_id": result.id,
                "task": task[:100],
                "domain": domain,
                "winner": result.winner,
                "confidence": result.confidence,
                "consensus": result.consensus_reached,
            }
            if belief_cruxes:
                metadata["crux_claims"] = belief_cruxes[:10]  # Limit to 10 cruxes

            memory_id = f"debate_outcome_{result.id[:8]}"
            self.continuum_memory.add(
                id=memory_id,
                content=memory_content,
                tier=tier,
                importance=importance,
                metadata=metadata,
            )
            logger.info(
                f"  [continuum] Stored outcome as {tier}-tier memory (importance: {importance:.2f})"
            )

            # Emit memory stored event
            self._emit_event(
                MemoryEventType.MEMORY_STORED,
                memory_id=memory_id,
                tier=tier.value if hasattr(tier, "value") else str(tier),
                importance=importance,
                domain=domain,
                debate_id=result.id,
            )

        except (AttributeError, TypeError, ValueError) as e:
            # Expected: memory system configuration or data format issues
            logger.warning(f"  [continuum] Failed to store outcome: {e}")
        except Exception as e:
            # Unexpected error - log with full context
            _, msg, exc_info = _build_error_action(e, "continuum")
            logger.exception(f"  [continuum] Unexpected error storing outcome: {msg}")

    def store_consensus_record(
        self,
        result: "DebateResult",
        task: str,
        belief_cruxes: Optional[list[str]] = None,
    ) -> None:
        """Store debate consensus and dissents in ConsensusMemory.

        This enables the DissentRetriever to find relevant historical dissents
        for future debates on similar topics.

        Args:
            result: The debate result containing votes and outcomes
            task: The original debate task/topic
            belief_cruxes: Optional list of identified crux claims to store
        """
        if not self.consensus_memory or not result.final_answer:
            return

        try:
            # Determine strength from confidence
            strength = self._confidence_to_strength(result.confidence)

            # Extract agreeing/dissenting agents from votes
            agreeing_agents = []
            dissenting_agents = []
            for vote in getattr(result, "votes", []):
                agent_name = getattr(vote, "agent", None)
                if not agent_name:
                    continue
                # Check if vote supports consensus (vote.choice matches winner or high confidence)
                supports = getattr(vote, "supports_consensus", None)
                if supports is None:
                    # Fallback: check if vote.choice matches winner
                    supports = getattr(vote, "choice", "") == result.winner
                if supports:
                    agreeing_agents.append(agent_name)
                else:
                    dissenting_agents.append(agent_name)

            # Get participating agents
            participating = [a.name for a in getattr(result, "agents", [])]
            if not participating:
                participating = agreeing_agents + dissenting_agents

            # Extract key claims from grounded verdict if available
            key_claims = []
            if belief_cruxes:
                key_claims = belief_cruxes[:10]  # Limit to top 10
            elif hasattr(result, "grounded_verdict") and result.grounded_verdict:
                claims = getattr(result.grounded_verdict, "claims", [])
                key_claims = [c.statement for c in claims[:5] if hasattr(c, "statement")]

            # Store consensus record
            domain = self._get_domain()
            record = self.consensus_memory.store_consensus(
                topic=task,
                conclusion=result.final_answer[:2000],  # Limit length
                strength=strength,
                confidence=result.confidence,
                participating_agents=participating,
                agreeing_agents=agreeing_agents,
                dissenting_agents=dissenting_agents,
                key_claims=key_claims,
                domain=domain,
                rounds=result.rounds_used,
                metadata={
                    "debate_id": result.id,
                    "winner": result.winner,
                    "consensus_reached": result.consensus_reached,
                    "crux_claims": belief_cruxes or [],
                },
            )

            logger.info(
                f"  [consensus] Stored record: {strength.value} consensus, "
                f"{len(agreeing_agents)} agreed, {len(dissenting_agents)} dissented"
            )

            # Store individual dissents for each dissenting agent
            for agent_name in dissenting_agents:
                self._store_agent_dissent(record.id, agent_name, result, task)

        except (AttributeError, TypeError, ValueError) as e:
            # Expected: consensus memory configuration or data format issues
            logger.warning(f"  [consensus] Failed to store record: {e}")
        except Exception as e:
            # Unexpected error - log with full context
            _, msg, exc_info = _build_error_action(e, "consensus")
            logger.exception(f"  [consensus] Unexpected error storing record: {msg}")

    def _confidence_to_strength(self, confidence: float) -> "ConsensusStrength":
        """Convert confidence score to ConsensusStrength enum."""
        from aragora.memory.consensus import ConsensusStrength

        if confidence >= 0.95:
            return ConsensusStrength.UNANIMOUS
        elif confidence >= 0.8:
            return ConsensusStrength.STRONG
        elif confidence >= 0.6:
            return ConsensusStrength.MODERATE
        elif confidence >= 0.5:
            return ConsensusStrength.WEAK
        elif confidence >= 0.3:
            return ConsensusStrength.SPLIT
        else:
            return ConsensusStrength.CONTESTED

    def _store_agent_dissent(
        self,
        consensus_id: str,
        agent_name: str,
        result: "DebateResult",
        task: str,
    ) -> None:
        """Store a dissent record for an agent that disagreed with consensus.

        Args:
            consensus_id: ID of the consensus record
            agent_name: Name of the dissenting agent
            result: The debate result
            task: The debate task
        """
        if not self.consensus_memory:
            return

        try:
            from aragora.memory.consensus import DissentType

            # Find the agent's last message to extract their reasoning
            agent_content = ""
            for msg in reversed(getattr(result, "messages", [])):
                if getattr(msg, "agent", None) == agent_name:
                    agent_content = getattr(msg, "content", "")[:500]
                    break

            # Find agent's vote for confidence
            agent_confidence = 0.5
            for vote in getattr(result, "votes", []):
                if getattr(vote, "agent", None) == agent_name:
                    agent_confidence = getattr(vote, "confidence", 0.5)
                    break

            # Determine dissent type based on confidence
            if agent_confidence >= 0.8:
                dissent_type = DissentType.FUNDAMENTAL_DISAGREEMENT
            elif agent_confidence >= 0.6:
                dissent_type = DissentType.ALTERNATIVE_APPROACH
            elif agent_confidence >= 0.4:
                dissent_type = DissentType.EDGE_CASE_CONCERN
            else:
                dissent_type = DissentType.MINOR_QUIBBLE

            self.consensus_memory.store_dissent(
                debate_id=consensus_id,
                agent_id=agent_name,
                dissent_type=dissent_type,
                content=agent_content or f"{agent_name} disagreed with the consensus",
                reasoning=f"Agent voted against consensus on: {task[:100]}",
                confidence=agent_confidence,
            )

            logger.debug(f"  [consensus] Stored dissent for {agent_name}")

        except (AttributeError, TypeError, ValueError, KeyError) as e:
            # Expected: missing data or format issues in dissent storage
            logger.debug(f"  [consensus] Failed to store dissent for {agent_name}: {e}")
        except Exception as e:
            # Unexpected error - log with more detail
            logger.warning(f"  [consensus] Unexpected error storing dissent for {agent_name}: {e}")

    def store_evidence(self, evidence_snippets: list, task: str) -> None:
        """Store collected evidence snippets in ContinuumMemory for future retrieval.

        Evidence from web research and local docs is valuable for future debates
        on similar topics. This stores each unique snippet with moderate importance.

        Also registers evidence with the EvidenceProvenanceBridge for provenance tracking.

        Args:
            evidence_snippets: List of evidence snippets to store
            task: The debate task these snippets relate to
        """
        if not self.continuum_memory or not evidence_snippets:
            return

        try:
            domain = self._get_domain()
            stored_count = 0

            # Get evidence bridge for provenance tracking (optional)
            evidence_bridge = None
            try:
                from aragora.reasoning.evidence_bridge import get_evidence_bridge
                evidence_bridge = get_evidence_bridge()
            except ImportError:
                pass  # Bridge not available

            for snippet in evidence_snippets[:10]:  # Limit to top 10 snippets
                # Get content from snippet (handle different formats)
                content = getattr(snippet, "content", str(snippet))[:500]
                source = getattr(snippet, "source", "unknown")
                relevance = getattr(snippet, "relevance", 0.5)

                if len(content) < 50:  # Skip too-short snippets
                    continue

                evidence_id = f"evidence_{hashlib.sha256(content.encode()).hexdigest()[:10]}"

                # Store as medium-tier memory with moderate importance
                try:
                    self.continuum_memory.add(
                        id=evidence_id,
                        content=f"[Evidence:{domain}] {content} (Source: {source})",
                        tier=MemoryTier.MEDIUM,
                        importance=min(0.7, relevance + 0.2),
                        metadata={
                            "task": task[:100],
                            "domain": domain,
                            "source": source,
                            "type": "evidence",
                        },
                    )
                    stored_count += 1

                    # Register with evidence bridge for provenance tracking
                    if evidence_bridge and hasattr(snippet, "id"):
                        try:
                            evidence_bridge.register_evidence(snippet)
                        except Exception as e:
                            logger.debug(f"Evidence bridge registration (non-fatal): {e}")

                except (AttributeError, TypeError, ValueError) as e:
                    # Expected: data format or memory configuration issues
                    logger.debug(f"Continuum storage error (non-fatal): {e}")
                except Exception as e:
                    # Unexpected error - still non-fatal but log with more detail
                    logger.warning(f"Continuum storage unexpected error (non-fatal): {e}")

            if stored_count > 0:
                logger.info(
                    f"  [continuum] Stored {stored_count} evidence snippets for future retrieval"
                )
                # Emit EVIDENCE_FOUND event for real-time panel updates
                self._emit_evidence_found(
                    stored_count, domain, task, evidence_snippets[:stored_count]
                )

        except (AttributeError, TypeError, ValueError) as e:
            # Expected: evidence format or memory configuration issues
            logger.warning(f"  [continuum] Failed to store evidence: {e}")
        except Exception as e:
            # Unexpected error - log with full context
            _, msg, exc_info = _build_error_action(e, "continuum")
            logger.exception(f"  [continuum] Unexpected error storing evidence: {msg}")

    def _emit_evidence_found(
        self,
        count: int,
        domain: str,
        task: str,
        snippets: list,
    ) -> None:
        """Emit EVIDENCE_FOUND event to WebSocket."""
        if not self.event_emitter:
            return

        try:
            from aragora.server.stream import StreamEvent, StreamEventType

            # Build snippet summaries for the event
            snippet_summaries = []
            for snippet in snippets[:5]:  # Limit to 5 in event
                content = getattr(snippet, "content", str(snippet))[:150]
                source = getattr(snippet, "source", "unknown")
                snippet_summaries.append(
                    {
                        "content": content,
                        "source": source,
                    }
                )

            self.event_emitter.emit(
                StreamEvent(
                    type=StreamEventType.EVIDENCE_FOUND,
                    loop_id=self.loop_id,
                    data={
                        "count": count,
                        "domain": domain,
                        "task": task[:100],
                        "snippets": snippet_summaries,
                    },
                )
            )
        except ImportError as e:
            # Expected: stream module not available
            logger.debug(f"Evidence event emission skipped (module unavailable): {e}")
        except (AttributeError, TypeError) as e:
            # Expected: emitter method or signature issues
            logger.debug(f"Evidence event emission error: {e}")
        except Exception as e:
            # Unexpected error
            logger.warning(f"Unexpected evidence event emission error: {e}")

    def update_memory_outcomes(self, result: "DebateResult") -> None:
        """Update retrieved memories based on debate outcome.

        Implements surprise-based learning: memories that led to successful
        debates get reinforced, those that didn't get demoted.

        Also records usage analytics for tier ROI tracking.

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

                    # Record usage for tier analytics if tracker available
                    if self.tier_analytics_tracker and mem_id in self._retrieved_tiers:
                        try:
                            # quality_before: neutral baseline (0.5)
                            # quality_after: debate outcome confidence
                            self.tier_analytics_tracker.record_usage(
                                memory_id=mem_id,
                                tier=self._retrieved_tiers[mem_id],
                                debate_id=result.id,
                                quality_before=0.5,
                                quality_after=result.confidence if success else 0.3,
                            )
                        except (AttributeError, TypeError, ValueError) as e:
                            # Expected: tier analytics configuration issues
                            logger.debug(
                                f"  [tier_analytics] Failed to record usage for {mem_id}: {e}"
                            )
                        except Exception as e:
                            # Unexpected error
                            logger.warning(
                                f"  [tier_analytics] Unexpected error recording usage for {mem_id}: {e}"
                            )

                except (AttributeError, TypeError, ValueError, KeyError) as e:
                    # Expected: memory update configuration or data issues
                    logger.debug(f"  [continuum] Failed to update memory {mem_id}: {e}")
                except Exception as e:
                    # Unexpected error
                    logger.warning(f"  [continuum] Unexpected error updating memory {mem_id}: {e}")

            if updated_count > 0:
                logger.info(
                    f"  [continuum] Updated {updated_count} memories with outcome (success={success})"
                )

            # Clear tracked IDs and tiers after update
            self._retrieved_ids = []
            self._retrieved_tiers = {}

        except (AttributeError, TypeError, ValueError) as e:
            # Expected: memory configuration or data format issues
            logger.warning(f"  [continuum] Failed to update memory outcomes: {e}")
        except Exception as e:
            # Unexpected error - log with full context
            _, msg, exc_info = _build_error_action(e, "continuum")
            logger.exception(f"  [continuum] Unexpected error updating memory outcomes: {msg}")

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
                    metric=top_similarity,
                )

            # Also emit to WebSocket stream for live dashboard
            if self.event_emitter:
                from aragora.server.stream import StreamEvent, StreamEventType

                self.event_emitter.emit(
                    StreamEvent(
                        type=StreamEventType.MEMORY_RECALL,
                        loop_id=self.loop_id,
                        data={
                            "query": task,
                            "hits": [
                                {"topic": excerpt, "similarity": round(sim, 2)}
                                for _, excerpt, sim in results[:3]
                            ],
                            "count": len(results),
                        },
                    )
                )

            lines = ["## HISTORICAL CONTEXT (Similar Past Debates)"]
            lines.append("Learn from these previous debates on similar topics:\n")

            for debate_id, excerpt, similarity in results:
                lines.append(f"**[{similarity:.0%} similar]** {excerpt}")
                lines.append("")  # blank line between entries

            return "\n".join(lines)
        except (AttributeError, TypeError, ValueError, KeyError) as e:
            # Expected: embedding search or formatting issues
            logger.debug(f"Historical context retrieval error: {e}")
            return ""
        except Exception as e:
            # Unexpected error
            logger.warning(f"Unexpected historical context error: {e}")
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
            result = self._format_patterns_for_prompt(
                [
                    {
                        "category": p.issue_type,
                        "pattern": (
                            f"{p.issue_text} → {p.suggestion_text}"
                            if p.suggestion_text
                            else p.issue_text
                        ),
                        "occurrences": p.success_count,
                        "avg_severity": p.avg_severity,
                    }
                    for p in patterns
                ]
            )

            # Cache the result
            self._patterns_cache = (now, result)
            return result
        except (AttributeError, TypeError, ValueError) as e:
            # Expected: critique store configuration or data issues
            logger.debug(f"Failed to retrieve patterns: {e}")
            return ""
        except Exception as e:
            # Unexpected error
            logger.warning(f"Unexpected error retrieving patterns: {e}")
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
            except (AttributeError, TypeError) as e:
                # Expected: spectator method or signature issues
                logger.debug(f"Spectator notification error: {e}")
            except Exception as e:
                # Unexpected error
                logger.warning(f"Unexpected spectator notification error: {e}")

    def track_retrieved_ids(
        self,
        ids: list[str],
        tiers: Optional[dict[str, MemoryTier]] = None,
    ) -> None:
        """Track retrieved memory IDs for later outcome updates.

        Args:
            ids: List of memory IDs that were retrieved
            tiers: Optional dict mapping memory ID to its tier (for analytics)
        """
        self._retrieved_ids = [i for i in ids if i]
        self._retrieved_tiers = tiers or {}

    def clear_retrieved_ids(self) -> None:
        """Clear tracked retrieved IDs and tier info."""
        self._retrieved_ids = []
        self._retrieved_tiers = {}

    @property
    def retrieved_ids(self) -> list[str]:
        """Get list of currently tracked memory IDs."""
        return self._retrieved_ids.copy()
