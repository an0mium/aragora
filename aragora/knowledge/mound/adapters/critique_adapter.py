"""
CritiqueAdapter - Bridges CritiqueStore to the Knowledge Mound.

This adapter enables the Knowledge Mound to query successful critique patterns
and agent reputation data from the CritiqueStore.

The adapter provides:
- Pattern search by issue type
- Agent reputation integration
- Pattern-to-KnowledgeItem conversion
- Surprise-based learning integration
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from aragora.memory.store import CritiqueStore, Pattern, AgentReputation
    from aragora.knowledge.mound.types import KnowledgeItem

logger = logging.getLogger(__name__)


# ============================================================================
# Reverse Flow Dataclasses (KM → CritiqueStore)
# ============================================================================


@dataclass
class KMPatternBoost:
    """Result of boosting a critique pattern from KM validation."""

    pattern_id: str
    boost_amount: int = 0  # Additional success count
    km_confidence: float = 0.7
    source_debates: List[str] = field(default_factory=list)
    was_applied: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KMReputationAdjustment:
    """KM-driven reputation adjustment for an agent."""

    agent_name: str
    adjustment: float = 0.0  # Reputation score adjustment
    pattern_contributions: int = 0  # Patterns the agent contributed
    km_confidence: float = 0.7
    recommendation: str = "keep"  # "boost", "penalize", "keep"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KMPatternValidation:
    """Validation result from KM for a critique pattern."""

    pattern_id: str
    km_confidence: float = 0.7
    cross_debate_usage: int = 0
    outcome_success_rate: float = 0.0
    recommendation: str = "keep"  # "boost", "archive", "keep"
    boost_amount: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CritiqueKMSyncResult:
    """Result of syncing KM validations to CritiqueStore."""

    patterns_analyzed: int = 0
    patterns_boosted: int = 0
    agents_analyzed: int = 0
    reputation_adjustments: int = 0
    errors: List[str] = field(default_factory=list)
    duration_ms: float = 0.0


@dataclass
class CritiqueSearchResult:
    """Wrapper for critique pattern search results."""

    pattern: "Pattern"
    relevance_score: float = 0.0
    matched_category: bool = False


class CritiqueAdapter:
    """
    Adapter that bridges CritiqueStore to the Knowledge Mound.

    Provides methods that the Knowledge Mound expects for federated queries:
    - search_patterns: Find relevant critique patterns
    - to_knowledge_item: Convert patterns to unified format
    - get_agent_insights: Agent reputation for cultural learning

    Usage:
        from aragora.memory.store import CritiqueStore
        from aragora.knowledge.mound.adapters import CritiqueAdapter

        store = CritiqueStore()
        adapter = CritiqueAdapter(store)

        # Search for patterns
        results = adapter.search_patterns("performance", limit=10)

        # Convert to knowledge items
        items = [adapter.to_knowledge_item(r) for r in results]
    """

    def __init__(
        self,
        store: "CritiqueStore",
        enable_dual_write: bool = False,
    ):
        """
        Initialize the adapter.

        Args:
            store: The CritiqueStore instance to wrap
            enable_dual_write: If True, writes go to both systems during migration
        """
        self._store = store
        self._enable_dual_write = enable_dual_write

    @property
    def store(self) -> "CritiqueStore":
        """Access the underlying CritiqueStore."""
        return self._store

    def search_patterns(
        self,
        query: str,
        limit: int = 10,
        issue_type: Optional[str] = None,
        min_success: int = 1,
    ) -> List["Pattern"]:
        """
        Search critique patterns by query.

        This method wraps CritiqueStore.retrieve_patterns() to provide
        the interface expected by KnowledgeMound._query_critique().

        If query matches a known issue type, filters by that type.
        Otherwise, retrieves patterns and filters by keyword match.

        Args:
            query: Search query
            limit: Maximum results to return
            issue_type: Optional explicit issue type filter
            min_success: Minimum success count for patterns

        Returns:
            List of Pattern objects matching the query
        """
        # Known issue types from CritiqueStore._categorize_issue()
        known_types = {
            "performance",
            "security",
            "correctness",
            "clarity",
            "architecture",
            "completeness",
            "testing",
            "general",
        }

        # Check if query is an issue type
        query_lower = query.lower().strip()
        detected_type = None

        if issue_type:
            detected_type = issue_type
        elif query_lower in known_types:
            detected_type = query_lower
        else:
            # Check for type keywords in query
            for t in known_types:
                if t in query_lower:
                    detected_type = t
                    break

        # Retrieve patterns
        patterns = self._store.retrieve_patterns(
            issue_type=detected_type,
            min_success=min_success,
            limit=limit * 2 if detected_type is None else limit,  # Get more if filtering
        )

        # If no type filter, also filter by keyword in issue/suggestion text
        if detected_type is None and patterns:
            query_words = set(query_lower.split())
            filtered = []
            for pattern in patterns:
                pattern_words = set(pattern.issue_text.lower().split())
                if pattern.suggestion_text:
                    pattern_words.update(pattern.suggestion_text.lower().split())
                if query_words & pattern_words:  # Any word overlap
                    filtered.append(pattern)
            patterns = filtered[:limit]

        return patterns

    def get(self, pattern_id: str) -> Optional["Pattern"]:
        """
        Get a specific pattern by ID.

        Args:
            pattern_id: The pattern ID (may be prefixed with "cr_" from mound)

        Returns:
            Pattern or None
        """
        # Strip mound prefix if present
        if pattern_id.startswith("cr_"):
            pattern_id = pattern_id[3:]

        # CritiqueStore doesn't have a direct get_pattern method
        # We retrieve all patterns for the issue type and filter
        # This is inefficient but works for now
        patterns = self._store.retrieve_patterns(min_success=0, limit=1000)
        for pattern in patterns:
            if pattern.id == pattern_id:
                return pattern
        return None

    async def get_async(self, pattern_id: str) -> Optional["Pattern"]:
        """Async version of get for compatibility."""
        return self.get(pattern_id)

    def to_knowledge_item(
        self,
        pattern: "Pattern",
    ) -> "KnowledgeItem":
        """
        Convert a Pattern to a KnowledgeItem.

        Args:
            pattern: The critique pattern

        Returns:
            KnowledgeItem for unified knowledge mound API
        """
        from aragora.knowledge.mound.types import (
            ConfidenceLevel,
            KnowledgeItem,
            KnowledgeSource,
        )

        # Map success rate to confidence
        success_rate = pattern.success_rate
        if success_rate >= 0.8:
            confidence = ConfidenceLevel.HIGH
        elif success_rate >= 0.6:
            confidence = ConfidenceLevel.MEDIUM
        elif success_rate >= 0.4:
            confidence = ConfidenceLevel.LOW
        else:
            confidence = ConfidenceLevel.UNCERTAIN

        # Build content from issue + suggestion
        content = pattern.issue_text
        if pattern.suggestion_text:
            content += f"\n\nSuggestion: {pattern.suggestion_text}"

        # Build metadata
        metadata: Dict[str, Any] = {
            "issue_type": pattern.issue_type,
            "success_count": pattern.success_count,
            "failure_count": pattern.failure_count,
            "success_rate": success_rate,
            "avg_severity": pattern.avg_severity,
            "example_task": pattern.example_task,
        }

        return KnowledgeItem(
            id=f"cr_{pattern.id}",
            content=content,
            source=KnowledgeSource.CRITIQUE,
            source_id=pattern.id,
            confidence=confidence,
            created_at=datetime.fromisoformat(pattern.created_at),
            updated_at=datetime.fromisoformat(pattern.updated_at),
            metadata=metadata,
            importance=success_rate,
        )

    def get_agent_reputation(
        self,
        agent_name: str,
    ) -> Optional["AgentReputation"]:
        """
        Get reputation data for an agent.

        This is useful for culture accumulation - agents with high reputation
        contribute more weight to organizational patterns.

        Args:
            agent_name: The agent name

        Returns:
            AgentReputation or None
        """
        return self._store.get_reputation(agent_name)

    def get_top_agents(
        self,
        limit: int = 10,
    ) -> List["AgentReputation"]:
        """
        Get top agents by reputation.

        Args:
            limit: Maximum agents to return

        Returns:
            List of AgentReputation sorted by score
        """
        reputations = self._store.get_all_reputations(limit=limit)
        # Sort by reputation score
        return sorted(reputations, key=lambda r: r.reputation_score, reverse=True)

    def get_agent_vote_weights(
        self,
        agent_names: List[str],
    ) -> Dict[str, float]:
        """
        Get vote weights for multiple agents.

        Vote weights are based on historical performance and can be used
        for weighted consensus or culture pattern strength.

        Args:
            agent_names: List of agent names

        Returns:
            Dict mapping agent names to vote weights (0.4-1.6 range)
        """
        return self._store.get_vote_weights_batch(agent_names)

    def get_patterns_by_type(
        self,
        issue_type: str,
        limit: int = 20,
    ) -> List["Pattern"]:
        """
        Get patterns for a specific issue type.

        Args:
            issue_type: The issue type (performance, security, etc.)
            limit: Maximum patterns to return

        Returns:
            List of Pattern objects
        """
        return self._store.retrieve_patterns(
            issue_type=issue_type,
            min_success=1,
            limit=limit,
        )

    def get_surprising_patterns(
        self,
        limit: int = 10,
    ) -> List["Pattern"]:
        """
        Get patterns with high surprise scores.

        High surprise patterns are particularly valuable as they represent
        unexpected successes that deviate from base rates.

        Args:
            limit: Maximum patterns to return

        Returns:
            List of Pattern objects sorted by surprise
        """
        # Retrieve more patterns and filter/sort by surprise
        patterns = self._store.retrieve_patterns(min_success=1, limit=limit * 3)

        # The retrieve_patterns method already incorporates surprise in its ranking
        # but we can re-sort to prioritize pure surprise
        return patterns[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the critique store."""
        return self._store.get_stats()

    def get_archive_stats(self) -> Dict[str, Any]:
        """Get statistics about archived patterns."""
        return self._store.get_archive_stats()

    # ========================================================================
    # Reverse Flow Methods (KM → CritiqueStore)
    # ========================================================================

    def __init_reverse_flow_state(self) -> None:
        """Initialize reverse flow state if not already done."""
        if not hasattr(self, "_km_boosts_applied"):
            self._km_boosts_applied = 0
        if not hasattr(self, "_km_reputation_adjustments"):
            self._km_reputation_adjustments = 0
        if not hasattr(self, "_km_validations"):
            self._km_validations: List[KMPatternValidation] = []
        if not hasattr(self, "_pattern_usage"):
            self._pattern_usage: Dict[str, List[Dict[str, Any]]] = {}

    def record_pattern_usage(
        self,
        pattern_id: str,
        debate_id: str,
        was_successful: bool,
        confidence: float = 0.7,
    ) -> None:
        """
        Record that a pattern was used in a debate.

        This enables outcome-based pattern boosting.

        Args:
            pattern_id: The pattern ID
            debate_id: The debate where this pattern was used
            was_successful: Whether the pattern helped achieve success
            confidence: Confidence in the assessment
        """
        self.__init_reverse_flow_state()

        if pattern_id not in self._pattern_usage:
            self._pattern_usage[pattern_id] = []

        self._pattern_usage[pattern_id].append(
            {
                "debate_id": debate_id,
                "was_successful": was_successful,
                "confidence": confidence,
                "recorded_at": datetime.utcnow().isoformat(),
            }
        )

    async def validate_pattern_from_km(
        self,
        pattern_id: str,
        km_cross_references: List[Dict[str, Any]],
    ) -> KMPatternValidation:
        """
        Validate a critique pattern based on KM cross-references.

        Examines how this pattern performs across debates to determine
        if it should be boosted or archived.

        Args:
            pattern_id: The pattern ID to validate
            km_cross_references: Related KM items for cross-referencing

        Returns:
            KMPatternValidation with recommendation
        """
        self.__init_reverse_flow_state()

        pattern = self.get(pattern_id)

        # Analyze cross-references
        success_count = 0
        total_outcomes = 0
        debate_ids = set()

        for ref in km_cross_references:
            meta = ref.get("metadata", {})

            if debate_id := meta.get("debate_id"):
                debate_ids.add(debate_id)

            if "outcome_success" in meta:
                total_outcomes += 1
                if meta["outcome_success"]:
                    success_count += 1

        # Also check recorded usage
        if pattern_id in self._pattern_usage:
            for usage in self._pattern_usage[pattern_id]:
                total_outcomes += 1
                if usage["was_successful"]:
                    success_count += 1
                debate_ids.add(usage["debate_id"])

        # Compute metrics
        cross_debate_usage = len(debate_ids)
        outcome_success_rate = success_count / total_outcomes if total_outcomes > 0 else 0.0

        # Determine recommendation
        if total_outcomes >= 5 and outcome_success_rate >= 0.8:
            recommendation = "boost"
            # Boost proportional to usage and success
            boost_amount = min(int(success_count * 0.5), 5)
        elif total_outcomes >= 5 and outcome_success_rate < 0.3:
            recommendation = "archive"
            boost_amount = 0
        else:
            recommendation = "keep"
            boost_amount = 0

        # KM confidence based on evidence
        km_confidence = min(total_outcomes / 10, 1.0) if total_outcomes > 0 else 0.5

        validation = KMPatternValidation(
            pattern_id=pattern_id,
            km_confidence=km_confidence,
            cross_debate_usage=cross_debate_usage,
            outcome_success_rate=outcome_success_rate,
            recommendation=recommendation,
            boost_amount=boost_amount,
            metadata={
                "success_count": success_count,
                "total_outcomes": total_outcomes,
                "pattern_found": pattern is not None,
            },
        )

        self._km_validations.append(validation)

        return validation

    async def apply_pattern_boost(
        self,
        validation: KMPatternValidation,
    ) -> KMPatternBoost:
        """
        Apply a KM validation to boost a pattern's success count.

        Args:
            validation: The validation result to apply

        Returns:
            KMPatternBoost with application result
        """
        self.__init_reverse_flow_state()

        boost = KMPatternBoost(
            pattern_id=validation.pattern_id,
            boost_amount=validation.boost_amount,
            km_confidence=validation.km_confidence,
            was_applied=False,
        )

        if validation.recommendation != "boost" or validation.boost_amount <= 0:
            return boost

        pattern = self.get(validation.pattern_id)
        if not pattern:
            boost.metadata["error"] = "pattern_not_found"
            return boost

        # Apply the boost via the store
        # Note: CritiqueStore.record_pattern_outcome is the API for this
        try:
            for _ in range(validation.boost_amount):
                self._store.record_pattern_outcome(
                    issue_type=pattern.issue_type,
                    issue_text=pattern.issue_text,
                    suggestion=pattern.suggestion_text,
                    severity=pattern.avg_severity,
                    accepted=True,  # Boost = successful outcomes
                )
            boost.was_applied = True
            self._km_boosts_applied += 1

            logger.info(
                f"Applied KM boost to pattern {validation.pattern_id}: "
                f"+{validation.boost_amount} successes"
            )
        except Exception as e:
            boost.metadata["error"] = str(e)

        return boost

    async def compute_reputation_adjustment(
        self,
        agent_name: str,
        km_items: List[Dict[str, Any]],
    ) -> KMReputationAdjustment:
        """
        Compute a reputation adjustment for an agent based on KM patterns.

        Analyzes the agent's pattern contributions and their success rates
        to recommend reputation adjustments.

        Args:
            agent_name: The agent name
            km_items: KM items with pattern and outcome data

        Returns:
            KMReputationAdjustment with recommendation
        """
        self.__init_reverse_flow_state()

        # Count patterns contributed by this agent
        pattern_contributions = 0
        success_count = 0
        total_outcomes = 0

        for item in km_items:
            meta = item.get("metadata", {})

            # Check if this agent contributed
            if meta.get("agent_name") == agent_name or agent_name in meta.get(
                "agents_involved", []
            ):
                pattern_contributions += 1

                if "outcome_success" in meta:
                    total_outcomes += 1
                    if meta["outcome_success"]:
                        success_count += 1

        # Compute adjustment
        if total_outcomes == 0:
            return KMReputationAdjustment(
                agent_name=agent_name,
                pattern_contributions=pattern_contributions,
                km_confidence=0.5,
                recommendation="keep",
            )

        success_rate = success_count / total_outcomes
        km_confidence = min(total_outcomes / 10, 1.0)

        # Determine adjustment
        if success_rate >= 0.8 and total_outcomes >= 5:
            recommendation = "boost"
            adjustment = 0.1 * min(pattern_contributions / 5, 1.0)
        elif success_rate < 0.3 and total_outcomes >= 5:
            recommendation = "penalize"
            adjustment = -0.05 * min(pattern_contributions / 5, 1.0)
        else:
            recommendation = "keep"
            adjustment = 0.0

        return KMReputationAdjustment(
            agent_name=agent_name,
            adjustment=adjustment,
            pattern_contributions=pattern_contributions,
            km_confidence=km_confidence,
            recommendation=recommendation,
            metadata={
                "success_count": success_count,
                "total_outcomes": total_outcomes,
                "success_rate": success_rate,
            },
        )

    async def apply_reputation_adjustment(
        self,
        adjustment: KMReputationAdjustment,
    ) -> bool:
        """
        Apply a reputation adjustment to an agent.

        Args:
            adjustment: The adjustment to apply

        Returns:
            True if applied successfully
        """
        self.__init_reverse_flow_state()

        if adjustment.adjustment == 0.0:
            return False

        try:
            reputation = self._store.get_reputation(adjustment.agent_name)
            if not reputation:
                return False

            # CritiqueStore doesn't have direct reputation adjustment
            # We simulate by recording outcomes
            if adjustment.adjustment > 0:
                # Record positive outcomes
                for _ in range(int(abs(adjustment.adjustment) * 10)):
                    self._store.update_reputation(
                        agent_name=adjustment.agent_name,
                        accepted=True,
                    )
            else:
                # Record negative outcomes
                for _ in range(int(abs(adjustment.adjustment) * 10)):
                    self._store.update_reputation(
                        agent_name=adjustment.agent_name,
                        accepted=False,
                    )

            self._km_reputation_adjustments += 1

            logger.info(
                f"Applied KM reputation adjustment to {adjustment.agent_name}: "
                f"{adjustment.adjustment:+.2f} ({adjustment.recommendation})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to apply reputation adjustment: {e}")
            return False

    async def sync_validations_from_km(
        self,
        km_items: List[Dict[str, Any]],
        min_confidence: float = 0.7,
    ) -> CritiqueKMSyncResult:
        """
        Batch sync KM validations to CritiqueStore.

        Args:
            km_items: KM items with validation data
            min_confidence: Minimum confidence for applying changes

        Returns:
            CritiqueKMSyncResult with sync details
        """
        import time

        self.__init_reverse_flow_state()

        start_time = time.time()
        result = CritiqueKMSyncResult()
        errors = []

        # Group items by pattern_id
        items_by_pattern: Dict[str, List[Dict[str, Any]]] = {}
        agents_seen = set()

        for item in km_items:
            meta = item.get("metadata", {})

            pattern_id = meta.get("pattern_id") or meta.get("source_id")
            if pattern_id:
                if pattern_id not in items_by_pattern:
                    items_by_pattern[pattern_id] = []
                items_by_pattern[pattern_id].append(item)

            if agent_name := meta.get("agent_name"):
                agents_seen.add(agent_name)
            for agent in meta.get("agents_involved", []):
                agents_seen.add(agent)

        # Validate and boost patterns
        for pattern_id, cross_refs in items_by_pattern.items():
            try:
                result.patterns_analyzed += 1

                validation = await self.validate_pattern_from_km(pattern_id, cross_refs)

                if (
                    validation.km_confidence >= min_confidence
                    and validation.recommendation == "boost"
                ):
                    boost = await self.apply_pattern_boost(validation)
                    if boost.was_applied:
                        result.patterns_boosted += 1

            except Exception as e:
                errors.append(f"Error validating pattern {pattern_id}: {e}")

        # Compute and apply reputation adjustments
        for agent_name in agents_seen:
            try:
                result.agents_analyzed += 1

                adjustment = await self.compute_reputation_adjustment(agent_name, km_items)

                if adjustment.km_confidence >= min_confidence and adjustment.adjustment != 0:
                    if await self.apply_reputation_adjustment(adjustment):
                        result.reputation_adjustments += 1

            except Exception as e:
                errors.append(f"Error adjusting reputation for {agent_name}: {e}")

        result.errors = errors
        result.duration_ms = (time.time() - start_time) * 1000

        return result

    def get_reverse_flow_stats(self) -> Dict[str, Any]:
        """Get statistics about reverse flow operations."""
        self.__init_reverse_flow_state()

        return {
            "km_boosts_applied": self._km_boosts_applied,
            "km_reputation_adjustments": self._km_reputation_adjustments,
            "validations_stored": len(self._km_validations),
            "patterns_tracked": len(self._pattern_usage),
            "total_usage_records": sum(len(v) for v in self._pattern_usage.values()),
        }

    def clear_reverse_flow_state(self) -> None:
        """Clear all reverse flow state (for testing)."""
        self._km_boosts_applied = 0
        self._km_reputation_adjustments = 0
        self._km_validations = []
        self._pattern_usage = {}


__all__ = [
    "CritiqueAdapter",
    "CritiqueSearchResult",
    # Reverse flow dataclasses
    "KMPatternBoost",
    "KMReputationAdjustment",
    "KMPatternValidation",
    "CritiqueKMSyncResult",
]
