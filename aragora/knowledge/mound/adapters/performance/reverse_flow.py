"""
Reverse flow methods: KM -> ELO pattern analysis and adjustments.

Handles:
- Analyzing Knowledge Mound items for agent performance patterns
- Computing ELO adjustment recommendations from KM patterns
- Applying adjustments to the ELO system
- Batch syncing KM patterns to ELO
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Protocol

from aragora.knowledge.mound.adapters.performance.models import (
    EloAdjustmentRecommendation,
    EloSyncResult,
    KMEloPattern,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class _ReverseFlowHostProtocol(Protocol):
    """Protocol for host class of ReverseFlowMixin."""

    _elo_system: Any | None
    _km_patterns: dict[str, list[KMEloPattern]]
    _pending_km_adjustments: list[EloAdjustmentRecommendation]
    _applied_km_adjustments: list[EloAdjustmentRecommendation]
    _km_adjustments_applied: int


class ReverseFlowMixin:
    """Mixin providing KM-to-ELO reverse flow methods.

    NOTE: Does NOT inherit from Protocol to preserve cooperative inheritance.
    Type checking uses _ReverseFlowHostProtocol via cast() as needed.

    Expects the following attributes on the host class:
    - _elo_system: Optional[EloSystem]
    - _km_patterns: dict[str, list[KMEloPattern]]  (lazily initialized)
    - _pending_km_adjustments: list[EloAdjustmentRecommendation]
    - _applied_km_adjustments: list[EloAdjustmentRecommendation]
    - _km_adjustments_applied: int
    """

    # Attribute declarations for mypy (provided by host class)
    _elo_system: Any | None
    _km_patterns: dict[str, list[KMEloPattern]]
    _pending_km_adjustments: list[EloAdjustmentRecommendation]
    _applied_km_adjustments: list[EloAdjustmentRecommendation]
    _km_adjustments_applied: int

    def _ReverseFlowMixin__init_reverse_flow_state(self) -> None:
        """Initialize state for reverse flow tracking."""
        if not hasattr(self, "_km_patterns"):
            self._km_patterns: dict[str, list[KMEloPattern]] = {}  # agent -> patterns
            self._pending_km_adjustments: list[EloAdjustmentRecommendation] = []
            self._applied_km_adjustments: list[EloAdjustmentRecommendation] = []
            self._km_adjustments_applied: int = 0

    async def analyze_km_patterns_for_agent(
        self,
        agent_name: str,
        km_items: list[dict[str, Any]],
        min_confidence: float = 0.7,
    ) -> list[KMEloPattern]:
        """
        Analyze Knowledge Mound items to detect patterns for an agent.

        Examines KM data to identify:
        - success_contributor: Agent's claims frequently lead to successful outcomes
        - contradiction_source: Agent's claims frequently contradicted
        - domain_expert: Agent shows consistent expertise in a domain
        - crux_resolver: Agent identifies and resolves debate cruxes

        Args:
            agent_name: The agent to analyze
            km_items: KM items mentioning this agent
            min_confidence: Minimum confidence threshold for patterns

        Returns:
            List of detected patterns
        """
        self._ReverseFlowMixin__init_reverse_flow_state()

        patterns: list[KMEloPattern] = []

        # Counters for pattern detection
        success_count = 0
        contradiction_count = 0
        domain_mentions: dict[str, int] = {}
        crux_resolutions = 0
        total_items = len(km_items)

        if total_items == 0:
            return patterns

        debate_ids: list[str] = []

        for item in km_items:
            metadata = item.get("metadata", {})
            debate_id = metadata.get("debate_id")
            if debate_id:
                debate_ids.append(debate_id)

            # Check for success contribution
            if metadata.get("outcome_success") or metadata.get("claim_validated"):
                success_count += 1

            # Check for contradictions
            if metadata.get("was_contradicted") or metadata.get("claim_invalidated"):
                contradiction_count += 1

            # Track domain mentions
            domain = metadata.get("domain") or item.get("domain")
            if domain:
                domain_mentions[domain] = domain_mentions.get(domain, 0) + 1

            # Check for crux resolution
            if metadata.get("crux_resolved") or metadata.get("key_insight"):
                crux_resolutions += 1

        # Detect success_contributor pattern
        success_rate = success_count / total_items
        if success_rate >= 0.6 and success_count >= 3:
            patterns.append(
                KMEloPattern(
                    agent_name=agent_name,
                    pattern_type="success_contributor",
                    confidence=min(0.95, success_rate + 0.1),
                    observation_count=success_count,
                    debate_ids=debate_ids[:10],  # Cap at 10
                    metadata={"success_rate": success_rate, "total_items": total_items},
                )
            )

        # Detect contradiction_source pattern (negative)
        contradiction_rate = contradiction_count / total_items
        if contradiction_rate >= 0.3 and contradiction_count >= 3:
            patterns.append(
                KMEloPattern(
                    agent_name=agent_name,
                    pattern_type="contradiction_source",
                    confidence=min(0.95, contradiction_rate + 0.2),
                    observation_count=contradiction_count,
                    debate_ids=debate_ids[:10],
                    metadata={"contradiction_rate": contradiction_rate},
                )
            )

        # Detect domain_expert patterns
        for domain, count in domain_mentions.items():
            if count >= 5:  # Need sufficient domain presence
                patterns.append(
                    KMEloPattern(
                        agent_name=agent_name,
                        pattern_type="domain_expert",
                        confidence=min(0.9, count / 20 + 0.5),
                        observation_count=count,
                        domain=domain,
                        debate_ids=debate_ids[:5],
                        metadata={"domain_item_count": count},
                    )
                )

        # Detect crux_resolver pattern
        if crux_resolutions >= 3:
            patterns.append(
                KMEloPattern(
                    agent_name=agent_name,
                    pattern_type="crux_resolver",
                    confidence=min(0.9, crux_resolutions / 10 + 0.5),
                    observation_count=crux_resolutions,
                    debate_ids=debate_ids[:10],
                    metadata={"crux_resolutions": crux_resolutions},
                )
            )

        # Filter by confidence threshold
        patterns = [p for p in patterns if p.confidence >= min_confidence]

        # Store patterns
        self._km_patterns[agent_name] = patterns

        logger.info(
            f"Analyzed KM patterns for {agent_name}: "
            f"found {len(patterns)} patterns from {total_items} items"
        )

        return patterns

    def compute_elo_adjustment(
        self,
        patterns: list[KMEloPattern],
        max_adjustment: float = 50.0,
    ) -> EloAdjustmentRecommendation | None:
        """
        Compute ELO adjustment recommendation from KM patterns.

        Args:
            patterns: List of KM patterns for an agent
            max_adjustment: Maximum absolute ELO change allowed

        Returns:
            EloAdjustmentRecommendation or None if no adjustment warranted
        """
        self._ReverseFlowMixin__init_reverse_flow_state()

        if not patterns:
            return None

        agent_name = patterns[0].agent_name
        total_adjustment = 0.0
        reasons: list[str] = []
        overall_confidence = 0.0
        domain_adjustments: dict[str, float] = {}

        for pattern in patterns:
            confidence_weight = pattern.confidence * (1 + min(0.5, pattern.observation_count / 20))

            if pattern.pattern_type == "success_contributor":
                # Boost for contributing to successful outcomes
                adj = 15.0 * confidence_weight
                total_adjustment += adj
                reasons.append(f"+{adj:.1f} success contributor ({pattern.observation_count} obs)")

            elif pattern.pattern_type == "contradiction_source":
                # Penalty for frequently contradicted claims
                adj = -10.0 * confidence_weight
                total_adjustment += adj
                reasons.append(f"{adj:.1f} contradictions ({pattern.observation_count} obs)")

            elif pattern.pattern_type == "domain_expert":
                # Domain-specific boost
                domain = pattern.domain or "general"
                adj = 20.0 * confidence_weight
                domain_adjustments[domain] = domain_adjustments.get(domain, 0) + adj
                reasons.append(f"+{adj:.1f} domain expert: {domain}")

            elif pattern.pattern_type == "crux_resolver":
                # Boost for resolving key debate cruxes
                adj = 12.0 * confidence_weight
                total_adjustment += adj
                reasons.append(f"+{adj:.1f} crux resolver ({pattern.observation_count} obs)")

            overall_confidence = max(overall_confidence, pattern.confidence)

        # Apply domain adjustments (take highest)
        if domain_adjustments:
            best_domain = max(domain_adjustments, key=lambda k: domain_adjustments.get(k) or 0.0)
            total_adjustment += domain_adjustments[best_domain]

        # Clamp to max adjustment
        total_adjustment = max(-max_adjustment, min(max_adjustment, total_adjustment))

        # Skip tiny adjustments
        if abs(total_adjustment) < 2.0:
            return None

        recommendation = EloAdjustmentRecommendation(
            agent_name=agent_name,
            adjustment=total_adjustment,
            reason="; ".join(reasons),
            patterns=patterns,
            confidence=overall_confidence,
            domain=list(domain_adjustments.keys())[0] if len(domain_adjustments) == 1 else None,
        )

        self._pending_km_adjustments.append(recommendation)

        logger.info(
            f"KM ELO adjustment recommended for {agent_name}: "
            f"{total_adjustment:+.1f} (confidence={overall_confidence:.2f})"
        )

        return recommendation

    async def apply_km_elo_adjustment(
        self,
        recommendation: EloAdjustmentRecommendation,
        force: bool = False,
    ) -> bool:
        """
        Apply a KM-based ELO adjustment to the underlying ELO system.

        Args:
            recommendation: The adjustment to apply
            force: If True, apply even if confidence is below threshold

        Returns:
            True if applied, False if skipped
        """
        self._ReverseFlowMixin__init_reverse_flow_state()

        if not self._elo_system:
            logger.warning("Cannot apply KM adjustment: no ELO system configured")
            return False

        if recommendation.applied:
            logger.debug(f"Adjustment for {recommendation.agent_name} already applied")
            return False

        if not force and recommendation.confidence < 0.7:
            logger.debug(
                f"Skipping low-confidence adjustment for {recommendation.agent_name}: "
                f"confidence={recommendation.confidence:.2f}"
            )
            return False

        agent_name = recommendation.agent_name

        try:
            # Get current rating
            current_rating = self._elo_system.get_rating(agent_name)
            if not current_rating:
                logger.warning(f"Agent {agent_name} not found in ELO system")
                return False

            # Apply adjustment
            new_elo = current_rating.elo + recommendation.adjustment

            # Use ELO system's update mechanism if available
            if hasattr(self._elo_system, "adjust_rating"):
                self._elo_system.adjust_rating(
                    agent_name,
                    adjustment=recommendation.adjustment,
                    reason=f"KM pattern: {recommendation.reason}",
                )
            else:
                # Fallback: direct rating modification
                current_rating.elo = new_elo

            # Mark as applied
            recommendation.applied = True
            self._applied_km_adjustments.append(recommendation)
            self._km_adjustments_applied += 1

            # Remove from pending
            if recommendation in self._pending_km_adjustments:
                self._pending_km_adjustments.remove(recommendation)

            logger.info(
                f"Applied KM ELO adjustment for {agent_name}: "
                f"{current_rating.elo - recommendation.adjustment:.0f} -> {new_elo:.0f}"
            )

            return True

        except (RuntimeError, ValueError, OSError, AttributeError) as e:
            logger.error(f"Error applying KM adjustment for {agent_name}: {e}")
            return False

    async def sync_km_to_elo(
        self,
        agent_patterns: dict[str, list[KMEloPattern]],
        max_adjustment: float = 50.0,
        min_confidence: float = 0.7,
        auto_apply: bool = False,
    ) -> EloSyncResult:
        """
        Batch sync KM patterns to ELO adjustments.

        Args:
            agent_patterns: Dict mapping agent names to their KM patterns
            max_adjustment: Maximum ELO adjustment per agent
            min_confidence: Minimum confidence to apply adjustment
            auto_apply: If True, automatically apply adjustments

        Returns:
            EloSyncResult with sync statistics
        """
        start_time = time.time()
        result = EloSyncResult()

        for agent_name, patterns in agent_patterns.items():
            result.total_patterns += len(patterns)

            # Compute adjustment recommendation
            recommendation = self.compute_elo_adjustment(patterns, max_adjustment)

            if recommendation:
                result.adjustments_recommended += 1

                if auto_apply and recommendation.confidence >= min_confidence:
                    applied = await self.apply_km_elo_adjustment(recommendation)
                    if applied:
                        result.adjustments_applied += 1
                        result.total_elo_change += recommendation.adjustment
                        if agent_name not in result.agents_affected:
                            result.agents_affected.append(agent_name)
                    else:
                        result.adjustments_skipped += 1
                else:
                    result.adjustments_skipped += 1

        result.duration_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"KM -> ELO sync complete: "
            f"patterns={result.total_patterns}, "
            f"recommended={result.adjustments_recommended}, "
            f"applied={result.adjustments_applied}, "
            f"total_change={result.total_elo_change:+.1f}"
        )

        return result

    def get_pending_adjustments(self) -> list[EloAdjustmentRecommendation]:
        """Get list of pending KM-based ELO adjustments."""
        self._ReverseFlowMixin__init_reverse_flow_state()
        return list(self._pending_km_adjustments)

    def get_applied_adjustments(
        self,
        limit: int = 50,
    ) -> list[EloAdjustmentRecommendation]:
        """Get list of applied KM-based ELO adjustments."""
        self._ReverseFlowMixin__init_reverse_flow_state()
        return self._applied_km_adjustments[-limit:]

    def get_agent_km_patterns(
        self,
        agent_name: str,
    ) -> list[KMEloPattern]:
        """Get stored KM patterns for an agent."""
        self._ReverseFlowMixin__init_reverse_flow_state()
        return self._km_patterns.get(agent_name, [])

    def clear_pending_adjustments(self) -> int:
        """Clear all pending adjustments. Returns count cleared."""
        self._ReverseFlowMixin__init_reverse_flow_state()
        count = len(self._pending_km_adjustments)
        self._pending_km_adjustments = []
        return count

    def get_reverse_flow_stats(self) -> dict[str, Any]:
        """Get statistics about KM -> ELO reverse flow."""
        self._ReverseFlowMixin__init_reverse_flow_state()

        total_patterns = sum(len(p) for p in self._km_patterns.values())
        avg_confidence = 0.0
        if total_patterns > 0:
            all_confidences = [
                p.confidence for patterns in self._km_patterns.values() for p in patterns
            ]
            avg_confidence = sum(all_confidences) / len(all_confidences)

        return {
            "agents_with_patterns": len(self._km_patterns),
            "total_patterns": total_patterns,
            "pending_adjustments": len(self._pending_km_adjustments),
            "applied_adjustments": self._km_adjustments_applied,
            "avg_pattern_confidence": round(avg_confidence, 3),
            "pattern_types": self._count_pattern_types(),
        }

    def _count_pattern_types(self) -> dict[str, int]:
        """Count patterns by type."""
        self._ReverseFlowMixin__init_reverse_flow_state()
        counts: dict[str, int] = {}
        for patterns in self._km_patterns.values():
            for p in patterns:
                counts[p.pattern_type] = counts.get(p.pattern_type, 0) + 1
        return counts


__all__ = ["ReverseFlowMixin"]
