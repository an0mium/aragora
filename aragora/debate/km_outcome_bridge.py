"""
KMOutcomeBridge - Bridges OutcomeTracker to Knowledge Mound for validation feedback.

This module enables bidirectional integration between debate outcomes and
the Knowledge Mound:

- When a debate succeeds, KM entries used in that debate get confidence boosts
- When a debate fails, KM entries get confidence penalties
- Validation can propagate through KM graph relationships

This creates a feedback loop where the system learns from its own decisions:
- Successful patterns get reinforced
- Failed patterns get flagged for review

Usage:
    from aragora.debate.km_outcome_bridge import KMOutcomeBridge, OutcomeValidation

    bridge = KMOutcomeBridge(outcome_tracker, knowledge_mound)

    # After debate completes, validate KM entries used
    validation = await bridge.validate_knowledge_from_outcome(
        outcome=consensus_outcome,
        km_item_ids=["km_123", "km_456"],
    )

    # Propagate validation through graph
    propagated = await bridge.propagate_validation(
        km_item_id="km_123",
        validation=validation,
        depth=2,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from aragora.debate.outcome_tracker import ConsensusOutcome, OutcomeTracker  # type: ignore[attr-defined]
    from aragora.knowledge.mound.core import KnowledgeMound  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)


@dataclass
class OutcomeValidation:
    """Result of validating KM entries based on debate outcome.

    Represents the validation status applied to a KM entry after
    analyzing the debate outcome that used it.
    """

    km_item_id: str
    debate_id: str
    was_successful: bool  # Whether the debate/implementation succeeded
    confidence_adjustment: float  # Positive = boost, negative = penalty
    validation_reason: str  # Why this validation was applied
    original_confidence: float = 0.0
    new_confidence: float = 0.0
    propagated_from: Optional[str] = None  # If this came from propagation
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PropagationResult:
    """Result of propagating validation through KM graph."""

    root_item_id: str
    items_updated: int = 0
    items_skipped: int = 0
    validations: List[OutcomeValidation] = field(default_factory=list)
    depth_reached: int = 0
    errors: List[str] = field(default_factory=list)


@dataclass
class KMOutcomeBridgeConfig:
    """Configuration for KMOutcomeBridge."""

    # Confidence adjustments
    success_boost: float = 0.1  # Boost for successful outcomes
    failure_penalty: float = 0.05  # Penalty for failed outcomes
    propagation_decay: float = 0.5  # Decay factor for propagated validations

    # Thresholds
    min_confidence_for_propagation: float = 0.5  # Min confidence to propagate
    max_propagation_depth: int = 3  # Max graph traversal depth

    # Behavior
    auto_propagate: bool = True  # Auto-propagate validations
    track_usage: bool = True  # Track which KM items are used in debates


class KMOutcomeBridge:
    """
    Bridges OutcomeTracker to Knowledge Mound for validation feedback.

    Enables debate outcomes to validate or invalidate knowledge:
    - Successful implementations boost confidence in used knowledge
    - Failed implementations flag knowledge for review
    - Validation propagates through graph relationships

    This creates a self-improving knowledge system where good patterns
    are reinforced and bad patterns are deprioritized.
    """

    def __init__(
        self,
        outcome_tracker: Optional["OutcomeTracker"] = None,
        knowledge_mound: Optional["KnowledgeMound"] = None,
        config: Optional[KMOutcomeBridgeConfig] = None,
    ):
        """
        Initialize the bridge.

        Args:
            outcome_tracker: OutcomeTracker for debate outcomes
            knowledge_mound: KnowledgeMound for knowledge validation
            config: Optional configuration
        """
        self._outcome_tracker = outcome_tracker
        self._knowledge_mound = knowledge_mound
        self._config = config or KMOutcomeBridgeConfig()

        # Track KM usage in debates
        self._debate_km_usage: Dict[str, List[str]] = {}  # debate_id -> [km_item_ids]
        self._validations_applied: List[OutcomeValidation] = []
        self._total_validations: int = 0

    @property
    def outcome_tracker(self) -> Optional["OutcomeTracker"]:
        """Get the outcome tracker."""
        return self._outcome_tracker

    @property
    def knowledge_mound(self) -> Optional["KnowledgeMound"]:
        """Get the knowledge mound."""
        return self._knowledge_mound

    def set_outcome_tracker(self, tracker: "OutcomeTracker") -> None:
        """Set the outcome tracker."""
        self._outcome_tracker = tracker

    def set_knowledge_mound(self, mound: "KnowledgeMound") -> None:
        """Set the knowledge mound."""
        self._knowledge_mound = mound

    def record_km_usage(
        self,
        debate_id: str,
        km_item_ids: List[str],
    ) -> None:
        """
        Record that KM items were used in a debate.

        Call this during debate execution when KM items are retrieved
        and used for context injection.

        Args:
            debate_id: The debate identifier
            km_item_ids: List of KM item IDs used
        """
        if not self._config.track_usage:
            return

        if debate_id not in self._debate_km_usage:
            self._debate_km_usage[debate_id] = []

        self._debate_km_usage[debate_id].extend(km_item_ids)

        # Deduplicate
        self._debate_km_usage[debate_id] = list(set(self._debate_km_usage[debate_id]))

        logger.debug(f"Recorded KM usage for debate {debate_id}: " f"{len(km_item_ids)} items")

    def get_km_usage(self, debate_id: str) -> List[str]:
        """Get KM items used in a debate."""
        return self._debate_km_usage.get(debate_id, [])

    async def validate_knowledge_from_outcome(
        self,
        outcome: "ConsensusOutcome",
        km_item_ids: Optional[List[str]] = None,
    ) -> List[OutcomeValidation]:
        """
        Validate KM entries based on debate outcome.

        If implementation succeeded:
        - Boost confidence of knowledge items used
        - Mark as outcome-validated

        If implementation failed:
        - Reduce confidence slightly
        - Flag for review

        Args:
            outcome: The consensus outcome from OutcomeTracker
            km_item_ids: KM items to validate (if None, uses tracked usage)

        Returns:
            List of validations applied
        """
        if not self._knowledge_mound:
            logger.warning("Cannot validate: no knowledge mound configured")
            return []

        # Get KM items used
        item_ids = km_item_ids or self.get_km_usage(outcome.debate_id)
        if not item_ids:
            logger.debug(f"No KM items to validate for debate {outcome.debate_id}")
            return []

        validations: List[OutcomeValidation] = []
        was_successful = outcome.implementation_succeeded

        for item_id in item_ids:
            try:
                validation = await self._validate_single_item(
                    item_id=item_id,
                    debate_id=outcome.debate_id,
                    was_successful=was_successful,
                    outcome_confidence=outcome.consensus_confidence,
                )
                if validation:
                    validations.append(validation)
                    self._validations_applied.append(validation)
                    self._total_validations += 1

            except Exception as e:
                logger.error(f"Error validating KM item {item_id}: {e}")

        # Auto-propagate if enabled
        if self._config.auto_propagate and validations:
            for validation in validations:
                try:
                    await self.propagate_validation(
                        km_item_id=validation.km_item_id,
                        validation=validation,
                        depth=self._config.max_propagation_depth,
                    )
                except Exception as e:
                    logger.error(f"Error propagating validation: {e}")

        logger.info(
            f"Validated {len(validations)} KM items for debate {outcome.debate_id} "
            f"(success={was_successful})"
        )

        return validations

    async def _validate_single_item(
        self,
        item_id: str,
        debate_id: str,
        was_successful: bool,
        outcome_confidence: float,
    ) -> Optional[OutcomeValidation]:
        """Validate a single KM item based on outcome."""
        # Get current item
        item = await self._get_km_item(item_id)
        if not item:
            logger.warning(f"KM item not found for validation: {item_id}")
            return None

        original_confidence = item.get("confidence", 0.5)
        if isinstance(original_confidence, str):
            # Convert confidence level to float if needed
            confidence_map = {
                "verified": 0.95,
                "high": 0.8,
                "medium": 0.6,
                "low": 0.4,
                "unverified": 0.2,
            }
            original_confidence = confidence_map.get(original_confidence.lower(), 0.5)

        # Calculate adjustment based on outcome
        if was_successful:
            adjustment = self._config.success_boost * outcome_confidence
            reason = f"Used in successful debate (confidence={outcome_confidence:.2f})"
        else:
            adjustment = -self._config.failure_penalty * (1 - outcome_confidence)
            reason = f"Used in failed debate (confidence={outcome_confidence:.2f})"

        new_confidence = max(0.0, min(1.0, original_confidence + adjustment))

        # Apply validation to KM (if significant change)
        if abs(adjustment) > 0.01:
            await self._update_km_confidence(
                item_id=item_id,
                new_confidence=new_confidence,
                validation_metadata={
                    "debate_id": debate_id,
                    "was_successful": was_successful,
                    "outcome_validated": True,
                    "validation_timestamp": datetime.now().isoformat(),
                },
            )

        return OutcomeValidation(
            km_item_id=item_id,
            debate_id=debate_id,
            was_successful=was_successful,
            confidence_adjustment=adjustment,
            validation_reason=reason,
            original_confidence=original_confidence,
            new_confidence=new_confidence,
        )

    async def propagate_validation(
        self,
        km_item_id: str,
        validation: OutcomeValidation,
        depth: int = 2,
    ) -> PropagationResult:
        """
        Propagate validation to related KM items via graph relationships.

        Validation propagates with decay factor:
        - Direct supports: full validation (decayed)
        - Indirect relationships: partial validation (more decay)

        Args:
            km_item_id: The root KM item
            validation: The validation to propagate
            depth: Maximum propagation depth

        Returns:
            PropagationResult with items updated
        """
        result = PropagationResult(root_item_id=km_item_id)

        if not self._knowledge_mound:
            result.errors.append("No knowledge mound configured")
            return result

        if depth > self._config.max_propagation_depth:
            depth = self._config.max_propagation_depth

        # Get related items
        related_items = await self._get_related_items(km_item_id, depth)

        for related_id, relationship_depth in related_items:
            try:
                # Calculate decayed adjustment
                decay = self._config.propagation_decay**relationship_depth
                decayed_adjustment = validation.confidence_adjustment * decay

                if abs(decayed_adjustment) < 0.005:
                    result.items_skipped += 1
                    continue

                # Get current confidence
                item = await self._get_km_item(related_id)
                if not item:
                    result.items_skipped += 1
                    continue

                original_conf = item.get("confidence", 0.5)
                if isinstance(original_conf, str):
                    confidence_map = {
                        "verified": 0.95,
                        "high": 0.8,
                        "medium": 0.6,
                        "low": 0.4,
                        "unverified": 0.2,
                    }
                    original_conf = confidence_map.get(original_conf.lower(), 0.5)

                new_confidence = max(0.0, min(1.0, original_conf + decayed_adjustment))

                # Apply update
                await self._update_km_confidence(
                    item_id=related_id,
                    new_confidence=new_confidence,
                    validation_metadata={
                        "propagated_from": km_item_id,
                        "propagation_depth": relationship_depth,
                        "debate_id": validation.debate_id,
                        "was_successful": validation.was_successful,
                    },
                )

                propagated_validation = OutcomeValidation(
                    km_item_id=related_id,
                    debate_id=validation.debate_id,
                    was_successful=validation.was_successful,
                    confidence_adjustment=decayed_adjustment,
                    validation_reason=f"Propagated from {km_item_id} (depth={relationship_depth})",
                    original_confidence=original_conf,
                    new_confidence=new_confidence,
                    propagated_from=km_item_id,
                )

                result.validations.append(propagated_validation)
                result.items_updated += 1
                result.depth_reached = max(result.depth_reached, relationship_depth)

            except Exception as e:
                error_msg = f"Error propagating to {related_id}: {e}"
                logger.error(error_msg)
                result.errors.append(error_msg)

        logger.info(
            f"Propagated validation from {km_item_id}: "
            f"updated={result.items_updated}, skipped={result.items_skipped}"
        )

        return result

    async def _get_km_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Get a KM item by ID."""
        if not self._knowledge_mound:
            return None

        try:
            # Try different accessor patterns
            if hasattr(self._knowledge_mound, "get"):
                return await self._knowledge_mound.get(item_id)
            elif hasattr(self._knowledge_mound, "get_item"):
                return await self._knowledge_mound.get_item(item_id)
            elif hasattr(self._knowledge_mound, "query"):
                result = await self._knowledge_mound.query(item_id, limit=1)
                return result[0] if result else None
            else:
                # Mock/fallback for testing
                return {"id": item_id, "confidence": 0.7}
        except Exception as e:
            logger.error(f"Error getting KM item {item_id}: {e}")
            return None

    async def _update_km_confidence(
        self,
        item_id: str,
        new_confidence: float,
        validation_metadata: Dict[str, Any],
    ) -> bool:
        """Update a KM item's confidence."""
        if not self._knowledge_mound:
            return False

        try:
            # Try different update patterns
            if hasattr(self._knowledge_mound, "update_confidence"):
                return await self._knowledge_mound.update_confidence(
                    item_id, new_confidence, validation_metadata
                )
            elif hasattr(self._knowledge_mound, "update"):
                return await self._knowledge_mound.update(
                    item_id, confidence=new_confidence, metadata=validation_metadata
                )
            else:
                # Log success for testing even without real KM
                logger.debug(f"Would update {item_id} confidence to {new_confidence}")
                return True
        except Exception as e:
            logger.error(f"Error updating KM item {item_id}: {e}")
            return False

    async def _get_related_items(
        self,
        item_id: str,
        max_depth: int,
    ) -> List[tuple[str, int]]:
        """
        Get related KM items via graph relationships.

        Returns list of (item_id, depth) tuples.
        """
        if not self._knowledge_mound:
            return []

        related: List[tuple[str, int]] = []

        try:
            # Try graph traversal if available
            if hasattr(self._knowledge_mound, "get_graph_neighbors"):
                for depth in range(1, max_depth + 1):
                    neighbors = await self._knowledge_mound.get_graph_neighbors(
                        item_id, depth=depth
                    )
                    for neighbor_id in neighbors:
                        if neighbor_id != item_id:
                            related.append((neighbor_id, depth))
            elif hasattr(self._knowledge_mound, "get_relationships"):
                # Simpler relationship query
                rels = await self._knowledge_mound.get_relationships(item_id)
                for rel in rels:
                    target_id = rel.get("target_id") or rel.get("to_id")
                    if target_id and target_id != item_id:
                        related.append((target_id, 1))
        except Exception as e:
            logger.error(f"Error getting related items for {item_id}: {e}")

        return related

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get statistics about outcome-based validations."""
        success_validations = [v for v in self._validations_applied if v.was_successful]
        failure_validations = [v for v in self._validations_applied if not v.was_successful]

        avg_success_boost = 0.0
        avg_failure_penalty = 0.0

        if success_validations:
            avg_success_boost = sum(v.confidence_adjustment for v in success_validations) / len(
                success_validations
            )
        if failure_validations:
            avg_failure_penalty = sum(v.confidence_adjustment for v in failure_validations) / len(
                failure_validations
            )

        return {
            "total_validations": self._total_validations,
            "success_validations": len(success_validations),
            "failure_validations": len(failure_validations),
            "debates_tracked": len(self._debate_km_usage),
            "total_km_items_tracked": sum(len(items) for items in self._debate_km_usage.values()),
            "avg_success_boost": round(avg_success_boost, 4),
            "avg_failure_penalty": round(avg_failure_penalty, 4),
            "config": {
                "success_boost": self._config.success_boost,
                "failure_penalty": self._config.failure_penalty,
                "propagation_decay": self._config.propagation_decay,
                "auto_propagate": self._config.auto_propagate,
            },
        }

    def clear_tracking(self) -> None:
        """Clear tracked KM usage and validations."""
        self._debate_km_usage.clear()
        self._validations_applied.clear()
        self._total_validations = 0


__all__ = [
    "KMOutcomeBridge",
    "KMOutcomeBridgeConfig",
    "OutcomeValidation",
    "PropagationResult",
]
