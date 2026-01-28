"""
Adapter Fusion Protocol for Knowledge Mound Phase A3.

This module provides the foundation for multi-adapter coordination and
conflict resolution in the Knowledge Mound system.

Key Components:
- FusionStrategy: Enum defining how to merge multiple adapter results
- ConflictResolution: Enum defining how to handle conflicting validations
- FusedValidation: Result of fusing multiple adapter validations
- AdapterFusionProtocol: Abstract base for fusion-capable adapters
- FusionCoordinator: Orchestrates multi-adapter fusion operations

Usage:
    from aragora.knowledge.mound.ops.fusion import (
        FusionCoordinator,
        FusionStrategy,
        ConflictResolution,
    )

    coordinator = FusionCoordinator()
    result = await coordinator.fuse_adapter_validations(
        validations=[...],
        strategy=FusionStrategy.WEIGHTED_AVERAGE,
    )
"""

from __future__ import annotations

import logging
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


class FusionStrategy(Enum):
    """Strategy for fusing multiple adapter validations."""

    WEIGHTED_AVERAGE = "weighted_average"
    """Compute weighted average of confidences using adapter reliability."""

    MAJORITY_VOTE = "majority_vote"
    """Use the confidence value that appears most frequently."""

    MAXIMUM_CONFIDENCE = "maximum_confidence"
    """Use the highest confidence value among all adapters."""

    MINIMUM_CONFIDENCE = "minimum_confidence"
    """Use the lowest confidence value (conservative approach)."""

    CONSENSUS_THRESHOLD = "consensus_threshold"
    """Only accept if agreement exceeds threshold, else escalate."""

    MEDIAN = "median"
    """Use the median confidence value across adapters."""


class ConflictResolution(Enum):
    """Strategy for resolving conflicting adapter validations."""

    PREFER_HIGHER_CONFIDENCE = "prefer_higher_confidence"
    """Prefer the validation with higher confidence score."""

    PREFER_MORE_SOURCES = "prefer_more_sources"
    """Prefer the validation with more supporting sources."""

    PREFER_NEWER = "prefer_newer"
    """Prefer the more recent validation."""

    PREFER_HIGHER_PRIORITY = "prefer_higher_priority"
    """Prefer the validation from higher-priority adapter."""

    MERGE = "merge"
    """Attempt to merge conflicting validations into a combined result."""

    ESCALATE = "escalate"
    """Escalate to human review when conflicts cannot be resolved."""


class FusionOutcome(Enum):
    """Outcome of a fusion operation."""

    SUCCESS = "success"
    """Fusion completed successfully with consensus."""

    PARTIAL = "partial"
    """Fusion completed but some adapters disagreed."""

    CONFLICT = "conflict"
    """Significant conflict detected, resolution applied."""

    ESCALATED = "escalated"
    """Conflict could not be resolved, escalated for review."""

    INSUFFICIENT_DATA = "insufficient_data"
    """Not enough adapter validations to perform fusion."""


@dataclass
class AdapterValidation:
    """A validation result from a single adapter."""

    adapter_name: str
    """Name of the adapter that produced this validation."""

    item_id: str
    """ID of the knowledge item being validated."""

    confidence: float
    """Confidence score from this adapter (0.0 to 1.0)."""

    is_valid: bool
    """Whether the adapter considers the item valid."""

    sources: List[str] = field(default_factory=list)
    """Sources supporting this validation."""

    reasoning: Optional[str] = None
    """Optional reasoning for the validation."""

    priority: int = 0
    """Adapter priority (higher = more authoritative)."""

    reliability: float = 1.0
    """Historical reliability of this adapter (0.0 to 1.0)."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When this validation was produced."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional adapter-specific metadata."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "adapter_name": self.adapter_name,
            "item_id": self.item_id,
            "confidence": self.confidence,
            "is_valid": self.is_valid,
            "sources": self.sources,
            "reasoning": self.reasoning,
            "priority": self.priority,
            "reliability": self.reliability,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class FusedValidation:
    """Result of fusing multiple adapter validations."""

    item_id: str
    """ID of the knowledge item that was validated."""

    fused_confidence: float
    """Combined confidence score after fusion (0.0 to 1.0)."""

    is_valid: bool
    """Combined validity determination."""

    strategy_used: FusionStrategy
    """The fusion strategy that was applied."""

    source_validations: List[AdapterValidation]
    """Original validations that were fused."""

    outcome: FusionOutcome = FusionOutcome.SUCCESS
    """Outcome of the fusion operation."""

    conflict_detected: bool = False
    """Whether a conflict was detected during fusion."""

    conflict_resolved: bool = False
    """Whether the conflict was successfully resolved."""

    resolution_method: Optional[ConflictResolution] = None
    """Method used to resolve conflict, if any."""

    agreement_ratio: float = 1.0
    """Ratio of adapters that agreed (0.0 to 1.0)."""

    confidence_variance: float = 0.0
    """Variance in confidence scores across adapters."""

    participating_adapters: List[str] = field(default_factory=list)
    """Names of adapters that participated in fusion."""

    escalation_reason: Optional[str] = None
    """Reason for escalation, if outcome is ESCALATED."""

    fused_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When the fusion was performed."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional fusion metadata."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "item_id": self.item_id,
            "fused_confidence": self.fused_confidence,
            "is_valid": self.is_valid,
            "strategy_used": self.strategy_used.value,
            "source_validations": [v.to_dict() for v in self.source_validations],
            "outcome": self.outcome.value,
            "conflict_detected": self.conflict_detected,
            "conflict_resolved": self.conflict_resolved,
            "resolution_method": self.resolution_method.value if self.resolution_method else None,
            "agreement_ratio": self.agreement_ratio,
            "confidence_variance": self.confidence_variance,
            "participating_adapters": self.participating_adapters,
            "escalation_reason": self.escalation_reason,
            "fused_at": self.fused_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class FusionConfig:
    """Configuration for fusion operations."""

    default_strategy: FusionStrategy = FusionStrategy.WEIGHTED_AVERAGE
    """Default strategy for fusing validations."""

    conflict_resolution: ConflictResolution = ConflictResolution.PREFER_HIGHER_CONFIDENCE
    """Default conflict resolution strategy."""

    min_adapters_for_fusion: int = 2
    """Minimum number of adapters required for fusion."""

    consensus_threshold: float = 0.7
    """Agreement threshold for CONSENSUS_THRESHOLD strategy."""

    conflict_threshold: float = 0.3
    """Confidence variance above which conflict is detected."""

    validity_threshold: float = 0.5
    """Threshold for determining is_valid from fused confidence."""

    weight_by_reliability: bool = True
    """Whether to weight by adapter reliability in WEIGHTED_AVERAGE."""

    weight_by_priority: bool = True
    """Whether to weight by adapter priority in WEIGHTED_AVERAGE."""

    escalate_on_deadlock: bool = True
    """Whether to escalate when consensus cannot be reached."""

    max_escalation_wait_seconds: float = 86400.0
    """Maximum time to wait for escalation resolution (24 hours)."""


class AdapterFusionCapable(Protocol):
    """Protocol for adapters that support fusion operations."""

    adapter_name: str
    """Unique identifier for this adapter."""

    def get_fusion_priority(self) -> int:
        """Return the fusion priority of this adapter (higher = more authoritative)."""
        ...

    def get_reliability_score(self) -> float:
        """Return the historical reliability score of this adapter (0.0 to 1.0)."""
        ...

    def supports_fusion(self) -> bool:
        """Return whether this adapter supports fusion operations."""
        ...


class AdapterFusionProtocol(ABC):
    """Abstract base class for fusion-capable adapters.

    Adapters implementing this protocol can participate in multi-adapter
    fusion operations where validations from multiple sources are combined.
    """

    @property
    @abstractmethod
    def adapter_name(self) -> str:
        """Return the unique name of this adapter."""
        ...

    @abstractmethod
    def get_fusion_priority(self) -> int:
        """Return the fusion priority of this adapter.

        Higher priority adapters have more influence in fusion operations.
        Default implementations should return 0.
        """
        ...

    @abstractmethod
    def get_reliability_score(self) -> float:
        """Return the historical reliability score (0.0 to 1.0).

        This is based on past validation accuracy. Adapters with higher
        reliability have more weight in fusion operations.
        """
        ...

    def supports_fusion(self) -> bool:
        """Return whether this adapter supports fusion.

        Override to return False to opt out of fusion operations.
        """
        return True

    async def prepare_for_fusion(
        self,
        item_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AdapterValidation:
        """Prepare a validation for fusion with other adapters.

        This method should return the adapter's validation for the given item,
        formatted as an AdapterValidation that can be fused with others.

        Args:
            item_id: ID of the knowledge item to validate.
            context: Optional context for the validation.

        Returns:
            AdapterValidation with this adapter's assessment.
        """
        raise NotImplementedError(f"{self.adapter_name} must implement prepare_for_fusion()")

    async def apply_fused_result(
        self,
        fused: FusedValidation,
    ) -> bool:
        """Apply the result of a fusion operation.

        After fusion is complete, this method is called to allow the adapter
        to update its internal state based on the fused result.

        Args:
            fused: The fused validation result.

        Returns:
            True if the result was successfully applied.
        """
        raise NotImplementedError(f"{self.adapter_name} must implement apply_fused_result()")


class FusionCoordinator:
    """Coordinates fusion operations across multiple adapters.

    This class orchestrates the process of collecting validations from
    multiple adapters and fusing them into a single consensus result.
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        """Initialize the fusion coordinator.

        Args:
            config: Optional fusion configuration. Uses defaults if not provided.
        """
        self.config = config or FusionConfig()
        self._fusion_history: List[FusedValidation] = []
        self._escalation_queue: List[FusedValidation] = []

    def fuse_validations(
        self,
        validations: List[AdapterValidation],
        strategy: Optional[FusionStrategy] = None,
        conflict_resolution: Optional[ConflictResolution] = None,
    ) -> FusedValidation:
        """Fuse multiple adapter validations into a single result.

        Args:
            validations: List of validations to fuse.
            strategy: Optional strategy override.
            conflict_resolution: Optional conflict resolution override.

        Returns:
            FusedValidation with the combined result.
        """
        if not validations:
            return self._empty_fusion_result()

        if len(validations) < self.config.min_adapters_for_fusion:
            return self._insufficient_data_result(validations)

        strategy = strategy or self.config.default_strategy
        resolution = conflict_resolution or self.config.conflict_resolution

        # Extract item_id (should be same across all validations)
        item_id = validations[0].item_id

        # Calculate confidence statistics
        confidences = [v.confidence for v in validations]
        variance = statistics.variance(confidences) if len(confidences) > 1 else 0.0

        # Detect conflict
        conflict_detected = variance > self.config.conflict_threshold

        # Calculate agreement ratio (validations with same is_valid)
        valid_count = sum(1 for v in validations if v.is_valid)
        agreement_ratio = max(valid_count, len(validations) - valid_count) / len(validations)

        # Apply fusion strategy
        fused_confidence = self._apply_strategy(validations, strategy)

        # Determine combined validity
        is_valid = fused_confidence >= self.config.validity_threshold

        # Handle conflicts
        conflict_resolved = False
        escalation_reason = None
        outcome = FusionOutcome.SUCCESS

        if conflict_detected:
            if strategy == FusionStrategy.CONSENSUS_THRESHOLD:
                if agreement_ratio < self.config.consensus_threshold:
                    if self.config.escalate_on_deadlock:
                        outcome = FusionOutcome.ESCALATED
                        escalation_reason = (
                            f"Agreement ratio {agreement_ratio:.2f} below "
                            f"threshold {self.config.consensus_threshold}"
                        )
                    else:
                        outcome = FusionOutcome.CONFLICT
                else:
                    conflict_resolved = True
                    outcome = FusionOutcome.PARTIAL
            else:
                # Apply conflict resolution
                resolved = self._resolve_conflict(validations, resolution)
                if resolved:
                    fused_confidence = resolved.confidence
                    is_valid = resolved.is_valid
                    conflict_resolved = True
                    outcome = FusionOutcome.PARTIAL
                else:
                    if self.config.escalate_on_deadlock:
                        outcome = FusionOutcome.ESCALATED
                        escalation_reason = f"Could not resolve conflict with {resolution.value}"
                    else:
                        outcome = FusionOutcome.CONFLICT

        result = FusedValidation(
            item_id=item_id,
            fused_confidence=fused_confidence,
            is_valid=is_valid,
            strategy_used=strategy,
            source_validations=validations,
            outcome=outcome,
            conflict_detected=conflict_detected,
            conflict_resolved=conflict_resolved,
            resolution_method=resolution if conflict_detected else None,
            agreement_ratio=agreement_ratio,
            confidence_variance=variance,
            participating_adapters=[v.adapter_name for v in validations],
            escalation_reason=escalation_reason,
        )

        self._fusion_history.append(result)
        if outcome == FusionOutcome.ESCALATED:
            self._escalation_queue.append(result)

        return result

    def _apply_strategy(
        self,
        validations: List[AdapterValidation],
        strategy: FusionStrategy,
    ) -> float:
        """Apply the fusion strategy to compute fused confidence.

        Args:
            validations: Validations to fuse.
            strategy: Strategy to apply.

        Returns:
            Fused confidence score.
        """
        confidences = [v.confidence for v in validations]

        if strategy == FusionStrategy.WEIGHTED_AVERAGE:
            return self._weighted_average(validations)

        elif strategy == FusionStrategy.MAJORITY_VOTE:
            # Bucket confidences and take most common bucket
            buckets = [round(c, 1) for c in confidences]
            most_common = max(set(buckets), key=buckets.count)
            # Return average of confidences in that bucket
            bucket_values = [c for c in confidences if round(c, 1) == most_common]
            return sum(bucket_values) / len(bucket_values)

        elif strategy == FusionStrategy.MAXIMUM_CONFIDENCE:
            return max(confidences)

        elif strategy == FusionStrategy.MINIMUM_CONFIDENCE:
            return min(confidences)

        elif strategy == FusionStrategy.MEDIAN:
            return statistics.median(confidences)

        elif strategy == FusionStrategy.CONSENSUS_THRESHOLD:
            # Use weighted average if consensus reached, otherwise use median
            return self._weighted_average(validations)

        else:
            # Default to weighted average
            return self._weighted_average(validations)

    def _weighted_average(self, validations: List[AdapterValidation]) -> float:
        """Compute weighted average of confidences.

        Weights are computed from reliability and priority scores.

        Args:
            validations: Validations to average.

        Returns:
            Weighted average confidence.
        """
        total_weight = 0.0
        weighted_sum = 0.0

        for v in validations:
            weight = 1.0

            if self.config.weight_by_reliability:
                weight *= v.reliability

            if self.config.weight_by_priority:
                # Normalize priority: 0 -> 1.0, 100 -> 2.0
                priority_factor = 1.0 + (v.priority / 100.0)
                weight *= priority_factor

            weighted_sum += v.confidence * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _resolve_conflict(
        self,
        validations: List[AdapterValidation],
        resolution: ConflictResolution,
    ) -> Optional[AdapterValidation]:
        """Resolve a conflict between validations.

        Args:
            validations: Conflicting validations.
            resolution: Resolution strategy.

        Returns:
            The winning validation, or None if cannot be resolved.
        """
        if not validations:
            return None

        if resolution == ConflictResolution.PREFER_HIGHER_CONFIDENCE:
            return max(validations, key=lambda v: v.confidence)

        elif resolution == ConflictResolution.PREFER_MORE_SOURCES:
            return max(validations, key=lambda v: len(v.sources))

        elif resolution == ConflictResolution.PREFER_NEWER:
            return max(validations, key=lambda v: v.timestamp)

        elif resolution == ConflictResolution.PREFER_HIGHER_PRIORITY:
            return max(validations, key=lambda v: v.priority)

        elif resolution == ConflictResolution.MERGE:
            # Merge: average confidences, combine sources, take newer timestamp
            merged_confidence = sum(v.confidence for v in validations) / len(validations)
            merged_sources = list(set(s for v in validations for s in v.sources))
            merged_valid = sum(1 for v in validations if v.is_valid) > len(validations) / 2

            return AdapterValidation(
                adapter_name="merged",
                item_id=validations[0].item_id,
                confidence=merged_confidence,
                is_valid=merged_valid,
                sources=merged_sources,
                reasoning="Merged from conflicting validations",
                priority=max(v.priority for v in validations),
                reliability=sum(v.reliability for v in validations) / len(validations),
                timestamp=max(v.timestamp for v in validations),
            )

        elif resolution == ConflictResolution.ESCALATE:
            # Cannot resolve automatically
            return None

        return None

    def _empty_fusion_result(self) -> FusedValidation:
        """Return an empty fusion result for when no validations provided."""
        return FusedValidation(
            item_id="",
            fused_confidence=0.0,
            is_valid=False,
            strategy_used=self.config.default_strategy,
            source_validations=[],
            outcome=FusionOutcome.INSUFFICIENT_DATA,
        )

    def _insufficient_data_result(
        self,
        validations: List[AdapterValidation],
    ) -> FusedValidation:
        """Return result when not enough validations for fusion."""
        item_id = validations[0].item_id if validations else ""

        # If only one validation, use it directly
        if len(validations) == 1:
            v = validations[0]
            return FusedValidation(
                item_id=item_id,
                fused_confidence=v.confidence,
                is_valid=v.is_valid,
                strategy_used=self.config.default_strategy,
                source_validations=validations,
                outcome=FusionOutcome.INSUFFICIENT_DATA,
                participating_adapters=[v.adapter_name],
                metadata={"reason": "Single adapter validation, no fusion performed"},
            )

        return FusedValidation(
            item_id=item_id,
            fused_confidence=0.0,
            is_valid=False,
            strategy_used=self.config.default_strategy,
            source_validations=validations,
            outcome=FusionOutcome.INSUFFICIENT_DATA,
            participating_adapters=[v.adapter_name for v in validations],
            metadata={
                "reason": f"Need {self.config.min_adapters_for_fusion} adapters, got {len(validations)}"
            },
        )

    def get_fusion_history(
        self,
        limit: int = 100,
        item_id: Optional[str] = None,
    ) -> List[FusedValidation]:
        """Get recent fusion history.

        Args:
            limit: Maximum number of results.
            item_id: Optional filter by item ID.

        Returns:
            List of recent fused validations.
        """
        history = self._fusion_history

        if item_id:
            history = [f for f in history if f.item_id == item_id]

        return history[-limit:]

    def get_escalation_queue(self) -> List[FusedValidation]:
        """Get validations awaiting escalation resolution."""
        return list(self._escalation_queue)

    def resolve_escalation(
        self,
        item_id: str,
        resolution: FusedValidation,
        resolver_id: str,
    ) -> bool:
        """Resolve an escalated fusion.

        Args:
            item_id: ID of the item to resolve.
            resolution: The resolution to apply.
            resolver_id: ID of the user/system resolving.

        Returns:
            True if resolution was applied.
        """
        for i, fused in enumerate(self._escalation_queue):
            if fused.item_id == item_id:
                # Update the resolution
                resolution.metadata["resolved_by"] = resolver_id
                resolution.metadata["resolved_at"] = datetime.now(timezone.utc).isoformat()
                resolution.outcome = FusionOutcome.SUCCESS
                resolution.escalation_reason = None

                # Remove from queue and update history
                self._escalation_queue.pop(i)
                self._fusion_history.append(resolution)
                return True

        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get fusion statistics.

        Returns:
            Dictionary with fusion metrics.
        """
        if not self._fusion_history:
            return {
                "total_fusions": 0,
                "success_rate": 0.0,
                "conflict_rate": 0.0,
                "escalation_rate": 0.0,
                "avg_agreement_ratio": 0.0,
                "pending_escalations": 0,
            }

        total = len(self._fusion_history)
        successes = sum(1 for f in self._fusion_history if f.outcome == FusionOutcome.SUCCESS)
        conflicts = sum(1 for f in self._fusion_history if f.conflict_detected)
        escalations = sum(1 for f in self._fusion_history if f.outcome == FusionOutcome.ESCALATED)
        avg_agreement = sum(f.agreement_ratio for f in self._fusion_history) / total

        return {
            "total_fusions": total,
            "success_rate": successes / total,
            "conflict_rate": conflicts / total,
            "escalation_rate": escalations / total,
            "avg_agreement_ratio": avg_agreement,
            "pending_escalations": len(self._escalation_queue),
            "by_strategy": self._stats_by_strategy(),
        }

    def _stats_by_strategy(self) -> Dict[str, int]:
        """Get fusion counts by strategy."""
        counts: Dict[str, int] = {}
        for f in self._fusion_history:
            strategy = f.strategy_used.value
            counts[strategy] = counts.get(strategy, 0) + 1
        return counts


# Singleton coordinator instance
_fusion_coordinator: Optional[FusionCoordinator] = None


def get_fusion_coordinator(config: Optional[FusionConfig] = None) -> FusionCoordinator:
    """Get or create the singleton fusion coordinator.

    Args:
        config: Optional configuration (only used on first call).

    Returns:
        FusionCoordinator instance.
    """
    global _fusion_coordinator
    if _fusion_coordinator is None:
        _fusion_coordinator = FusionCoordinator(config)
    return _fusion_coordinator


__all__ = [
    # Enums
    "FusionStrategy",
    "ConflictResolution",
    "FusionOutcome",
    # Dataclasses
    "AdapterValidation",
    "FusedValidation",
    "FusionConfig",
    # Protocol/ABC
    "AdapterFusionCapable",
    "AdapterFusionProtocol",
    # Coordinator
    "FusionCoordinator",
    "get_fusion_coordinator",
]
