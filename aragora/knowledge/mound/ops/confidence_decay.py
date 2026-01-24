"""
Confidence Decay for Knowledge Mound.

Implements dynamic confidence adjustment over time:
- Time-based decay for aging knowledge
- Usage-based confidence boosting
- Validation-based adjustments
- Contradiction-driven decay

Phase A2 - Knowledge Quality Assurance
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from aragora.knowledge.mound.facade import KnowledgeMound

logger = logging.getLogger(__name__)


class DecayModel(str, Enum):
    """Models for confidence decay calculation."""

    EXPONENTIAL = "exponential"  # Fast initial decay, slow tail
    LINEAR = "linear"  # Constant decay rate
    STEP = "step"  # Discrete confidence levels
    CUSTOM = "custom"  # User-defined decay function


class ConfidenceEvent(str, Enum):
    """Events that affect confidence."""

    CREATED = "created"
    ACCESSED = "accessed"
    CITED = "cited"
    VALIDATED = "validated"
    INVALIDATED = "invalidated"
    CONTRADICTED = "contradicted"
    UPDATED = "updated"
    DECAYED = "decayed"


@dataclass
class ConfidenceAdjustment:
    """Record of a confidence adjustment."""

    id: str
    item_id: str
    event: ConfidenceEvent
    old_confidence: float
    new_confidence: float
    reason: str
    adjusted_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "item_id": self.item_id,
            "event": self.event.value,
            "old_confidence": self.old_confidence,
            "new_confidence": self.new_confidence,
            "reason": self.reason,
            "adjusted_at": self.adjusted_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class DecayConfig:
    """Configuration for confidence decay."""

    # Decay model
    model: DecayModel = DecayModel.EXPONENTIAL

    # Time-based decay
    half_life_days: float = 90.0  # Days until confidence halves
    min_confidence: float = 0.1  # Floor for decayed confidence
    max_confidence: float = 1.0  # Ceiling for boosted confidence

    # Usage-based boosting
    access_boost: float = 0.01  # Confidence boost per access
    citation_boost: float = 0.05  # Confidence boost when cited
    validation_boost: float = 0.1  # Confidence boost when validated

    # Penalty adjustments
    invalidation_penalty: float = 0.3  # Confidence drop when invalidated
    contradiction_penalty: float = 0.2  # Confidence drop when contradicted

    # Batch processing
    batch_size: int = 100
    decay_interval_hours: int = 24  # How often to run decay

    # Domain-specific half-lives (optional overrides)
    domain_half_lives: Dict[str, float] = field(
        default_factory=lambda: {
            "technology": 30.0,  # Tech knowledge decays faster
            "science": 180.0,  # Scientific knowledge more stable
            "legal": 365.0,  # Legal knowledge very stable
            "news": 7.0,  # News decays very fast
        }
    )


@dataclass
class DecayReport:
    """Report of confidence decay results."""

    workspace_id: str
    items_processed: int
    items_decayed: int
    items_boosted: int
    average_confidence_change: float
    adjustments: List[ConfidenceAdjustment]
    processed_at: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "workspace_id": self.workspace_id,
            "items_processed": self.items_processed,
            "items_decayed": self.items_decayed,
            "items_boosted": self.items_boosted,
            "average_confidence_change": self.average_confidence_change,
            "adjustments": [a.to_dict() for a in self.adjustments],
            "processed_at": self.processed_at.isoformat(),
            "duration_ms": self.duration_ms,
        }


class ConfidenceDecayManager:
    """Manages confidence decay for knowledge items."""

    def __init__(self, config: Optional[DecayConfig] = None):
        """Initialize the decay manager."""
        self.config = config or DecayConfig()
        self._adjustments: List[ConfidenceAdjustment] = []
        self._last_decay_run: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()

    def calculate_decay(
        self,
        current_confidence: float,
        age_days: float,
        domain: Optional[str] = None,
    ) -> float:
        """Calculate decayed confidence based on age.

        Args:
            current_confidence: Current confidence level (0-1)
            age_days: Age of the knowledge item in days
            domain: Optional domain for domain-specific decay

        Returns:
            New confidence level after decay
        """
        # Get half-life for domain
        half_life = self.config.domain_half_lives.get(domain or "", self.config.half_life_days)

        if self.config.model == DecayModel.EXPONENTIAL:
            # Exponential decay: C(t) = C0 * (0.5)^(t/half_life)
            decay_factor = math.pow(0.5, age_days / half_life)
            new_confidence = current_confidence * decay_factor

        elif self.config.model == DecayModel.LINEAR:
            # Linear decay: C(t) = C0 - (C0 * t / (2 * half_life))
            decay_rate = current_confidence / (2 * half_life)
            new_confidence = current_confidence - (decay_rate * age_days)

        elif self.config.model == DecayModel.STEP:
            # Step decay: discrete confidence levels
            if age_days < half_life * 0.5:
                new_confidence = current_confidence
            elif age_days < half_life:
                new_confidence = current_confidence * 0.75
            elif age_days < half_life * 2:
                new_confidence = current_confidence * 0.5
            else:
                new_confidence = current_confidence * 0.25

        else:
            new_confidence = current_confidence

        # Apply floor
        return max(self.config.min_confidence, new_confidence)

    def calculate_boost(
        self,
        current_confidence: float,
        event: ConfidenceEvent,
    ) -> float:
        """Calculate confidence boost from an event.

        Args:
            current_confidence: Current confidence level (0-1)
            event: Event that triggers the boost

        Returns:
            New confidence level after boost
        """
        boost = 0.0

        if event == ConfidenceEvent.ACCESSED:
            boost = self.config.access_boost
        elif event == ConfidenceEvent.CITED:
            boost = self.config.citation_boost
        elif event == ConfidenceEvent.VALIDATED:
            boost = self.config.validation_boost
        elif event == ConfidenceEvent.INVALIDATED:
            boost = -self.config.invalidation_penalty
        elif event == ConfidenceEvent.CONTRADICTED:
            boost = -self.config.contradiction_penalty

        new_confidence = current_confidence + boost

        # Clamp to valid range
        return max(
            self.config.min_confidence,
            min(self.config.max_confidence, new_confidence),
        )

    async def apply_decay(
        self,
        mound: "KnowledgeMound",
        workspace_id: str,
        force: bool = False,
    ) -> DecayReport:
        """Apply confidence decay to all items in a workspace.

        Args:
            mound: KnowledgeMound instance
            workspace_id: Workspace to process
            force: Force decay even if recently run

        Returns:
            DecayReport with results
        """
        import time
        import uuid

        start_time = time.time()

        # Check if we should run
        if not force:
            last_run = self._last_decay_run.get(workspace_id)
            if last_run:
                hours_since = (datetime.now() - last_run).total_seconds() / 3600
                if hours_since < self.config.decay_interval_hours:
                    logger.debug(
                        f"Skipping decay for {workspace_id}, " f"last run {hours_since:.1f}h ago"
                    )
                    return DecayReport(
                        workspace_id=workspace_id,
                        items_processed=0,
                        items_decayed=0,
                        items_boosted=0,
                        average_confidence_change=0.0,
                        adjustments=[],
                    )

        # Get items
        result = await mound.query(
            workspace_id=workspace_id,
            query="",
            limit=10000,
        )
        items = result.items if hasattr(result, "items") else []

        adjustments: List[ConfidenceAdjustment] = []
        items_decayed = 0
        items_boosted = 0
        total_change = 0.0

        now = datetime.now()

        for item in items:
            # Get current confidence
            old_confidence = getattr(item, "confidence", 0.5)
            if old_confidence is None:
                old_confidence = 0.5

            # Calculate age
            created_at = getattr(item, "created_at", None)
            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at)
                except ValueError:
                    created_at = now
            elif created_at is None:
                created_at = now

            age_days = (now - created_at).total_seconds() / 86400

            # Get domain if available
            domain = None
            topics = getattr(item, "topics", []) or []
            if topics:
                domain = topics[0].lower() if topics[0] else None

            # Calculate new confidence
            new_confidence = self.calculate_decay(old_confidence, age_days, domain)

            # Only record if changed
            if abs(new_confidence - old_confidence) > 0.001:
                adjustment = ConfidenceAdjustment(
                    id=str(uuid.uuid4()),
                    item_id=item.id,
                    event=ConfidenceEvent.DECAYED,
                    old_confidence=old_confidence,
                    new_confidence=new_confidence,
                    reason=f"Time-based decay after {age_days:.1f} days",
                    metadata={"age_days": age_days, "domain": domain},
                )
                adjustments.append(adjustment)

                change = new_confidence - old_confidence
                total_change += change

                if change < 0:
                    items_decayed += 1
                else:
                    items_boosted += 1

                # Update item confidence (if mound supports it)
                try:
                    if hasattr(mound, "update_confidence"):
                        # KnowledgeMound inherits update_confidence from CRUDMixin
                        await mound.update_confidence(item.id, new_confidence)  # type: ignore[misc]
                except Exception as e:
                    logger.warning(f"Failed to update confidence for {item.id}: {e}")

        # Record run time
        self._last_decay_run[workspace_id] = now

        # Store adjustments
        async with self._lock:
            self._adjustments.extend(adjustments)
            # Keep only recent adjustments
            if len(self._adjustments) > 10000:
                self._adjustments = self._adjustments[-10000:]

        duration_ms = (time.time() - start_time) * 1000
        avg_change = total_change / len(items) if items else 0.0

        return DecayReport(
            workspace_id=workspace_id,
            items_processed=len(items),
            items_decayed=items_decayed,
            items_boosted=items_boosted,
            average_confidence_change=avg_change,
            adjustments=adjustments,
            duration_ms=duration_ms,
        )

    async def record_event(
        self,
        mound: "KnowledgeMound",
        item_id: str,
        event: ConfidenceEvent,
        reason: str = "",
    ) -> Optional[ConfidenceAdjustment]:
        """Record a confidence-affecting event.

        Args:
            mound: KnowledgeMound instance
            item_id: Item affected
            event: Event type
            reason: Optional reason description

        Returns:
            ConfidenceAdjustment if confidence changed
        """
        import uuid

        # Get item
        item = await mound.get(item_id)  # type: ignore[arg-type,misc]
        if not item:
            return None

        old_confidence = getattr(item, "confidence", 0.5)
        if old_confidence is None:
            old_confidence = 0.5

        new_confidence = self.calculate_boost(old_confidence, event)

        if abs(new_confidence - old_confidence) < 0.001:
            return None

        adjustment = ConfidenceAdjustment(
            id=str(uuid.uuid4()),
            item_id=item_id,
            event=event,
            old_confidence=old_confidence,
            new_confidence=new_confidence,
            reason=reason or f"Event: {event.value}",
        )

        # Update item
        try:
            if hasattr(mound, "update_confidence"):
                # KnowledgeMound inherits update_confidence from CRUDMixin
                await mound.update_confidence(item_id, new_confidence)  # type: ignore[arg-type]
        except Exception as e:
            logger.warning(f"Failed to update confidence for {item_id}: {e}")

        async with self._lock:
            self._adjustments.append(adjustment)

        return adjustment

    async def get_adjustment_history(
        self,
        item_id: Optional[str] = None,
        event_type: Optional[ConfidenceEvent] = None,
        limit: int = 100,
    ) -> List[ConfidenceAdjustment]:
        """Get confidence adjustment history.

        Args:
            item_id: Filter by item ID
            event_type: Filter by event type
            limit: Maximum results

        Returns:
            List of adjustments
        """
        async with self._lock:
            results = self._adjustments

            if item_id:
                results = [a for a in results if a.item_id == item_id]

            if event_type:
                results = [a for a in results if a.event == event_type]

            return results[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get decay manager statistics."""
        by_event: Dict[str, int] = {}
        total_positive = 0
        total_negative = 0

        for adj in self._adjustments:
            by_event[adj.event.value] = by_event.get(adj.event.value, 0) + 1
            change = adj.new_confidence - adj.old_confidence
            if change > 0:
                total_positive += 1
            elif change < 0:
                total_negative += 1

        return {
            "total_adjustments": len(self._adjustments),
            "by_event": by_event,
            "positive_adjustments": total_positive,
            "negative_adjustments": total_negative,
            "last_decay_runs": {k: v.isoformat() for k, v in self._last_decay_run.items()},
        }


class ConfidenceDecayMixin:
    """Mixin for confidence decay operations on KnowledgeMound."""

    _decay_manager: Optional[ConfidenceDecayManager] = None

    def _get_decay_manager(self) -> ConfidenceDecayManager:
        """Get or create decay manager."""
        if self._decay_manager is None:
            self._decay_manager = ConfidenceDecayManager()
        return self._decay_manager

    async def apply_confidence_decay(
        self,
        workspace_id: str,
        force: bool = False,
    ) -> DecayReport:
        """Apply confidence decay to workspace items.

        Args:
            workspace_id: Workspace to process
            force: Force decay even if recently run

        Returns:
            DecayReport with results
        """
        manager = self._get_decay_manager()
        return await manager.apply_decay(self, workspace_id, force)  # type: ignore[arg-type]

    async def record_confidence_event(
        self,
        item_id: str,
        event: ConfidenceEvent,
        reason: str = "",
    ) -> Optional[ConfidenceAdjustment]:
        """Record a confidence-affecting event.

        Args:
            item_id: Item affected
            event: Event type
            reason: Optional reason

        Returns:
            ConfidenceAdjustment if confidence changed
        """
        manager = self._get_decay_manager()
        return await manager.record_event(self, item_id, event, reason)  # type: ignore[arg-type]

    async def get_confidence_history(
        self,
        item_id: Optional[str] = None,
        event_type: Optional[ConfidenceEvent] = None,
        limit: int = 100,
    ) -> List[ConfidenceAdjustment]:
        """Get confidence adjustment history."""
        manager = self._get_decay_manager()
        return await manager.get_adjustment_history(item_id, event_type, limit)

    def get_decay_stats(self) -> Dict[str, Any]:
        """Get confidence decay statistics."""
        manager = self._get_decay_manager()
        return manager.get_stats()


# Singleton instance
_decay_manager: Optional[ConfidenceDecayManager] = None


def get_decay_manager() -> ConfidenceDecayManager:
    """Get the global confidence decay manager instance."""
    global _decay_manager
    if _decay_manager is None:
        _decay_manager = ConfidenceDecayManager()
    return _decay_manager
