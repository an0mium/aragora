"""
Tier Manager for Continuum Memory System.

Extracts tier configuration and transition logic from ContinuumMemory,
providing a configurable component for managing memory tier lifecycles.

Usage:
    from aragora.memory.tier_manager import TierManager, MemoryTier

    manager = TierManager()

    # Check if entry should promote/demote
    if manager.should_promote(entry):
        new_tier = manager.get_next_tier(entry.tier, "faster")

    # Get tier metrics
    stats = manager.get_transition_stats()
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MemoryTier(Enum):
    """Memory update frequency tiers."""
    FAST = "fast"       # Updates on every event
    MEDIUM = "medium"   # Updates per debate round
    SLOW = "slow"       # Updates per nomic cycle
    GLACIAL = "glacial" # Updates monthly


# Tier ordering from slowest to fastest
TIER_ORDER: List[MemoryTier] = [
    MemoryTier.GLACIAL,
    MemoryTier.SLOW,
    MemoryTier.MEDIUM,
    MemoryTier.FAST,
]


@dataclass
class TierConfig:
    """Configuration for a memory tier."""
    name: str
    half_life_hours: float
    update_frequency: str
    base_learning_rate: float
    decay_rate: float
    promotion_threshold: float  # Surprise score to promote to faster tier
    demotion_threshold: float   # Stability score to demote to slower tier

    @property
    def half_life_seconds(self) -> float:
        """Half-life in seconds."""
        return self.half_life_hours * 3600


# Default tier configurations (HOPE-inspired nested learning)
DEFAULT_TIER_CONFIGS: Dict[MemoryTier, TierConfig] = {
    MemoryTier.FAST: TierConfig(
        name="fast",
        half_life_hours=1,
        update_frequency="event",
        base_learning_rate=0.3,
        decay_rate=0.95,
        promotion_threshold=1.0,  # Can't promote higher
        demotion_threshold=0.2,   # Very stable patterns demote
    ),
    MemoryTier.MEDIUM: TierConfig(
        name="medium",
        half_life_hours=24,
        update_frequency="round",
        base_learning_rate=0.1,
        decay_rate=0.99,
        promotion_threshold=0.7,  # High surprise promotes to fast
        demotion_threshold=0.3,
    ),
    MemoryTier.SLOW: TierConfig(
        name="slow",
        half_life_hours=168,  # 7 days
        update_frequency="cycle",
        base_learning_rate=0.03,
        decay_rate=0.999,
        promotion_threshold=0.6,  # Medium surprise promotes to medium
        demotion_threshold=0.4,
    ),
    MemoryTier.GLACIAL: TierConfig(
        name="glacial",
        half_life_hours=720,  # 30 days
        update_frequency="monthly",
        base_learning_rate=0.01,
        decay_rate=0.9999,
        promotion_threshold=0.5,  # Low bar to promote (rarely happens)
        demotion_threshold=1.0,   # Can't demote lower
    ),
}


@dataclass
class TierTransitionMetrics:
    """Metrics for tier transition tracking."""
    promotions: Dict[str, int] = field(default_factory=dict)  # tier -> count
    demotions: Dict[str, int] = field(default_factory=dict)   # tier -> count
    total_promotions: int = 0
    total_demotions: int = 0
    last_reset: str = field(default_factory=lambda: datetime.now().isoformat())
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record_promotion(self, from_tier: MemoryTier, to_tier: MemoryTier) -> None:
        """Record a promotion event (thread-safe)."""
        with self._lock:
            key = f"{from_tier.value}->{to_tier.value}"
            self.promotions[key] = self.promotions.get(key, 0) + 1
            self.total_promotions += 1

    def record_demotion(self, from_tier: MemoryTier, to_tier: MemoryTier) -> None:
        """Record a demotion event (thread-safe)."""
        with self._lock:
            key = f"{from_tier.value}->{to_tier.value}"
            self.demotions[key] = self.demotions.get(key, 0) + 1
            self.total_demotions += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization (thread-safe)."""
        with self._lock:
            return {
                "promotions": dict(self.promotions),
                "demotions": dict(self.demotions),
                "total_promotions": self.total_promotions,
                "total_demotions": self.total_demotions,
                "last_reset": self.last_reset,
            }

    def reset(self) -> None:
        """Reset metrics (thread-safe)."""
        with self._lock:
            self.promotions = {}
            self.demotions = {}
            self.total_promotions = 0
            self.total_demotions = 0
            self.last_reset = datetime.now().isoformat()


class TierManager:
    """
    Manager for memory tier configuration and transitions.

    Provides:
    - Tier configuration lookup
    - Transition decision logic
    - Transition metrics tracking
    - Custom tier configuration support
    """

    def __init__(
        self,
        configs: Optional[Dict[MemoryTier, TierConfig]] = None,
        promotion_cooldown_hours: float = 24.0,
        min_updates_for_demotion: int = 10,
    ):
        """
        Initialize the tier manager.

        Args:
            configs: Optional custom tier configurations. Uses defaults if not provided.
            promotion_cooldown_hours: Minimum hours between promotions for same entry.
            min_updates_for_demotion: Minimum updates before demotion is considered.
        """
        self._configs = configs or DEFAULT_TIER_CONFIGS.copy()
        self._promotion_cooldown_hours = promotion_cooldown_hours
        self._min_updates_for_demotion = min_updates_for_demotion
        self._metrics = TierTransitionMetrics()

    def get_config(self, tier: MemoryTier) -> TierConfig:
        """Get configuration for a tier."""
        return self._configs[tier]

    def get_all_configs(self) -> Dict[MemoryTier, TierConfig]:
        """Get all tier configurations."""
        return self._configs.copy()

    def update_config(self, tier: MemoryTier, config: TierConfig) -> None:
        """Update configuration for a tier."""
        self._configs[tier] = config
        logger.info(f"Updated tier config for {tier.value}")

    def get_tier_index(self, tier: MemoryTier) -> int:
        """Get the index of a tier in the ordering (0=slowest, 3=fastest)."""
        return TIER_ORDER.index(tier)

    def get_next_tier(self, current_tier: MemoryTier, direction: str) -> Optional[MemoryTier]:
        """
        Get the next tier in the specified direction.

        Args:
            current_tier: Current memory tier
            direction: "faster" for promotion, "slower" for demotion

        Returns:
            Next tier or None if already at boundary
        """
        idx = self.get_tier_index(current_tier)

        if direction == "faster":
            if idx >= len(TIER_ORDER) - 1:
                return None
            return TIER_ORDER[idx + 1]
        elif direction == "slower":
            if idx <= 0:
                return None
            return TIER_ORDER[idx - 1]
        else:
            raise ValueError(f"Invalid direction: {direction}")

    def should_promote(
        self,
        tier: MemoryTier,
        surprise_score: float,
        last_promotion_at: Optional[str] = None,
    ) -> bool:
        """
        Check if a memory entry should be promoted to a faster tier.

        Args:
            tier: Current tier
            surprise_score: Current surprise score (0-1)
            last_promotion_at: ISO timestamp of last promotion, if any

        Returns:
            True if entry should be promoted
        """
        # Already at fastest tier
        if tier == MemoryTier.FAST:
            return False

        # Check cooldown
        if last_promotion_at:
            last_dt = datetime.fromisoformat(last_promotion_at)
            hours_since = (datetime.now() - last_dt).total_seconds() / 3600
            if hours_since < self._promotion_cooldown_hours:
                return False

        # Check threshold
        config = self._configs[tier]
        return surprise_score > config.promotion_threshold

    def should_demote(
        self,
        tier: MemoryTier,
        surprise_score: float,
        update_count: int,
    ) -> bool:
        """
        Check if a memory entry should be demoted to a slower tier.

        Args:
            tier: Current tier
            surprise_score: Current surprise score (0-1)
            update_count: Number of times this entry has been updated

        Returns:
            True if entry should be demoted
        """
        # Already at slowest tier
        if tier == MemoryTier.GLACIAL:
            return False

        # Need enough updates to be confident about stability
        if update_count < self._min_updates_for_demotion:
            return False

        # Check threshold (stability = 1 - surprise)
        config = self._configs[tier]
        stability_score = 1.0 - surprise_score
        return stability_score > config.demotion_threshold

    def calculate_decay_factor(self, tier: MemoryTier, hours_elapsed: float) -> float:
        """
        Calculate the decay factor for importance based on time elapsed.

        Uses exponential decay with tier-specific half-life.

        Args:
            tier: Memory tier
            hours_elapsed: Hours since last update

        Returns:
            Decay multiplier (0-1)
        """
        config = self._configs[tier]
        half_life = config.half_life_hours

        if hours_elapsed <= 0:
            return 1.0

        # Exponential decay: factor = 0.5^(t/half_life)
        return 0.5 ** (hours_elapsed / half_life)

    def record_promotion(self, from_tier: MemoryTier, to_tier: MemoryTier) -> None:
        """Record a promotion event for metrics."""
        self._metrics.record_promotion(from_tier, to_tier)

    def record_demotion(self, from_tier: MemoryTier, to_tier: MemoryTier) -> None:
        """Record a demotion event for metrics."""
        self._metrics.record_demotion(from_tier, to_tier)

    def get_metrics(self) -> TierTransitionMetrics:
        """Get current transition metrics."""
        return self._metrics

    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get metrics as a dictionary."""
        return self._metrics.to_dict()

    def reset_metrics(self) -> None:
        """Reset transition metrics."""
        self._metrics.reset()

    @property
    def promotion_cooldown_hours(self) -> float:
        """Get promotion cooldown in hours."""
        return self._promotion_cooldown_hours

    @promotion_cooldown_hours.setter
    def promotion_cooldown_hours(self, value: float) -> None:
        """Set promotion cooldown in hours."""
        self._promotion_cooldown_hours = max(0, value)

    @property
    def min_updates_for_demotion(self) -> int:
        """Get minimum updates required for demotion."""
        return self._min_updates_for_demotion

    @min_updates_for_demotion.setter
    def min_updates_for_demotion(self, value: int) -> None:
        """Set minimum updates required for demotion."""
        self._min_updates_for_demotion = max(1, value)


# Use ServiceRegistry for singleton management
# Backward-compatible - these functions still work but delegate to registry


def get_tier_manager() -> TierManager:
    """Get the default TierManager instance.

    Uses ServiceRegistry for centralized singleton management.
    """
    from aragora.services import ServiceRegistry

    registry = ServiceRegistry.get()
    if not registry.has(TierManager):
        registry.register(TierManager, TierManager())
    return registry.resolve(TierManager)


def reset_tier_manager() -> None:
    """Reset the default TierManager (useful for testing).

    Removes TierManager from the ServiceRegistry.
    """
    from aragora.services import ServiceRegistry

    ServiceRegistry.get().unregister(TierManager)
