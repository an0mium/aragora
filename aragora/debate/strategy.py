"""
Memory-aware debate strategy for adaptive round planning.

Uses ContinuumMemory tier confidence to inform debate duration.
High-confidence memories from glacial tier suggest shorter debates,
while novel topics without prior knowledge warrant more exploration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from aragora.memory.continuum import ContinuumMemory, ContinuumMemoryEntry

logger = logging.getLogger(__name__)


@dataclass
class StrategyRecommendation:
    """Recommendation from debate strategy analysis."""

    estimated_rounds: int
    confidence: float
    reasoning: str
    relevant_memories: List[str]  # Memory IDs used in analysis


class DebateStrategy:
    """
    Memory-aware debate strategy for adaptive round planning.

    Uses ContinuumMemory to:
    1. Check if similar tasks have been solved with high confidence
    2. Recommend appropriate number of debate rounds
    3. Provide relevant context from prior debates
    """

    # Default round estimates by scenario
    QUICK_VALIDATION_ROUNDS = 2  # Known topic with high confidence
    STANDARD_DEBATE_ROUNDS = 3  # Some prior knowledge
    EXPLORATION_ROUNDS = 5  # Novel topic, no prior knowledge

    # Confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.9
    MEDIUM_CONFIDENCE_THRESHOLD = 0.7

    def __init__(
        self,
        continuum_memory: Optional["ContinuumMemory"] = None,
        quick_validation_rounds: int = 2,
        standard_rounds: int = 3,
        exploration_rounds: int = 5,
        high_confidence_threshold: float = 0.9,
        medium_confidence_threshold: float = 0.7,
    ):
        """Initialize the debate strategy.

        Args:
            continuum_memory: ContinuumMemory instance for retrieving prior knowledge
            quick_validation_rounds: Rounds for high-confidence known topics
            standard_rounds: Rounds for partially known topics
            exploration_rounds: Rounds for novel topics
            high_confidence_threshold: Success rate threshold for quick debates
            medium_confidence_threshold: Success rate threshold for standard debates
        """
        self.continuum_memory = continuum_memory
        self.quick_validation_rounds = quick_validation_rounds
        self.standard_rounds = standard_rounds
        self.exploration_rounds = exploration_rounds
        self.high_confidence_threshold = high_confidence_threshold
        self.medium_confidence_threshold = medium_confidence_threshold

    def estimate_rounds(
        self,
        task: str,
        default_rounds: Optional[int] = None,
        respect_minimum: bool = True,
    ) -> StrategyRecommendation:
        """Estimate optimal number of debate rounds based on memory confidence.

        Args:
            task: The debate task to analyze
            default_rounds: Default rounds if no memory available (uses exploration_rounds)
            respect_minimum: If True, never recommend fewer than 2 rounds

        Returns:
            StrategyRecommendation with estimated rounds and reasoning
        """
        default = default_rounds or self.exploration_rounds

        if not self.continuum_memory:
            return StrategyRecommendation(
                estimated_rounds=default,
                confidence=0.0,
                reasoning="No memory system available, using default exploration rounds",
                relevant_memories=[],
            )

        # Check glacial tier first (foundational, long-term knowledge)
        try:
            from aragora.memory.tier_manager import MemoryTier

            glacial_memories = self.continuum_memory.retrieve(
                query=task,
                tiers=[MemoryTier.GLACIAL],
                limit=5,
                min_importance=0.5,
            )

            if glacial_memories:
                max_confidence = max(m.success_rate for m in glacial_memories)
                memory_ids = [m.id for m in glacial_memories]

                if max_confidence >= self.high_confidence_threshold:
                    rounds = (
                        max(2, self.quick_validation_rounds)
                        if respect_minimum
                        else self.quick_validation_rounds
                    )
                    return StrategyRecommendation(
                        estimated_rounds=rounds,
                        confidence=max_confidence,
                        reasoning=f"High-confidence glacial memory found (success_rate={max_confidence:.0%}). "
                        f"Quick validation debate recommended.",
                        relevant_memories=memory_ids,
                    )

            # Check slow tier (cross-session learning)
            slow_memories = self.continuum_memory.retrieve(
                query=task,
                tiers=[MemoryTier.SLOW],
                limit=5,
                min_importance=0.5,
            )

            if slow_memories:
                max_confidence = max(m.success_rate for m in slow_memories)
                memory_ids = [m.id for m in slow_memories]

                if max_confidence >= self.medium_confidence_threshold:
                    return StrategyRecommendation(
                        estimated_rounds=self.standard_rounds,
                        confidence=max_confidence,
                        reasoning=f"Medium-confidence slow-tier memory found (success_rate={max_confidence:.0%}). "
                        f"Standard debate recommended.",
                        relevant_memories=memory_ids,
                    )

            # Check medium tier (session memory)
            medium_memories = self.continuum_memory.retrieve(
                query=task,
                tiers=[MemoryTier.MEDIUM],
                limit=3,
                min_importance=0.3,
            )

            if medium_memories:
                max_confidence = max(m.success_rate for m in medium_memories)
                memory_ids = [m.id for m in medium_memories]

                # Some recent experience, but not consolidated enough for quick debate
                return StrategyRecommendation(
                    estimated_rounds=self.standard_rounds,
                    confidence=max_confidence * 0.8,  # Discount for recency
                    reasoning=f"Recent medium-tier memory found (success_rate={max_confidence:.0%}). "
                    f"Standard debate to verify and consolidate.",
                    relevant_memories=memory_ids,
                )

        except Exception as e:
            logger.warning(f"[debate_strategy] Error querying memory: {e}")
            return StrategyRecommendation(
                estimated_rounds=default,
                confidence=0.0,
                reasoning=f"Memory query failed: {e}. Using default rounds.",
                relevant_memories=[],
            )

        # No relevant memories found - exploration debate
        return StrategyRecommendation(
            estimated_rounds=self.exploration_rounds,
            confidence=0.0,
            reasoning="No relevant prior knowledge found. Exploration debate recommended.",
            relevant_memories=[],
        )

    async def estimate_rounds_async(
        self,
        task: str,
        default_rounds: Optional[int] = None,
        respect_minimum: bool = True,
    ) -> StrategyRecommendation:
        """Async version of estimate_rounds.

        Uses the async retrieval interface of ContinuumMemory.
        """
        default = default_rounds or self.exploration_rounds

        if not self.continuum_memory:
            return StrategyRecommendation(
                estimated_rounds=default,
                confidence=0.0,
                reasoning="No memory system available, using default exploration rounds",
                relevant_memories=[],
            )

        try:
            from aragora.memory.tier_manager import MemoryTier

            # Query all relevant tiers in parallel
            glacial_memories = await self.continuum_memory.retrieve_async(
                query=task,
                tiers=[MemoryTier.GLACIAL],
                limit=5,
                min_importance=0.5,
            )

            if glacial_memories:
                max_confidence = max(m.success_rate for m in glacial_memories)
                memory_ids = [m.id for m in glacial_memories]

                if max_confidence >= self.high_confidence_threshold:
                    rounds = (
                        max(2, self.quick_validation_rounds)
                        if respect_minimum
                        else self.quick_validation_rounds
                    )
                    return StrategyRecommendation(
                        estimated_rounds=rounds,
                        confidence=max_confidence,
                        reasoning=f"High-confidence glacial memory found (success_rate={max_confidence:.0%}). "
                        f"Quick validation debate recommended.",
                        relevant_memories=memory_ids,
                    )

            slow_memories = await self.continuum_memory.retrieve_async(
                query=task,
                tiers=[MemoryTier.SLOW],
                limit=5,
                min_importance=0.5,
            )

            if slow_memories:
                max_confidence = max(m.success_rate for m in slow_memories)
                memory_ids = [m.id for m in slow_memories]

                if max_confidence >= self.medium_confidence_threshold:
                    return StrategyRecommendation(
                        estimated_rounds=self.standard_rounds,
                        confidence=max_confidence,
                        reasoning=f"Medium-confidence slow-tier memory found (success_rate={max_confidence:.0%}). "
                        f"Standard debate recommended.",
                        relevant_memories=memory_ids,
                    )

            medium_memories = await self.continuum_memory.retrieve_async(
                query=task,
                tiers=[MemoryTier.MEDIUM],
                limit=3,
                min_importance=0.3,
            )

            if medium_memories:
                max_confidence = max(m.success_rate for m in medium_memories)
                memory_ids = [m.id for m in medium_memories]

                return StrategyRecommendation(
                    estimated_rounds=self.standard_rounds,
                    confidence=max_confidence * 0.8,
                    reasoning=f"Recent medium-tier memory found (success_rate={max_confidence:.0%}). "
                    f"Standard debate to verify and consolidate.",
                    relevant_memories=memory_ids,
                )

        except Exception as e:
            logger.warning(f"[debate_strategy] Async memory query failed: {e}")
            return StrategyRecommendation(
                estimated_rounds=default,
                confidence=0.0,
                reasoning=f"Memory query failed: {e}. Using default rounds.",
                relevant_memories=[],
            )

        return StrategyRecommendation(
            estimated_rounds=self.exploration_rounds,
            confidence=0.0,
            reasoning="No relevant prior knowledge found. Exploration debate recommended.",
            relevant_memories=[],
        )

    def get_relevant_context(self, task: str, limit: int = 3) -> List["ContinuumMemoryEntry"]:
        """Get relevant memories to include as debate context.

        Prioritizes glacial and slow tier memories with high success rates.

        Args:
            task: The debate task
            limit: Maximum number of memories to return

        Returns:
            List of relevant memory entries
        """
        if not self.continuum_memory:
            return []

        try:
            from aragora.memory.tier_manager import MemoryTier

            # Prioritize stable, high-quality memories
            memories = self.continuum_memory.retrieve(
                query=task,
                tiers=[MemoryTier.GLACIAL, MemoryTier.SLOW],
                limit=limit,
                min_importance=0.5,
            )

            return memories

        except Exception as e:
            logger.warning(f"[debate_strategy] Error getting context: {e}")
            return []
