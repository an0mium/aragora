"""
RLM Bridge to Selection Feedback Bridge.

Bridges RLM compression metrics into the SelectionFeedbackLoop,
enabling selection optimization for agents that work well with compressed context.

This closes the loop between:
1. RLM Compressor: Tracks compression quality, fidelity, and sub-call efficiency
2. SelectionFeedbackLoop: Adjusts selection weights based on performance metrics

By connecting them, we enable:
- Selecting agents that maintain quality under compression
- Prioritizing agents efficient at navigating hierarchical context
- Optimizing for long debates where RLM is essential

Usage:
    from aragora.rlm.rlm_selection_bridge import RLMSelectionBridge

    bridge = RLMSelectionBridge(
        rlm_bridge=rlm_bridge,
        selection_feedback=feedback_loop,
    )

    # After compression/query operations
    bridge.record_rlm_operation(agent_name, compression_result)

    # Get selection boost for agents good with RLM
    boost = bridge.get_compression_boost("claude")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from aragora.debate.selection_feedback import SelectionFeedbackLoop
    from aragora.rlm.bridge import RLMBridge
    from aragora.rlm.types import CompressionResult, RLMResult

logger = logging.getLogger(__name__)


@dataclass
class AgentRLMStats:
    """RLM performance statistics for a single agent."""

    agent_name: str
    total_operations: int = 0
    total_compressions: int = 0
    total_queries: int = 0

    # Compression quality metrics
    total_fidelity: float = 0.0
    avg_compression_ratio: float = 0.0
    total_compression_ratio: float = 0.0

    # Query metrics
    total_confidence: float = 0.0
    total_sub_calls: int = 0
    total_tokens_processed: int = 0
    total_query_time: float = 0.0

    # Efficiency
    total_cache_hits: int = 0
    total_operations_time: float = 0.0

    last_updated: datetime = field(default_factory=datetime.now)

    @property
    def avg_fidelity(self) -> float:
        """Average compression fidelity (0-1)."""
        if self.total_compressions == 0:
            return 1.0
        return self.total_fidelity / self.total_compressions

    @property
    def avg_confidence(self) -> float:
        """Average query confidence (0-1)."""
        if self.total_queries == 0:
            return 0.5
        return self.total_confidence / self.total_queries

    @property
    def avg_sub_calls_per_query(self) -> float:
        """Average sub-LM calls per query (lower = more efficient)."""
        if self.total_queries == 0:
            return 0.0
        return self.total_sub_calls / self.total_queries

    @property
    def efficiency_score(self) -> float:
        """Combined efficiency score (0-1, higher = better)."""
        if self.total_operations == 0:
            return 0.5

        # Combine fidelity, confidence, and sub-call efficiency
        fidelity_score = self.avg_fidelity
        confidence_score = self.avg_confidence

        # Sub-calls: 0-3 is great, 5+ is mediocre, 10+ is poor
        if self.avg_sub_calls_per_query <= 3:
            sub_call_score = 1.0
        elif self.avg_sub_calls_per_query <= 10:
            sub_call_score = 1.0 - (self.avg_sub_calls_per_query - 3) / 7 * 0.5
        else:
            sub_call_score = 0.5 - min(0.3, (self.avg_sub_calls_per_query - 10) / 20)

        # Cache hits bonus
        cache_ratio = self.total_cache_hits / max(1, self.total_operations)
        cache_bonus = min(0.1, cache_ratio * 0.2)

        return (fidelity_score * 0.3 + confidence_score * 0.3 +
                sub_call_score * 0.3 + cache_bonus + 0.1)


@dataclass
class RLMSelectionBridgeConfig:
    """Configuration for the RLM-selection bridge."""

    # Minimum operations before applying boost
    min_operations_for_boost: int = 5

    # Boost weight for compression efficiency
    compression_boost_weight: float = 0.15

    # Boost weight for query efficiency
    query_boost_weight: float = 0.15

    # Minimum fidelity threshold for positive boost
    min_fidelity_threshold: float = 0.7

    # Maximum boost (prevents over-selection)
    max_boost: float = 0.25

    # Penalty for low fidelity
    low_fidelity_penalty: float = 0.1


@dataclass
class RLMSelectionBridge:
    """Bridges RLM compression/query metrics into SelectionFeedbackLoop decisions.

    Key integration points:
    1. Tracks compression fidelity per agent
    2. Monitors query efficiency (sub-calls, confidence)
    3. Computes selection boosts for RLM-efficient agents
    4. Enables optimization for long debates requiring RLM
    """

    rlm_bridge: Optional["RLMBridge"] = None
    selection_feedback: Optional["SelectionFeedbackLoop"] = None
    config: RLMSelectionBridgeConfig = field(
        default_factory=RLMSelectionBridgeConfig
    )

    # Internal state
    _agent_stats: Dict[str, AgentRLMStats] = field(default_factory=dict, repr=False)
    _rlm_adjustments: Dict[str, float] = field(default_factory=dict, repr=False)

    def record_compression(
        self,
        agent_name: str,
        compression_result: "CompressionResult",
    ) -> float:
        """Record a compression operation result.

        Args:
            agent_name: Name of the agent
            compression_result: Result from RLM compression

        Returns:
            Current RLM boost for this agent
        """
        stats = self._get_or_create_stats(agent_name)

        stats.total_operations += 1
        stats.total_compressions += 1
        stats.total_fidelity += compression_result.estimated_fidelity
        stats.total_operations_time += compression_result.time_seconds
        stats.total_cache_hits += compression_result.cache_hits

        # Compute average compression ratio across levels
        if compression_result.compression_ratio:
            avg_ratio = sum(compression_result.compression_ratio.values()) / len(
                compression_result.compression_ratio
            )
            stats.total_compression_ratio += avg_ratio
            stats.avg_compression_ratio = (
                stats.total_compression_ratio / stats.total_compressions
            )

        stats.last_updated = datetime.now()

        # Recompute adjustment
        adjustment = self._compute_adjustment(stats)
        self._rlm_adjustments[agent_name] = adjustment

        logger.debug(
            f"rlm_compression_recorded agent={agent_name} "
            f"fidelity={compression_result.estimated_fidelity:.2f} "
            f"adjustment={adjustment:.3f}"
        )

        return adjustment

    def record_query(
        self,
        agent_name: str,
        query_result: "RLMResult",
    ) -> float:
        """Record an RLM query result.

        Args:
            agent_name: Name of the agent
            query_result: Result from RLM query

        Returns:
            Current RLM boost for this agent
        """
        stats = self._get_or_create_stats(agent_name)

        stats.total_operations += 1
        stats.total_queries += 1
        stats.total_confidence += query_result.confidence
        stats.total_sub_calls += query_result.sub_calls_made
        stats.total_tokens_processed += query_result.tokens_processed
        stats.total_query_time += query_result.time_seconds
        stats.last_updated = datetime.now()

        # Recompute adjustment
        adjustment = self._compute_adjustment(stats)
        self._rlm_adjustments[agent_name] = adjustment

        logger.debug(
            f"rlm_query_recorded agent={agent_name} "
            f"confidence={query_result.confidence:.2f} "
            f"sub_calls={query_result.sub_calls_made} "
            f"adjustment={adjustment:.3f}"
        )

        return adjustment

    def record_rlm_operation(
        self,
        agent_name: str,
        result: Any,
    ) -> float:
        """Record any RLM operation result.

        Automatically detects result type and delegates.

        Args:
            agent_name: Name of the agent
            result: CompressionResult or RLMResult

        Returns:
            Current RLM boost for this agent
        """
        # Import here to avoid circular imports
        from aragora.rlm.types import CompressionResult, RLMResult

        if isinstance(result, CompressionResult):
            return self.record_compression(agent_name, result)
        elif isinstance(result, RLMResult):
            return self.record_query(agent_name, result)
        else:
            logger.warning(f"Unknown RLM result type: {type(result)}")
            return self._rlm_adjustments.get(agent_name, 0.0)

    def _get_or_create_stats(self, agent_name: str) -> AgentRLMStats:
        """Get or create RLM stats for an agent."""
        if agent_name not in self._agent_stats:
            self._agent_stats[agent_name] = AgentRLMStats(agent_name=agent_name)
        return self._agent_stats[agent_name]

    def _compute_adjustment(self, stats: AgentRLMStats) -> float:
        """Compute selection adjustment based on RLM stats.

        Args:
            stats: Agent's RLM statistics

        Returns:
            Adjustment factor (positive = boost, negative = penalty)
        """
        if stats.total_operations < self.config.min_operations_for_boost:
            return 0.0

        adjustment = 0.0

        # Compression fidelity component
        if stats.total_compressions > 0:
            if stats.avg_fidelity >= self.config.min_fidelity_threshold:
                # Boost for high fidelity
                fidelity_boost = (
                    (stats.avg_fidelity - self.config.min_fidelity_threshold)
                    / (1.0 - self.config.min_fidelity_threshold)
                    * self.config.compression_boost_weight
                )
                adjustment += fidelity_boost
            else:
                # Penalty for low fidelity
                fidelity_penalty = (
                    (self.config.min_fidelity_threshold - stats.avg_fidelity)
                    / self.config.min_fidelity_threshold
                    * self.config.low_fidelity_penalty
                )
                adjustment -= fidelity_penalty

        # Query efficiency component
        if stats.total_queries > 0:
            # High confidence = boost
            if stats.avg_confidence > 0.6:
                confidence_boost = (
                    (stats.avg_confidence - 0.6) / 0.4
                    * self.config.query_boost_weight
                )
                adjustment += confidence_boost

            # Low sub-calls = efficiency boost
            if stats.avg_sub_calls_per_query < 5:
                efficiency_boost = (
                    (5 - stats.avg_sub_calls_per_query) / 5
                    * self.config.query_boost_weight * 0.5
                )
                adjustment += efficiency_boost

        # Apply bounds
        return max(-self.config.low_fidelity_penalty,
                   min(self.config.max_boost, adjustment))

    def get_compression_boost(self, agent_name: str) -> float:
        """Get the compression efficiency boost for an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Positive boost value (0 if no boost)
        """
        return max(0.0, self._rlm_adjustments.get(agent_name, 0.0))

    def get_compression_penalty(self, agent_name: str) -> float:
        """Get the compression inefficiency penalty for an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Negative penalty value (0 if no penalty)
        """
        return min(0.0, self._rlm_adjustments.get(agent_name, 0.0))

    def get_combined_adjustment(self, agent_name: str) -> float:
        """Get the combined RLM adjustment for an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Combined adjustment (boost + penalty)
        """
        return self._rlm_adjustments.get(agent_name, 0.0)

    def get_all_adjustments(self) -> Dict[str, float]:
        """Get RLM adjustments for all tracked agents.

        Returns:
            Dict mapping agent names to adjustments
        """
        return dict(self._rlm_adjustments)

    def get_rlm_efficient_agents(self, threshold: float = 0.7) -> List[str]:
        """Get agents with high RLM efficiency.

        Args:
            threshold: Minimum efficiency score (default 0.7)

        Returns:
            List of agent names with high efficiency
        """
        return [
            agent_name
            for agent_name, stats in self._agent_stats.items()
            if stats.total_operations >= self.config.min_operations_for_boost
            and stats.efficiency_score >= threshold
        ]

    def get_best_agents_for_long_debates(self, top_n: int = 5) -> List[str]:
        """Get agents best suited for long debates requiring RLM.

        Args:
            top_n: Number of agents to return

        Returns:
            List of agent names sorted by RLM efficiency
        """
        eligible_agents = [
            (name, stats.efficiency_score)
            for name, stats in self._agent_stats.items()
            if stats.total_operations >= self.config.min_operations_for_boost
        ]

        eligible_agents.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in eligible_agents[:top_n]]

    def sync_to_selection_feedback(self) -> int:
        """Sync RLM adjustments to the SelectionFeedbackLoop.

        Applies RLM-based adjustments to the feedback loop's selection weights.

        Returns:
            Number of agents updated
        """
        if self.selection_feedback is None:
            logger.debug("No selection feedback loop attached")
            return 0

        updated = 0
        for agent_name, adjustment in self._rlm_adjustments.items():
            if abs(adjustment) < 0.01:
                continue

            state = self.selection_feedback.get_agent_state(agent_name)
            if state:
                current = self.selection_feedback.get_selection_adjustment(agent_name)
                self.selection_feedback._selection_adjustments[agent_name] = (
                    current + adjustment
                )
                updated += 1

        logger.info(f"rlm_selection_synced agents_updated={updated}")
        return updated

    def get_agent_stats(self, agent_name: str) -> Optional[AgentRLMStats]:
        """Get RLM statistics for an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            AgentRLMStats if available
        """
        return self._agent_stats.get(agent_name)

    def get_all_stats(self) -> Dict[str, AgentRLMStats]:
        """Get RLM statistics for all agents.

        Returns:
            Dict mapping agent names to stats
        """
        return dict(self._agent_stats)

    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics.

        Returns:
            Dict with bridge metrics
        """
        efficient_agents = self.get_rlm_efficient_agents()

        return {
            "agents_tracked": len(self._agent_stats),
            "total_operations": sum(
                s.total_operations for s in self._agent_stats.values()
            ),
            "total_compressions": sum(
                s.total_compressions for s in self._agent_stats.values()
            ),
            "total_queries": sum(
                s.total_queries for s in self._agent_stats.values()
            ),
            "efficient_agents": len(efficient_agents),
            "avg_adjustment": (
                sum(self._rlm_adjustments.values()) / len(self._rlm_adjustments)
                if self._rlm_adjustments
                else 0.0
            ),
        }

    def reset(self) -> None:
        """Reset all RLM statistics."""
        self._agent_stats.clear()
        self._rlm_adjustments.clear()
        logger.debug("RLMSelectionBridge reset")


def create_rlm_selection_bridge(
    rlm_bridge: Optional["RLMBridge"] = None,
    selection_feedback: Optional["SelectionFeedbackLoop"] = None,
    **config_kwargs: Any,
) -> RLMSelectionBridge:
    """Create and configure an RLMSelectionBridge.

    Args:
        rlm_bridge: RLMBridge instance
        selection_feedback: SelectionFeedbackLoop instance
        **config_kwargs: Additional configuration options

    Returns:
        Configured RLMSelectionBridge instance
    """
    config = RLMSelectionBridgeConfig(**config_kwargs)
    return RLMSelectionBridge(
        rlm_bridge=rlm_bridge,
        selection_feedback=selection_feedback,
        config=config,
    )


__all__ = [
    "RLMSelectionBridge",
    "RLMSelectionBridgeConfig",
    "AgentRLMStats",
    "create_rlm_selection_bridge",
]
