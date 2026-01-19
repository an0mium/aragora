"""
RL Training loop for RLM context management.

Implements the training loop that:
1. Collects trajectories from RLM queries
2. Computes rewards using debate outcomes
3. Updates policies using policy gradient methods

Based on Prime Intellect's approach (arXiv:2512.24601).

Usage:
    from aragora.rlm.training import Trainer, TrainerConfig

    config = TrainerConfig(
        batch_size=32,
        learning_rate=0.001,
    )
    trainer = Trainer(config)

    # Training loop
    for epoch in range(epochs):
        metrics = trainer.train_step()
        print(f"Epoch {epoch}: reward={metrics['avg_reward']:.3f}")
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

from .buffer import ExperienceBuffer, Step, Trajectory
from .policy import Policy, PolicyState, RefinementPolicy, StrategyPolicy
from .reward import DebateOutcomeReward, RewardModel

if TYPE_CHECKING:
    from aragora.rlm.bridge import AragoraRLM
    from aragora.rlm.types import RLMContext, RLMResult

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Configuration for the RL trainer."""

    # Training parameters
    batch_size: int = 32
    learning_rate: float = 0.001
    discount_factor: float = 0.99  # Gamma for future rewards
    entropy_bonus: float = 0.01  # Encourage exploration

    # Experience collection
    trajectories_per_step: int = 10
    max_trajectory_length: int = 50

    # Buffer
    buffer_size: int = 10000
    prioritized_replay: bool = False
    priority_alpha: float = 0.6

    # Policy update
    update_frequency: int = 5  # Update every N trajectories
    target_update_frequency: int = 100  # Update target network

    # Refinement
    max_refinement_iterations: int = 5

    # Logging
    log_interval: int = 10
    checkpoint_interval: int = 100


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""

    epoch: int = 0
    total_trajectories: int = 0
    avg_reward: float = 0.0
    avg_steps: float = 0.0
    success_rate: float = 0.0
    avg_confidence: float = 0.0
    avg_iterations: float = 0.0

    # Component rewards
    consensus_reward: float = 0.0
    efficiency_reward: float = 0.0
    confidence_reward: float = 0.0
    iteration_reward: float = 0.0

    # Policy stats
    strategy_distribution: dict[str, float] = field(default_factory=dict)
    exploration_rate: float = 0.0

    # Timing
    training_time_seconds: float = 0.0
    collection_time_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "epoch": self.epoch,
            "total_trajectories": self.total_trajectories,
            "avg_reward": self.avg_reward,
            "avg_steps": self.avg_steps,
            "success_rate": self.success_rate,
            "avg_confidence": self.avg_confidence,
            "avg_iterations": self.avg_iterations,
            "consensus_reward": self.consensus_reward,
            "efficiency_reward": self.efficiency_reward,
            "confidence_reward": self.confidence_reward,
            "iteration_reward": self.iteration_reward,
            "strategy_distribution": self.strategy_distribution,
            "exploration_rate": self.exploration_rate,
            "training_time_seconds": self.training_time_seconds,
            "collection_time_seconds": self.collection_time_seconds,
        }


class Trainer:
    """
    RL Trainer for RLM context management.

    Trains policies to optimize:
    - Strategy selection (which decomposition approach to use)
    - Refinement decisions (when to continue vs. accept)
    - Context navigation (which nodes to examine)
    """

    def __init__(
        self,
        config: Optional[TrainerConfig] = None,
        strategy_policy: Optional[StrategyPolicy] = None,
        refinement_policy: Optional[RefinementPolicy] = None,
        reward_model: Optional[RewardModel] = None,
        experience_buffer: Optional[ExperienceBuffer] = None,
    ):
        """
        Initialize the trainer.

        Args:
            config: Training configuration
            strategy_policy: Policy for strategy selection
            refinement_policy: Policy for refinement decisions
            reward_model: Model for computing rewards
            experience_buffer: Buffer for storing trajectories
        """
        self.config = config or TrainerConfig()

        # Policies
        self.strategy_policy = strategy_policy or StrategyPolicy(
            exploration_rate=0.1
        )
        self.refinement_policy = refinement_policy or RefinementPolicy(
            max_iterations=self.config.max_refinement_iterations
        )

        # Reward computation
        self.reward_model = reward_model or DebateOutcomeReward()

        # Experience storage
        self.experience_buffer = experience_buffer or ExperienceBuffer(
            max_size=self.config.buffer_size,
            priority_alpha=self.config.priority_alpha if self.config.prioritized_replay else 0.0,
        )

        # Training state
        self._epoch = 0
        self._total_trajectories = 0
        self._callbacks: list[Callable[[TrainingMetrics], None]] = []

    def add_callback(self, callback: Callable[[TrainingMetrics], None]) -> None:
        """Add a callback to be called after each training step."""
        self._callbacks.append(callback)

    async def collect_trajectory(
        self,
        rlm: "AragoraRLM",
        query: str,
        context: "RLMContext",
    ) -> Trajectory:
        """
        Collect a single trajectory by running an RLM query.

        Args:
            rlm: The RLM instance to use
            query: Query to answer
            context: Compressed context to query

        Returns:
            Completed trajectory with outcome
        """
        # Create trajectory
        trajectory = Trajectory(
            query=query,
            context_tokens=context.original_tokens,
            source_type=context.source_type,
        )

        # Build policy state
        state = PolicyState(
            query=query,
            context_tokens=context.original_tokens,
            abstraction_levels=len(context.levels),
            source_type=context.source_type,
        )

        # Select strategy using policy
        strategy = self.strategy_policy.act(state)
        trajectory.strategy = strategy

        # Record strategy selection step
        trajectory.add_step(Step(
            action=f"strategy = '{strategy}'",
            action_type="strategy",
            state={"query": query, "context_tokens": context.original_tokens},
        ))

        # Execute query with refinement
        start_time = time.time()
        try:
            result = await rlm.query_with_refinement(
                query,
                context,
                strategy,
                max_iterations=self.config.max_refinement_iterations,
            )

            # Record refinement steps
            for i, history in enumerate(result.refinement_history or []):
                trajectory.add_step(Step(
                    action=f"iteration_{i+1}",
                    action_type="refinement",
                    observation=history[:500] if history else "",
                    tokens_examined=result.tokens_processed // max(1, result.iteration),
                    sub_calls=result.sub_calls_made // max(1, result.iteration),
                ))

            # Finalize trajectory
            duration = time.time() - start_time
            trajectory.finalize(
                answer=result.answer,
                outcome={
                    "consensus_reached": result.ready,
                    "agreement_score": result.confidence,
                    "success": result.ready and result.confidence > 0.5,
                },
                stats={
                    "sub_calls_made": result.sub_calls_made,
                    "confidence": result.confidence,
                    "iterations": result.iteration,
                    "ready": result.ready,
                    "tokens_processed": result.tokens_processed,
                    "duration_seconds": duration,
                },
            )

        except Exception as e:
            logger.error(f"Error collecting trajectory: {e}")
            trajectory.finalize(
                answer="",
                outcome={"success": False, "error": str(e)},
                stats={"confidence": 0.0, "ready": False},
            )

        return trajectory

    async def collect_trajectories(
        self,
        rlm: "AragoraRLM",
        queries: list[tuple[str, "RLMContext"]],
    ) -> list[Trajectory]:
        """
        Collect multiple trajectories.

        Args:
            rlm: The RLM instance
            queries: List of (query, context) pairs

        Returns:
            List of completed trajectories
        """
        trajectories = []
        for query, context in queries:
            trajectory = await self.collect_trajectory(rlm, query, context)
            trajectories.append(trajectory)
            self._total_trajectories += 1

        return trajectories

    def compute_rewards(
        self,
        trajectories: list[Trajectory],
    ) -> list[tuple[Trajectory, float, dict[str, float]]]:
        """
        Compute rewards for trajectories.

        Args:
            trajectories: List of trajectories

        Returns:
            List of (trajectory, total_reward, component_rewards)
        """
        results = []
        for trajectory in trajectories:
            total_reward = self.reward_model.compute(trajectory)
            components = self.reward_model.compute_components(trajectory)
            results.append((trajectory, total_reward, components))
        return results

    def update_policies(
        self,
        trajectories: list[Trajectory],
        rewards: list[tuple[Trajectory, float, dict[str, float]]],
    ) -> None:
        """
        Update policies based on trajectory rewards.

        Uses a simple REINFORCE-style update for now.

        Args:
            trajectories: Collected trajectories
            rewards: Computed rewards for each trajectory
        """
        # Group trajectories by strategy
        strategy_rewards: dict[str, list[float]] = {}
        for trajectory, total_reward, _ in rewards:
            strategy = trajectory.strategy
            if strategy not in strategy_rewards:
                strategy_rewards[strategy] = []
            strategy_rewards[strategy].append(total_reward)

        # Update strategy policy weights
        avg_reward = sum(r for _, r, _ in rewards) / len(rewards) if rewards else 0
        for strategy, r_list in strategy_rewards.items():
            strategy_avg = sum(r_list) / len(r_list)
            advantage = strategy_avg - avg_reward

            # Update weights for features that were active
            self.strategy_policy.update_weights(
                strategy,
                "base",
                advantage,
                self.config.learning_rate,
            )

        # Decay exploration rate
        current_exploration = self.strategy_policy.exploration_rate
        self.strategy_policy.exploration_rate = max(
            0.01,  # Minimum exploration
            current_exploration * 0.999,
        )

    def store_trajectories(
        self,
        rewards: list[tuple[Trajectory, float, dict[str, float]]],
    ) -> None:
        """
        Store trajectories in experience buffer.

        Args:
            rewards: List of (trajectory, reward, components)
        """
        for trajectory, total_reward, _ in rewards:
            # Use |reward| as priority for prioritized replay
            priority = abs(total_reward) + 0.1
            self.experience_buffer.add(trajectory, priority)

    async def train_step(
        self,
        rlm: "AragoraRLM",
        queries: list[tuple[str, "RLMContext"]],
    ) -> TrainingMetrics:
        """
        Execute one training step.

        Args:
            rlm: The RLM instance
            queries: Query-context pairs for this step

        Returns:
            Training metrics for this step
        """
        self._epoch += 1
        metrics = TrainingMetrics(epoch=self._epoch)

        # Collect trajectories
        collection_start = time.time()
        trajectories = await self.collect_trajectories(rlm, queries)
        metrics.collection_time_seconds = time.time() - collection_start

        if not trajectories:
            return metrics

        # Compute rewards
        training_start = time.time()
        rewards = self.compute_rewards(trajectories)

        # Update policies
        self.update_policies(trajectories, rewards)

        # Store in buffer
        self.store_trajectories(rewards)

        metrics.training_time_seconds = time.time() - training_start

        # Compute metrics
        metrics.total_trajectories = self._total_trajectories
        metrics.avg_reward = sum(r for _, r, _ in rewards) / len(rewards)
        metrics.avg_steps = sum(len(t.steps) for t in trajectories) / len(trajectories)
        metrics.success_rate = sum(
            1 for t in trajectories if t.outcome.get("success", False)
        ) / len(trajectories)
        metrics.avg_confidence = sum(
            t.stats.get("confidence", 0) for t in trajectories
        ) / len(trajectories)
        metrics.avg_iterations = sum(
            t.stats.get("iterations", 1) for t in trajectories
        ) / len(trajectories)

        # Component rewards
        all_components: dict[str, list[float]] = {}
        for _, _, components in rewards:
            for key, value in components.items():
                if key not in all_components:
                    all_components[key] = []
                all_components[key].append(value)

        metrics.consensus_reward = sum(all_components.get("consensus", [0])) / max(
            1, len(all_components.get("consensus", []))
        )
        metrics.efficiency_reward = sum(all_components.get("efficiency", [0])) / max(
            1, len(all_components.get("efficiency", []))
        )
        metrics.confidence_reward = sum(all_components.get("confidence", [0])) / max(
            1, len(all_components.get("confidence", []))
        )
        metrics.iteration_reward = sum(all_components.get("iteration", [0])) / max(
            1, len(all_components.get("iteration", []))
        )

        # Strategy distribution
        strategy_counts: dict[str, int] = {}
        for t in trajectories:
            strategy_counts[t.strategy] = strategy_counts.get(t.strategy, 0) + 1
        total = len(trajectories)
        metrics.strategy_distribution = {
            s: c / total for s, c in strategy_counts.items()
        }

        metrics.exploration_rate = self.strategy_policy.exploration_rate

        # Invoke callbacks
        for callback in self._callbacks:
            try:
                callback(metrics)
            except Exception as e:
                logger.error(f"Callback error: {e}")

        return metrics

    async def train(
        self,
        rlm: "AragoraRLM",
        query_generator: Callable[[], list[tuple[str, "RLMContext"]]],
        epochs: int,
        on_epoch: Optional[Callable[[int, TrainingMetrics], None]] = None,
    ) -> list[TrainingMetrics]:
        """
        Run full training loop.

        Args:
            rlm: The RLM instance
            query_generator: Function that generates query-context pairs
            epochs: Number of training epochs
            on_epoch: Callback after each epoch

        Returns:
            List of metrics for each epoch
        """
        all_metrics = []

        for epoch in range(epochs):
            # Generate queries for this epoch
            queries = query_generator()

            # Run training step
            metrics = await self.train_step(rlm, queries)
            all_metrics.append(metrics)

            # Log progress
            if epoch % self.config.log_interval == 0:
                logger.info(
                    f"Epoch {epoch}: reward={metrics.avg_reward:.3f}, "
                    f"success={metrics.success_rate:.1%}, "
                    f"confidence={metrics.avg_confidence:.2f}"
                )

            # Callback
            if on_epoch:
                on_epoch(epoch, metrics)

            # Checkpoint
            if epoch % self.config.checkpoint_interval == 0 and epoch > 0:
                self._save_checkpoint(epoch)

        return all_metrics

    def _save_checkpoint(self, epoch: int) -> None:
        """Save training checkpoint."""
        logger.info(f"Saving checkpoint at epoch {epoch}")
        # Save experience buffer
        self.experience_buffer.save(f"rlm_buffer_epoch_{epoch}.json")

    def get_buffer_stats(self) -> dict[str, Any]:
        """Get experience buffer statistics."""
        return self.experience_buffer.get_stats()

    def get_policy_stats(self) -> dict[str, Any]:
        """Get policy statistics."""
        return {
            "strategy_policy": {
                "strategies": self.strategy_policy.strategies,
                "exploration_rate": self.strategy_policy.exploration_rate,
                "weights": dict(self.strategy_policy._weights),
            },
            "refinement_policy": {
                "confidence_threshold": self.refinement_policy.confidence_threshold,
                "max_iterations": self.refinement_policy.max_iterations,
            },
        }


__all__ = [
    "TrainerConfig",
    "TrainingMetrics",
    "Trainer",
]
