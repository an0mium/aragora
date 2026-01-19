"""
RLM Training Module.

Provides infrastructure for RL training of context management strategies,
based on Prime Intellect's approach to learning optimal REPL interactions.

Components:
- reward: Reward models for computing training signals from debate outcomes
- buffer: Experience buffer for storing and sampling trajectories
- policy: Policy interfaces for strategy selection and refinement decisions

Usage:
    from aragora.rlm.training import (
        ExperienceBuffer,
        Trajectory,
        DebateOutcomeReward,
        StrategyPolicy,
    )

    # Create experience buffer
    buffer = ExperienceBuffer(max_size=10000)

    # Create reward model
    reward_model = DebateOutcomeReward()

    # Create policy
    policy = StrategyPolicy()

    # Record trajectory
    trajectory = Trajectory(query="What is the consensus?")
    trajectory.add_step(Step(action="strategy = 'grep'"))
    trajectory.finalize(
        answer="The consensus is...",
        outcome={"consensus_reached": True, "quality_score": 0.8},
    )

    # Compute reward
    reward = reward_model.compute(trajectory)

    # Store in buffer
    buffer.add(trajectory)

See docs/RLM_TRAINING.md for detailed documentation.
"""

from .buffer import ExperienceBuffer, Step, Trajectory
from .policy import (
    CompositePolicy,
    Policy,
    PolicyState,
    RefinementPolicy,
    StrategyPolicy,
)
from .reward import (
    CompositeReward,
    DebateOutcomeReward,
    RewardConfig,
    RewardModel,
    SparseReward,
)
from .trainer import Trainer, TrainerConfig, TrainingMetrics

__all__ = [
    # Buffer
    "ExperienceBuffer",
    "Step",
    "Trajectory",
    # Reward
    "RewardConfig",
    "RewardModel",
    "DebateOutcomeReward",
    "CompositeReward",
    "SparseReward",
    # Policy
    "PolicyState",
    "Policy",
    "StrategyPolicy",
    "RefinementPolicy",
    "CompositePolicy",
    # Trainer
    "Trainer",
    "TrainerConfig",
    "TrainingMetrics",
]
