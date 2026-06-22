"""
Tests for the RLM training module.

Tests the experience buffer, reward models, and trainer components
for reinforcement learning training of context management strategies.
"""

import pytest
from unittest.mock import MagicMock

from aragora.rlm.training.buffer import ExperienceBuffer, Step, Trajectory
from aragora.rlm.training.reward import (
    CompositeReward,
    DebateOutcomeReward,
    RewardConfig,
    SparseReward,
)
from aragora.rlm.training.trainer import TrainerConfig, TrainingMetrics


# =============================================================================
# Step Tests
# =============================================================================


class TestStep:
    """Tests for the Step dataclass."""

    def test_step_creation_defaults(self):
        """Test creating a step with defaults."""
        step = Step()
        assert step.state == {}
        assert step.action == ""
        assert step.action_type == "code"
        assert step.observation == ""
        assert step.tokens_examined == 0
        assert step.sub_calls == 0
        assert step.timestamp  # Auto-generated

    def test_step_creation_with_values(self):
        """Test creating a step with values."""
        step = Step(
            state={"context_size": 1000},
            action="strategy = 'grep'",
            action_type="strategy",
            observation="Found 5 matches",
            tokens_examined=500,
            sub_calls=2,
            duration_seconds=1.5,
        )
        assert step.state == {"context_size": 1000}
        assert step.action == "strategy = 'grep'"
        assert step.action_type == "strategy"
        assert step.observation == "Found 5 matches"
        assert step.tokens_examined == 500
        assert step.sub_calls == 2
        assert step.duration_seconds == 1.5


# =============================================================================
# Trajectory Tests
# =============================================================================


class TestTrajectory:
    """Tests for the Trajectory dataclass."""

    def test_trajectory_creation_defaults(self):
        """Test creating a trajectory with defaults."""
        trajectory = Trajectory()
        assert trajectory.trajectory_id  # Auto-generated
        assert trajectory.query == ""
        assert trajectory.strategy == "auto"
        assert trajectory.steps == []
        assert trajectory.final_answer == ""
        assert trajectory.is_terminal is False
        assert trajectory.created_at  # Auto-generated

    def test_trajectory_creation_with_values(self):
        """Test creating a trajectory with values."""
        trajectory = Trajectory(
            query="What is the consensus?",
            strategy="grep",
            context_tokens=4000,
            source_type="code",
        )
        assert trajectory.query == "What is the consensus?"
        assert trajectory.strategy == "grep"
        assert trajectory.context_tokens == 4000
        assert trajectory.source_type == "code"

    def test_trajectory_add_step(self):
        """Test adding steps to a trajectory."""
        trajectory = Trajectory(query="Test query")
        step1 = Step(action="step 1", tokens_examined=100)
        step2 = Step(action="step 2", tokens_examined=200)

        trajectory.add_step(step1)
        trajectory.add_step(step2)

        assert len(trajectory.steps) == 2
        assert trajectory.steps[0].action == "step 1"
        assert trajectory.steps[1].action == "step 2"

    def test_trajectory_finalize(self):
        """Test finalizing a trajectory."""
        trajectory = Trajectory(query="Test query")
        trajectory.add_step(Step(action="step 1", tokens_examined=100, sub_calls=1))
        trajectory.add_step(Step(action="step 2", tokens_examined=200, sub_calls=2))

        trajectory.finalize(
            answer="The answer is 42",
            outcome={"consensus_reached": True, "quality_score": 0.85},
        )

        assert trajectory.final_answer == "The answer is 42"
        assert trajectory.is_terminal is True
        assert trajectory.outcome["consensus_reached"] is True
        assert "total_steps" in trajectory.stats
        assert trajectory.stats["total_steps"] == 2

    def test_trajectory_compute_stats(self):
        """Test automatic stats computation on finalize."""
        trajectory = Trajectory(query="Test", strategy="tree")
        trajectory.add_step(Step(tokens_examined=100, sub_calls=1, duration_seconds=0.5))
        trajectory.add_step(Step(tokens_examined=200, sub_calls=2, duration_seconds=1.0))
        trajectory.add_step(Step(tokens_examined=150, sub_calls=0, duration_seconds=0.3))

        trajectory.finalize(answer="Done", outcome={})

        assert trajectory.stats["total_steps"] == 3
        assert trajectory.stats["total_tokens_examined"] == 450
        assert trajectory.stats["sub_calls_made"] == 3
        assert trajectory.stats["total_duration"] == pytest.approx(1.8)
        assert trajectory.stats["strategy"] == "tree"

    def test_trajectory_to_dict(self):
        """Test trajectory serialization."""
        trajectory = Trajectory(query="Test query", strategy="grep")
        trajectory.add_step(Step(action="step 1", observation="result 1"))
        trajectory.finalize(answer="Final answer", outcome={"success": True})

        d = trajectory.to_dict()

        assert d["query"] == "Test query"
        assert d["strategy"] == "grep"
        assert len(d["steps"]) == 1
        assert d["final_answer"] == "Final answer"
        assert d["outcome"]["success"] is True

    def test_trajectory_from_dict(self):
        """Test trajectory deserialization."""
        data = {
            "trajectory_id": "test-123",
            "query": "Test query",
            "strategy": "semantic",
            "steps": [
                {"action": "step 1", "tokens_examined": 100},
                {"action": "step 2", "tokens_examined": 200},
            ],
            "final_answer": "Answer",
            "outcome": {"consensus_reached": True},
            "stats": {"total_steps": 2},
        }

        trajectory = Trajectory.from_dict(data)

        assert trajectory.trajectory_id == "test-123"
        assert trajectory.query == "Test query"
        assert trajectory.strategy == "semantic"
        assert len(trajectory.steps) == 2
        assert trajectory.final_answer == "Answer"
        assert trajectory.is_terminal is True


# =============================================================================
# ExperienceBuffer Tests
# =============================================================================


class TestExperienceBuffer:
    """Tests for the ExperienceBuffer class."""

    def test_buffer_creation(self):
        """Test buffer creation with defaults."""
        buffer = ExperienceBuffer()
        assert len(buffer) == 0
        assert buffer.max_size == 10000
        assert buffer.priority_alpha == 0.0

    def test_buffer_creation_custom(self):
        """Test buffer creation with custom settings."""
        buffer = ExperienceBuffer(max_size=100, priority_alpha=0.6)
        assert buffer.max_size == 100
        assert buffer.priority_alpha == 0.6

    def test_buffer_add(self):
        """Test adding trajectories to buffer."""
        buffer = ExperienceBuffer(max_size=10)

        trajectory = Trajectory(query="Test")
        trajectory.finalize(answer="Answer", outcome={"success": True})

        buffer.add(trajectory)

        assert len(buffer) == 1

    def test_buffer_add_non_terminal_warning(self, caplog):
        """Test warning when adding non-terminal trajectory."""
        import logging

        buffer = ExperienceBuffer()

        trajectory = Trajectory(query="Test")  # Not finalized

        with caplog.at_level(logging.WARNING):
            buffer.add(trajectory)

        assert "non-terminal" in caplog.text.lower()

    def test_buffer_fifo_eviction(self):
        """Test FIFO eviction when buffer is full."""
        buffer = ExperienceBuffer(max_size=3)

        for i in range(5):
            trajectory = Trajectory(query=f"Query {i}")
            trajectory.finalize(answer=f"Answer {i}", outcome={})
            buffer.add(trajectory)

        assert len(buffer) == 3
        # First two should have been evicted
        queries = [t.query for t in buffer._buffer]
        assert "Query 0" not in queries
        assert "Query 1" not in queries
        assert "Query 4" in queries

    def test_buffer_sample_uniform(self):
        """Test uniform sampling from buffer."""
        buffer = ExperienceBuffer(max_size=100)

        for i in range(10):
            trajectory = Trajectory(query=f"Query {i}")
            trajectory.finalize(answer=f"Answer {i}", outcome={})
            buffer.add(trajectory)

        batch = buffer.sample(batch_size=5)

        assert len(batch) == 5
        assert all(isinstance(t, Trajectory) for t in batch)

    def test_buffer_sample_empty(self):
        """Test sampling from empty buffer."""
        buffer = ExperienceBuffer()
        batch = buffer.sample(batch_size=5)
        assert batch == []

    def test_buffer_sample_larger_than_buffer(self):
        """Test sampling when batch_size > buffer size."""
        buffer = ExperienceBuffer()

        for i in range(3):
            trajectory = Trajectory(query=f"Query {i}")
            trajectory.finalize(answer=f"Answer {i}", outcome={})
            buffer.add(trajectory)

        batch = buffer.sample(batch_size=10)

        assert len(batch) == 3  # Can only return what's available


# =============================================================================
# RewardConfig Tests
# =============================================================================


class TestRewardConfig:
    """Tests for RewardConfig dataclass."""

    def test_config_defaults(self):
        """Test default configuration."""
        config = RewardConfig()
        assert config.consensus_weight == 0.4
        assert config.efficiency_weight == 0.2
        assert config.confidence_weight == 0.2
        assert config.iteration_penalty_weight == 0.1
        assert config.quality_weight == 0.1
        assert config.max_sub_calls == 10
        assert config.max_iterations == 5

    def test_config_weights_sum_to_one(self):
        """Test that default weights sum to 1.0."""
        config = RewardConfig()
        total = (
            config.consensus_weight
            + config.efficiency_weight
            + config.confidence_weight
            + config.iteration_penalty_weight
            + config.quality_weight
        )
        assert total == pytest.approx(1.0)

    def test_config_custom(self):
        """Test custom configuration."""
        config = RewardConfig(
            consensus_weight=0.6,
            efficiency_weight=0.4,
            confidence_weight=0.0,
            iteration_penalty_weight=0.0,
            quality_weight=0.0,
        )
        assert config.consensus_weight == 0.6
        assert config.efficiency_weight == 0.4


# =============================================================================
# DebateOutcomeReward Tests
# =============================================================================


class TestDebateOutcomeReward:
    """Tests for DebateOutcomeReward reward model."""

    def test_reward_creation(self):
        """Test reward model creation."""
        reward = DebateOutcomeReward()
        assert isinstance(reward.config, RewardConfig)

    def test_compute_with_consensus(self):
        """Test reward computation with consensus reached."""
        reward = DebateOutcomeReward()

        trajectory = Trajectory(query="Test")
        trajectory.add_step(Step(sub_calls=2))
        trajectory.finalize(
            answer="Answer",
            outcome={
                "consensus_reached": True,
                "agreement_score": 0.9,
            },
        )
        trajectory.stats["sub_calls_made"] = 2
        trajectory.stats["confidence"] = 0.8
        trajectory.stats["iterations"] = 1

        total_reward = reward.compute(trajectory)

        # Should be positive with consensus
        assert total_reward > 0

    def test_compute_without_consensus(self):
        """Test reward computation without consensus."""
        reward = DebateOutcomeReward()

        trajectory = Trajectory(query="Test")
        trajectory.finalize(
            answer="Answer",
            outcome={
                "consensus_reached": False,
                "agreement_score": 0.3,
            },
        )
        trajectory.stats["sub_calls_made"] = 15
        trajectory.stats["confidence"] = 0.3
        trajectory.stats["iterations"] = 5
        trajectory.stats["ready"] = False

        total_reward = reward.compute(trajectory)

        # Should be negative without consensus and with penalties
        assert total_reward < 0

    def test_compute_components(self):
        """Test individual component computation."""
        reward = DebateOutcomeReward()

        trajectory = Trajectory(query="Test")
        trajectory.finalize(
            answer="Answer",
            outcome={
                "consensus_reached": True,
                "agreement_score": 0.8,
                "quality_score": 0.9,
            },
        )
        trajectory.stats["sub_calls_made"] = 5
        trajectory.stats["confidence"] = 0.85
        trajectory.stats["iterations"] = 2
        trajectory.stats["ready"] = True

        components = reward.compute_components(trajectory)

        assert "consensus" in components
        assert "efficiency" in components
        assert "confidence" in components
        assert "iteration" in components
        assert "quality" in components

    def test_consensus_reward_strong(self):
        """Test consensus reward with strong agreement."""
        reward = DebateOutcomeReward()

        trajectory = Trajectory()
        trajectory.finalize(
            answer="",
            outcome={"consensus_reached": True, "agreement_score": 1.0},
        )

        components = reward.compute_components(trajectory)
        # Consensus with perfect agreement should give max consensus reward
        assert components["consensus"] > 0

    def test_efficiency_reward_low_subcalls(self):
        """Test efficiency reward with few sub-calls."""
        reward = DebateOutcomeReward()

        trajectory = Trajectory()
        trajectory.finalize(answer="", outcome={})
        trajectory.stats["sub_calls_made"] = 2

        components = reward.compute_components(trajectory)
        assert components["efficiency"] > 0  # Few sub-calls is good

    def test_efficiency_reward_high_subcalls(self):
        """Test efficiency reward with many sub-calls."""
        reward = DebateOutcomeReward()

        trajectory = Trajectory()
        trajectory.finalize(answer="", outcome={})
        trajectory.stats["sub_calls_made"] = 20  # Exceeds max

        components = reward.compute_components(trajectory)
        assert components["efficiency"] < 0  # Too many sub-calls is bad

    def test_iteration_reward_no_refinement(self):
        """Test iteration reward with no refinement needed."""
        reward = DebateOutcomeReward()

        trajectory = Trajectory()
        trajectory.finalize(answer="", outcome={})
        trajectory.stats["iterations"] = 1

        components = reward.compute_components(trajectory)
        # Perfect - no refinement needed
        assert components["iteration"] > 0


# =============================================================================
# CompositeReward Tests
# =============================================================================


class TestCompositeReward:
    """Tests for CompositeReward reward model."""

    def test_composite_empty(self):
        """Test composite reward with no models."""
        composite = CompositeReward()
        trajectory = Trajectory()
        trajectory.finalize(answer="", outcome={})

        reward = composite.compute(trajectory)
        assert reward == 0.0

    def test_composite_single_model(self):
        """Test composite reward with single model."""
        composite = CompositeReward()
        debate_reward = DebateOutcomeReward()
        composite.add_model(debate_reward, weight=1.0)

        trajectory = Trajectory()
        trajectory.finalize(
            answer="",
            outcome={"consensus_reached": True, "agreement_score": 0.9},
        )
        trajectory.stats["sub_calls_made"] = 5
        trajectory.stats["confidence"] = 0.8
        trajectory.stats["iterations"] = 1

        reward = composite.compute(trajectory)
        direct_reward = debate_reward.compute(trajectory)

        assert reward == pytest.approx(direct_reward)

    def test_composite_weighted_models(self):
        """Test composite reward with weighted models."""
        composite = CompositeReward()

        # Mock reward model that returns fixed value
        mock_model1 = MagicMock()
        mock_model1.compute.return_value = 1.0
        mock_model1.compute_components.return_value = {"test": 1.0}

        mock_model2 = MagicMock()
        mock_model2.compute.return_value = -1.0
        mock_model2.compute_components.return_value = {"test": -1.0}

        composite.add_model(mock_model1, weight=0.75)
        composite.add_model(mock_model2, weight=0.25)

        trajectory = Trajectory()

        reward = composite.compute(trajectory)

        # Weighted average: (1.0*0.75 + -1.0*0.25) / 1.0 = 0.5
        assert reward == pytest.approx(0.5)


# =============================================================================
# SparseReward Tests
# =============================================================================


class TestSparseReward:
    """Tests for SparseReward reward model."""

    def test_sparse_defaults(self):
        """Test sparse reward defaults."""
        sparse = SparseReward()
        assert sparse.success_reward == 1.0
        assert sparse.failure_penalty == -1.0
        assert sparse.partial_reward == 0.0

    def test_sparse_custom(self):
        """Test sparse reward with custom values."""
        sparse = SparseReward(
            success_reward=10.0,
            failure_penalty=-5.0,
            partial_reward=2.0,
        )
        assert sparse.success_reward == 10.0
        assert sparse.failure_penalty == -5.0
        assert sparse.partial_reward == 2.0


# =============================================================================
# TrainerConfig Tests
# =============================================================================


class TestTrainerConfig:
    """Tests for TrainerConfig dataclass."""

    def test_config_defaults(self):
        """Test default trainer configuration."""
        config = TrainerConfig()
        assert config.batch_size == 32
        assert config.learning_rate == 0.001
        assert config.discount_factor == 0.99
        assert config.entropy_bonus == 0.01
        assert config.buffer_size == 10000
        assert config.max_refinement_iterations == 5

    def test_config_custom(self):
        """Test custom trainer configuration."""
        config = TrainerConfig(
            batch_size=64,
            learning_rate=0.0001,
            prioritized_replay=True,
            priority_alpha=0.8,
        )
        assert config.batch_size == 64
        assert config.learning_rate == 0.0001
        assert config.prioritized_replay is True
        assert config.priority_alpha == 0.8


# =============================================================================
# TrainingMetrics Tests
# =============================================================================


class TestTrainingMetrics:
    """Tests for TrainingMetrics dataclass."""

    def test_metrics_defaults(self):
        """Test default training metrics."""
        metrics = TrainingMetrics()
        assert metrics.epoch == 0
        assert metrics.total_trajectories == 0
        assert metrics.avg_reward == 0.0
        assert metrics.success_rate == 0.0
        assert metrics.strategy_distribution == {}

    def test_metrics_with_values(self):
        """Test training metrics with values."""
        metrics = TrainingMetrics(
            epoch=10,
            total_trajectories=1000,
            avg_reward=0.75,
            success_rate=0.85,
            strategy_distribution={"grep": 0.3, "tree": 0.5, "semantic": 0.2},
        )
        assert metrics.epoch == 10
        assert metrics.total_trajectories == 1000
        assert metrics.avg_reward == 0.75
        assert metrics.success_rate == 0.85
        assert len(metrics.strategy_distribution) == 3

    def test_metrics_to_dict(self):
        """Test metrics serialization."""
        metrics = TrainingMetrics(
            epoch=5,
            avg_reward=0.6,
            consensus_reward=0.4,
            efficiency_reward=0.1,
        )

        d = metrics.to_dict()

        assert d["epoch"] == 5
        assert d["avg_reward"] == 0.6
        assert d["consensus_reward"] == 0.4
        assert d["efficiency_reward"] == 0.1
        assert "strategy_distribution" in d


# =============================================================================
# Integration Tests
# =============================================================================


class TestRLMTrainingIntegration:
    """Integration tests for RLM training components."""

    def test_full_trajectory_flow(self):
        """Test complete trajectory recording and reward computation."""
        # Create trajectory
        trajectory = Trajectory(
            query="What is the consensus on implementing feature X?",
            strategy="auto",
        )

        # Add steps
        trajectory.add_step(
            Step(
                action="strategy = 'grep'",
                action_type="strategy",
                observation="Switching to grep strategy",
                tokens_examined=0,
                sub_calls=0,
                duration_seconds=0.1,
            )
        )

        trajectory.add_step(
            Step(
                action="search('feature X')",
                action_type="code",
                observation="Found 5 discussions about feature X",
                tokens_examined=2000,
                sub_calls=3,
                duration_seconds=1.5,
            )
        )

        trajectory.add_step(
            Step(
                action="answer = synthesize(results)",
                action_type="final",
                observation="Synthesized consensus from discussions",
                tokens_examined=500,
                sub_calls=1,
                duration_seconds=2.0,
            )
        )

        # Finalize
        trajectory.finalize(
            answer="The consensus is to implement feature X with approach A",
            outcome={
                "consensus_reached": True,
                "agreement_score": 0.85,
                "quality_score": 0.9,
            },
        )

        # Verify trajectory
        assert len(trajectory.steps) == 3
        assert trajectory.is_terminal is True
        assert trajectory.stats["total_steps"] == 3
        assert trajectory.stats["total_tokens_examined"] == 2500
        assert trajectory.stats["sub_calls_made"] == 4

        # Compute reward
        reward_model = DebateOutcomeReward()
        reward = reward_model.compute(trajectory)
        components = reward_model.compute_components(trajectory)

        # Should be positive with good outcome
        assert reward > 0
        assert "consensus" in components
        assert components["consensus"] > 0  # Consensus reached

        # Store in buffer
        buffer = ExperienceBuffer(max_size=100)
        buffer.add(trajectory)
        assert len(buffer) == 1

        # Sample from buffer
        batch = buffer.sample(batch_size=1)
        assert len(batch) == 1
        assert batch[0].trajectory_id == trajectory.trajectory_id

    def test_multiple_trajectories_statistics(self):
        """Test statistics across multiple trajectories."""
        buffer = ExperienceBuffer(max_size=100)
        reward_model = DebateOutcomeReward()

        rewards = []
        for i in range(10):
            trajectory = Trajectory(query=f"Query {i}")
            trajectory.add_step(Step(sub_calls=i % 5))
            trajectory.finalize(
                answer=f"Answer {i}",
                outcome={
                    "consensus_reached": i % 2 == 0,
                    "agreement_score": 0.5 + (i % 5) * 0.1,
                },
            )
            trajectory.stats["sub_calls_made"] = i % 5
            trajectory.stats["confidence"] = 0.5 + (i % 5) * 0.1
            trajectory.stats["iterations"] = 1 + (i % 3)

            buffer.add(trajectory)
            rewards.append(reward_model.compute(trajectory))

        assert len(buffer) == 10
        assert len(rewards) == 10

        # Verify we get variety in rewards
        assert min(rewards) != max(rewards)
