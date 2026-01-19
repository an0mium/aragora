"""
Tests for RLM training module.

Tests:
- Trajectory creation and serialization
- ExperienceBuffer storage and sampling
- Reward model computation
- Policy interfaces
"""

import pytest
import tempfile
import os

from aragora.rlm.training import (
    ExperienceBuffer,
    Step,
    Trajectory,
    DebateOutcomeReward,
    RewardConfig,
    StrategyPolicy,
    RefinementPolicy,
    PolicyState,
)


class TestStep:
    """Test Step dataclass."""

    def test_step_creation(self):
        """Test creating a step."""
        step = Step(
            action="print(get_summary())",
            action_type="code",
            observation="Summary content...",
            tokens_examined=100,
        )
        assert step.action == "print(get_summary())"
        assert step.action_type == "code"
        assert step.tokens_examined == 100

    def test_step_has_timestamp(self):
        """Test step gets timestamp automatically."""
        step = Step(action="test")
        assert step.timestamp != ""


class TestTrajectory:
    """Test Trajectory dataclass."""

    def test_trajectory_creation(self):
        """Test creating a trajectory."""
        trajectory = Trajectory(
            query="What is the consensus?",
            strategy="grep",
        )
        assert trajectory.query == "What is the consensus?"
        assert trajectory.strategy == "grep"
        assert trajectory.trajectory_id != ""

    def test_trajectory_add_step(self):
        """Test adding steps to trajectory."""
        trajectory = Trajectory(query="test")
        step = Step(action="code1", tokens_examined=50)
        trajectory.add_step(step)

        assert len(trajectory.steps) == 1
        assert trajectory.steps[0].tokens_examined == 50

    def test_trajectory_finalize(self):
        """Test finalizing trajectory."""
        trajectory = Trajectory(query="test")
        trajectory.add_step(Step(action="code", tokens_examined=100))

        trajectory.finalize(
            answer="Final answer",
            outcome={"consensus_reached": True},
        )

        assert trajectory.is_terminal
        assert trajectory.final_answer == "Final answer"
        assert trajectory.outcome["consensus_reached"] is True
        assert "total_steps" in trajectory.stats

    def test_trajectory_to_dict(self):
        """Test trajectory serialization."""
        trajectory = Trajectory(query="test", strategy="peek")
        trajectory.add_step(Step(action="code"))
        trajectory.finalize("answer", {"success": True})

        data = trajectory.to_dict()
        assert data["query"] == "test"
        assert data["strategy"] == "peek"
        assert len(data["steps"]) == 1

    def test_trajectory_from_dict(self):
        """Test trajectory deserialization."""
        data = {
            "trajectory_id": "test123",
            "query": "test query",
            "strategy": "grep",
            "steps": [
                {"action": "code", "action_type": "code", "observation": "out"}
            ],
            "final_answer": "answer",
            "outcome": {"success": True},
            "stats": {"total_steps": 1},
        }

        trajectory = Trajectory.from_dict(data)
        assert trajectory.trajectory_id == "test123"
        assert trajectory.query == "test query"
        assert len(trajectory.steps) == 1


class TestExperienceBuffer:
    """Test ExperienceBuffer class."""

    def test_buffer_creation(self):
        """Test creating experience buffer."""
        buffer = ExperienceBuffer(max_size=100)
        assert len(buffer) == 0
        assert buffer.max_size == 100

    def test_buffer_add_trajectory(self):
        """Test adding trajectory to buffer."""
        buffer = ExperienceBuffer(max_size=100)
        trajectory = Trajectory(query="test")
        trajectory.finalize("answer", {"success": True})

        buffer.add(trajectory)
        assert len(buffer) == 1

    def test_buffer_sample(self):
        """Test sampling from buffer."""
        buffer = ExperienceBuffer(max_size=100)

        for i in range(10):
            t = Trajectory(query=f"query {i}")
            t.finalize(f"answer {i}", {"success": True})
            buffer.add(t)

        sample = buffer.sample(5)
        assert len(sample) == 5
        assert all(isinstance(t, Trajectory) for t in sample)

    def test_buffer_sample_empty(self):
        """Test sampling from empty buffer."""
        buffer = ExperienceBuffer()
        sample = buffer.sample(5)
        assert sample == []

    def test_buffer_fifo_eviction(self):
        """Test FIFO eviction when capacity is reached."""
        buffer = ExperienceBuffer(max_size=3)

        for i in range(5):
            t = Trajectory(query=f"query {i}")
            t.finalize(f"answer {i}", {})
            buffer.add(t)

        assert len(buffer) == 3

    def test_buffer_save_load(self):
        """Test saving and loading buffer."""
        buffer = ExperienceBuffer(max_size=100)

        for i in range(3):
            t = Trajectory(query=f"query {i}")
            t.finalize(f"answer {i}", {"success": True})
            buffer.add(t)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            filepath = f.name

        try:
            buffer.save(filepath)
            loaded = ExperienceBuffer.load(filepath)

            assert len(loaded) == 3
        finally:
            os.unlink(filepath)

    def test_buffer_get_stats(self):
        """Test buffer statistics."""
        buffer = ExperienceBuffer(max_size=100)

        for i in range(5):
            t = Trajectory(query=f"query {i}", strategy="grep")
            t.add_step(Step(action="code"))
            t.finalize(f"answer", {"success": i % 2 == 0})
            buffer.add(t)

        stats = buffer.get_stats()
        assert stats["size"] == 5
        assert stats["success_rate"] == 0.6  # 3 out of 5
        assert "grep" in stats["strategies"]


class TestDebateOutcomeReward:
    """Test DebateOutcomeReward model."""

    def test_reward_creation(self):
        """Test creating reward model."""
        reward = DebateOutcomeReward()
        assert reward.config is not None

    def test_reward_compute_success(self):
        """Test reward computation for successful trajectory."""
        reward = DebateOutcomeReward()

        trajectory = Trajectory(query="test")
        trajectory.finalize(
            "answer",
            outcome={
                "consensus_reached": True,
                "agreement_score": 0.9,
                "quality_score": 0.8,
            },
            stats={
                "sub_calls_made": 3,
                "confidence": 0.85,
                "iterations": 1,
                "ready": True,
            },
        )

        total_reward = reward.compute(trajectory)
        assert total_reward > 0  # Should be positive for success

    def test_reward_compute_failure(self):
        """Test reward computation for failed trajectory."""
        reward = DebateOutcomeReward()

        trajectory = Trajectory(query="test")
        trajectory.finalize(
            "answer",
            outcome={
                "consensus_reached": False,
                "agreement_score": 0.3,
            },
            stats={
                "sub_calls_made": 15,
                "confidence": 0.2,
                "iterations": 5,
                "ready": False,
            },
        )

        total_reward = reward.compute(trajectory)
        assert total_reward < 0  # Should be negative for failure

    def test_reward_components(self):
        """Test reward component breakdown."""
        reward = DebateOutcomeReward()

        trajectory = Trajectory(query="test")
        trajectory.finalize(
            "answer",
            outcome={"consensus_reached": True},
            stats={"confidence": 0.8},
        )

        components = reward.compute_components(trajectory)
        assert "consensus" in components
        assert "efficiency" in components
        assert "confidence" in components
        assert "iteration" in components


class TestStrategyPolicy:
    """Test StrategyPolicy class."""

    def test_policy_creation(self):
        """Test creating strategy policy."""
        policy = StrategyPolicy()
        assert len(policy.strategies) > 0

    def test_policy_act(self):
        """Test policy action selection."""
        policy = StrategyPolicy(exploration_rate=0.0)
        state = PolicyState(
            query="test query",
            context_tokens=100000,
        )

        action = policy.act(state)
        assert action in policy.strategies

    def test_policy_action_probs(self):
        """Test action probability distribution."""
        policy = StrategyPolicy()
        state = PolicyState(query="test")

        probs = policy.get_action_probs(state)
        assert sum(probs.values()) == pytest.approx(1.0, rel=0.01)
        assert all(p >= 0 for p in probs.values())

    def test_policy_exploration(self):
        """Test exploration behavior."""
        policy = StrategyPolicy(exploration_rate=1.0)  # Always explore
        state = PolicyState(query="test")

        # With full exploration, should get variety
        actions = [policy.act(state) for _ in range(100)]
        unique_actions = set(actions)
        assert len(unique_actions) > 1


class TestRefinementPolicy:
    """Test RefinementPolicy class."""

    def test_policy_accepts_high_confidence(self):
        """Test policy accepts high confidence answers."""
        policy = RefinementPolicy(confidence_threshold=0.8)
        state = PolicyState(previous_confidence=0.9)

        action = policy.act(state)
        assert action == "accept"

    def test_policy_refines_low_confidence(self):
        """Test policy refines low confidence answers."""
        policy = RefinementPolicy(confidence_threshold=0.8)
        state = PolicyState(previous_confidence=0.3)

        action = policy.act(state)
        assert action == "refine"

    def test_policy_forces_accept_at_max_iterations(self):
        """Test policy forces acceptance at max iterations."""
        policy = RefinementPolicy(max_iterations=5)
        state = PolicyState(
            iteration=4,  # 0-indexed, so this is the 5th iteration
            previous_confidence=0.3,
        )

        action = policy.act(state)
        assert action == "accept"


class TestPolicyState:
    """Test PolicyState class."""

    def test_state_to_feature_vector(self):
        """Test converting state to feature vector."""
        state = PolicyState(
            context_tokens=10000,
            abstraction_levels=3,
            iteration=2,
            previous_confidence=0.7,
        )

        features = state.to_feature_vector()
        assert len(features) == 7  # Base features
        assert all(isinstance(f, float) for f in features)
