"""
Extended tests for RLM training pipeline.

Tests additional functionality not covered in test_training.py:
- CompositeReward and SparseReward models
- Prioritized experience replay
- Outcome/strategy filtering for sampling
- Edge cases for buffer and reward computation
"""

import tempfile
import os
import pytest

from aragora.rlm.training.buffer import Step, Trajectory, ExperienceBuffer
from aragora.rlm.training.reward import (
    RewardConfig,
    DebateOutcomeReward,
    CompositeReward,
    SparseReward,
)


class TestStepEdgeCases:
    """Additional tests for Step dataclass."""

    def test_step_default_values(self):
        """Test step has proper default values."""
        step = Step()
        assert step.state == {}
        assert step.action == ""
        assert step.action_type == "code"
        assert step.observation == ""
        assert step.tokens_examined == 0
        assert step.sub_calls == 0
        assert step.duration_seconds == 0.0

    def test_step_custom_state(self):
        """Test step with custom state dict."""
        state = {
            "context_tokens": 5000,
            "current_strategy": "grep",
            "iteration": 2,
        }
        step = Step(
            state=state,
            action="search('pattern')",
            action_type="strategy",
            tokens_examined=1500,
            sub_calls=3,
            duration_seconds=2.5,
        )

        assert step.state["context_tokens"] == 5000
        assert step.sub_calls == 3
        assert step.duration_seconds == 2.5

    def test_step_preserves_timestamp_if_provided(self):
        """Test step preserves timestamp if explicitly provided."""
        custom_ts = "2024-01-15T10:30:00"
        step = Step(timestamp=custom_ts)
        assert step.timestamp == custom_ts


class TestTrajectoryEdgeCases:
    """Additional tests for Trajectory dataclass."""

    def test_trajectory_compute_stats(self):
        """Test trajectory computes correct aggregate stats."""
        trajectory = Trajectory(query="test", strategy="grep")

        trajectory.add_step(Step(tokens_examined=100, sub_calls=2, duration_seconds=1.0))
        trajectory.add_step(Step(tokens_examined=200, sub_calls=1, duration_seconds=0.5))
        trajectory.add_step(Step(tokens_examined=150, sub_calls=3, duration_seconds=0.8))

        trajectory.finalize("answer", {"success": True})

        assert trajectory.stats["total_steps"] == 3
        assert trajectory.stats["total_tokens_examined"] == 450
        assert trajectory.stats["sub_calls_made"] == 6
        assert trajectory.stats["total_duration"] == pytest.approx(2.3, rel=0.01)
        assert trajectory.stats["strategy"] == "grep"

    def test_trajectory_finalize_with_custom_stats(self):
        """Test trajectory accepts custom stats on finalize."""
        trajectory = Trajectory(query="test")
        trajectory.add_step(Step(tokens_examined=100))

        custom_stats = {
            "total_steps": 1,
            "custom_metric": 0.95,
            "evaluation_source": "human",
        }

        trajectory.finalize("answer", {"success": True}, stats=custom_stats)

        assert trajectory.stats["custom_metric"] == 0.95
        assert trajectory.stats["evaluation_source"] == "human"

    def test_trajectory_preserves_id_if_provided(self):
        """Test trajectory preserves ID if explicitly provided."""
        trajectory = Trajectory(trajectory_id="custom_id_123", query="test")
        assert trajectory.trajectory_id == "custom_id_123"

    def test_trajectory_from_dict_handles_missing_fields(self):
        """Test from_dict handles minimal data gracefully."""
        data = {"query": "minimal test"}

        trajectory = Trajectory.from_dict(data)

        assert trajectory.query == "minimal test"
        assert trajectory.strategy == "auto"
        assert trajectory.steps == []
        assert trajectory.outcome == {}

    def test_trajectory_serialization_truncates_long_content(self):
        """Test to_dict truncates very long content."""
        trajectory = Trajectory(query="test")

        # Create very long observation and answer
        long_observation = "x" * 1000
        long_answer = "y" * 2000

        trajectory.add_step(Step(observation=long_observation))
        trajectory.finalize(long_answer, {"success": True})

        data = trajectory.to_dict()

        # Observation truncated to 500 chars
        assert len(data["steps"][0]["observation"]) == 500
        # Answer truncated to 1000 chars
        assert len(data["final_answer"]) == 1000


class TestExperienceBufferPrioritized:
    """Tests for prioritized experience replay."""

    def test_buffer_prioritized_sampling(self):
        """Test prioritized sampling favors high priority items."""
        buffer = ExperienceBuffer(max_size=100, priority_alpha=1.0)

        # Add low priority trajectories
        for i in range(10):
            t = Trajectory(query=f"low_{i}")
            t.finalize("answer", {"success": False})
            buffer.add(t, priority=0.1)

        # Add one high priority trajectory
        high_priority = Trajectory(query="high_priority")
        high_priority.finalize("answer", {"success": True})
        buffer.add(high_priority, priority=10.0)

        # Sample many times and check high priority appears more often
        high_priority_count = 0
        for _ in range(100):
            sample = buffer.sample(1)
            if sample[0].query == "high_priority":
                high_priority_count += 1

        # With such extreme priority difference, high priority should
        # appear in most samples (statistically > 30%)
        assert high_priority_count > 30

    def test_buffer_update_priority(self):
        """Test updating priority of existing trajectory."""
        buffer = ExperienceBuffer(max_size=100, priority_alpha=1.0)

        t1 = Trajectory(trajectory_id="t1", query="test1")
        t1.finalize("answer", {"success": True})
        buffer.add(t1, priority=1.0)

        t2 = Trajectory(trajectory_id="t2", query="test2")
        t2.finalize("answer", {"success": True})
        buffer.add(t2, priority=1.0)

        # Update t1 to very high priority
        buffer.update_priority("t1", 100.0)

        # Now t1 should appear much more often
        t1_count = 0
        for _ in range(100):
            sample = buffer.sample(1)
            if sample[0].trajectory_id == "t1":
                t1_count += 1

        assert t1_count > 70  # Should dominate with 100x priority

    def test_buffer_uniform_with_zero_alpha(self):
        """Test uniform sampling when alpha is 0."""
        buffer = ExperienceBuffer(max_size=100, priority_alpha=0.0)

        # Add trajectories with very different priorities
        for i in range(10):
            t = Trajectory(query=f"traj_{i}")
            t.finalize("answer", {"success": True})
            # Huge priority difference
            buffer.add(t, priority=(i + 1) * 100)

        # With alpha=0, should be uniform sampling
        # Each trajectory should appear roughly 10% of the time
        query_counts = {f"traj_{i}": 0 for i in range(10)}
        for _ in range(1000):
            sample = buffer.sample(1)
            query_counts[sample[0].query] += 1

        # Check relatively uniform (each between 5% and 15%)
        for count in query_counts.values():
            assert 50 <= count <= 150


class TestExperienceBufferFiltering:
    """Tests for outcome and strategy filtering."""

    def test_sample_by_outcome_success_only(self):
        """Test filtering to successful trajectories only."""
        buffer = ExperienceBuffer(max_size=100)

        for i in range(10):
            t = Trajectory(query=f"traj_{i}")
            t.finalize("answer", {"success": i % 2 == 0})  # 5 success, 5 failure
            buffer.add(t)

        successes = buffer.sample_by_outcome(100, success_only=True)

        assert len(successes) == 5
        assert all(t.outcome.get("success") for t in successes)

    def test_sample_by_outcome_failure_only(self):
        """Test filtering to failed trajectories only."""
        buffer = ExperienceBuffer(max_size=100)

        for i in range(10):
            t = Trajectory(query=f"traj_{i}")
            t.finalize("answer", {"success": i % 2 == 0})
            buffer.add(t)

        failures = buffer.sample_by_outcome(100, failure_only=True)

        assert len(failures) == 5
        assert all(not t.outcome.get("success") for t in failures)

    def test_sample_by_outcome_empty_result(self):
        """Test filtering returns empty when no matches."""
        buffer = ExperienceBuffer(max_size=100)

        # Only add failures
        for i in range(5):
            t = Trajectory(query=f"traj_{i}")
            t.finalize("answer", {"success": False})
            buffer.add(t)

        successes = buffer.sample_by_outcome(10, success_only=True)
        assert successes == []

    def test_sample_by_strategy(self):
        """Test filtering by strategy."""
        buffer = ExperienceBuffer(max_size=100)

        strategies = ["grep", "peek", "semantic", "grep", "grep"]
        for i, strategy in enumerate(strategies):
            t = Trajectory(query=f"traj_{i}", strategy=strategy)
            t.finalize("answer", {"success": True})
            buffer.add(t)

        grep_trajs = buffer.sample_by_strategy("grep", 100)

        assert len(grep_trajs) == 3
        assert all(t.strategy == "grep" for t in grep_trajs)

    def test_sample_by_strategy_batch_limit(self):
        """Test batch_size is respected in strategy filtering."""
        buffer = ExperienceBuffer(max_size=100)

        for i in range(10):
            t = Trajectory(query=f"traj_{i}", strategy="grep")
            t.finalize("answer", {"success": True})
            buffer.add(t)

        sample = buffer.sample_by_strategy("grep", 3)
        assert len(sample) == 3

    def test_sample_by_strategy_no_matches(self):
        """Test filtering returns empty when no strategy matches."""
        buffer = ExperienceBuffer(max_size=100)

        t = Trajectory(query="test", strategy="grep")
        t.finalize("answer", {"success": True})
        buffer.add(t)

        sample = buffer.sample_by_strategy("semantic", 10)
        assert sample == []


class TestExperienceBufferOperations:
    """Additional buffer operation tests."""

    def test_buffer_clear(self):
        """Test clearing the buffer."""
        buffer = ExperienceBuffer(max_size=100)

        for i in range(5):
            t = Trajectory(query=f"traj_{i}")
            t.finalize("answer", {"success": True})
            buffer.add(t)

        assert len(buffer) == 5

        buffer.clear()

        assert len(buffer) == 0
        assert buffer.sample(1) == []

    def test_buffer_stats_empty(self):
        """Test stats for empty buffer."""
        buffer = ExperienceBuffer(max_size=100)

        stats = buffer.get_stats()

        assert stats["size"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["avg_steps"] == 0.0
        assert stats["strategies"] == {}

    def test_buffer_sample_larger_than_size(self):
        """Test sampling more than buffer size returns all."""
        buffer = ExperienceBuffer(max_size=100)

        for i in range(3):
            t = Trajectory(query=f"traj_{i}")
            t.finalize("answer", {"success": True})
            buffer.add(t)

        sample = buffer.sample(100)
        assert len(sample) == 3

    def test_buffer_save_load_preserves_priorities(self):
        """Test save/load preserves priority values."""
        buffer = ExperienceBuffer(max_size=100, priority_alpha=0.5)

        for i in range(3):
            t = Trajectory(query=f"traj_{i}")
            t.finalize("answer", {"success": True})
            buffer.add(t, priority=float(i + 1))

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            buffer.save(filepath)
            loaded = ExperienceBuffer.load(filepath)

            assert len(loaded) == 3
            assert loaded.priority_alpha == 0.5
            # Verify priorities are preserved (check by sampling behavior)
            assert list(loaded._priorities) == [1.0, 2.0, 3.0]
        finally:
            os.unlink(filepath)


class TestCompositeReward:
    """Tests for CompositeReward model."""

    def test_composite_empty_models(self):
        """Test composite reward with no models returns 0."""
        composite = CompositeReward()

        trajectory = Trajectory(query="test")
        trajectory.finalize("answer", {"success": True})

        reward = composite.compute(trajectory)
        assert reward == 0.0

    def test_composite_single_model(self):
        """Test composite with single model returns that model's reward."""
        composite = CompositeReward()
        debate_reward = DebateOutcomeReward()
        composite.add_model(debate_reward, weight=1.0)

        trajectory = Trajectory(query="test")
        trajectory.finalize(
            "answer",
            {"consensus_reached": True, "agreement_score": 0.8},
            stats={"confidence": 0.9, "iterations": 1, "ready": True},
        )

        composite_result = composite.compute(trajectory)
        direct_result = debate_reward.compute(trajectory)

        assert composite_result == pytest.approx(direct_result, rel=0.01)

    def test_composite_weighted_average(self):
        """Test composite computes weighted average."""
        composite = CompositeReward()

        model1 = DebateOutcomeReward(
            config=RewardConfig(
                consensus_weight=1.0,
                efficiency_weight=0.0,
                confidence_weight=0.0,
                iteration_penalty_weight=0.0,
                quality_weight=0.0,
            )
        )
        model2 = SparseReward(success_reward=2.0, failure_penalty=-2.0)

        composite.add_model(model1, weight=1.0)
        composite.add_model(model2, weight=1.0)

        trajectory = Trajectory(query="test")
        trajectory.finalize(
            "answer",
            {"success": True, "consensus_reached": True, "agreement_score": 1.0},
        )

        reward = composite.compute(trajectory)

        # model1 should give ~1.0 (consensus with agreement 1.0)
        # model2 should give 2.0 (success)
        # Average should be around 1.5
        assert 1.0 < reward < 2.0

    def test_composite_zero_weight_ignored(self):
        """Test models with zero weight don't contribute."""
        composite = CompositeReward()

        model1 = SparseReward(success_reward=10.0)
        model2 = SparseReward(success_reward=0.0, failure_penalty=0.0)

        composite.add_model(model1, weight=0.0)  # Should be ignored
        composite.add_model(model2, weight=1.0)

        trajectory = Trajectory(query="test")
        trajectory.finalize("answer", {"success": True})

        reward = composite.compute(trajectory)
        assert reward == 0.0

    def test_composite_components_include_model_name(self):
        """Test component names include model type."""
        composite = CompositeReward()
        composite.add_model(DebateOutcomeReward(), weight=1.0)
        composite.add_model(SparseReward(), weight=1.0)

        trajectory = Trajectory(query="test")
        trajectory.finalize("answer", {"success": True})

        components = composite.compute_components(trajectory)

        # Should have components from both models
        debate_components = [k for k in components if "DebateOutcomeReward" in k]
        sparse_components = [k for k in components if "SparseReward" in k]

        assert len(debate_components) > 0
        assert len(sparse_components) > 0


class TestSparseReward:
    """Tests for SparseReward model."""

    def test_sparse_non_terminal_returns_zero(self):
        """Test sparse reward returns 0 for non-terminal trajectory."""
        reward_model = SparseReward()

        trajectory = Trajectory(query="test")
        # Don't finalize - not terminal

        reward = reward_model.compute(trajectory)
        assert reward == 0.0

    def test_sparse_success(self):
        """Test sparse reward for successful outcome."""
        reward_model = SparseReward(
            success_reward=1.5,
            failure_penalty=-0.5,
        )

        trajectory = Trajectory(query="test")
        trajectory.finalize("answer", {"success": True})

        reward = reward_model.compute(trajectory)
        assert reward == 1.5

    def test_sparse_failure(self):
        """Test sparse reward for failed outcome."""
        reward_model = SparseReward(
            success_reward=1.0,
            failure_penalty=-2.0,
        )

        trajectory = Trajectory(query="test")
        trajectory.finalize("answer", {"success": False})

        reward = reward_model.compute(trajectory)
        assert reward == -2.0

    def test_sparse_partial_success(self):
        """Test sparse reward for partial success."""
        reward_model = SparseReward(
            success_reward=1.0,
            failure_penalty=-1.0,
            partial_reward=0.3,
        )

        trajectory = Trajectory(query="test")
        trajectory.finalize(
            "answer",
            {
                "success": False,
                "partial_success": True,
            },
        )

        reward = reward_model.compute(trajectory)
        assert reward == 0.3

    def test_sparse_empty_outcome(self):
        """Test sparse reward with empty outcome dict."""
        reward_model = SparseReward(failure_penalty=-0.8)

        trajectory = Trajectory(query="test")
        trajectory.is_terminal = True  # Mark terminal but no outcome
        trajectory.outcome = {}

        reward = reward_model.compute(trajectory)
        assert reward == -0.8

    def test_sparse_components(self):
        """Test sparse reward components."""
        reward_model = SparseReward(success_reward=1.0)

        trajectory = Trajectory(query="test")
        trajectory.finalize("answer", {"success": True})

        components = reward_model.compute_components(trajectory)

        assert "sparse" in components
        assert components["sparse"] == 1.0


class TestDebateOutcomeRewardEdgeCases:
    """Edge case tests for DebateOutcomeReward."""

    def test_reward_no_outcome(self):
        """Test reward when trajectory has no outcome."""
        reward_model = DebateOutcomeReward()

        trajectory = Trajectory(query="test")
        trajectory.is_terminal = True
        trajectory.outcome = {}

        components = reward_model.compute_components(trajectory)

        # Consensus should be penalized
        assert components["consensus"] < 0

    def test_reward_near_consensus(self):
        """Test partial credit for near-consensus."""
        reward_model = DebateOutcomeReward()

        trajectory = Trajectory(query="test")
        trajectory.finalize(
            "answer",
            {
                "consensus_reached": False,
                "agreement_score": 0.75,  # Near consensus
            },
        )

        components = reward_model.compute_components(trajectory)

        # Should get partial credit
        assert components["consensus"] > 0

    def test_reward_excessive_sub_calls(self):
        """Test penalty for excessive sub-calls."""
        config = RewardConfig(max_sub_calls=5)
        reward_model = DebateOutcomeReward(config=config)

        trajectory = Trajectory(query="test")
        trajectory.finalize(
            "answer",
            {"consensus_reached": True},
            stats={"sub_calls_made": 20},  # Way over limit
        )

        components = reward_model.compute_components(trajectory)

        # Efficiency should be negative
        assert components["efficiency"] < 0

    def test_reward_no_quality_score(self):
        """Test quality component is 0 when no score available."""
        reward_model = DebateOutcomeReward()

        trajectory = Trajectory(query="test")
        trajectory.finalize(
            "answer",
            {"consensus_reached": True},  # No quality_score
        )

        components = reward_model.compute_components(trajectory)

        assert components["quality"] == 0.0

    def test_reward_iteration_penalties(self):
        """Test iteration penalty at various levels."""
        config = RewardConfig(max_iterations=5)
        reward_model = DebateOutcomeReward(config=config)

        # Test 1 iteration (best case)
        t1 = Trajectory(query="test")
        t1.finalize("answer", {}, stats={"iterations": 1, "ready": True})
        c1 = reward_model.compute_components(t1)

        # Test 3 iterations
        t2 = Trajectory(query="test")
        t2.finalize("answer", {}, stats={"iterations": 3, "ready": True})
        c2 = reward_model.compute_components(t2)

        # Test max iterations not ready
        t3 = Trajectory(query="test")
        t3.finalize("answer", {}, stats={"iterations": 6, "ready": False})
        c3 = reward_model.compute_components(t3)

        # Verify ordering
        assert c1["iteration"] > c2["iteration"] > c3["iteration"]
        assert c3["iteration"] < 0  # Should be penalty

    def test_reward_config_weights_sum(self):
        """Test that default config weights sum to 1.0."""
        config = RewardConfig()

        total = (
            config.consensus_weight
            + config.efficiency_weight
            + config.confidence_weight
            + config.iteration_penalty_weight
            + config.quality_weight
        )

        assert total == pytest.approx(1.0, rel=0.01)


class TestRewardConfigCustomization:
    """Tests for RewardConfig customization."""

    def test_custom_config_affects_reward(self):
        """Test custom config values affect reward computation."""
        # Config that only values consensus
        config = RewardConfig(
            consensus_weight=1.0,
            efficiency_weight=0.0,
            confidence_weight=0.0,
            iteration_penalty_weight=0.0,
            quality_weight=0.0,
        )
        reward_model = DebateOutcomeReward(config=config)

        trajectory = Trajectory(query="test")
        trajectory.finalize(
            "answer",
            {
                "consensus_reached": True,
                "agreement_score": 1.0,
            },
            stats={
                "sub_calls_made": 100,  # Very inefficient
                "confidence": 0.1,  # Low confidence
                "iterations": 10,  # Many iterations
            },
        )

        reward = reward_model.compute(trajectory)

        # With only consensus weight, should be high despite other poor metrics
        assert reward > 0.5

    def test_config_scaling_factors(self):
        """Test config scaling factors affect computation."""
        config = RewardConfig(
            max_sub_calls=2,  # Very strict limit
            efficiency_weight=1.0,
            consensus_weight=0.0,
            confidence_weight=0.0,
            iteration_penalty_weight=0.0,
            quality_weight=0.0,
        )
        reward_model = DebateOutcomeReward(config=config)

        # Test with 3 sub-calls (just over limit)
        trajectory = Trajectory(query="test")
        trajectory.finalize("answer", {}, stats={"sub_calls_made": 3})

        components = reward_model.compute_components(trajectory)

        # Should be penalized for exceeding limit
        assert components["efficiency"] < 0
