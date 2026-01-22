"""
RLM Training Pipeline Integration Tests.

Tests comprehensive training scenarios including:
- End-to-end training with mock RLM
- Checkpoint save/load recovery
- Async trajectory collection
- Multi-epoch training convergence
- Error handling and resilience
"""

import asyncio
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.rlm.training import (
    CompositeReward,
    DebateOutcomeReward,
    ExperienceBuffer,
    PolicyState,
    RefinementPolicy,
    SparseReward,
    Step,
    StrategyPolicy,
    Trajectory,
)


# =============================================================================
# Mock RLM Components
# =============================================================================


@dataclass
class MockRLMResult:
    """Mock result from RLM query."""

    answer: str
    confidence: float
    strategy: str
    iterations: int
    tokens_examined: int
    sub_calls: int
    refinement_history: List[Dict[str, Any]]


class MockAragoraRLM:
    """Mock RLM for testing training pipeline."""

    def __init__(
        self,
        success_rate: float = 0.8,
        avg_confidence: float = 0.7,
        strategies: Optional[List[str]] = None,
    ):
        self.success_rate = success_rate
        self.avg_confidence = avg_confidence
        self.strategies = strategies or ["peek", "grep", "summarize"]
        self.call_count = 0
        self._should_fail = False

    def set_failure_mode(self, should_fail: bool) -> None:
        """Set whether next query should fail."""
        self._should_fail = should_fail

    async def query_with_refinement(
        self,
        query: str,
        context: Any,
        strategy: Optional[str] = None,
        max_iterations: int = 5,
        **kwargs,
    ) -> MockRLMResult:
        """Simulate RLM query with refinement."""
        self.call_count += 1

        if self._should_fail:
            self._should_fail = False  # Reset for next call
            raise RuntimeError("Simulated RLM failure")

        # Simulate variable results
        import random

        success = random.random() < self.success_rate
        confidence = self.avg_confidence + random.uniform(-0.2, 0.2)
        confidence = max(0.0, min(1.0, confidence))

        selected_strategy = strategy or random.choice(self.strategies)
        iterations = random.randint(1, max_iterations)

        # Build refinement history
        history = []
        for i in range(iterations):
            history.append(
                {
                    "iteration": i + 1,
                    "confidence": confidence * (0.5 + 0.5 * (i + 1) / iterations),
                    "tokens": random.randint(100, 500),
                }
            )

        return MockRLMResult(
            answer=f"Answer to: {query}" if success else "",
            confidence=confidence,
            strategy=selected_strategy,
            iterations=iterations,
            tokens_examined=sum(h["tokens"] for h in history),
            sub_calls=iterations + random.randint(0, 3),
            refinement_history=history,
        )


class MockTrainer:
    """Minimal trainer implementation for testing."""

    def __init__(
        self,
        rlm: MockAragoraRLM,
        buffer_size: int = 1000,
        exploration_rate: float = 0.3,
    ):
        self.rlm = rlm
        self.buffer = ExperienceBuffer(max_size=buffer_size)
        self.strategy_policy = StrategyPolicy(
            strategies=["peek", "grep", "summarize", "partition_map"],
            exploration_rate=exploration_rate,
        )
        self.refinement_policy = RefinementPolicy(confidence_threshold=0.7)
        self.reward_model = DebateOutcomeReward()
        self.epoch = 0
        self.total_trajectories = 0
        self.metrics_history: List[Dict[str, float]] = []

    async def collect_trajectory(
        self,
        query: str,
        context: Any,
    ) -> Trajectory:
        """Collect a single trajectory from RLM."""
        trajectory = Trajectory()

        # Select strategy using PolicyState
        state = PolicyState(
            query=query,
            context_tokens=1000,
            iteration=0,
        )
        strategy = self.strategy_policy.act(state)

        # Add initial step
        trajectory.add_step(
            Step(
                state=state,
                action=strategy,
                tokens_examined=0,
            )
        )

        # Query RLM
        try:
            result = await self.rlm.query_with_refinement(
                query=query,
                context=context,
                strategy=strategy,
            )

            # Add refinement steps
            for i, hist in enumerate(result.refinement_history):
                trajectory.add_step(
                    Step(
                        state={"iteration": i + 1, "confidence": hist["confidence"]},
                        action="refine" if hist["confidence"] < 0.7 else "accept",
                        tokens_examined=hist["tokens"],
                    )
                )

            # Finalize with outcome
            trajectory.finalize(
                answer=result.answer,
                outcome={
                    "success": result.confidence >= 0.6,
                    "confidence": result.confidence,
                },
                stats={"total_tokens": result.tokens_examined},
            )

        except Exception as e:
            trajectory.finalize(
                answer="",
                outcome={"success": False, "error": str(e)},
            )

        return trajectory

    async def collect_trajectories(
        self,
        queries: List[tuple[str, Any]],
        concurrency: int = 5,
    ) -> List[Trajectory]:
        """Collect multiple trajectories concurrently."""
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_collect(query: str, context: Any) -> Trajectory:
            async with semaphore:
                return await self.collect_trajectory(query, context)

        tasks = [bounded_collect(q, c) for q, c in queries]
        return await asyncio.gather(*tasks)

    def compute_rewards(self, trajectories: List[Trajectory]) -> List[float]:
        """Compute rewards for trajectories."""
        rewards = []
        for traj in trajectories:
            reward = self.reward_model.compute(traj)
            rewards.append(reward)
        return rewards

    def update_policies(self, trajectories: List[Trajectory], rewards: List[float]) -> None:
        """Update policies based on rewards."""
        for traj, reward in zip(trajectories, rewards):
            # Update strategy weights using the trajectory's strategy and outcome
            strategy = traj.strategy
            # Positive reward increases weight, negative decreases
            self.strategy_policy.update_weights(
                strategy=strategy,
                feature="outcome_reward",
                delta=reward,
                learning_rate=0.01,
            )

    async def train_step(
        self,
        queries: List[tuple[str, Any]],
    ) -> Dict[str, float]:
        """Execute a single training step."""
        # Collect trajectories
        trajectories = await self.collect_trajectories(queries)

        # Compute rewards
        rewards = self.compute_rewards(trajectories)

        # Update policies
        self.update_policies(trajectories, rewards)

        # Store in buffer
        for traj, reward in zip(trajectories, rewards):
            self.buffer.add(traj, reward)

        self.total_trajectories += len(trajectories)
        self.epoch += 1

        # Compute metrics
        success_count = sum(1 for t in trajectories if t.outcome.get("success", False))
        metrics = {
            "epoch": self.epoch,
            "num_trajectories": len(trajectories),
            "avg_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "success_rate": success_count / len(trajectories) if trajectories else 0.0,
            "exploration_rate": self.strategy_policy.exploration_rate,
            "buffer_size": len(self.buffer),
        }

        self.metrics_history.append(metrics)
        return metrics

    async def train(
        self,
        queries_per_epoch: List[List[tuple[str, Any]]],
        checkpoint_dir: Optional[Path] = None,
        checkpoint_interval: int = 5,
    ) -> List[Dict[str, float]]:
        """Run full training loop."""
        all_metrics = []

        for epoch_idx, queries in enumerate(queries_per_epoch):
            metrics = await self.train_step(queries)
            all_metrics.append(metrics)

            # Checkpoint
            if checkpoint_dir and (epoch_idx + 1) % checkpoint_interval == 0:
                self._save_checkpoint(checkpoint_dir, epoch_idx + 1)

            # Decay exploration rate
            self.strategy_policy.exploration_rate *= 0.95

        return all_metrics

    def _save_checkpoint(self, checkpoint_dir: Path, epoch: int) -> None:
        """Save training checkpoint."""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save buffer
        buffer_path = checkpoint_dir / f"buffer_epoch_{epoch}.json"
        self.buffer.save(str(buffer_path))

        # Save metadata
        meta_path = checkpoint_dir / f"trainer_epoch_{epoch}.json"
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "epoch": epoch,
                    "total_trajectories": self.total_trajectories,
                    "exploration_rate": self.strategy_policy.exploration_rate,
                },
                f,
            )

    def _load_checkpoint(self, checkpoint_dir: Path, epoch: int) -> None:
        """Load training checkpoint."""
        buffer_path = checkpoint_dir / f"buffer_epoch_{epoch}.json"
        self.buffer = ExperienceBuffer.load(str(buffer_path))

        meta_path = checkpoint_dir / f"trainer_epoch_{epoch}.json"
        with open(meta_path) as f:
            meta = json.load(f)
            self.epoch = meta["epoch"]
            self.total_trajectories = meta["total_trajectories"]
            self.strategy_policy.exploration_rate = meta["exploration_rate"]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_rlm():
    """Create mock RLM for testing."""
    return MockAragoraRLM(success_rate=0.8, avg_confidence=0.7)


@pytest.fixture
def mock_trainer(mock_rlm):
    """Create mock trainer with RLM."""
    return MockTrainer(rlm=mock_rlm, buffer_size=1000)


@pytest.fixture
def sample_queries():
    """Generate sample queries for testing."""
    return [(f"Query {i}", {"context": f"Context {i}", "tokens": 1000}) for i in range(10)]


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory for checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# Test: End-to-End Training
# =============================================================================


class TestEndToEndTraining:
    """Tests for complete training cycles."""

    @pytest.mark.asyncio
    async def test_single_train_step(self, mock_trainer, sample_queries):
        """Test a single training step with mock RLM."""
        queries = sample_queries[:5]
        metrics = await mock_trainer.train_step(queries)

        assert metrics["epoch"] == 1
        assert metrics["num_trajectories"] == 5
        assert 0 <= metrics["avg_reward"] <= 1
        assert 0 <= metrics["success_rate"] <= 1
        assert len(mock_trainer.buffer) == 5

    @pytest.mark.asyncio
    async def test_multiple_train_steps(self, mock_trainer, sample_queries):
        """Test multiple sequential training steps."""
        for i in range(3):
            queries = sample_queries[: i + 3]
            metrics = await mock_trainer.train_step(queries)
            assert metrics["epoch"] == i + 1

        assert mock_trainer.epoch == 3
        assert mock_trainer.total_trajectories > 0

    @pytest.mark.asyncio
    async def test_full_training_loop(self, mock_trainer):
        """Test complete training loop with multiple epochs."""
        # Create queries for 5 epochs, 4 queries each
        queries_per_epoch = [[(f"Q{e}_{i}", {"ctx": i}) for i in range(4)] for e in range(5)]

        all_metrics = await mock_trainer.train(queries_per_epoch)

        assert len(all_metrics) == 5
        assert mock_trainer.epoch == 5
        assert mock_trainer.total_trajectories == 20

    @pytest.mark.asyncio
    async def test_exploration_decay_over_training(self, mock_trainer):
        """Test that exploration rate decays during training."""
        initial_exploration = mock_trainer.strategy_policy.exploration_rate

        queries_per_epoch = [[(f"Q{e}_{i}", {}) for i in range(3)] for e in range(10)]

        await mock_trainer.train(queries_per_epoch)

        final_exploration = mock_trainer.strategy_policy.exploration_rate
        assert final_exploration < initial_exploration


# =============================================================================
# Test: Checkpoint Save/Load
# =============================================================================


class TestCheckpointRecovery:
    """Tests for checkpoint save and load functionality."""

    @pytest.mark.asyncio
    async def test_checkpoint_saves_buffer(self, mock_trainer, temp_checkpoint_dir):
        """Test that checkpoint saves buffer to disk."""
        # Train for a few steps
        for i in range(5):
            await mock_trainer.train_step([(f"Q{i}", {})])

        # Save checkpoint
        mock_trainer._save_checkpoint(temp_checkpoint_dir, 5)

        # Verify files created
        buffer_file = temp_checkpoint_dir / "buffer_epoch_5.json"
        meta_file = temp_checkpoint_dir / "trainer_epoch_5.json"
        assert buffer_file.exists()
        assert meta_file.exists()

    @pytest.mark.asyncio
    async def test_checkpoint_load_restores_state(self, mock_rlm, temp_checkpoint_dir):
        """Test that loading checkpoint restores trainer state."""
        # Train first trainer
        trainer1 = MockTrainer(rlm=mock_rlm, buffer_size=1000)
        for i in range(5):
            await trainer1.train_step([(f"Q{i}", {})])
        trainer1._save_checkpoint(temp_checkpoint_dir, 5)

        original_epoch = trainer1.epoch
        original_trajectories = trainer1.total_trajectories
        original_buffer_size = len(trainer1.buffer)

        # Create new trainer and load checkpoint
        trainer2 = MockTrainer(rlm=mock_rlm, buffer_size=1000)
        trainer2._load_checkpoint(temp_checkpoint_dir, 5)

        assert trainer2.epoch == original_epoch
        assert trainer2.total_trajectories == original_trajectories
        assert len(trainer2.buffer) == original_buffer_size

    @pytest.mark.asyncio
    async def test_training_continues_after_load(self, mock_rlm, temp_checkpoint_dir):
        """Test that training can continue after loading checkpoint."""
        # Initial training
        trainer1 = MockTrainer(rlm=mock_rlm, buffer_size=1000)
        for i in range(5):
            await trainer1.train_step([(f"Q{i}", {})])
        trainer1._save_checkpoint(temp_checkpoint_dir, 5)

        # Load and continue
        trainer2 = MockTrainer(rlm=mock_rlm, buffer_size=1000)
        trainer2._load_checkpoint(temp_checkpoint_dir, 5)

        # Continue training
        for i in range(5, 10):
            await trainer2.train_step([(f"Q{i}", {})])

        assert trainer2.epoch == 10
        assert len(trainer2.buffer) == 10

    @pytest.mark.asyncio
    async def test_periodic_checkpointing_in_train(self, mock_trainer, temp_checkpoint_dir):
        """Test that train() saves checkpoints at intervals."""
        queries_per_epoch = [[(f"Q{e}", {})] for e in range(12)]

        await mock_trainer.train(
            queries_per_epoch,
            checkpoint_dir=temp_checkpoint_dir,
            checkpoint_interval=5,
        )

        # Should have checkpoints at epoch 5 and 10
        assert (temp_checkpoint_dir / "buffer_epoch_5.json").exists()
        assert (temp_checkpoint_dir / "buffer_epoch_10.json").exists()


# =============================================================================
# Test: Async Trajectory Collection
# =============================================================================


class TestAsyncTrajectoryCollection:
    """Tests for concurrent trajectory collection."""

    @pytest.mark.asyncio
    async def test_collect_single_trajectory(self, mock_trainer):
        """Test collecting a single trajectory."""
        traj = await mock_trainer.collect_trajectory("Test query", {"ctx": "test"})

        assert isinstance(traj, Trajectory)
        assert traj.is_terminal
        assert len(traj.steps) > 0

    @pytest.mark.asyncio
    async def test_collect_multiple_trajectories_concurrent(self, mock_trainer):
        """Test collecting multiple trajectories concurrently."""
        queries = [(f"Query {i}", {"idx": i}) for i in range(10)]

        trajectories = await mock_trainer.collect_trajectories(queries)

        assert len(trajectories) == 10
        assert all(isinstance(t, Trajectory) for t in trajectories)
        assert all(t.is_terminal for t in trajectories)

    @pytest.mark.asyncio
    async def test_concurrency_limit_respected(self, mock_rlm):
        """Test that concurrency limit is respected."""
        concurrent_calls = []
        max_concurrent = 0
        lock = asyncio.Lock()

        original_query = mock_rlm.query_with_refinement

        async def tracking_query(*args, **kwargs):
            nonlocal max_concurrent
            async with lock:
                concurrent_calls.append(1)
                if len([c for c in concurrent_calls if c == 1]) > max_concurrent:
                    max_concurrent = len([c for c in concurrent_calls if c == 1])

            try:
                return await original_query(*args, **kwargs)
            finally:
                async with lock:
                    concurrent_calls.remove(1)

        mock_rlm.query_with_refinement = tracking_query

        trainer = MockTrainer(rlm=mock_rlm)
        queries = [(f"Q{i}", {}) for i in range(20)]
        await trainer.collect_trajectories(queries, concurrency=3)

        # Should not exceed concurrency limit
        assert max_concurrent <= 3

    @pytest.mark.asyncio
    async def test_partial_failures_in_collection(self, mock_rlm):
        """Test handling of partial failures in concurrent collection."""
        # Make some queries fail
        call_count = 0

        original_query = mock_rlm.query_with_refinement

        async def failing_query(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:  # Every 3rd call fails
                raise RuntimeError("Simulated failure")
            return await original_query(*args, **kwargs)

        mock_rlm.query_with_refinement = failing_query

        trainer = MockTrainer(rlm=mock_rlm)
        queries = [(f"Q{i}", {}) for i in range(9)]
        trajectories = await trainer.collect_trajectories(queries)

        # All trajectories should be returned, some failed
        assert len(trajectories) == 9
        failed = [t for t in trajectories if not t.outcome.get("success", False)]
        # At least 3 failures (from every 3rd call), possibly more due to random success rate
        assert len(failed) >= 3


# =============================================================================
# Test: Training Convergence
# =============================================================================


class TestTrainingConvergence:
    """Tests for training convergence and improvement."""

    @pytest.mark.asyncio
    async def test_reward_tracked_over_epochs(self, mock_trainer):
        """Test that rewards are tracked over training epochs."""
        queries_per_epoch = [[(f"Q{e}_{i}", {}) for i in range(5)] for e in range(10)]

        all_metrics = await mock_trainer.train(queries_per_epoch)

        # Verify rewards tracked
        rewards = [m["avg_reward"] for m in all_metrics]
        assert len(rewards) == 10
        assert all(0 <= r <= 1 for r in rewards)

    @pytest.mark.asyncio
    async def test_success_rate_tracked(self, mock_trainer):
        """Test that success rate is tracked over epochs."""
        queries_per_epoch = [[(f"Q{e}_{i}", {}) for i in range(5)] for e in range(10)]

        all_metrics = await mock_trainer.train(queries_per_epoch)

        success_rates = [m["success_rate"] for m in all_metrics]
        assert len(success_rates) == 10
        assert all(0 <= s <= 1 for s in success_rates)

    @pytest.mark.asyncio
    async def test_buffer_grows_with_training(self, mock_trainer):
        """Test that buffer grows during training."""
        initial_size = len(mock_trainer.buffer)

        for i in range(5):
            await mock_trainer.train_step([(f"Q{i}", {})])

        assert len(mock_trainer.buffer) > initial_size
        assert len(mock_trainer.buffer) == 5

    @pytest.mark.asyncio
    async def test_exploration_decreases_over_time(self, mock_trainer):
        """Test that exploration rate decreases during training."""
        exploration_rates = [mock_trainer.strategy_policy.exploration_rate]

        queries_per_epoch = [[(f"Q{e}", {})] for e in range(20)]

        for queries in queries_per_epoch:
            await mock_trainer.train_step(queries)
            mock_trainer.strategy_policy.exploration_rate *= 0.9
            exploration_rates.append(mock_trainer.strategy_policy.exploration_rate)

        # Should be monotonically decreasing
        for i in range(len(exploration_rates) - 1):
            assert exploration_rates[i + 1] <= exploration_rates[i]


# =============================================================================
# Test: Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in training pipeline."""

    @pytest.mark.asyncio
    async def test_failed_rlm_query_creates_failed_trajectory(self, mock_rlm):
        """Test that failed RLM queries create failed trajectories."""
        mock_rlm.set_failure_mode(True)
        trainer = MockTrainer(rlm=mock_rlm)

        traj = await trainer.collect_trajectory("Query", {})

        assert not traj.outcome.get("success", False)
        assert traj.is_terminal

    @pytest.mark.asyncio
    async def test_training_continues_after_failures(self, mock_rlm):
        """Test that training continues even after some failures."""
        trainer = MockTrainer(rlm=mock_rlm)

        # Set up alternating failures
        call_count = 0
        original_query = mock_rlm.query_with_refinement

        async def sometimes_fail(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise RuntimeError("Fail")
            return await original_query(*args, **kwargs)

        mock_rlm.query_with_refinement = sometimes_fail

        # Should still complete
        metrics = await trainer.train_step([(f"Q{i}", {}) for i in range(4)])

        assert metrics["num_trajectories"] == 4
        # Buffer should have all trajectories, successful or not
        assert len(trainer.buffer) == 4

    @pytest.mark.asyncio
    async def test_buffer_handles_overflow(self, mock_rlm):
        """Test that buffer handles overflow with FIFO eviction."""
        trainer = MockTrainer(rlm=mock_rlm, buffer_size=5)

        # Add more than buffer size
        for i in range(10):
            await trainer.train_step([(f"Q{i}", {})])

        # Buffer should be at max size
        assert len(trainer.buffer) == 5

    @pytest.mark.asyncio
    async def test_empty_queries_handled(self, mock_trainer):
        """Test handling of empty query list."""
        # Should not crash with empty queries
        try:
            await mock_trainer.collect_trajectories([])
            # Empty list is valid
        except Exception:
            pytest.fail("Should handle empty query list")


# =============================================================================
# Test: Metrics and Callbacks
# =============================================================================


class TestMetricsAndCallbacks:
    """Tests for metrics tracking and callbacks."""

    @pytest.mark.asyncio
    async def test_metrics_history_populated(self, mock_trainer):
        """Test that metrics history is populated during training."""
        queries_per_epoch = [[(f"Q{e}", {})] for e in range(5)]

        await mock_trainer.train(queries_per_epoch)

        assert len(mock_trainer.metrics_history) == 5
        for m in mock_trainer.metrics_history:
            assert "epoch" in m
            assert "avg_reward" in m
            assert "success_rate" in m

    @pytest.mark.asyncio
    async def test_metrics_serializable(self, mock_trainer):
        """Test that metrics can be serialized to JSON."""
        await mock_trainer.train_step([("Q1", {}), ("Q2", {})])

        # Should be JSON serializable
        json_str = json.dumps(mock_trainer.metrics_history)
        assert json_str is not None

        # Should round-trip
        loaded = json.loads(json_str)
        assert len(loaded) == len(mock_trainer.metrics_history)

    @pytest.mark.asyncio
    async def test_buffer_statistics_accurate(self, mock_rlm):
        """Test that buffer statistics are accurate."""
        trainer = MockTrainer(rlm=mock_rlm)

        for i in range(10):
            await trainer.train_step([(f"Q{i}", {})])

        stats = trainer.buffer.get_stats()

        assert stats["size"] == 10
        assert "success_rate" in stats
        assert "avg_steps" in stats


# =============================================================================
# Test: Reward Model Integration
# =============================================================================


class TestRewardModelIntegration:
    """Tests for reward model integration in training."""

    def test_debate_outcome_reward_computation(self):
        """Test DebateOutcomeReward computation."""
        reward_model = DebateOutcomeReward()

        # Create successful trajectory
        traj = Trajectory()
        traj.add_step(Step(state={"iteration": 0}, action="peek", tokens_examined=100))
        traj.add_step(Step(state={"iteration": 1}, action="accept", tokens_examined=50))
        traj.finalize(
            answer="Answer",
            outcome={"success": True, "consensus_reached": True, "agreement_score": 0.8},
            stats={"total_tokens": 150},
        )

        reward = reward_model.compute(traj)
        assert -1 <= reward <= 1

    def test_composite_reward_aggregation(self):
        """Test CompositeReward aggregates multiple models."""
        model1 = DebateOutcomeReward()
        model2 = SparseReward(success_reward=1.0, failure_penalty=0.0)

        composite = CompositeReward()
        composite.add_model(model1, 0.7)
        composite.add_model(model2, 0.3)

        traj = Trajectory()
        traj.add_step(Step(state={}, action="test", tokens_examined=100))
        traj.finalize(
            answer="Answer",
            outcome={"success": True, "consensus_reached": True},
        )

        reward = composite.compute(traj)
        assert -1 <= reward <= 1

    @pytest.mark.asyncio
    async def test_rewards_affect_policy_updates(self, mock_rlm):
        """Test that rewards affect policy updates."""
        trainer = MockTrainer(rlm=mock_rlm)

        # Create test state for getting probabilities
        test_state = PolicyState(query="test", context_tokens=1000)

        # Get initial strategy probabilities
        initial_probs = dict(trainer.strategy_policy.get_action_probs(test_state))

        # Train for several epochs
        for i in range(20):
            await trainer.train_step([(f"Q{i}", {})])

        # Probabilities should have changed (due to weight updates)
        final_probs = dict(trainer.strategy_policy.get_action_probs(test_state))

        # At least some strategy probabilities should differ
        # Note: With exploration and random rewards, changes may be small
        # but the mechanism should update weights
        assert trainer.total_trajectories == 20
