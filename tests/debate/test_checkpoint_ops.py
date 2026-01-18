"""
Tests for CheckpointOperations.

Tests the checkpoint and memory operations for Arena debates:
- Checkpoint creation during debate rounds
- Memory outcome storage and updates
- Evidence storage in memory
"""

from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.checkpoint_ops import CheckpointOperations


class MockDebateContext:
    """Mock DebateContext for testing."""

    def __init__(self, debate_id: str = "test-debate"):
        self.debate_id = debate_id
        self.result = MagicMock()
        self.result.messages = []
        self.result.critiques = []
        self.result.votes = {}
        self.result.final_answer = "Test consensus"


class MockCheckpointManager:
    """Mock CheckpointManager for testing."""

    def __init__(self):
        self.checkpoints_created = []
        self.should_checkpoint_result = True

    def should_checkpoint(self, debate_id: str, round_num: int) -> bool:
        return self.should_checkpoint_result

    async def create_checkpoint(self, **kwargs: Any) -> None:
        self.checkpoints_created.append(kwargs)


class MockMemoryManager:
    """Mock MemoryManager for testing."""

    def __init__(self):
        self.stored_outcomes = []
        self.stored_evidence = []
        self.tracked_ids = []
        self.outcomes_updated = False

    def store_debate_outcome(
        self, result: Any, task: str, belief_cruxes: Optional[list[str]] = None
    ) -> None:
        self.stored_outcomes.append({
            "result": result,
            "task": task,
            "belief_cruxes": belief_cruxes,
        })

    def store_evidence(self, evidence_snippets: list, task: str) -> None:
        self.stored_evidence.append({
            "snippets": evidence_snippets,
            "task": task,
        })

    def track_retrieved_ids(self, ids: set, tiers: Optional[dict] = None) -> None:
        # Copy the sets/dicts to preserve values before cache clearing
        self.tracked_ids.append({"ids": set(ids), "tiers": dict(tiers) if tiers else None})

    def update_memory_outcomes(self, result: Any) -> None:
        self.outcomes_updated = True


class MockStateCache:
    """Mock DebateStateCache for testing."""

    def __init__(self):
        self.continuum_retrieved_ids: set[str] = {"id-1", "id-2"}
        self.continuum_retrieved_tiers: dict[str, str] = {
            "id-1": "fast",
            "id-2": "slow",
        }
        self.cleared = False

    def clear_continuum_tracking(self) -> None:
        self.continuum_retrieved_ids.clear()
        self.continuum_retrieved_tiers.clear()
        self.cleared = True


class TestCheckpointOperationsInit:
    """Test CheckpointOperations initialization."""

    def test_init_with_all_dependencies(self):
        """Test initialization with all dependencies."""
        checkpoint_mgr = MockCheckpointManager()
        memory_mgr = MockMemoryManager()
        cache = MockStateCache()

        ops = CheckpointOperations(
            checkpoint_manager=checkpoint_mgr,
            memory_manager=memory_mgr,
            cache=cache,
        )

        assert ops.checkpoint_manager == checkpoint_mgr
        assert ops.memory_manager == memory_mgr
        assert ops._cache == cache

    def test_init_without_dependencies(self):
        """Test initialization without dependencies."""
        ops = CheckpointOperations()

        assert ops.checkpoint_manager is None
        assert ops.memory_manager is None
        assert ops._cache is None


class TestCreateCheckpoint:
    """Test create_checkpoint method."""

    @pytest.mark.asyncio
    async def test_create_checkpoint_success(self):
        """Test successful checkpoint creation."""
        checkpoint_mgr = MockCheckpointManager()
        ops = CheckpointOperations(checkpoint_manager=checkpoint_mgr)

        ctx = MockDebateContext()
        ctx.result.messages = [MagicMock(content="msg1")]
        ctx.result.critiques = [MagicMock(text="critique1")]
        ctx.result.votes = {"agent1": "choice1"}

        env = MagicMock()
        env.task = "Test task"

        agents = [MagicMock(name="agent1")]
        protocol = MagicMock()
        protocol.rounds = 3

        await ops.create_checkpoint(ctx, round_num=1, env=env, agents=agents, protocol=protocol)

        assert len(checkpoint_mgr.checkpoints_created) == 1
        checkpoint = checkpoint_mgr.checkpoints_created[0]
        assert checkpoint["debate_id"] == "test-debate"
        assert checkpoint["task"] == "Test task"
        assert checkpoint["current_round"] == 1
        assert checkpoint["total_rounds"] == 3
        assert checkpoint["phase"] == "revision"

    @pytest.mark.asyncio
    async def test_create_checkpoint_skipped_when_not_needed(self):
        """Test checkpoint is skipped when should_checkpoint returns False."""
        checkpoint_mgr = MockCheckpointManager()
        checkpoint_mgr.should_checkpoint_result = False
        ops = CheckpointOperations(checkpoint_manager=checkpoint_mgr)

        ctx = MockDebateContext()
        env = MagicMock(task="Test task")
        agents = []
        protocol = MagicMock(rounds=3)

        await ops.create_checkpoint(ctx, round_num=1, env=env, agents=agents, protocol=protocol)

        assert len(checkpoint_mgr.checkpoints_created) == 0

    @pytest.mark.asyncio
    async def test_create_checkpoint_no_manager(self):
        """Test checkpoint creation when no manager configured."""
        ops = CheckpointOperations()

        ctx = MockDebateContext()
        env = MagicMock(task="Test task")
        agents = []
        protocol = MagicMock(rounds=3)

        # Should not raise
        await ops.create_checkpoint(ctx, round_num=1, env=env, agents=agents, protocol=protocol)

    @pytest.mark.asyncio
    async def test_create_checkpoint_handles_errors(self):
        """Test checkpoint creation handles errors gracefully."""
        checkpoint_mgr = MagicMock()
        checkpoint_mgr.should_checkpoint.return_value = True
        checkpoint_mgr.create_checkpoint = AsyncMock(side_effect=IOError("Disk full"))
        ops = CheckpointOperations(checkpoint_manager=checkpoint_mgr)

        ctx = MockDebateContext()
        env = MagicMock(task="Test task")
        agents = []
        protocol = MagicMock(rounds=3)

        # Should not raise, just log warning
        await ops.create_checkpoint(ctx, round_num=1, env=env, agents=agents, protocol=protocol)


class TestStoreDebateOutcome:
    """Test store_debate_outcome method."""

    def test_store_debate_outcome_success(self):
        """Test successful outcome storage."""
        memory_mgr = MockMemoryManager()
        ops = CheckpointOperations(memory_manager=memory_mgr)

        result = MagicMock()
        task = "Test debate task"
        belief_cruxes = ["crux1", "crux2"]

        ops.store_debate_outcome(result, task, belief_cruxes=belief_cruxes)

        assert len(memory_mgr.stored_outcomes) == 1
        stored = memory_mgr.stored_outcomes[0]
        assert stored["result"] == result
        assert stored["task"] == task
        assert stored["belief_cruxes"] == belief_cruxes

    def test_store_debate_outcome_no_manager(self):
        """Test outcome storage when no manager configured."""
        ops = CheckpointOperations()

        result = MagicMock()
        task = "Test debate task"

        # Should not raise
        ops.store_debate_outcome(result, task)

    def test_store_debate_outcome_without_cruxes(self):
        """Test outcome storage without belief cruxes."""
        memory_mgr = MockMemoryManager()
        ops = CheckpointOperations(memory_manager=memory_mgr)

        result = MagicMock()
        task = "Test debate task"

        ops.store_debate_outcome(result, task)

        assert len(memory_mgr.stored_outcomes) == 1
        stored = memory_mgr.stored_outcomes[0]
        assert stored["belief_cruxes"] is None


class TestStoreEvidence:
    """Test store_evidence method."""

    def test_store_evidence_success(self):
        """Test successful evidence storage."""
        memory_mgr = MockMemoryManager()
        ops = CheckpointOperations(memory_manager=memory_mgr)

        evidence = [{"text": "Evidence 1"}, {"text": "Evidence 2"}]
        task = "Test task"

        ops.store_evidence(evidence, task)

        assert len(memory_mgr.stored_evidence) == 1
        stored = memory_mgr.stored_evidence[0]
        assert stored["snippets"] == evidence
        assert stored["task"] == task

    def test_store_evidence_no_manager(self):
        """Test evidence storage when no manager configured."""
        ops = CheckpointOperations()

        evidence = [{"text": "Evidence 1"}]
        task = "Test task"

        # Should not raise
        ops.store_evidence(evidence, task)


class TestUpdateMemoryOutcomes:
    """Test update_memory_outcomes method."""

    def test_update_memory_outcomes_success(self):
        """Test successful memory outcome update."""
        memory_mgr = MockMemoryManager()
        cache = MockStateCache()
        # Store original values before they get cleared
        original_ids = set(cache.continuum_retrieved_ids)
        original_tiers = dict(cache.continuum_retrieved_tiers)
        ops = CheckpointOperations(memory_manager=memory_mgr, cache=cache)

        result = MagicMock()

        ops.update_memory_outcomes(result)

        assert len(memory_mgr.tracked_ids) == 1
        # The tracked IDs were passed before cache clearing
        assert memory_mgr.tracked_ids[0]["ids"] == original_ids
        assert memory_mgr.tracked_ids[0]["tiers"] == original_tiers
        assert memory_mgr.outcomes_updated is True
        assert cache.cleared is True

    def test_update_memory_outcomes_no_manager(self):
        """Test memory outcome update when no manager configured."""
        ops = CheckpointOperations()

        result = MagicMock()

        # Should not raise
        ops.update_memory_outcomes(result)

    def test_update_memory_outcomes_no_cache(self):
        """Test memory outcome update when no cache configured."""
        memory_mgr = MockMemoryManager()
        ops = CheckpointOperations(memory_manager=memory_mgr)

        result = MagicMock()

        # Should not raise (early return)
        ops.update_memory_outcomes(result)

        assert memory_mgr.outcomes_updated is False
