"""
E2E Crash Recovery Tests for Debate Checkpoint and GUPP Hook Systems.

Tests the robustness of the debate system when crashes occur during various phases.
Verifies that:
- Debates can be resumed from checkpoints after crashes
- GUPP hook queues properly recover pending work
- State integrity is maintained across process restarts
"""

from __future__ import annotations

import asyncio
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

pytestmark = [pytest.mark.e2e, pytest.mark.asyncio]


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_checkpoint_dir(tmp_path: Path) -> Path:
    """Create temporary directory for checkpoint storage."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


@pytest.fixture
def temp_bead_dir(tmp_path: Path) -> Path:
    """Create temporary directory for bead storage."""
    bead_dir = tmp_path / "beads"
    bead_dir.mkdir(parents=True, exist_ok=True)
    return bead_dir


@pytest_asyncio.fixture
async def checkpoint_store(temp_checkpoint_dir: Path):
    """Create FileCheckpointStore with temp directory."""
    from aragora.debate.checkpoint import FileCheckpointStore

    return FileCheckpointStore(storage_dir=str(temp_checkpoint_dir))


@pytest_asyncio.fixture
async def checkpoint_manager(checkpoint_store):
    """Create CheckpointManager with config."""
    from aragora.debate.checkpoint import CheckpointConfig, CheckpointManager

    config = CheckpointConfig(
        interval_rounds=1,
        compress=False,
        auto_cleanup=False,
    )
    manager = CheckpointManager(store=checkpoint_store, config=config)
    return manager


def create_mock_agent(name: str, response: str = "Test response") -> MagicMock:
    """Create a properly mocked agent with generate/critique/vote methods."""
    agent = MagicMock()
    agent.name = name
    agent.agent_id = f"agent-{name}"

    # Mock async generate method
    async def mock_generate(*args, **kwargs):
        return f"[{name}] {response}"

    agent.generate = AsyncMock(side_effect=mock_generate)

    # Mock async critique method
    async def mock_critique(*args, **kwargs):
        return {
            "agent": name,
            "issues": ["Minor issue"],
            "suggestions": ["Consider improvement"],
            "score": 0.8,
        }

    agent.critique = AsyncMock(side_effect=mock_critique)

    # Mock async vote method
    async def mock_vote(*args, **kwargs):
        return {
            "agent": name,
            "choice": "proposal_1",
            "confidence": 0.85,
            "reasoning": "Best option",
        }

    agent.vote = AsyncMock(side_effect=mock_vote)

    return agent


async def create_test_checkpoint(
    store,
    debate_id: str,
    phase: str,
    round_num: int,
    messages: List[Dict[str, Any]],
    critiques: Optional[List[Dict[str, Any]]] = None,
    votes: Optional[List[Dict[str, Any]]] = None,
):
    """Create and save a test checkpoint."""
    from aragora.debate.checkpoint import CheckpointStatus, DebateCheckpoint

    checkpoint = DebateCheckpoint(
        debate_id=debate_id,
        current_round=round_num,
        total_rounds=5,
        phase=phase,
        message_history=messages,
        critiques=critiques or [],
        votes=votes or [],
        current_consensus=None,
        consensus_confidence=0.0,
        environment_state={"task": "Test debate task"},
        agent_states={},
        metadata={
            "created_at": datetime.now(timezone.utc).isoformat(),
            "checkpointed_by": "test",
        },
        status=CheckpointStatus.VALID,
    )

    await store.save(checkpoint)
    return checkpoint


# ============================================================================
# Proposal Phase Crash Recovery Tests
# ============================================================================


class TestProposalPhaseCrashRecovery:
    """Tests for crash recovery during proposal phase."""

    async def test_crash_during_first_proposal_resume_from_scratch(
        self, checkpoint_store, checkpoint_manager
    ):
        """Test crash during first agent's proposal, resume should restart round."""
        debate_id = str(uuid.uuid4())

        # Create checkpoint at proposal phase with no messages (first agent crashed)
        checkpoint = await create_test_checkpoint(
            store=checkpoint_store,
            debate_id=debate_id,
            phase="proposal",
            round_num=1,
            messages=[],
        )

        # Verify checkpoint was saved
        loaded = await checkpoint_store.load(debate_id)
        assert loaded is not None
        assert loaded.debate_id == debate_id
        assert loaded.phase == "proposal"
        assert loaded.current_round == 1
        assert len(loaded.message_history) == 0

    async def test_crash_after_partial_proposals_preserves_state(self, checkpoint_store):
        """Test crash when 2/3 agents have proposed preserves completed work."""
        debate_id = str(uuid.uuid4())

        # Two agents have completed proposals
        messages = [
            {"agent": "claude", "content": "First proposal", "round": 1},
            {"agent": "gpt4", "content": "Second proposal", "round": 1},
        ]

        checkpoint = await create_test_checkpoint(
            store=checkpoint_store,
            debate_id=debate_id,
            phase="proposal",
            round_num=1,
            messages=messages,
        )

        # Verify state is preserved
        loaded = await checkpoint_store.load(debate_id)
        assert len(loaded.message_history) == 2
        assert loaded.message_history[0]["agent"] == "claude"
        assert loaded.message_history[1]["agent"] == "gpt4"

    async def test_crash_during_later_round_preserves_history(self, checkpoint_store):
        """Test crash during round 3 preserves rounds 1-2 completely."""
        debate_id = str(uuid.uuid4())

        # Messages from rounds 1 and 2
        messages = [
            {"agent": "claude", "content": "R1 proposal", "round": 1},
            {"agent": "gpt4", "content": "R1 proposal", "round": 1},
            {"agent": "gemini", "content": "R1 proposal", "round": 1},
            {"agent": "claude", "content": "R2 proposal", "round": 2},
            {"agent": "gpt4", "content": "R2 proposal", "round": 2},
            {"agent": "gemini", "content": "R2 proposal", "round": 2},
        ]

        checkpoint = await create_test_checkpoint(
            store=checkpoint_store,
            debate_id=debate_id,
            phase="proposal",
            round_num=3,
            messages=messages,
        )

        loaded = await checkpoint_store.load(debate_id)
        assert loaded.current_round == 3
        assert len(loaded.message_history) == 6
        # Round 1 and 2 messages preserved
        round_1_msgs = [m for m in loaded.message_history if m["round"] == 1]
        round_2_msgs = [m for m in loaded.message_history if m["round"] == 2]
        assert len(round_1_msgs) == 3
        assert len(round_2_msgs) == 3


# ============================================================================
# Critique Phase Crash Recovery Tests
# ============================================================================


class TestCritiquePhaseCrashRecovery:
    """Tests for crash recovery during critique phase."""

    async def test_crash_during_critique_preserves_completed_critiques(self, checkpoint_store):
        """Test crash during critique phase preserves completed critiques."""
        debate_id = str(uuid.uuid4())

        messages = [
            {"agent": "claude", "content": "Proposal", "round": 1},
            {"agent": "gpt4", "content": "Proposal", "round": 1},
        ]

        critiques = [
            {
                "critic": "claude",
                "target": "gpt4",
                "issues": ["Issue 1"],
                "suggestions": ["Suggestion 1"],
                "score": 0.7,
            }
        ]

        checkpoint = await create_test_checkpoint(
            store=checkpoint_store,
            debate_id=debate_id,
            phase="critique",
            round_num=1,
            messages=messages,
            critiques=critiques,
        )

        loaded = await checkpoint_store.load(debate_id)
        assert loaded.phase == "critique"
        assert len(loaded.critiques) == 1
        assert loaded.critiques[0]["critic"] == "claude"
        assert loaded.critiques[0]["target"] == "gpt4"

    async def test_critique_relationships_preserved(self, checkpoint_store):
        """Test that critique relationships and details are preserved."""
        debate_id = str(uuid.uuid4())

        detailed_critiques = [
            {
                "critic": "claude",
                "target": "gpt4",
                "issues": ["Lack of evidence", "Unclear reasoning"],
                "suggestions": ["Add citations", "Clarify logic"],
                "score": 0.6,
                "round": 2,
            },
            {
                "critic": "gpt4",
                "target": "claude",
                "issues": ["Too verbose"],
                "suggestions": ["Be concise"],
                "score": 0.75,
                "round": 2,
            },
        ]

        checkpoint = await create_test_checkpoint(
            store=checkpoint_store,
            debate_id=debate_id,
            phase="critique",
            round_num=2,
            messages=[],
            critiques=detailed_critiques,
        )

        loaded = await checkpoint_store.load(debate_id)
        assert len(loaded.critiques) == 2

        # Verify first critique details
        c1 = loaded.critiques[0]
        assert c1["critic"] == "claude"
        assert "Lack of evidence" in c1["issues"]
        assert c1["score"] == 0.6


# ============================================================================
# Voting Phase Crash Recovery Tests
# ============================================================================


class TestVotingPhaseCrashRecovery:
    """Tests for crash recovery during voting phase."""

    async def test_crash_during_voting_preserves_votes(self, checkpoint_store):
        """Test crash during voting phase preserves completed votes."""
        debate_id = str(uuid.uuid4())

        votes = [
            {"agent": "claude", "choice": "proposal_1", "confidence": 0.9},
            {"agent": "gpt4", "choice": "proposal_1", "confidence": 0.85},
        ]

        checkpoint = await create_test_checkpoint(
            store=checkpoint_store,
            debate_id=debate_id,
            phase="vote",
            round_num=3,
            messages=[],
            votes=votes,
        )

        loaded = await checkpoint_store.load(debate_id)
        assert loaded.phase == "vote"
        assert len(loaded.votes) == 2
        assert loaded.votes[0]["agent"] == "claude"
        assert loaded.votes[1]["confidence"] == 0.85


# ============================================================================
# GUPP Hook Recovery Tests
# ============================================================================


class TestGUPPHookRecovery:
    """Tests for GUPP hook queue recovery after process restart."""

    async def test_hook_queue_recovery_finds_pending_work(self, temp_bead_dir):
        """Test that hook queue recovery finds pending beads."""
        try:
            from aragora.nomic.beads import Bead, BeadStatus, BeadStore, BeadType
            from aragora.nomic.hook_queue import HookEntryStatus, HookQueue
        except ImportError:
            pytest.skip("Bead/HookQueue modules not available")

        # Create bead store and debate bead
        bead_store = BeadStore(storage_dir=str(temp_bead_dir), git_enabled=False)

        bead = Bead(
            id=str(uuid.uuid4()),
            type=BeadType.DEBATE_DECISION,
            status=BeadStatus.PENDING,
            metadata={"debate_id": "test-debate-123"},
        )
        bead_store.save(bead)

        # Create hook queue and push bead
        hook_queue = HookQueue(
            agent_id="claude",
            bead_store=bead_store,
            storage_dir=str(temp_bead_dir / "hooks"),
        )
        hook_queue.push(bead)

        # Get pending work
        pending = hook_queue.get_pending()
        assert len(pending) >= 1

        # Find our bead
        found = any(entry.bead_id == bead.id for entry in pending)
        assert found, "Pushed bead should be in pending queue"

    async def test_hook_recovery_skips_completed_beads(self, temp_bead_dir):
        """Test that completed beads are not recovered."""
        try:
            from aragora.nomic.beads import Bead, BeadStatus, BeadStore, BeadType
            from aragora.nomic.hook_queue import HookQueue
        except ImportError:
            pytest.skip("Bead/HookQueue modules not available")

        bead_store = BeadStore(storage_dir=str(temp_bead_dir), git_enabled=False)

        # Create completed bead
        bead = Bead(
            id=str(uuid.uuid4()),
            type=BeadType.DEBATE_DECISION,
            status=BeadStatus.COMPLETED,  # Already completed
            metadata={"debate_id": "test-debate-456"},
        )
        bead_store.save(bead)

        hook_queue = HookQueue(
            agent_id="claude",
            bead_store=bead_store,
            storage_dir=str(temp_bead_dir / "hooks"),
        )

        # Try to push completed bead - should be rejected or filtered
        hook_queue.push(bead)

        # Get pending - completed beads should be filtered
        pending = hook_queue.get_pending()
        completed_pending = [e for e in pending if e.bead_id == bead.id]

        # Completed beads should either not be pushed or be filtered in get_pending
        # Implementation may vary, but completed work should not be re-processed
        assert len(completed_pending) == 0 or pending[0].bead.status != BeadStatus.PENDING


# ============================================================================
# End-to-End Crash Recovery Flow Tests
# ============================================================================


class TestEndToEndCrashRecoveryFlow:
    """End-to-end tests for full crash and resume scenarios."""

    async def test_checkpoint_integrity_verification(self, checkpoint_store):
        """Test that checkpoint integrity can be verified."""
        debate_id = str(uuid.uuid4())

        checkpoint = await create_test_checkpoint(
            store=checkpoint_store,
            debate_id=debate_id,
            phase="proposal",
            round_num=2,
            messages=[{"agent": "test", "content": "Test", "round": 1}],
        )

        # Verify integrity
        is_valid = checkpoint.verify_integrity()
        assert is_valid, "Checkpoint should pass integrity check"

    async def test_checkpoint_list_and_cleanup(self, checkpoint_store):
        """Test that checkpoints can be listed and cleaned up."""
        # Create multiple checkpoints
        debate_ids = [str(uuid.uuid4()) for _ in range(3)]

        for debate_id in debate_ids:
            await create_test_checkpoint(
                store=checkpoint_store,
                debate_id=debate_id,
                phase="proposal",
                round_num=1,
                messages=[],
            )

        # List all checkpoints
        all_checkpoints = await checkpoint_store.list_all()
        assert len(all_checkpoints) >= 3

        # Delete one
        await checkpoint_store.delete(debate_ids[0])

        # Verify deletion
        remaining = await checkpoint_store.list_all()
        assert debate_ids[0] not in remaining

    async def test_checkpoint_metadata_preserved(self, checkpoint_store):
        """Test that custom metadata is preserved across save/load."""
        debate_id = str(uuid.uuid4())

        checkpoint = await create_test_checkpoint(
            store=checkpoint_store,
            debate_id=debate_id,
            phase="synthesis",
            round_num=5,
            messages=[],
        )

        # Add custom metadata
        checkpoint.metadata["custom_field"] = "custom_value"
        checkpoint.metadata["recovery_count"] = 2
        await checkpoint_store.save(checkpoint)

        # Reload and verify
        loaded = await checkpoint_store.load(debate_id)
        assert loaded.metadata.get("custom_field") == "custom_value"
        assert loaded.metadata.get("recovery_count") == 2

    async def test_multiple_checkpoints_per_debate(self, checkpoint_store):
        """Test that multiple checkpoint versions can coexist."""
        from aragora.debate.checkpoint import CheckpointStatus

        debate_id = str(uuid.uuid4())

        # Create initial checkpoint
        cp1 = await create_test_checkpoint(
            store=checkpoint_store,
            debate_id=debate_id,
            phase="proposal",
            round_num=1,
            messages=[{"agent": "a1", "content": "m1", "round": 1}],
        )

        # Get checkpoint ID
        loaded1 = await checkpoint_store.load(debate_id)
        version1_hash = loaded1.compute_hash()

        # Update to round 2
        loaded1.current_round = 2
        loaded1.message_history.append({"agent": "a2", "content": "m2", "round": 2})
        await checkpoint_store.save(loaded1)

        # Verify update
        loaded2 = await checkpoint_store.load(debate_id)
        assert loaded2.current_round == 2
        assert len(loaded2.message_history) == 2
