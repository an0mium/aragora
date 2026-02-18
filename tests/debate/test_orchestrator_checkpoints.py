"""Tests for orchestrator_checkpoints.py — checkpoint save/restore/list/cleanup helpers.

Tests cover:
- save_checkpoint: Creates checkpoint via manager, returns checkpoint_id or None
- restore_from_checkpoint: Restores DebateContext from checkpoint data
- list_checkpoints: Delegates to checkpoint_manager.store.list_checkpoints
- cleanup_checkpoints: Deletes old checkpoints beyond keep_latest count
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Helpers to build mock objects
# =============================================================================


def make_checkpoint_manager():
    """Create a mock CheckpointManager."""
    mgr = MagicMock()
    mgr.store = MagicMock()
    mgr.create_checkpoint = AsyncMock()
    mgr.resume_from_checkpoint = AsyncMock()
    mgr.store.list_checkpoints = AsyncMock()
    mgr.store.delete = AsyncMock()
    return mgr


def make_env(task="Test debate task"):
    """Create a mock Environment."""
    env = MagicMock()
    env.task = task
    return env


def make_protocol(rounds=3):
    """Create a mock DebateProtocol."""
    proto = MagicMock()
    proto.rounds = rounds
    return proto


def make_agents(names=("agent-a", "agent-b")):
    """Create a list of mock agents."""
    agents = []
    for name in names:
        a = MagicMock()
        a.name = name
        agents.append(a)
    return agents


def make_checkpoint_record(checkpoint_id="ckpt-001", debate_id="debate-001",
                            created_at="2026-02-17T10:00:00"):
    """Return a checkpoint metadata dict as returned by list_checkpoints."""
    return {
        "checkpoint_id": checkpoint_id,
        "debate_id": debate_id,
        "task": "Test task",
        "current_round": 2,
        "created_at": created_at,
        "status": "complete",
    }


def make_resumed(original_debate_id="debate-001", checkpoint_id="ckpt-001",
                 current_round=2, critiques_dicts=None):
    """Create a mock resumed checkpoint object."""
    resumed = MagicMock()
    resumed.original_debate_id = original_debate_id

    ckpt = MagicMock()
    ckpt.task = "Test debate task"
    ckpt.current_round = current_round
    ckpt.consensus_confidence = 0.75
    ckpt.current_consensus = "Agents agreed on X"
    ckpt.critiques = critiques_dicts or []

    resumed.checkpoint = ckpt
    resumed.messages = []
    resumed.votes = []
    return resumed


# =============================================================================
# Tests: save_checkpoint
# =============================================================================


class TestSaveCheckpoint:
    """Tests for save_checkpoint()."""

    @pytest.mark.asyncio
    async def test_returns_none_when_manager_is_none(self):
        from aragora.debate.orchestrator_checkpoints import save_checkpoint

        result = await save_checkpoint(
            checkpoint_manager=None,
            debate_id="debate-001",
            env=make_env(),
            protocol=make_protocol(),
            agents=make_agents(),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_manager_is_falsy_empty_string(self):
        from aragora.debate.orchestrator_checkpoints import save_checkpoint

        result = await save_checkpoint(
            checkpoint_manager="",
            debate_id="debate-001",
            env=make_env(),
            protocol=make_protocol(),
            agents=make_agents(),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_checkpoint_id_on_success(self):
        from aragora.debate.orchestrator_checkpoints import save_checkpoint

        mgr = make_checkpoint_manager()
        ckpt = MagicMock()
        ckpt.checkpoint_id = "ckpt-xyz"
        mgr.create_checkpoint.return_value = ckpt

        result = await save_checkpoint(
            checkpoint_manager=mgr,
            debate_id="debate-001",
            env=make_env(),
            protocol=make_protocol(),
            agents=make_agents(),
        )
        assert result == "ckpt-xyz"

    @pytest.mark.asyncio
    async def test_calls_create_checkpoint_with_correct_debate_id(self):
        from aragora.debate.orchestrator_checkpoints import save_checkpoint

        mgr = make_checkpoint_manager()
        ckpt = MagicMock()
        ckpt.checkpoint_id = "ckpt-abc"
        mgr.create_checkpoint.return_value = ckpt

        await save_checkpoint(
            checkpoint_manager=mgr,
            debate_id="debate-42",
            env=make_env("My task"),
            protocol=make_protocol(rounds=5),
            agents=make_agents(),
        )

        mgr.create_checkpoint.assert_awaited_once()
        call_kwargs = mgr.create_checkpoint.call_args.kwargs
        assert call_kwargs["debate_id"] == "debate-42"
        assert call_kwargs["task"] == "My task"
        assert call_kwargs["total_rounds"] == 5

    @pytest.mark.asyncio
    async def test_calls_create_checkpoint_with_phase_and_round(self):
        from aragora.debate.orchestrator_checkpoints import save_checkpoint

        mgr = make_checkpoint_manager()
        ckpt = MagicMock()
        ckpt.checkpoint_id = "ckpt-phase"
        mgr.create_checkpoint.return_value = ckpt

        await save_checkpoint(
            checkpoint_manager=mgr,
            debate_id="debate-001",
            env=make_env(),
            protocol=make_protocol(),
            agents=make_agents(),
            phase="critique",
            current_round=2,
        )

        call_kwargs = mgr.create_checkpoint.call_args.kwargs
        assert call_kwargs["phase"] == "critique"
        assert call_kwargs["current_round"] == 2

    @pytest.mark.asyncio
    async def test_passes_messages_critiques_votes(self):
        from aragora.debate.orchestrator_checkpoints import save_checkpoint

        mgr = make_checkpoint_manager()
        ckpt = MagicMock()
        ckpt.checkpoint_id = "ckpt-data"
        mgr.create_checkpoint.return_value = ckpt

        messages = [MagicMock(), MagicMock()]
        critiques = [MagicMock()]
        votes = [MagicMock(), MagicMock(), MagicMock()]

        await save_checkpoint(
            checkpoint_manager=mgr,
            debate_id="debate-001",
            env=make_env(),
            protocol=make_protocol(),
            agents=make_agents(),
            messages=messages,
            critiques=critiques,
            votes=votes,
        )

        call_kwargs = mgr.create_checkpoint.call_args.kwargs
        assert call_kwargs["messages"] == messages
        assert call_kwargs["critiques"] == critiques
        assert call_kwargs["votes"] == votes

    @pytest.mark.asyncio
    async def test_defaults_to_empty_lists_when_none_passed(self):
        from aragora.debate.orchestrator_checkpoints import save_checkpoint

        mgr = make_checkpoint_manager()
        ckpt = MagicMock()
        ckpt.checkpoint_id = "ckpt-empty"
        mgr.create_checkpoint.return_value = ckpt

        await save_checkpoint(
            checkpoint_manager=mgr,
            debate_id="debate-001",
            env=make_env(),
            protocol=make_protocol(),
            agents=make_agents(),
            messages=None,
            critiques=None,
            votes=None,
        )

        call_kwargs = mgr.create_checkpoint.call_args.kwargs
        assert call_kwargs["messages"] == []
        assert call_kwargs["critiques"] == []
        assert call_kwargs["votes"] == []

    @pytest.mark.asyncio
    async def test_passes_current_consensus(self):
        from aragora.debate.orchestrator_checkpoints import save_checkpoint

        mgr = make_checkpoint_manager()
        ckpt = MagicMock()
        ckpt.checkpoint_id = "ckpt-con"
        mgr.create_checkpoint.return_value = ckpt

        await save_checkpoint(
            checkpoint_manager=mgr,
            debate_id="debate-001",
            env=make_env(),
            protocol=make_protocol(),
            agents=make_agents(),
            current_consensus="Agreed on solution Y",
        )

        call_kwargs = mgr.create_checkpoint.call_args.kwargs
        assert call_kwargs["current_consensus"] == "Agreed on solution Y"

    @pytest.mark.asyncio
    async def test_returns_none_on_oserror(self):
        from aragora.debate.orchestrator_checkpoints import save_checkpoint

        mgr = make_checkpoint_manager()
        mgr.create_checkpoint.side_effect = OSError("Disk full")

        result = await save_checkpoint(
            checkpoint_manager=mgr,
            debate_id="debate-001",
            env=make_env(),
            protocol=make_protocol(),
            agents=make_agents(),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_value_error(self):
        from aragora.debate.orchestrator_checkpoints import save_checkpoint

        mgr = make_checkpoint_manager()
        mgr.create_checkpoint.side_effect = ValueError("Bad value")

        result = await save_checkpoint(
            checkpoint_manager=mgr,
            debate_id="debate-001",
            env=make_env(),
            protocol=make_protocol(),
            agents=make_agents(),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_runtime_error(self):
        from aragora.debate.orchestrator_checkpoints import save_checkpoint

        mgr = make_checkpoint_manager()
        mgr.create_checkpoint.side_effect = RuntimeError("Runtime failure")

        result = await save_checkpoint(
            checkpoint_manager=mgr,
            debate_id="debate-001",
            env=make_env(),
            protocol=make_protocol(),
            agents=make_agents(),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_uses_empty_task_when_env_is_none(self):
        from aragora.debate.orchestrator_checkpoints import save_checkpoint

        mgr = make_checkpoint_manager()
        ckpt = MagicMock()
        ckpt.checkpoint_id = "ckpt-noenv"
        mgr.create_checkpoint.return_value = ckpt

        result = await save_checkpoint(
            checkpoint_manager=mgr,
            debate_id="debate-001",
            env=None,
            protocol=make_protocol(),
            agents=make_agents(),
        )

        assert result == "ckpt-noenv"
        call_kwargs = mgr.create_checkpoint.call_args.kwargs
        assert call_kwargs["task"] == ""

    @pytest.mark.asyncio
    async def test_uses_zero_rounds_when_protocol_is_none(self):
        from aragora.debate.orchestrator_checkpoints import save_checkpoint

        mgr = make_checkpoint_manager()
        ckpt = MagicMock()
        ckpt.checkpoint_id = "ckpt-noproto"
        mgr.create_checkpoint.return_value = ckpt

        await save_checkpoint(
            checkpoint_manager=mgr,
            debate_id="debate-001",
            env=make_env(),
            protocol=None,
            agents=make_agents(),
        )

        call_kwargs = mgr.create_checkpoint.call_args.kwargs
        assert call_kwargs["total_rounds"] == 0

    @pytest.mark.asyncio
    async def test_default_phase_is_manual(self):
        from aragora.debate.orchestrator_checkpoints import save_checkpoint

        mgr = make_checkpoint_manager()
        ckpt = MagicMock()
        ckpt.checkpoint_id = "ckpt-def"
        mgr.create_checkpoint.return_value = ckpt

        await save_checkpoint(
            checkpoint_manager=mgr,
            debate_id="debate-001",
            env=make_env(),
            protocol=make_protocol(),
            agents=[],
        )

        call_kwargs = mgr.create_checkpoint.call_args.kwargs
        assert call_kwargs["phase"] == "manual"


# =============================================================================
# Tests: restore_from_checkpoint
# =============================================================================


class TestRestoreFromCheckpoint:
    """Tests for restore_from_checkpoint()."""

    @pytest.mark.asyncio
    async def test_returns_none_when_manager_is_none(self):
        from aragora.debate.orchestrator_checkpoints import restore_from_checkpoint

        result = await restore_from_checkpoint(
            checkpoint_manager=None,
            checkpoint_id="ckpt-001",
            env=make_env(),
            agents=make_agents(),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_manager_is_falsy(self):
        from aragora.debate.orchestrator_checkpoints import restore_from_checkpoint

        result = await restore_from_checkpoint(
            checkpoint_manager=0,
            checkpoint_id="ckpt-001",
            env=make_env(),
            agents=make_agents(),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_resumed_is_falsy(self):
        from aragora.debate.orchestrator_checkpoints import restore_from_checkpoint

        mgr = make_checkpoint_manager()
        mgr.resume_from_checkpoint.return_value = None

        result = await restore_from_checkpoint(
            checkpoint_manager=mgr,
            checkpoint_id="ckpt-001",
            env=make_env(),
            agents=make_agents(),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_debate_context_on_success(self):
        from aragora.debate.debate_state import DebateContext
        from aragora.debate.orchestrator_checkpoints import restore_from_checkpoint

        mgr = make_checkpoint_manager()
        mgr.resume_from_checkpoint.return_value = make_resumed()

        result = await restore_from_checkpoint(
            checkpoint_manager=mgr,
            checkpoint_id="ckpt-001",
            env=make_env(),
            agents=make_agents(),
        )
        assert result is not None
        assert isinstance(result, DebateContext)

    @pytest.mark.asyncio
    async def test_sets_restored_from_checkpoint_attribute(self):
        from aragora.debate.orchestrator_checkpoints import restore_from_checkpoint

        mgr = make_checkpoint_manager()
        mgr.resume_from_checkpoint.return_value = make_resumed(checkpoint_id="ckpt-abc")

        result = await restore_from_checkpoint(
            checkpoint_manager=mgr,
            checkpoint_id="ckpt-abc",
            env=make_env(),
            agents=make_agents(),
        )
        assert result._restored_from_checkpoint == "ckpt-abc"

    @pytest.mark.asyncio
    async def test_sets_checkpoint_resume_round_attribute(self):
        from aragora.debate.orchestrator_checkpoints import restore_from_checkpoint

        mgr = make_checkpoint_manager()
        mgr.resume_from_checkpoint.return_value = make_resumed(current_round=4)

        result = await restore_from_checkpoint(
            checkpoint_manager=mgr,
            checkpoint_id="ckpt-001",
            env=make_env(),
            agents=make_agents(),
        )
        assert result._checkpoint_resume_round == 4

    @pytest.mark.asyncio
    async def test_context_debate_id_from_resumed(self):
        from aragora.debate.orchestrator_checkpoints import restore_from_checkpoint

        mgr = make_checkpoint_manager()
        mgr.resume_from_checkpoint.return_value = make_resumed(
            original_debate_id="orig-debate-99"
        )

        result = await restore_from_checkpoint(
            checkpoint_manager=mgr,
            checkpoint_id="ckpt-001",
            env=make_env(),
            agents=make_agents(),
        )
        assert result.debate_id == "orig-debate-99"

    @pytest.mark.asyncio
    async def test_context_domain_passed_through(self):
        from aragora.debate.orchestrator_checkpoints import restore_from_checkpoint

        mgr = make_checkpoint_manager()
        mgr.resume_from_checkpoint.return_value = make_resumed()

        result = await restore_from_checkpoint(
            checkpoint_manager=mgr,
            checkpoint_id="ckpt-001",
            env=make_env(),
            agents=make_agents(),
            domain="healthcare",
        )
        assert result.domain == "healthcare"

    @pytest.mark.asyncio
    async def test_context_org_id_passed_through(self):
        from aragora.debate.orchestrator_checkpoints import restore_from_checkpoint

        mgr = make_checkpoint_manager()
        mgr.resume_from_checkpoint.return_value = make_resumed()

        result = await restore_from_checkpoint(
            checkpoint_manager=mgr,
            checkpoint_id="ckpt-001",
            env=make_env(),
            agents=make_agents(),
            org_id="org-456",
        )
        assert result.org_id == "org-456"

    @pytest.mark.asyncio
    async def test_context_hook_manager_passed_through(self):
        from aragora.debate.orchestrator_checkpoints import restore_from_checkpoint

        mgr = make_checkpoint_manager()
        mgr.resume_from_checkpoint.return_value = make_resumed()
        hook_mgr = MagicMock()

        result = await restore_from_checkpoint(
            checkpoint_manager=mgr,
            checkpoint_id="ckpt-001",
            env=make_env(),
            agents=make_agents(),
            hook_manager=hook_mgr,
        )
        assert result.hook_manager is hook_mgr

    @pytest.mark.asyncio
    async def test_result_task_from_checkpoint(self):
        from aragora.debate.orchestrator_checkpoints import restore_from_checkpoint

        mgr = make_checkpoint_manager()
        resumed = make_resumed()
        resumed.checkpoint.task = "A very specific task"
        mgr.resume_from_checkpoint.return_value = resumed

        result = await restore_from_checkpoint(
            checkpoint_manager=mgr,
            checkpoint_id="ckpt-001",
            env=make_env(),
            agents=make_agents(),
        )
        assert result.result.task == "A very specific task"

    @pytest.mark.asyncio
    async def test_result_rounds_used_from_checkpoint(self):
        from aragora.debate.orchestrator_checkpoints import restore_from_checkpoint

        mgr = make_checkpoint_manager()
        mgr.resume_from_checkpoint.return_value = make_resumed(current_round=3)

        result = await restore_from_checkpoint(
            checkpoint_manager=mgr,
            checkpoint_id="ckpt-001",
            env=make_env(),
            agents=make_agents(),
        )
        assert result.result.rounds_used == 3

    @pytest.mark.asyncio
    async def test_result_final_answer_from_checkpoint(self):
        from aragora.debate.orchestrator_checkpoints import restore_from_checkpoint

        mgr = make_checkpoint_manager()
        resumed = make_resumed()
        resumed.checkpoint.current_consensus = "The answer is 42"
        mgr.resume_from_checkpoint.return_value = resumed

        result = await restore_from_checkpoint(
            checkpoint_manager=mgr,
            checkpoint_id="ckpt-001",
            env=make_env(),
            agents=make_agents(),
        )
        assert result.result.final_answer == "The answer is 42"

    @pytest.mark.asyncio
    async def test_critiques_reconstructed_from_dicts(self):
        from aragora.core import Critique
        from aragora.debate.orchestrator_checkpoints import restore_from_checkpoint

        mgr = make_checkpoint_manager()
        critique_dicts = [
            {
                "agent": "agent-a",
                "target_agent": "agent-b",
                "target_content": "some proposal",
                "issues": ["issue1"],
                "suggestions": ["suggestion1"],
                "severity": 5.0,
                "reasoning": "because reasons",
            }
        ]
        mgr.resume_from_checkpoint.return_value = make_resumed(critiques_dicts=critique_dicts)

        result = await restore_from_checkpoint(
            checkpoint_manager=mgr,
            checkpoint_id="ckpt-001",
            env=make_env(),
            agents=make_agents(),
        )
        assert len(result.result.critiques) == 1
        c = result.result.critiques[0]
        assert isinstance(c, Critique)
        assert c.agent == "agent-a"
        assert c.target_agent == "agent-b"
        assert c.severity == 5.0
        assert c.reasoning == "because reasons"

    @pytest.mark.asyncio
    async def test_critiques_defaults_when_keys_missing(self):
        from aragora.core import Critique
        from aragora.debate.orchestrator_checkpoints import restore_from_checkpoint

        mgr = make_checkpoint_manager()
        # Dict with only partial keys
        critique_dicts = [{}]
        mgr.resume_from_checkpoint.return_value = make_resumed(critiques_dicts=critique_dicts)

        result = await restore_from_checkpoint(
            checkpoint_manager=mgr,
            checkpoint_id="ckpt-001",
            env=make_env(),
            agents=make_agents(),
        )
        assert len(result.result.critiques) == 1
        c = result.result.critiques[0]
        assert isinstance(c, Critique)
        assert c.agent == ""
        assert c.severity == 0.0
        assert c.issues == []

    @pytest.mark.asyncio
    async def test_multiple_critiques_reconstructed(self):
        from aragora.debate.orchestrator_checkpoints import restore_from_checkpoint

        mgr = make_checkpoint_manager()
        critique_dicts = [
            {"agent": "a1", "target_agent": "a2", "target_content": "", "issues": [],
             "suggestions": [], "severity": 2.0, "reasoning": "r1"},
            {"agent": "a2", "target_agent": "a1", "target_content": "", "issues": [],
             "suggestions": [], "severity": 7.5, "reasoning": "r2"},
        ]
        mgr.resume_from_checkpoint.return_value = make_resumed(critiques_dicts=critique_dicts)

        result = await restore_from_checkpoint(
            checkpoint_manager=mgr,
            checkpoint_id="ckpt-001",
            env=make_env(),
            agents=make_agents(),
        )
        assert len(result.result.critiques) == 2

    @pytest.mark.asyncio
    async def test_calls_resume_with_correct_ids(self):
        from aragora.debate.orchestrator_checkpoints import restore_from_checkpoint

        mgr = make_checkpoint_manager()
        mgr.resume_from_checkpoint.return_value = make_resumed()

        await restore_from_checkpoint(
            checkpoint_manager=mgr,
            checkpoint_id="ckpt-special-99",
            env=make_env(),
            agents=make_agents(),
            resumed_by="user-alice",
        )

        mgr.resume_from_checkpoint.assert_awaited_once_with(
            checkpoint_id="ckpt-special-99",
            resumed_by="user-alice",
        )

    @pytest.mark.asyncio
    async def test_correlation_id_includes_checkpoint_id_prefix(self):
        from aragora.debate.orchestrator_checkpoints import restore_from_checkpoint

        mgr = make_checkpoint_manager()
        mgr.resume_from_checkpoint.return_value = make_resumed()

        result = await restore_from_checkpoint(
            checkpoint_manager=mgr,
            checkpoint_id="abcdefgh-1234",
            env=make_env(),
            agents=make_agents(),
        )
        # correlation_id is f"resumed-{checkpoint_id[:8]}"
        assert result.correlation_id == "resumed-abcdefgh"

    @pytest.mark.asyncio
    async def test_returns_none_on_attribute_error(self):
        from aragora.debate.orchestrator_checkpoints import restore_from_checkpoint

        mgr = make_checkpoint_manager()
        mgr.resume_from_checkpoint.side_effect = AttributeError("No attr")

        result = await restore_from_checkpoint(
            checkpoint_manager=mgr,
            checkpoint_id="ckpt-001",
            env=make_env(),
            agents=make_agents(),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_key_error(self):
        from aragora.debate.orchestrator_checkpoints import restore_from_checkpoint

        mgr = make_checkpoint_manager()
        mgr.resume_from_checkpoint.side_effect = KeyError("missing_key")

        result = await restore_from_checkpoint(
            checkpoint_manager=mgr,
            checkpoint_id="ckpt-001",
            env=make_env(),
            agents=make_agents(),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_type_error(self):
        from aragora.debate.orchestrator_checkpoints import restore_from_checkpoint

        mgr = make_checkpoint_manager()
        mgr.resume_from_checkpoint.side_effect = TypeError("type mismatch")

        result = await restore_from_checkpoint(
            checkpoint_manager=mgr,
            checkpoint_id="ckpt-001",
            env=make_env(),
            agents=make_agents(),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_result_consensus_reached_is_false(self):
        from aragora.debate.orchestrator_checkpoints import restore_from_checkpoint

        mgr = make_checkpoint_manager()
        mgr.resume_from_checkpoint.return_value = make_resumed()

        result = await restore_from_checkpoint(
            checkpoint_manager=mgr,
            checkpoint_id="ckpt-001",
            env=make_env(),
            agents=make_agents(),
        )
        # consensus_reached is always False on restore (hardcoded)
        assert result.result.consensus_reached is False

    @pytest.mark.asyncio
    async def test_result_messages_from_resumed(self):
        from aragora.debate.orchestrator_checkpoints import restore_from_checkpoint

        mgr = make_checkpoint_manager()
        resumed = make_resumed()
        resumed.messages = [MagicMock(), MagicMock()]
        mgr.resume_from_checkpoint.return_value = resumed

        result = await restore_from_checkpoint(
            checkpoint_manager=mgr,
            checkpoint_id="ckpt-001",
            env=make_env(),
            agents=make_agents(),
        )
        assert result.result.messages == resumed.messages

    @pytest.mark.asyncio
    async def test_result_votes_from_resumed(self):
        from aragora.debate.orchestrator_checkpoints import restore_from_checkpoint

        mgr = make_checkpoint_manager()
        resumed = make_resumed()
        resumed.votes = [MagicMock()]
        mgr.resume_from_checkpoint.return_value = resumed

        result = await restore_from_checkpoint(
            checkpoint_manager=mgr,
            checkpoint_id="ckpt-001",
            env=make_env(),
            agents=make_agents(),
        )
        assert result.result.votes == resumed.votes


# =============================================================================
# Tests: list_checkpoints
# =============================================================================


class TestListCheckpoints:
    """Tests for list_checkpoints()."""

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_manager_is_none(self):
        from aragora.debate.orchestrator_checkpoints import list_checkpoints

        result = await list_checkpoints(checkpoint_manager=None)
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_manager_is_falsy(self):
        from aragora.debate.orchestrator_checkpoints import list_checkpoints

        result = await list_checkpoints(checkpoint_manager=False)
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_list_of_dicts(self):
        from aragora.debate.orchestrator_checkpoints import list_checkpoints

        mgr = make_checkpoint_manager()
        expected = [
            make_checkpoint_record("ckpt-1", "debate-1"),
            make_checkpoint_record("ckpt-2", "debate-1"),
        ]
        mgr.store.list_checkpoints.return_value = expected

        result = await list_checkpoints(checkpoint_manager=mgr, debate_id="debate-1")
        assert result == expected

    @pytest.mark.asyncio
    async def test_calls_store_list_checkpoints_with_debate_id(self):
        from aragora.debate.orchestrator_checkpoints import list_checkpoints

        mgr = make_checkpoint_manager()
        mgr.store.list_checkpoints.return_value = []

        await list_checkpoints(checkpoint_manager=mgr, debate_id="debate-99", limit=50)

        mgr.store.list_checkpoints.assert_awaited_once_with(
            debate_id="debate-99",
            limit=50,
        )

    @pytest.mark.asyncio
    async def test_default_limit_is_100(self):
        from aragora.debate.orchestrator_checkpoints import list_checkpoints

        mgr = make_checkpoint_manager()
        mgr.store.list_checkpoints.return_value = []

        await list_checkpoints(checkpoint_manager=mgr, debate_id="debate-1")

        call_kwargs = mgr.store.list_checkpoints.call_args.kwargs
        assert call_kwargs["limit"] == 100

    @pytest.mark.asyncio
    async def test_debate_id_none_lists_all(self):
        from aragora.debate.orchestrator_checkpoints import list_checkpoints

        mgr = make_checkpoint_manager()
        mgr.store.list_checkpoints.return_value = []

        await list_checkpoints(checkpoint_manager=mgr, debate_id=None)

        call_kwargs = mgr.store.list_checkpoints.call_args.kwargs
        assert call_kwargs["debate_id"] is None

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_oserror(self):
        from aragora.debate.orchestrator_checkpoints import list_checkpoints

        mgr = make_checkpoint_manager()
        mgr.store.list_checkpoints.side_effect = OSError("Storage error")

        result = await list_checkpoints(checkpoint_manager=mgr)
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_attribute_error(self):
        from aragora.debate.orchestrator_checkpoints import list_checkpoints

        mgr = make_checkpoint_manager()
        mgr.store.list_checkpoints.side_effect = AttributeError("No store")

        result = await list_checkpoints(checkpoint_manager=mgr)
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_value_error(self):
        from aragora.debate.orchestrator_checkpoints import list_checkpoints

        mgr = make_checkpoint_manager()
        mgr.store.list_checkpoints.side_effect = ValueError("Bad param")

        result = await list_checkpoints(checkpoint_manager=mgr)
        assert result == []


# =============================================================================
# Tests: cleanup_checkpoints
# =============================================================================


class TestCleanupCheckpoints:
    """Tests for cleanup_checkpoints()."""

    @pytest.mark.asyncio
    async def test_returns_zero_when_manager_is_none(self):
        from aragora.debate.orchestrator_checkpoints import cleanup_checkpoints

        result = await cleanup_checkpoints(
            checkpoint_manager=None,
            debate_id="debate-001",
        )
        assert result == 0

    @pytest.mark.asyncio
    async def test_returns_zero_when_manager_is_falsy(self):
        from aragora.debate.orchestrator_checkpoints import cleanup_checkpoints

        result = await cleanup_checkpoints(
            checkpoint_manager=0,
            debate_id="debate-001",
        )
        assert result == 0

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_extra_checkpoints(self):
        from aragora.debate.orchestrator_checkpoints import cleanup_checkpoints

        mgr = make_checkpoint_manager()
        mgr.store.list_checkpoints.return_value = [
            make_checkpoint_record("ckpt-1", "debate-001", "2026-02-17T10:00:00"),
        ]

        result = await cleanup_checkpoints(
            checkpoint_manager=mgr,
            debate_id="debate-001",
            keep_latest=1,
        )
        assert result == 0
        mgr.store.delete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_deletes_extra_checkpoints(self):
        from aragora.debate.orchestrator_checkpoints import cleanup_checkpoints

        mgr = make_checkpoint_manager()
        mgr.store.list_checkpoints.return_value = [
            make_checkpoint_record("ckpt-1", "debate-001", "2026-02-17T10:00:00"),
            make_checkpoint_record("ckpt-2", "debate-001", "2026-02-17T09:00:00"),
            make_checkpoint_record("ckpt-3", "debate-001", "2026-02-17T08:00:00"),
        ]
        mgr.store.delete.return_value = True

        result = await cleanup_checkpoints(
            checkpoint_manager=mgr,
            debate_id="debate-001",
            keep_latest=1,
        )
        assert result == 2

    @pytest.mark.asyncio
    async def test_keeps_latest_two_checkpoints(self):
        from aragora.debate.orchestrator_checkpoints import cleanup_checkpoints

        mgr = make_checkpoint_manager()
        mgr.store.list_checkpoints.return_value = [
            make_checkpoint_record("ckpt-1", "debate-001", "2026-02-17T10:00:00"),
            make_checkpoint_record("ckpt-2", "debate-001", "2026-02-17T09:00:00"),
            make_checkpoint_record("ckpt-3", "debate-001", "2026-02-17T08:00:00"),
            make_checkpoint_record("ckpt-4", "debate-001", "2026-02-17T07:00:00"),
        ]
        mgr.store.delete.return_value = True

        result = await cleanup_checkpoints(
            checkpoint_manager=mgr,
            debate_id="debate-001",
            keep_latest=2,
        )
        assert result == 2

    @pytest.mark.asyncio
    async def test_sorts_by_created_at_newest_first(self):
        """Verify that the newest checkpoints are kept (not deleted)."""
        from aragora.debate.orchestrator_checkpoints import cleanup_checkpoints

        mgr = make_checkpoint_manager()
        # Provide out-of-order list; newest should be kept
        mgr.store.list_checkpoints.return_value = [
            make_checkpoint_record("ckpt-old", "debate-001", "2026-02-17T07:00:00"),
            make_checkpoint_record("ckpt-new", "debate-001", "2026-02-17T11:00:00"),
            make_checkpoint_record("ckpt-mid", "debate-001", "2026-02-17T09:00:00"),
        ]
        deleted_ids = []

        async def fake_delete(cp_id):
            deleted_ids.append(cp_id)
            return True

        mgr.store.delete = fake_delete

        await cleanup_checkpoints(
            checkpoint_manager=mgr,
            debate_id="debate-001",
            keep_latest=1,
        )

        # Newest (ckpt-new, 11:00:00) should NOT be deleted
        assert "ckpt-new" not in deleted_ids
        # Older ones should be deleted
        assert "ckpt-old" in deleted_ids
        assert "ckpt-mid" in deleted_ids

    @pytest.mark.asyncio
    async def test_calls_list_checkpoints_with_limit_1000(self):
        from aragora.debate.orchestrator_checkpoints import cleanup_checkpoints

        mgr = make_checkpoint_manager()
        mgr.store.list_checkpoints.return_value = []

        await cleanup_checkpoints(
            checkpoint_manager=mgr,
            debate_id="debate-001",
        )

        mgr.store.list_checkpoints.assert_awaited_once_with(
            debate_id="debate-001",
            limit=1000,
        )

    @pytest.mark.asyncio
    async def test_does_not_count_failed_deletes(self):
        from aragora.debate.orchestrator_checkpoints import cleanup_checkpoints

        mgr = make_checkpoint_manager()
        mgr.store.list_checkpoints.return_value = [
            make_checkpoint_record("ckpt-1", "debate-001", "2026-02-17T10:00:00"),
            make_checkpoint_record("ckpt-2", "debate-001", "2026-02-17T09:00:00"),
            make_checkpoint_record("ckpt-3", "debate-001", "2026-02-17T08:00:00"),
        ]
        # First delete succeeds, second fails
        mgr.store.delete.side_effect = [True, False]

        result = await cleanup_checkpoints(
            checkpoint_manager=mgr,
            debate_id="debate-001",
            keep_latest=1,
        )
        # Only 1 successful delete
        assert result == 1

    @pytest.mark.asyncio
    async def test_returns_zero_on_oserror(self):
        from aragora.debate.orchestrator_checkpoints import cleanup_checkpoints

        mgr = make_checkpoint_manager()
        mgr.store.list_checkpoints.side_effect = OSError("Storage error")

        result = await cleanup_checkpoints(
            checkpoint_manager=mgr,
            debate_id="debate-001",
        )
        assert result == 0

    @pytest.mark.asyncio
    async def test_returns_zero_on_attribute_error(self):
        from aragora.debate.orchestrator_checkpoints import cleanup_checkpoints

        mgr = make_checkpoint_manager()
        mgr.store.list_checkpoints.side_effect = AttributeError("No store")

        result = await cleanup_checkpoints(
            checkpoint_manager=mgr,
            debate_id="debate-001",
        )
        assert result == 0

    @pytest.mark.asyncio
    async def test_default_keep_latest_is_one(self):
        from aragora.debate.orchestrator_checkpoints import cleanup_checkpoints

        mgr = make_checkpoint_manager()
        mgr.store.list_checkpoints.return_value = [
            make_checkpoint_record("ckpt-1", "debate-001", "2026-02-17T10:00:00"),
            make_checkpoint_record("ckpt-2", "debate-001", "2026-02-17T09:00:00"),
        ]
        mgr.store.delete.return_value = True

        result = await cleanup_checkpoints(
            checkpoint_manager=mgr,
            debate_id="debate-001",
            # keep_latest not specified — defaults to 1
        )
        assert result == 1

    @pytest.mark.asyncio
    async def test_empty_checkpoints_list_returns_zero(self):
        from aragora.debate.orchestrator_checkpoints import cleanup_checkpoints

        mgr = make_checkpoint_manager()
        mgr.store.list_checkpoints.return_value = []

        result = await cleanup_checkpoints(
            checkpoint_manager=mgr,
            debate_id="debate-001",
            keep_latest=1,
        )
        assert result == 0
        mgr.store.delete.assert_not_awaited()
