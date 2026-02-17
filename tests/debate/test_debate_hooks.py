"""Tests for DebateHooks post-debate lifecycle hooks.

This module tests the DebateHooks class which handles post-debate
processing callbacks for round completion and debate completion events.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from aragora.debate.debate_hooks import DebateHooks, HooksConfig


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_position_ledger() -> Mock:
    """Create a mock position ledger."""
    ledger = Mock()
    ledger.record_position = Mock()
    return ledger


@pytest.fixture
def mock_elo_system() -> Mock:
    """Create a mock ELO system."""
    elo = Mock()
    elo.update_relationships_batch = Mock()
    return elo


@pytest.fixture
def mock_memory_manager() -> Mock:
    """Create a mock memory manager."""
    manager = Mock()
    manager.store_debate_outcome = Mock()
    manager.track_retrieved_ids = Mock()
    manager.update_memory_outcomes = Mock()
    manager.store_evidence = Mock()
    return manager


@pytest.fixture
def mock_evidence_grounder() -> Mock:
    """Create a mock evidence grounder."""
    grounder = Mock()
    grounder.verify_claims_formally = AsyncMock()
    grounder.create_grounded_verdict = Mock()
    return grounder


@pytest.fixture
def mock_calibration_tracker() -> Mock:
    """Create a mock calibration tracker."""
    tracker = Mock()
    tracker.record_prediction = Mock()
    tracker.get_calibration_curve = Mock(return_value=None)
    return tracker


@pytest.fixture
def mock_event_emitter() -> Mock:
    """Create a mock event emitter."""
    emitter = Mock()
    emitter.emit_calibration_update = Mock()
    return emitter


@pytest.fixture
def mock_debate_context() -> Mock:
    """Create a mock debate context."""
    ctx = Mock()
    ctx.debate_id = "test-debate-123"
    ctx.domain = "general"
    return ctx


@pytest.fixture
def mock_debate_result() -> Mock:
    """Create a mock debate result."""
    result = Mock()
    result.final_answer = "This is the final answer."
    result.confidence = 0.85
    result.consensus_reached = True
    result.rounds_used = 3
    result.winner = "agent1"
    result.grounded_verdict = None
    result.votes = []
    return result


@pytest.fixture
def mock_agents() -> list[Mock]:
    """Create a list of mock agents."""
    agents = []
    for i in range(3):
        agent = Mock()
        agent.name = f"agent{i + 1}"
        agents.append(agent)
    return agents


@pytest.fixture
def mock_votes() -> list[Mock]:
    """Create a list of mock votes."""
    votes = []
    for i in range(3):
        vote = Mock()
        vote.agent = f"agent{i + 1}"
        vote.choice = "agent1"
        vote.confidence = 0.8
        votes.append(vote)
    return votes


# =============================================================================
# Test DebateHooks Initialization
# =============================================================================


class TestDebateHooksInit:
    """Test DebateHooks initialization."""

    def test_init_with_no_subsystems(self) -> None:
        """DebateHooks can be created with no subsystems."""
        hooks = DebateHooks()
        assert hooks.position_ledger is None
        assert hooks.elo_system is None
        assert hooks.memory_manager is None
        assert hooks.evidence_grounder is None
        assert hooks.calibration_tracker is None
        assert hooks.event_emitter is None
        assert hooks.slack_webhook_url is None

    def test_init_with_all_subsystems(
        self,
        mock_position_ledger: Mock,
        mock_elo_system: Mock,
        mock_memory_manager: Mock,
        mock_evidence_grounder: Mock,
        mock_calibration_tracker: Mock,
        mock_event_emitter: Mock,
    ) -> None:
        """DebateHooks can be created with all subsystems."""
        hooks = DebateHooks(
            position_ledger=mock_position_ledger,
            elo_system=mock_elo_system,
            memory_manager=mock_memory_manager,
            evidence_grounder=mock_evidence_grounder,
            calibration_tracker=mock_calibration_tracker,
            event_emitter=mock_event_emitter,
            slack_webhook_url="https://hooks.slack.com/test",
        )
        assert hooks.position_ledger is mock_position_ledger
        assert hooks.elo_system is mock_elo_system
        assert hooks.memory_manager is mock_memory_manager
        assert hooks.evidence_grounder is mock_evidence_grounder
        assert hooks.calibration_tracker is mock_calibration_tracker
        assert hooks.event_emitter is mock_event_emitter
        assert hooks.slack_webhook_url == "https://hooks.slack.com/test"

    def test_init_tracking_state(self) -> None:
        """DebateHooks initializes tracking state correctly."""
        hooks = DebateHooks()
        assert hooks._continuum_retrieved_ids == []
        assert hooks._continuum_retrieved_tiers == {}


# =============================================================================
# Test Round Hooks
# =============================================================================


class TestOnRoundComplete:
    """Test on_round_complete hook."""

    def test_no_position_ledger_returns_early(self, mock_debate_context: Mock) -> None:
        """on_round_complete returns early if no position ledger."""
        hooks = DebateHooks()
        proposals = {"agent1": "Proposal 1", "agent2": "Proposal 2"}

        # Should not raise
        hooks.on_round_complete(mock_debate_context, round_num=1, proposals=proposals)

    def test_records_positions_for_all_agents(
        self,
        mock_position_ledger: Mock,
        mock_debate_context: Mock,
    ) -> None:
        """on_round_complete records positions for all agents."""
        hooks = DebateHooks(position_ledger=mock_position_ledger)
        proposals = {"agent1": "Proposal 1", "agent2": "Proposal 2"}

        hooks.on_round_complete(mock_debate_context, round_num=1, proposals=proposals)

        assert mock_position_ledger.record_position.call_count == 2
        calls = mock_position_ledger.record_position.call_args_list

        # Check first call
        assert calls[0].kwargs["agent_name"] == "agent1"
        assert calls[0].kwargs["claim"] == "Proposal 1"
        assert calls[0].kwargs["debate_id"] == "test-debate-123"
        assert calls[0].kwargs["round_num"] == 1

        # Check second call
        assert calls[1].kwargs["agent_name"] == "agent2"
        assert calls[1].kwargs["claim"] == "Proposal 2"

    def test_passes_domain_to_position_ledger(
        self,
        mock_position_ledger: Mock,
        mock_debate_context: Mock,
    ) -> None:
        """on_round_complete passes domain to position ledger."""
        hooks = DebateHooks(position_ledger=mock_position_ledger)
        proposals = {"agent1": "Proposal 1"}

        hooks.on_round_complete(
            mock_debate_context,
            round_num=2,
            proposals=proposals,
            domain="technology",
        )

        mock_position_ledger.record_position.assert_called_once()
        call_kwargs = mock_position_ledger.record_position.call_args.kwargs
        assert call_kwargs["domain"] == "technology"

    def test_truncates_long_proposals(
        self,
        mock_position_ledger: Mock,
        mock_debate_context: Mock,
    ) -> None:
        """on_round_complete truncates proposals longer than 1000 chars."""
        hooks = DebateHooks(position_ledger=mock_position_ledger)
        long_content = "x" * 2000
        proposals = {"agent1": long_content}

        hooks.on_round_complete(mock_debate_context, round_num=1, proposals=proposals)

        call_kwargs = mock_position_ledger.record_position.call_args.kwargs
        assert len(call_kwargs["claim"]) == 1000


class TestRecordPosition:
    """Test _record_position helper method."""

    def test_no_position_ledger_returns_early(self) -> None:
        """_record_position returns early if no position ledger."""
        hooks = DebateHooks()
        # Should not raise
        hooks._record_position(
            agent_name="agent1",
            content="Test content",
            debate_id="test-123",
            round_num=1,
        )

    def test_handles_attribute_error_gracefully(self, mock_position_ledger: Mock) -> None:
        """_record_position handles AttributeError gracefully."""
        mock_position_ledger.record_position.side_effect = AttributeError("test")
        hooks = DebateHooks(position_ledger=mock_position_ledger)

        # Should not raise
        hooks._record_position(
            agent_name="agent1",
            content="Test content",
            debate_id="test-123",
            round_num=1,
        )

    def test_handles_type_error_gracefully(self, mock_position_ledger: Mock) -> None:
        """_record_position handles TypeError gracefully."""
        mock_position_ledger.record_position.side_effect = TypeError("test")
        hooks = DebateHooks(position_ledger=mock_position_ledger)

        # Should not raise
        hooks._record_position(
            agent_name="agent1",
            content="Test content",
            debate_id="test-123",
            round_num=1,
        )

    def test_handles_value_error_gracefully(self, mock_position_ledger: Mock) -> None:
        """_record_position handles ValueError gracefully."""
        mock_position_ledger.record_position.side_effect = ValueError("test")
        hooks = DebateHooks(position_ledger=mock_position_ledger)

        # Should not raise
        hooks._record_position(
            agent_name="agent1",
            content="Test content",
            debate_id="test-123",
            round_num=1,
        )

    def test_handles_key_error_gracefully(self, mock_position_ledger: Mock) -> None:
        """_record_position handles KeyError gracefully."""
        mock_position_ledger.record_position.side_effect = KeyError("test")
        hooks = DebateHooks(position_ledger=mock_position_ledger)

        # Should not raise
        hooks._record_position(
            agent_name="agent1",
            content="Test content",
            debate_id="test-123",
            round_num=1,
        )

    def test_handles_runtime_error_gracefully(self, mock_position_ledger: Mock) -> None:
        """_record_position handles RuntimeError gracefully."""
        mock_position_ledger.record_position.side_effect = RuntimeError("test")
        hooks = DebateHooks(position_ledger=mock_position_ledger)

        # Should not raise
        hooks._record_position(
            agent_name="agent1",
            content="Test content",
            debate_id="test-123",
            round_num=1,
        )


# =============================================================================
# Test Debate Completion Hooks
# =============================================================================


class TestOnDebateComplete:
    """Test on_debate_complete hook."""

    @pytest.mark.asyncio
    async def test_calls_all_hooks(
        self,
        mock_debate_context: Mock,
        mock_debate_result: Mock,
        mock_agents: list[Mock],
        mock_elo_system: Mock,
        mock_memory_manager: Mock,
        mock_evidence_grounder: Mock,
        mock_calibration_tracker: Mock,
    ) -> None:
        """on_debate_complete calls all post-debate hooks."""
        hooks = DebateHooks(
            elo_system=mock_elo_system,
            memory_manager=mock_memory_manager,
            evidence_grounder=mock_evidence_grounder,
            calibration_tracker=mock_calibration_tracker,
        )

        await hooks.on_debate_complete(
            ctx=mock_debate_context,
            result=mock_debate_result,
            agents=mock_agents,
            task="Test task",
        )

        # Verify all hooks were called
        mock_elo_system.update_relationships_batch.assert_called()
        mock_memory_manager.store_debate_outcome.assert_called()
        mock_calibration_tracker.record_prediction.assert_called()

    @pytest.mark.asyncio
    async def test_extracts_participant_names(
        self,
        mock_debate_context: Mock,
        mock_debate_result: Mock,
        mock_agents: list[Mock],
        mock_elo_system: Mock,
    ) -> None:
        """on_debate_complete extracts participant names from agents."""
        hooks = DebateHooks(elo_system=mock_elo_system)

        await hooks.on_debate_complete(
            ctx=mock_debate_context,
            result=mock_debate_result,
            agents=mock_agents,
            task="Test task",
        )

        # Check the batch call includes all agent pairs
        call_args = mock_elo_system.update_relationships_batch.call_args
        updates = call_args[0][0]

        # 3 agents = 3 pairs: (agent1, agent2), (agent1, agent3), (agent2, agent3)
        assert len(updates) == 3

    @pytest.mark.asyncio
    async def test_passes_belief_cruxes(
        self,
        mock_debate_context: Mock,
        mock_debate_result: Mock,
        mock_agents: list[Mock],
        mock_memory_manager: Mock,
    ) -> None:
        """on_debate_complete passes belief cruxes to memory storage."""
        hooks = DebateHooks(memory_manager=mock_memory_manager)
        cruxes = ["crux1", "crux2"]

        await hooks.on_debate_complete(
            ctx=mock_debate_context,
            result=mock_debate_result,
            agents=mock_agents,
            task="Test task",
            belief_cruxes=cruxes,
        )

        call_kwargs = mock_memory_manager.store_debate_outcome.call_args.kwargs
        assert call_kwargs["belief_cruxes"] == cruxes


class TestUpdateRelationships:
    """Test _update_relationships helper method."""

    def test_no_elo_system_returns_early(self) -> None:
        """_update_relationships returns early if no ELO system."""
        hooks = DebateHooks()
        # Should not raise
        hooks._update_relationships(
            debate_id="test-123",
            participants=["agent1", "agent2"],
            winner="agent1",
            votes=[],
        )

    def test_creates_batch_updates_for_all_pairs(self, mock_elo_system: Mock) -> None:
        """_update_relationships creates batch updates for all participant pairs."""
        hooks = DebateHooks(elo_system=mock_elo_system)

        hooks._update_relationships(
            debate_id="test-123",
            participants=["agent1", "agent2", "agent3"],
            winner="agent1",
            votes=[],
        )

        call_args = mock_elo_system.update_relationships_batch.call_args
        updates = call_args[0][0]

        # 3 agents = 3 pairs
        assert len(updates) == 3

        # Check first update
        assert updates[0]["agent_a"] == "agent1"
        assert updates[0]["agent_b"] == "agent2"
        assert updates[0]["debate_increment"] == 1
        assert updates[0]["a_win"] == 1  # agent1 is winner
        assert updates[0]["b_win"] == 0

    def test_detects_agreement_from_votes(
        self, mock_elo_system: Mock, mock_votes: list[Mock]
    ) -> None:
        """_update_relationships detects agreement when agents vote for same choice."""
        hooks = DebateHooks(elo_system=mock_elo_system)

        hooks._update_relationships(
            debate_id="test-123",
            participants=["agent1", "agent2"],
            winner=None,
            votes=mock_votes[:2],  # Both vote for agent1
        )

        call_args = mock_elo_system.update_relationships_batch.call_args
        updates = call_args[0][0]

        assert updates[0]["agreement_increment"] == 1

    def test_handles_no_winner(self, mock_elo_system: Mock) -> None:
        """_update_relationships handles case with no winner."""
        hooks = DebateHooks(elo_system=mock_elo_system)

        hooks._update_relationships(
            debate_id="test-123",
            participants=["agent1", "agent2"],
            winner=None,
            votes=[],
        )

        call_args = mock_elo_system.update_relationships_batch.call_args
        updates = call_args[0][0]

        assert updates[0]["a_win"] == 0
        assert updates[0]["b_win"] == 0

    def test_handles_attribute_error_gracefully(self, mock_elo_system: Mock) -> None:
        """_update_relationships handles AttributeError gracefully."""
        mock_elo_system.update_relationships_batch.side_effect = AttributeError("test")
        hooks = DebateHooks(elo_system=mock_elo_system)

        # Should not raise
        hooks._update_relationships(
            debate_id="test-123",
            participants=["agent1", "agent2"],
            winner=None,
            votes=[],
        )


class TestStoreDebateOutcome:
    """Test _store_debate_outcome helper method."""

    def test_no_memory_manager_returns_early(self, mock_debate_result: Mock) -> None:
        """_store_debate_outcome returns early if no memory manager."""
        hooks = DebateHooks()
        # Should not raise
        hooks._store_debate_outcome(mock_debate_result, "Test task")

    def test_stores_outcome_with_task(
        self, mock_memory_manager: Mock, mock_debate_result: Mock
    ) -> None:
        """_store_debate_outcome stores outcome with task."""
        hooks = DebateHooks(memory_manager=mock_memory_manager)

        hooks._store_debate_outcome(mock_debate_result, "Test task")

        mock_memory_manager.store_debate_outcome.assert_called_once_with(
            mock_debate_result, "Test task", belief_cruxes=None
        )

    def test_normalizes_belief_cruxes(
        self, mock_memory_manager: Mock, mock_debate_result: Mock
    ) -> None:
        """_store_debate_outcome normalizes belief cruxes to strings."""
        hooks = DebateHooks(memory_manager=mock_memory_manager)
        cruxes = [123, "crux2", 456]

        hooks._store_debate_outcome(mock_debate_result, "Test task", cruxes)

        call_kwargs = mock_memory_manager.store_debate_outcome.call_args.kwargs
        assert call_kwargs["belief_cruxes"] == ["123", "crux2", "456"]

    def test_limits_belief_cruxes_to_10(
        self, mock_memory_manager: Mock, mock_debate_result: Mock
    ) -> None:
        """_store_debate_outcome limits belief cruxes to 10 items."""
        hooks = DebateHooks(memory_manager=mock_memory_manager)
        cruxes = [f"crux{i}" for i in range(20)]

        hooks._store_debate_outcome(mock_debate_result, "Test task", cruxes)

        call_kwargs = mock_memory_manager.store_debate_outcome.call_args.kwargs
        assert len(call_kwargs["belief_cruxes"]) == 10

    def test_handles_attribute_error_gracefully(
        self, mock_memory_manager: Mock, mock_debate_result: Mock
    ) -> None:
        """_store_debate_outcome handles AttributeError gracefully."""
        mock_memory_manager.store_debate_outcome.side_effect = AttributeError("test")
        hooks = DebateHooks(memory_manager=mock_memory_manager)

        # Should not raise
        hooks._store_debate_outcome(mock_debate_result, "Test task")


class TestUpdateMemoryOutcomes:
    """Test _update_memory_outcomes helper method."""

    def test_no_memory_manager_returns_early(self, mock_debate_result: Mock) -> None:
        """_update_memory_outcomes returns early if no memory manager."""
        hooks = DebateHooks()
        # Should not raise
        hooks._update_memory_outcomes(mock_debate_result)

    def test_updates_outcomes_when_ids_tracked(
        self, mock_memory_manager: Mock, mock_debate_result: Mock
    ) -> None:
        """_update_memory_outcomes updates outcomes when IDs are tracked."""
        hooks = DebateHooks(memory_manager=mock_memory_manager)
        hooks._continuum_retrieved_ids = ["id1", "id2"]
        hooks._continuum_retrieved_tiers = {"id1": "fast", "id2": "slow"}

        hooks._update_memory_outcomes(mock_debate_result)

        mock_memory_manager.track_retrieved_ids.assert_called_once_with(
            ["id1", "id2"], tiers={"id1": "fast", "id2": "slow"}
        )
        mock_memory_manager.update_memory_outcomes.assert_called_once_with(mock_debate_result)

    def test_clears_tracking_after_update(
        self, mock_memory_manager: Mock, mock_debate_result: Mock
    ) -> None:
        """_update_memory_outcomes clears tracking state after update."""
        hooks = DebateHooks(memory_manager=mock_memory_manager)
        hooks._continuum_retrieved_ids = ["id1", "id2"]
        hooks._continuum_retrieved_tiers = {"id1": "fast"}

        hooks._update_memory_outcomes(mock_debate_result)

        assert hooks._continuum_retrieved_ids == []
        assert hooks._continuum_retrieved_tiers == {}

    def test_no_update_when_no_ids_tracked(
        self, mock_memory_manager: Mock, mock_debate_result: Mock
    ) -> None:
        """_update_memory_outcomes skips update when no IDs tracked."""
        hooks = DebateHooks(memory_manager=mock_memory_manager)

        hooks._update_memory_outcomes(mock_debate_result)

        mock_memory_manager.track_retrieved_ids.assert_not_called()
        mock_memory_manager.update_memory_outcomes.assert_not_called()


class TestUpdateCalibration:
    """Test _update_calibration helper method."""

    def test_no_calibration_tracker_returns_early(
        self, mock_debate_context: Mock, mock_debate_result: Mock
    ) -> None:
        """_update_calibration returns early if no calibration tracker."""
        hooks = DebateHooks()
        # Should not raise
        hooks._update_calibration(
            ctx=mock_debate_context,
            result=mock_debate_result,
            participants=["agent1", "agent2"],
        )

    def test_records_prediction_for_each_participant(
        self,
        mock_calibration_tracker: Mock,
        mock_debate_context: Mock,
        mock_debate_result: Mock,
    ) -> None:
        """_update_calibration records prediction for each participant."""
        hooks = DebateHooks(calibration_tracker=mock_calibration_tracker)

        hooks._update_calibration(
            ctx=mock_debate_context,
            result=mock_debate_result,
            participants=["agent1", "agent2", "agent3"],
        )

        assert mock_calibration_tracker.record_prediction.call_count == 3

    def test_winner_is_marked_correct(
        self,
        mock_calibration_tracker: Mock,
        mock_debate_context: Mock,
        mock_debate_result: Mock,
    ) -> None:
        """_update_calibration marks winner as correct."""
        mock_debate_result.winner = "agent1"
        hooks = DebateHooks(calibration_tracker=mock_calibration_tracker)

        hooks._update_calibration(
            ctx=mock_debate_context,
            result=mock_debate_result,
            participants=["agent1", "agent2"],
        )

        # Find the call for agent1
        calls = mock_calibration_tracker.record_prediction.call_args_list
        agent1_call = [c for c in calls if c.kwargs["agent"] == "agent1"][0]
        assert agent1_call.kwargs["correct"] is True

        # agent2 should be incorrect
        agent2_call = [c for c in calls if c.kwargs["agent"] == "agent2"][0]
        assert agent2_call.kwargs["correct"] is False

    def test_uses_vote_confidence_when_available(
        self,
        mock_calibration_tracker: Mock,
        mock_debate_context: Mock,
        mock_debate_result: Mock,
        mock_votes: list[Mock],
    ) -> None:
        """_update_calibration uses vote confidence when available."""
        mock_debate_result.votes = mock_votes[:1]  # Just agent1's vote
        mock_votes[0].confidence = 0.95
        hooks = DebateHooks(calibration_tracker=mock_calibration_tracker)

        hooks._update_calibration(
            ctx=mock_debate_context,
            result=mock_debate_result,
            participants=["agent1"],
        )

        call_kwargs = mock_calibration_tracker.record_prediction.call_args.kwargs
        assert call_kwargs["confidence"] == 0.95

    def test_emits_calibration_events(
        self,
        mock_calibration_tracker: Mock,
        mock_event_emitter: Mock,
        mock_debate_context: Mock,
        mock_debate_result: Mock,
    ) -> None:
        """_update_calibration emits calibration events when emitter available."""

        # Set up calibration curve data
        @dataclass
        class MockBucket:
            range_start: float
            range_end: float
            total_predictions: int
            correct_predictions: int
            accuracy: float

        mock_calibration_tracker.get_calibration_curve.return_value = [
            MockBucket(0.5, 0.6, 10, 8, 0.8),
            MockBucket(0.6, 0.7, 5, 4, 0.8),
        ]

        hooks = DebateHooks(
            calibration_tracker=mock_calibration_tracker,
            event_emitter=mock_event_emitter,
        )

        hooks._update_calibration(
            ctx=mock_debate_context,
            result=mock_debate_result,
            participants=["agent1"],
        )

        mock_event_emitter.emit_calibration_update.assert_called_once()
        call_kwargs = mock_event_emitter.emit_calibration_update.call_args.kwargs
        assert call_kwargs["agent_name"] == "agent1"
        assert call_kwargs["prediction_count"] == 15
        assert call_kwargs["accuracy"] == 12 / 15


class TestVerifyClaims:
    """Test _verify_claims helper method."""

    @pytest.mark.asyncio
    async def test_no_evidence_grounder_returns_early(self, mock_debate_result: Mock) -> None:
        """_verify_claims returns early if no evidence grounder."""
        hooks = DebateHooks()
        # Should not raise
        await hooks._verify_claims(mock_debate_result)

    @pytest.mark.asyncio
    async def test_no_grounded_verdict_returns_early(
        self, mock_evidence_grounder: Mock, mock_debate_result: Mock
    ) -> None:
        """_verify_claims returns early if no grounded verdict."""
        mock_debate_result.grounded_verdict = None
        hooks = DebateHooks(evidence_grounder=mock_evidence_grounder)

        await hooks._verify_claims(mock_debate_result)

        mock_evidence_grounder.verify_claims_formally.assert_not_called()

    @pytest.mark.asyncio
    async def test_calls_verify_claims_formally(
        self, mock_evidence_grounder: Mock, mock_debate_result: Mock
    ) -> None:
        """_verify_claims calls verify_claims_formally with grounded verdict."""
        mock_debate_result.grounded_verdict = {"claim": "test"}
        hooks = DebateHooks(evidence_grounder=mock_evidence_grounder)

        await hooks._verify_claims(mock_debate_result)

        mock_evidence_grounder.verify_claims_formally.assert_called_once_with({"claim": "test"})

    @pytest.mark.asyncio
    async def test_handles_timeout_gracefully(
        self, mock_evidence_grounder: Mock, mock_debate_result: Mock
    ) -> None:
        """_verify_claims handles asyncio.TimeoutError gracefully."""
        mock_debate_result.grounded_verdict = {"claim": "test"}
        mock_evidence_grounder.verify_claims_formally.side_effect = asyncio.TimeoutError()
        hooks = DebateHooks(evidence_grounder=mock_evidence_grounder)

        # Should not raise
        await hooks._verify_claims(mock_debate_result)

    @pytest.mark.asyncio
    async def test_handles_cancelled_gracefully(
        self, mock_evidence_grounder: Mock, mock_debate_result: Mock
    ) -> None:
        """_verify_claims handles asyncio.CancelledError gracefully."""
        mock_debate_result.grounded_verdict = {"claim": "test"}
        mock_evidence_grounder.verify_claims_formally.side_effect = asyncio.CancelledError()
        hooks = DebateHooks(evidence_grounder=mock_evidence_grounder)

        # Should not raise
        await hooks._verify_claims(mock_debate_result)

    @pytest.mark.asyncio
    async def test_handles_value_error_gracefully(
        self, mock_evidence_grounder: Mock, mock_debate_result: Mock
    ) -> None:
        """_verify_claims handles ValueError gracefully."""
        mock_debate_result.grounded_verdict = {"claim": "test"}
        mock_evidence_grounder.verify_claims_formally.side_effect = ValueError("test")
        hooks = DebateHooks(evidence_grounder=mock_evidence_grounder)

        # Should not raise
        await hooks._verify_claims(mock_debate_result)


# =============================================================================
# Test Memory Tracking
# =============================================================================


class TestTrackRetrievedMemories:
    """Test track_retrieved_memories method."""

    def test_stores_ids_and_tiers(self) -> None:
        """track_retrieved_memories stores IDs and tiers."""
        hooks = DebateHooks()
        ids = ["id1", "id2", "id3"]
        tiers = {"id1": "fast", "id2": "medium", "id3": "slow"}

        hooks.track_retrieved_memories(ids, tiers)

        assert hooks._continuum_retrieved_ids == ids
        assert hooks._continuum_retrieved_tiers == tiers

    def test_replaces_previous_tracking(self) -> None:
        """track_retrieved_memories replaces previous tracking data."""
        hooks = DebateHooks()
        hooks._continuum_retrieved_ids = ["old_id"]
        hooks._continuum_retrieved_tiers = {"old_id": "fast"}

        hooks.track_retrieved_memories(["new_id"], {"new_id": "slow"})

        assert hooks._continuum_retrieved_ids == ["new_id"]
        assert hooks._continuum_retrieved_tiers == {"new_id": "slow"}


# =============================================================================
# Test Evidence Storage
# =============================================================================


class TestStoreEvidence:
    """Test store_evidence method."""

    def test_no_memory_manager_returns_early(self) -> None:
        """store_evidence returns early if no memory manager."""
        hooks = DebateHooks()
        # Should not raise
        hooks.store_evidence([{"snippet": "test"}], "Test task")

    def test_stores_evidence_snippets(self, mock_memory_manager: Mock) -> None:
        """store_evidence stores evidence snippets."""
        hooks = DebateHooks(memory_manager=mock_memory_manager)
        snippets = [{"snippet": "Evidence 1"}, {"snippet": "Evidence 2"}]

        hooks.store_evidence(snippets, "Test task")

        mock_memory_manager.store_evidence.assert_called_once_with(snippets, "Test task")

    def test_handles_attribute_error_gracefully(self, mock_memory_manager: Mock) -> None:
        """store_evidence handles AttributeError gracefully."""
        mock_memory_manager.store_evidence.side_effect = AttributeError("test")
        hooks = DebateHooks(memory_manager=mock_memory_manager)

        # Should not raise
        hooks.store_evidence([{"snippet": "test"}], "Test task")


# =============================================================================
# Test Grounded Verdict Creation
# =============================================================================


class TestCreateGroundedVerdict:
    """Test create_grounded_verdict method."""

    def test_no_evidence_grounder_returns_none(self, mock_debate_result: Mock) -> None:
        """create_grounded_verdict returns None if no evidence grounder."""
        hooks = DebateHooks()
        result = hooks.create_grounded_verdict(mock_debate_result)
        assert result is None

    def test_no_final_answer_returns_none(
        self, mock_evidence_grounder: Mock, mock_debate_result: Mock
    ) -> None:
        """create_grounded_verdict returns None if no final answer."""
        mock_debate_result.final_answer = ""
        hooks = DebateHooks(evidence_grounder=mock_evidence_grounder)

        result = hooks.create_grounded_verdict(mock_debate_result)

        assert result is None
        mock_evidence_grounder.create_grounded_verdict.assert_not_called()

    def test_creates_verdict_with_answer_and_confidence(
        self, mock_evidence_grounder: Mock, mock_debate_result: Mock
    ) -> None:
        """create_grounded_verdict creates verdict with answer and confidence."""
        mock_evidence_grounder.create_grounded_verdict.return_value = {"verdict": "test"}
        hooks = DebateHooks(evidence_grounder=mock_evidence_grounder)

        result = hooks.create_grounded_verdict(mock_debate_result)

        assert result == {"verdict": "test"}
        mock_evidence_grounder.create_grounded_verdict.assert_called_once_with(
            final_answer="This is the final answer.",
            confidence=0.85,
        )

    def test_handles_exception_gracefully(
        self, mock_evidence_grounder: Mock, mock_debate_result: Mock
    ) -> None:
        """create_grounded_verdict handles exceptions gracefully."""
        mock_evidence_grounder.create_grounded_verdict.side_effect = RuntimeError("test")
        hooks = DebateHooks(evidence_grounder=mock_evidence_grounder)

        result = hooks.create_grounded_verdict(mock_debate_result)

        assert result is None


# =============================================================================
# Test Slack Webhook Notifications
# =============================================================================


class TestNotifySlackWebhook:
    """Test _notify_slack_webhook method."""

    def test_no_webhook_url_returns_early(self, mock_debate_result: Mock) -> None:
        """_notify_slack_webhook returns early if no webhook URL."""
        hooks = DebateHooks()
        # Should not raise
        hooks._notify_slack_webhook(mock_debate_result, "Test task", ["agent1", "agent2"])

    @patch("httpx.post")
    @patch("threading.Thread")
    def test_sends_webhook_in_background_thread(
        self,
        mock_thread_class: Mock,
        mock_post: Mock,
        mock_debate_result: Mock,
    ) -> None:
        """_notify_slack_webhook sends webhook in background thread."""
        hooks = DebateHooks(slack_webhook_url="https://hooks.slack.com/test")

        hooks._notify_slack_webhook(mock_debate_result, "Test task", ["agent1", "agent2"])

        # Verify thread was created and started
        mock_thread_class.assert_called_once()
        mock_thread_class.return_value.start.assert_called_once()

    @patch("httpx.post")
    @patch("threading.Thread")
    def test_builds_correct_slack_payload(
        self,
        mock_thread_class: Mock,
        mock_post: Mock,
        mock_debate_result: Mock,
    ) -> None:
        """_notify_slack_webhook builds correct Slack Block Kit payload."""
        hooks = DebateHooks(slack_webhook_url="https://hooks.slack.com/test")

        hooks._notify_slack_webhook(mock_debate_result, "Test task question", ["agent1", "agent2"])

        # Get the target function that was passed to Thread
        call_kwargs = mock_thread_class.call_args.kwargs
        assert call_kwargs["daemon"] is True

    def test_handles_import_error_gracefully(self, mock_debate_result: Mock) -> None:
        """_notify_slack_webhook handles import errors gracefully."""
        hooks = DebateHooks(slack_webhook_url="https://hooks.slack.com/test")

        # Mock httpx to raise ImportError
        with patch.dict("sys.modules", {"httpx": None}):
            # Should not raise
            hooks._notify_slack_webhook(mock_debate_result, "Test task", ["agent1", "agent2"])


# =============================================================================
# Test Diagnostics
# =============================================================================


class TestGetStatus:
    """Test get_status method."""

    def test_reports_all_subsystems_none(self) -> None:
        """get_status reports all subsystems as unavailable when None."""
        hooks = DebateHooks()
        status = hooks.get_status()

        assert status["subsystems"]["position_ledger"] is False
        assert status["subsystems"]["elo_system"] is False
        assert status["subsystems"]["memory_manager"] is False
        assert status["subsystems"]["evidence_grounder"] is False
        assert status["subsystems"]["slack_webhook"] is False

    def test_reports_available_subsystems(
        self,
        mock_position_ledger: Mock,
        mock_elo_system: Mock,
        mock_memory_manager: Mock,
        mock_evidence_grounder: Mock,
    ) -> None:
        """get_status reports available subsystems correctly."""
        hooks = DebateHooks(
            position_ledger=mock_position_ledger,
            elo_system=mock_elo_system,
            memory_manager=mock_memory_manager,
            evidence_grounder=mock_evidence_grounder,
            slack_webhook_url="https://hooks.slack.com/test",
        )
        status = hooks.get_status()

        assert status["subsystems"]["position_ledger"] is True
        assert status["subsystems"]["elo_system"] is True
        assert status["subsystems"]["memory_manager"] is True
        assert status["subsystems"]["evidence_grounder"] is True
        assert status["subsystems"]["slack_webhook"] is True

    def test_reports_tracking_state(self) -> None:
        """get_status reports tracking state."""
        hooks = DebateHooks()
        hooks._continuum_retrieved_ids = ["id1", "id2", "id3"]

        status = hooks.get_status()

        assert status["tracking"]["retrieved_memory_count"] == 3


# =============================================================================
# Test HooksConfig
# =============================================================================


class TestHooksConfig:
    """Test HooksConfig dataclass."""

    def test_init_with_defaults(self) -> None:
        """HooksConfig initializes with None defaults."""
        config = HooksConfig()
        assert config.position_ledger is None
        assert config.elo_system is None
        assert config.memory_manager is None
        assert config.evidence_grounder is None
        assert config.slack_webhook_url is None

    def test_init_with_all_fields(
        self,
        mock_position_ledger: Mock,
        mock_elo_system: Mock,
        mock_memory_manager: Mock,
        mock_evidence_grounder: Mock,
    ) -> None:
        """HooksConfig initializes with all fields."""
        config = HooksConfig(
            position_ledger=mock_position_ledger,
            elo_system=mock_elo_system,
            memory_manager=mock_memory_manager,
            evidence_grounder=mock_evidence_grounder,
            slack_webhook_url="https://hooks.slack.com/test",
        )
        assert config.position_ledger is mock_position_ledger
        assert config.elo_system is mock_elo_system
        assert config.memory_manager is mock_memory_manager
        assert config.evidence_grounder is mock_evidence_grounder
        assert config.slack_webhook_url == "https://hooks.slack.com/test"

    def test_create_hooks_with_no_subsystems(self) -> None:
        """create_hooks creates DebateHooks with no subsystems."""
        config = HooksConfig()
        hooks = config.create_hooks()

        assert isinstance(hooks, DebateHooks)
        assert hooks.position_ledger is None
        assert hooks.elo_system is None
        assert hooks.memory_manager is None
        assert hooks.evidence_grounder is None
        assert hooks.slack_webhook_url is None

    def test_create_hooks_with_all_subsystems(
        self,
        mock_position_ledger: Mock,
        mock_elo_system: Mock,
        mock_memory_manager: Mock,
        mock_evidence_grounder: Mock,
    ) -> None:
        """create_hooks creates DebateHooks with all subsystems."""
        config = HooksConfig(
            position_ledger=mock_position_ledger,
            elo_system=mock_elo_system,
            memory_manager=mock_memory_manager,
            evidence_grounder=mock_evidence_grounder,
            slack_webhook_url="https://hooks.slack.com/test",
        )
        hooks = config.create_hooks()

        assert isinstance(hooks, DebateHooks)
        assert hooks.position_ledger is mock_position_ledger
        assert hooks.elo_system is mock_elo_system
        assert hooks.memory_manager is mock_memory_manager
        assert hooks.evidence_grounder is mock_evidence_grounder
        assert hooks.slack_webhook_url == "https://hooks.slack.com/test"


# =============================================================================
# Test Error Isolation
# =============================================================================


class TestErrorIsolation:
    """Test that errors in hooks don't cascade."""

    @pytest.mark.asyncio
    async def test_relationship_error_doesnt_prevent_memory_storage(
        self,
        mock_elo_system: Mock,
        mock_memory_manager: Mock,
        mock_debate_context: Mock,
        mock_debate_result: Mock,
        mock_agents: list[Mock],
    ) -> None:
        """Error in relationship update doesn't prevent memory storage."""
        mock_elo_system.update_relationships_batch.side_effect = RuntimeError("test")
        hooks = DebateHooks(
            elo_system=mock_elo_system,
            memory_manager=mock_memory_manager,
        )

        await hooks.on_debate_complete(
            ctx=mock_debate_context,
            result=mock_debate_result,
            agents=mock_agents,
            task="Test task",
        )

        # Memory storage should still be called
        mock_memory_manager.store_debate_outcome.assert_called_once()

    @pytest.mark.asyncio
    async def test_memory_error_doesnt_prevent_calibration(
        self,
        mock_memory_manager: Mock,
        mock_calibration_tracker: Mock,
        mock_debate_context: Mock,
        mock_debate_result: Mock,
        mock_agents: list[Mock],
    ) -> None:
        """Error in memory storage doesn't prevent calibration update."""
        mock_memory_manager.store_debate_outcome.side_effect = RuntimeError("test")
        hooks = DebateHooks(
            memory_manager=mock_memory_manager,
            calibration_tracker=mock_calibration_tracker,
        )

        await hooks.on_debate_complete(
            ctx=mock_debate_context,
            result=mock_debate_result,
            agents=mock_agents,
            task="Test task",
        )

        # Calibration should still be called
        mock_calibration_tracker.record_prediction.assert_called()

    def test_logging_on_errors(
        self, mock_position_ledger: Mock, mock_debate_context: Mock, caplog
    ) -> None:
        """Errors are logged with appropriate warning level."""
        mock_position_ledger.record_position.side_effect = ValueError("test error")
        hooks = DebateHooks(position_ledger=mock_position_ledger)

        with caplog.at_level(logging.WARNING):
            hooks.on_round_complete(
                mock_debate_context,
                round_num=1,
                proposals={"agent1": "Proposal"},
            )

        assert "Position ledger error" in caplog.text


# =============================================================================
# Test Integration Scenarios
# =============================================================================


class TestIntegrationScenarios:
    """Test complete integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_debate_lifecycle(
        self,
        mock_position_ledger: Mock,
        mock_elo_system: Mock,
        mock_memory_manager: Mock,
        mock_evidence_grounder: Mock,
        mock_calibration_tracker: Mock,
        mock_debate_context: Mock,
        mock_debate_result: Mock,
        mock_agents: list[Mock],
    ) -> None:
        """Test complete debate lifecycle with all hooks."""
        hooks = DebateHooks(
            position_ledger=mock_position_ledger,
            elo_system=mock_elo_system,
            memory_manager=mock_memory_manager,
            evidence_grounder=mock_evidence_grounder,
            calibration_tracker=mock_calibration_tracker,
        )

        # Simulate round 1
        hooks.on_round_complete(
            mock_debate_context,
            round_num=1,
            proposals={"agent1": "Proposal 1", "agent2": "Proposal 2"},
        )

        # Simulate round 2
        hooks.on_round_complete(
            mock_debate_context,
            round_num=2,
            proposals={"agent1": "Revised 1", "agent2": "Revised 2"},
        )

        # Track retrieved memories
        hooks.track_retrieved_memories(["mem1", "mem2"], {"mem1": "fast"})

        # Complete debate
        await hooks.on_debate_complete(
            ctx=mock_debate_context,
            result=mock_debate_result,
            agents=mock_agents,
            task="Test task",
        )

        # Verify all hooks were called
        assert mock_position_ledger.record_position.call_count == 4  # 2 agents x 2 rounds
        mock_elo_system.update_relationships_batch.assert_called_once()
        mock_memory_manager.store_debate_outcome.assert_called_once()
        mock_memory_manager.track_retrieved_ids.assert_called_once()
        mock_calibration_tracker.record_prediction.assert_called()

    @pytest.mark.asyncio
    async def test_minimal_hooks_configuration(
        self,
        mock_debate_context: Mock,
        mock_debate_result: Mock,
        mock_agents: list[Mock],
    ) -> None:
        """Test debate lifecycle with minimal hooks configuration."""
        # Only position ledger
        position_ledger = Mock()
        position_ledger.record_position = Mock()

        hooks = DebateHooks(position_ledger=position_ledger)

        # Round complete
        hooks.on_round_complete(
            mock_debate_context,
            round_num=1,
            proposals={"agent1": "Proposal"},
        )

        # Debate complete - should not raise despite missing subsystems
        await hooks.on_debate_complete(
            ctx=mock_debate_context,
            result=mock_debate_result,
            agents=mock_agents,
            task="Test task",
        )

        position_ledger.record_position.assert_called_once()

    def test_hooks_status_reflects_configuration(
        self,
        mock_position_ledger: Mock,
        mock_memory_manager: Mock,
    ) -> None:
        """get_status accurately reflects hook configuration."""
        hooks = DebateHooks(
            position_ledger=mock_position_ledger,
            memory_manager=mock_memory_manager,
        )
        hooks._continuum_retrieved_ids = ["id1", "id2"]

        status = hooks.get_status()

        assert status["subsystems"]["position_ledger"] is True
        assert status["subsystems"]["memory_manager"] is True
        assert status["subsystems"]["elo_system"] is False
        assert status["tracking"]["retrieved_memory_count"] == 2
