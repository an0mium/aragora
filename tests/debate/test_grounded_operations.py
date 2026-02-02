"""Tests for GroundedOperations module.

Tests cover:
1. Grounded reasoning operations
2. Evidence grounding
3. Claim verification
4. Edge cases
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.grounded_operations import GroundedOperations


# ============================================================================
# Fixtures
# ============================================================================


@dataclass
class MockVote:
    """Mock Vote object for testing."""

    agent: str
    choice: str
    reasoning: str = ""
    confidence: float = 1.0


@dataclass
class MockDebateResult:
    """Mock DebateResult for testing."""

    final_answer: str = ""
    confidence: float = 0.8
    grounded_verdict: Any = None


@pytest.fixture
def mock_position_ledger() -> MagicMock:
    """Create a mock position ledger."""
    ledger = MagicMock()
    ledger.record_position = MagicMock()
    return ledger


@pytest.fixture
def mock_elo_system() -> MagicMock:
    """Create a mock ELO system."""
    elo = MagicMock()
    elo.update_relationships_batch = MagicMock()
    return elo


@pytest.fixture
def mock_evidence_grounder() -> MagicMock:
    """Create a mock evidence grounder."""
    grounder = MagicMock()
    grounder.create_grounded_verdict = MagicMock(return_value=MagicMock(grounding_score=0.85))
    grounder.verify_claims_formally = AsyncMock(return_value=(2, 0))
    return grounder


@pytest.fixture
def grounded_ops(
    mock_position_ledger: MagicMock,
    mock_elo_system: MagicMock,
    mock_evidence_grounder: MagicMock,
) -> GroundedOperations:
    """Create a GroundedOperations instance with all dependencies."""
    return GroundedOperations(
        position_ledger=mock_position_ledger,
        elo_system=mock_elo_system,
        evidence_grounder=mock_evidence_grounder,
    )


@pytest.fixture
def minimal_ops() -> GroundedOperations:
    """Create a GroundedOperations instance with no dependencies."""
    return GroundedOperations()


# ============================================================================
# Initialization Tests
# ============================================================================


class TestGroundedOperationsInit:
    """Test GroundedOperations initialization."""

    def test_init_with_all_dependencies(
        self,
        mock_position_ledger: MagicMock,
        mock_elo_system: MagicMock,
        mock_evidence_grounder: MagicMock,
    ) -> None:
        """Test initialization with all dependencies."""
        ops = GroundedOperations(
            position_ledger=mock_position_ledger,
            elo_system=mock_elo_system,
            evidence_grounder=mock_evidence_grounder,
        )
        assert ops.position_ledger is mock_position_ledger
        assert ops.elo_system is mock_elo_system
        assert ops.evidence_grounder is mock_evidence_grounder

    def test_init_with_no_dependencies(self) -> None:
        """Test initialization with no dependencies."""
        ops = GroundedOperations()
        assert ops.position_ledger is None
        assert ops.elo_system is None
        assert ops.evidence_grounder is None

    def test_init_with_partial_dependencies(
        self,
        mock_position_ledger: MagicMock,
    ) -> None:
        """Test initialization with partial dependencies."""
        ops = GroundedOperations(position_ledger=mock_position_ledger)
        assert ops.position_ledger is mock_position_ledger
        assert ops.elo_system is None
        assert ops.evidence_grounder is None


# ============================================================================
# Position Recording Tests
# ============================================================================


class TestRecordPosition:
    """Test position recording functionality."""

    def test_record_position_success(
        self,
        grounded_ops: GroundedOperations,
        mock_position_ledger: MagicMock,
    ) -> None:
        """Test successful position recording."""
        grounded_ops.record_position(
            agent_name="claude",
            content="This is my position on the matter.",
            debate_id="debate-123",
            round_num=1,
            confidence=0.85,
            domain="technology",
        )

        mock_position_ledger.record_position.assert_called_once_with(
            agent_name="claude",
            claim="This is my position on the matter.",
            confidence=0.85,
            debate_id="debate-123",
            round_num=1,
            domain="technology",
        )

    def test_record_position_truncates_long_content(
        self,
        grounded_ops: GroundedOperations,
        mock_position_ledger: MagicMock,
    ) -> None:
        """Test that long content is truncated to 1000 chars."""
        long_content = "x" * 2000
        grounded_ops.record_position(
            agent_name="claude",
            content=long_content,
            debate_id="debate-123",
            round_num=1,
        )

        call_args = mock_position_ledger.record_position.call_args
        assert len(call_args.kwargs["claim"]) == 1000

    def test_record_position_default_confidence(
        self,
        grounded_ops: GroundedOperations,
        mock_position_ledger: MagicMock,
    ) -> None:
        """Test default confidence value of 0.7."""
        grounded_ops.record_position(
            agent_name="claude",
            content="Test position",
            debate_id="debate-123",
            round_num=1,
        )

        call_args = mock_position_ledger.record_position.call_args
        assert call_args.kwargs["confidence"] == 0.7

    def test_record_position_no_ledger(
        self,
        minimal_ops: GroundedOperations,
    ) -> None:
        """Test that no error is raised when position_ledger is None."""
        # Should not raise
        minimal_ops.record_position(
            agent_name="claude",
            content="Test position",
            debate_id="debate-123",
            round_num=1,
        )

    def test_record_position_handles_attribute_error(
        self,
        mock_position_ledger: MagicMock,
    ) -> None:
        """Test handling of AttributeError from ledger."""
        mock_position_ledger.record_position.side_effect = AttributeError("Missing attribute")
        ops = GroundedOperations(position_ledger=mock_position_ledger)

        # Should not raise, just log warning
        ops.record_position(
            agent_name="claude",
            content="Test position",
            debate_id="debate-123",
            round_num=1,
        )

    def test_record_position_handles_type_error(
        self,
        mock_position_ledger: MagicMock,
    ) -> None:
        """Test handling of TypeError from ledger."""
        mock_position_ledger.record_position.side_effect = TypeError("Type mismatch")
        ops = GroundedOperations(position_ledger=mock_position_ledger)

        # Should not raise, just log warning
        ops.record_position(
            agent_name="claude",
            content="Test position",
            debate_id="debate-123",
            round_num=1,
        )

    def test_record_position_handles_value_error(
        self,
        mock_position_ledger: MagicMock,
    ) -> None:
        """Test handling of ValueError from ledger."""
        mock_position_ledger.record_position.side_effect = ValueError("Invalid value")
        ops = GroundedOperations(position_ledger=mock_position_ledger)

        # Should not raise, just log warning
        ops.record_position(
            agent_name="claude",
            content="Test position",
            debate_id="debate-123",
            round_num=1,
        )

    def test_record_position_handles_key_error(
        self,
        mock_position_ledger: MagicMock,
    ) -> None:
        """Test handling of KeyError from ledger."""
        mock_position_ledger.record_position.side_effect = KeyError("Missing key")
        ops = GroundedOperations(position_ledger=mock_position_ledger)

        # Should not raise, just log warning
        ops.record_position(
            agent_name="claude",
            content="Test position",
            debate_id="debate-123",
            round_num=1,
        )

    def test_record_position_handles_runtime_error(
        self,
        mock_position_ledger: MagicMock,
    ) -> None:
        """Test handling of RuntimeError from ledger."""
        mock_position_ledger.record_position.side_effect = RuntimeError("Runtime issue")
        ops = GroundedOperations(position_ledger=mock_position_ledger)

        # Should not raise, just log warning
        ops.record_position(
            agent_name="claude",
            content="Test position",
            debate_id="debate-123",
            round_num=1,
        )

    def test_record_position_handles_os_error(
        self,
        mock_position_ledger: MagicMock,
    ) -> None:
        """Test handling of OSError from ledger."""
        mock_position_ledger.record_position.side_effect = OSError("I/O error")
        ops = GroundedOperations(position_ledger=mock_position_ledger)

        # Should not raise, just log warning
        ops.record_position(
            agent_name="claude",
            content="Test position",
            debate_id="debate-123",
            round_num=1,
        )


# ============================================================================
# Relationship Update Tests
# ============================================================================


class TestUpdateRelationships:
    """Test agent relationship update functionality."""

    def test_update_relationships_success(
        self,
        grounded_ops: GroundedOperations,
        mock_elo_system: MagicMock,
    ) -> None:
        """Test successful relationship update."""
        participants = ["claude", "gpt4", "gemini"]
        votes = [
            MockVote(agent="claude", choice="option_a"),
            MockVote(agent="gpt4", choice="option_a"),
            MockVote(agent="gemini", choice="option_b"),
        ]

        grounded_ops.update_relationships(
            debate_id="debate-123",
            participants=participants,
            winner="claude",
            votes=votes,
        )

        mock_elo_system.update_relationships_batch.assert_called_once()
        call_args = mock_elo_system.update_relationships_batch.call_args
        updates = call_args[0][0]

        # Should have 3 pairs: (claude, gpt4), (claude, gemini), (gpt4, gemini)
        assert len(updates) == 3

    def test_update_relationships_computes_correct_pairs(
        self,
        grounded_ops: GroundedOperations,
        mock_elo_system: MagicMock,
    ) -> None:
        """Test that all participant pairs are computed correctly."""
        participants = ["agent_a", "agent_b", "agent_c", "agent_d"]
        votes = []

        grounded_ops.update_relationships(
            debate_id="debate-123",
            participants=participants,
            winner=None,
            votes=votes,
        )

        call_args = mock_elo_system.update_relationships_batch.call_args
        updates = call_args[0][0]

        # 4 participants = 6 pairs (n*(n-1)/2)
        assert len(updates) == 6

        # Verify all pairs are unique and correct
        pairs = [(u["agent_a"], u["agent_b"]) for u in updates]
        expected_pairs = [
            ("agent_a", "agent_b"),
            ("agent_a", "agent_c"),
            ("agent_a", "agent_d"),
            ("agent_b", "agent_c"),
            ("agent_b", "agent_d"),
            ("agent_c", "agent_d"),
        ]
        assert sorted(pairs) == sorted(expected_pairs)

    def test_update_relationships_tracks_agreement(
        self,
        grounded_ops: GroundedOperations,
        mock_elo_system: MagicMock,
    ) -> None:
        """Test that agreement is tracked correctly."""
        participants = ["claude", "gpt4"]
        votes = [
            MockVote(agent="claude", choice="same_choice"),
            MockVote(agent="gpt4", choice="same_choice"),
        ]

        grounded_ops.update_relationships(
            debate_id="debate-123",
            participants=participants,
            winner=None,
            votes=votes,
        )

        call_args = mock_elo_system.update_relationships_batch.call_args
        updates = call_args[0][0]

        # Both voted same, so agreement_increment should be 1
        assert updates[0]["agreement_increment"] == 1

    def test_update_relationships_tracks_disagreement(
        self,
        grounded_ops: GroundedOperations,
        mock_elo_system: MagicMock,
    ) -> None:
        """Test that disagreement is tracked correctly."""
        participants = ["claude", "gpt4"]
        votes = [
            MockVote(agent="claude", choice="option_a"),
            MockVote(agent="gpt4", choice="option_b"),
        ]

        grounded_ops.update_relationships(
            debate_id="debate-123",
            participants=participants,
            winner=None,
            votes=votes,
        )

        call_args = mock_elo_system.update_relationships_batch.call_args
        updates = call_args[0][0]

        # Different votes, so agreement_increment should be 0
        assert updates[0]["agreement_increment"] == 0

    def test_update_relationships_tracks_wins(
        self,
        grounded_ops: GroundedOperations,
        mock_elo_system: MagicMock,
    ) -> None:
        """Test that wins are tracked correctly."""
        participants = ["claude", "gpt4"]
        votes = []

        grounded_ops.update_relationships(
            debate_id="debate-123",
            participants=participants,
            winner="claude",
            votes=votes,
        )

        call_args = mock_elo_system.update_relationships_batch.call_args
        updates = call_args[0][0]

        assert updates[0]["a_win"] == 1
        assert updates[0]["b_win"] == 0

    def test_update_relationships_tracks_b_wins(
        self,
        grounded_ops: GroundedOperations,
        mock_elo_system: MagicMock,
    ) -> None:
        """Test that wins for agent_b are tracked correctly."""
        participants = ["claude", "gpt4"]
        votes = []

        grounded_ops.update_relationships(
            debate_id="debate-123",
            participants=participants,
            winner="gpt4",
            votes=votes,
        )

        call_args = mock_elo_system.update_relationships_batch.call_args
        updates = call_args[0][0]

        assert updates[0]["a_win"] == 0
        assert updates[0]["b_win"] == 1

    def test_update_relationships_no_winner(
        self,
        grounded_ops: GroundedOperations,
        mock_elo_system: MagicMock,
    ) -> None:
        """Test relationship update when there is no winner."""
        participants = ["claude", "gpt4"]
        votes = []

        grounded_ops.update_relationships(
            debate_id="debate-123",
            participants=participants,
            winner=None,
            votes=votes,
        )

        call_args = mock_elo_system.update_relationships_batch.call_args
        updates = call_args[0][0]

        assert updates[0]["a_win"] == 0
        assert updates[0]["b_win"] == 0

    def test_update_relationships_no_elo_system(
        self,
        minimal_ops: GroundedOperations,
    ) -> None:
        """Test that no error is raised when elo_system is None."""
        # Should not raise
        minimal_ops.update_relationships(
            debate_id="debate-123",
            participants=["claude", "gpt4"],
            winner="claude",
            votes=[],
        )

    def test_update_relationships_single_participant(
        self,
        grounded_ops: GroundedOperations,
        mock_elo_system: MagicMock,
    ) -> None:
        """Test with single participant (no pairs to update)."""
        grounded_ops.update_relationships(
            debate_id="debate-123",
            participants=["claude"],
            winner="claude",
            votes=[],
        )

        call_args = mock_elo_system.update_relationships_batch.call_args
        updates = call_args[0][0]
        assert len(updates) == 0

    def test_update_relationships_empty_participants(
        self,
        grounded_ops: GroundedOperations,
        mock_elo_system: MagicMock,
    ) -> None:
        """Test with empty participants list."""
        grounded_ops.update_relationships(
            debate_id="debate-123",
            participants=[],
            winner=None,
            votes=[],
        )

        call_args = mock_elo_system.update_relationships_batch.call_args
        updates = call_args[0][0]
        assert len(updates) == 0

    def test_update_relationships_handles_attribute_error(
        self,
        mock_elo_system: MagicMock,
    ) -> None:
        """Test handling of AttributeError from ELO system."""
        mock_elo_system.update_relationships_batch.side_effect = AttributeError("Missing attr")
        ops = GroundedOperations(elo_system=mock_elo_system)

        # Should not raise, just log warning
        ops.update_relationships(
            debate_id="debate-123",
            participants=["claude", "gpt4"],
            winner="claude",
            votes=[],
        )

    def test_update_relationships_handles_type_error(
        self,
        mock_elo_system: MagicMock,
    ) -> None:
        """Test handling of TypeError from ELO system."""
        mock_elo_system.update_relationships_batch.side_effect = TypeError("Type issue")
        ops = GroundedOperations(elo_system=mock_elo_system)

        # Should not raise, just log warning
        ops.update_relationships(
            debate_id="debate-123",
            participants=["claude", "gpt4"],
            winner="claude",
            votes=[],
        )

    def test_update_relationships_handles_key_error(
        self,
        mock_elo_system: MagicMock,
    ) -> None:
        """Test handling of KeyError from ELO system."""
        mock_elo_system.update_relationships_batch.side_effect = KeyError("Missing key")
        ops = GroundedOperations(elo_system=mock_elo_system)

        # Should not raise, just log warning
        ops.update_relationships(
            debate_id="debate-123",
            participants=["claude", "gpt4"],
            winner="claude",
            votes=[],
        )

    def test_update_relationships_handles_value_error(
        self,
        mock_elo_system: MagicMock,
    ) -> None:
        """Test handling of ValueError from ELO system."""
        mock_elo_system.update_relationships_batch.side_effect = ValueError("Invalid value")
        ops = GroundedOperations(elo_system=mock_elo_system)

        # Should not raise, just log warning
        ops.update_relationships(
            debate_id="debate-123",
            participants=["claude", "gpt4"],
            winner="claude",
            votes=[],
        )

    def test_update_relationships_handles_runtime_error(
        self,
        mock_elo_system: MagicMock,
    ) -> None:
        """Test handling of RuntimeError from ELO system."""
        mock_elo_system.update_relationships_batch.side_effect = RuntimeError("Runtime issue")
        ops = GroundedOperations(elo_system=mock_elo_system)

        # Should not raise, just log warning
        ops.update_relationships(
            debate_id="debate-123",
            participants=["claude", "gpt4"],
            winner="claude",
            votes=[],
        )

    def test_update_relationships_handles_os_error(
        self,
        mock_elo_system: MagicMock,
    ) -> None:
        """Test handling of OSError from ELO system."""
        mock_elo_system.update_relationships_batch.side_effect = OSError("I/O error")
        ops = GroundedOperations(elo_system=mock_elo_system)

        # Should not raise, just log warning
        ops.update_relationships(
            debate_id="debate-123",
            participants=["claude", "gpt4"],
            winner="claude",
            votes=[],
        )

    def test_update_relationships_handles_votes_without_agent_attr(
        self,
        grounded_ops: GroundedOperations,
        mock_elo_system: MagicMock,
    ) -> None:
        """Test handling of votes missing agent attribute."""
        participants = ["claude", "gpt4"]
        # Create a mock without 'agent' attribute
        bad_vote = MagicMock(spec=[])  # Empty spec means no attributes

        grounded_ops.update_relationships(
            debate_id="debate-123",
            participants=participants,
            winner=None,
            votes=[bad_vote],
        )

        # Should still work, just won't include vote in choices
        mock_elo_system.update_relationships_batch.assert_called_once()

    def test_update_relationships_handles_votes_without_choice_attr(
        self,
        grounded_ops: GroundedOperations,
        mock_elo_system: MagicMock,
    ) -> None:
        """Test handling of votes missing choice attribute."""
        participants = ["claude", "gpt4"]
        # Create a mock with only 'agent' attribute
        bad_vote = MagicMock(spec=["agent"])
        bad_vote.agent = "claude"

        grounded_ops.update_relationships(
            debate_id="debate-123",
            participants=participants,
            winner=None,
            votes=[bad_vote],
        )

        # Should still work, just won't include vote in choices
        mock_elo_system.update_relationships_batch.assert_called_once()


# ============================================================================
# Grounded Verdict Tests
# ============================================================================


class TestCreateGroundedVerdict:
    """Test grounded verdict creation."""

    def test_create_grounded_verdict_success(
        self,
        grounded_ops: GroundedOperations,
        mock_evidence_grounder: MagicMock,
    ) -> None:
        """Test successful verdict creation."""
        result = MockDebateResult(
            final_answer="The answer is 42.",
            confidence=0.95,
        )

        verdict = grounded_ops.create_grounded_verdict(result)

        mock_evidence_grounder.create_grounded_verdict.assert_called_once_with(
            final_answer="The answer is 42.",
            confidence=0.95,
        )
        assert verdict is not None

    def test_create_grounded_verdict_no_final_answer(
        self,
        grounded_ops: GroundedOperations,
        mock_evidence_grounder: MagicMock,
    ) -> None:
        """Test verdict creation with empty final answer."""
        result = MockDebateResult(final_answer="", confidence=0.5)

        verdict = grounded_ops.create_grounded_verdict(result)

        assert verdict is None
        mock_evidence_grounder.create_grounded_verdict.assert_not_called()

    def test_create_grounded_verdict_no_evidence_grounder(
        self,
        minimal_ops: GroundedOperations,
    ) -> None:
        """Test verdict creation without evidence grounder."""
        result = MockDebateResult(final_answer="Some answer", confidence=0.8)

        verdict = minimal_ops.create_grounded_verdict(result)

        assert verdict is None

    def test_create_grounded_verdict_passes_confidence(
        self,
        grounded_ops: GroundedOperations,
        mock_evidence_grounder: MagicMock,
    ) -> None:
        """Test that confidence is passed correctly."""
        result = MockDebateResult(final_answer="Answer", confidence=0.73)

        grounded_ops.create_grounded_verdict(result)

        call_args = mock_evidence_grounder.create_grounded_verdict.call_args
        assert call_args.kwargs["confidence"] == 0.73

    def test_create_grounded_verdict_returns_grounder_result(
        self,
        grounded_ops: GroundedOperations,
        mock_evidence_grounder: MagicMock,
    ) -> None:
        """Test that the grounder result is returned."""
        expected_verdict = MagicMock(grounding_score=0.9)
        mock_evidence_grounder.create_grounded_verdict.return_value = expected_verdict
        result = MockDebateResult(final_answer="Answer", confidence=0.8)

        verdict = grounded_ops.create_grounded_verdict(result)

        assert verdict is expected_verdict


# ============================================================================
# Formal Verification Tests
# ============================================================================


class TestVerifyClaimsFormally:
    """Test formal claim verification."""

    @pytest.mark.asyncio
    async def test_verify_claims_formally_success(
        self,
        grounded_ops: GroundedOperations,
        mock_evidence_grounder: MagicMock,
    ) -> None:
        """Test successful formal verification."""
        verdict = MagicMock()
        result = MockDebateResult(
            final_answer="Test",
            confidence=0.8,
            grounded_verdict=verdict,
        )

        await grounded_ops.verify_claims_formally(result)

        mock_evidence_grounder.verify_claims_formally.assert_called_once_with(verdict)

    @pytest.mark.asyncio
    async def test_verify_claims_formally_no_verdict(
        self,
        grounded_ops: GroundedOperations,
        mock_evidence_grounder: MagicMock,
    ) -> None:
        """Test verification with no grounded verdict."""
        result = MockDebateResult(
            final_answer="Test",
            confidence=0.8,
            grounded_verdict=None,
        )

        await grounded_ops.verify_claims_formally(result)

        mock_evidence_grounder.verify_claims_formally.assert_not_called()

    @pytest.mark.asyncio
    async def test_verify_claims_formally_no_evidence_grounder(
        self,
        minimal_ops: GroundedOperations,
    ) -> None:
        """Test verification without evidence grounder."""
        verdict = MagicMock()
        result = MockDebateResult(
            final_answer="Test",
            confidence=0.8,
            grounded_verdict=verdict,
        )

        # Should not raise
        await minimal_ops.verify_claims_formally(result)


# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_record_position_with_empty_content(
        self,
        grounded_ops: GroundedOperations,
        mock_position_ledger: MagicMock,
    ) -> None:
        """Test recording position with empty content."""
        grounded_ops.record_position(
            agent_name="claude",
            content="",
            debate_id="debate-123",
            round_num=1,
        )

        call_args = mock_position_ledger.record_position.call_args
        assert call_args.kwargs["claim"] == ""

    def test_record_position_with_exactly_1000_chars(
        self,
        grounded_ops: GroundedOperations,
        mock_position_ledger: MagicMock,
    ) -> None:
        """Test recording position with exactly 1000 characters."""
        content = "x" * 1000
        grounded_ops.record_position(
            agent_name="claude",
            content=content,
            debate_id="debate-123",
            round_num=1,
        )

        call_args = mock_position_ledger.record_position.call_args
        assert len(call_args.kwargs["claim"]) == 1000

    def test_record_position_with_special_characters(
        self,
        grounded_ops: GroundedOperations,
        mock_position_ledger: MagicMock,
    ) -> None:
        """Test recording position with special characters."""
        content = "Test with special chars: \n\t\r\0 and unicode: \u2603"
        grounded_ops.record_position(
            agent_name="claude",
            content=content,
            debate_id="debate-123",
            round_num=1,
        )

        call_args = mock_position_ledger.record_position.call_args
        assert call_args.kwargs["claim"] == content

    def test_update_relationships_with_many_participants(
        self,
        grounded_ops: GroundedOperations,
        mock_elo_system: MagicMock,
    ) -> None:
        """Test relationship update with many participants."""
        participants = [f"agent_{i}" for i in range(10)]

        grounded_ops.update_relationships(
            debate_id="debate-123",
            participants=participants,
            winner=None,
            votes=[],
        )

        call_args = mock_elo_system.update_relationships_batch.call_args
        updates = call_args[0][0]

        # 10 participants = 45 pairs (10*9/2)
        assert len(updates) == 45

    def test_update_relationships_debate_increment_always_1(
        self,
        grounded_ops: GroundedOperations,
        mock_elo_system: MagicMock,
    ) -> None:
        """Test that debate_increment is always 1."""
        participants = ["claude", "gpt4", "gemini"]

        grounded_ops.update_relationships(
            debate_id="debate-123",
            participants=participants,
            winner=None,
            votes=[],
        )

        call_args = mock_elo_system.update_relationships_batch.call_args
        updates = call_args[0][0]

        for update in updates:
            assert update["debate_increment"] == 1

    def test_create_grounded_verdict_with_none_result_final_answer(
        self,
        grounded_ops: GroundedOperations,
        mock_evidence_grounder: MagicMock,
    ) -> None:
        """Test verdict creation when result.final_answer is falsy."""
        result = MagicMock()
        result.final_answer = None

        verdict = grounded_ops.create_grounded_verdict(result)

        assert verdict is None


class TestModuleExports:
    """Test module exports and __all__."""

    def test_grounded_operations_in_all(self) -> None:
        """Test that GroundedOperations is in __all__."""
        from aragora.debate import grounded_operations

        assert "GroundedOperations" in grounded_operations.__all__

    def test_can_import_grounded_operations(self) -> None:
        """Test that GroundedOperations can be imported."""
        from aragora.debate.grounded_operations import GroundedOperations

        assert GroundedOperations is not None


class TestLogging:
    """Test logging behavior."""

    def test_record_position_logs_warning_on_error(
        self,
        mock_position_ledger: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that warnings are logged on position ledger errors."""
        mock_position_ledger.record_position.side_effect = ValueError("Test error")
        ops = GroundedOperations(position_ledger=mock_position_ledger)

        import logging

        with caplog.at_level(logging.WARNING):
            ops.record_position(
                agent_name="claude",
                content="Test",
                debate_id="debate-123",
                round_num=1,
            )

        assert "Position ledger error" in caplog.text

    def test_update_relationships_logs_warning_on_error(
        self,
        mock_elo_system: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that warnings are logged on ELO system errors."""
        mock_elo_system.update_relationships_batch.side_effect = ValueError("Test error")
        ops = GroundedOperations(elo_system=mock_elo_system)

        import logging

        with caplog.at_level(logging.WARNING):
            ops.update_relationships(
                debate_id="debate-123",
                participants=["claude", "gpt4"],
                winner="claude",
                votes=[],
            )

        assert "Relationship update error" in caplog.text
