"""
Tests for aragora/debate/phases/consensus_storage.py

Covers:
- ConsensusStorage initialization
- store_consensus_outcome: all early-exit branches
- store_consensus_outcome: confidence → strength mapping
- store_consensus_outcome: vote-based agreeing/dissenting agent separation
- store_consensus_outcome: belief crux extraction from result
- store_consensus_outcome: dissent delegation
- store_consensus_outcome: ImportError and RuntimeError handling
- _confidence_to_strength: all five enum levels
- _store_dissenting_views: reasoning fallback, per-vote storage, error resilience
- store_cruxes: all early-exit branches
- store_cruxes: crux extraction from belief_network, dissenting_views, votes
- store_cruxes: update_cruxes delegation and error resilience
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from aragora.debate.phases.consensus_storage import ConsensusStorage


# ---------------------------------------------------------------------------
# Lightweight mock data structures
# ---------------------------------------------------------------------------


@dataclass
class MockVote:
    agent: str
    choice: str
    reasoning: str = ""
    confidence: float = 0.5


@dataclass
class MockResult:
    final_answer: str = "The best approach is X"
    confidence: float = 0.85
    rounds_used: int = 3
    winner: str = "claude"
    votes: list = field(default_factory=list)
    dissenting_views: list = field(default_factory=list)
    belief_cruxes: list = field(default_factory=list)


def make_ctx(
    *,
    result=None,
    agents=None,
    domain="general",
    debate_id="debate-123",
    choice_mapping=None,
    belief_network=None,
):
    """Build a minimal MagicMock DebateContext."""
    ctx = MagicMock()
    ctx.result = result
    ctx.env.task = "Should we do X or Y?"
    ctx.agents = agents or [MagicMock(name_attr="agent1")]
    for a in ctx.agents:
        if not hasattr(a, "name") or isinstance(a.name, MagicMock):
            a.name = "agent1"
    ctx.domain = domain
    ctx.debate_id = debate_id
    ctx.choice_mapping = choice_mapping or {}
    if belief_network is None:
        # Ensure hasattr(ctx, 'belief_network') returns False by default
        del ctx.belief_network
    else:
        ctx.belief_network = belief_network
    return ctx


def make_consensus_memory():
    """Build a MagicMock ConsensusMemory."""
    mem = MagicMock()
    record = MagicMock()
    record.id = "consensus-abc"
    mem.store_consensus.return_value = record
    return mem


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def consensus_memory():
    return make_consensus_memory()


@pytest.fixture
def storage(consensus_memory):
    return ConsensusStorage(consensus_memory=consensus_memory)


@pytest.fixture
def storage_no_memory():
    return ConsensusStorage()


@pytest.fixture
def simple_result():
    return MockResult()


@pytest.fixture
def simple_ctx(simple_result):
    agents = [MagicMock(), MagicMock()]
    agents[0].name = "claude"
    agents[1].name = "gpt4"
    return make_ctx(result=simple_result, agents=agents)


# ===========================================================================
# Tests: ConsensusStorage.__init__
# ===========================================================================


class TestInit:
    def test_default_no_memory(self):
        cs = ConsensusStorage()
        assert cs.consensus_memory is None

    def test_with_memory(self, consensus_memory):
        cs = ConsensusStorage(consensus_memory=consensus_memory)
        assert cs.consensus_memory is consensus_memory

    def test_keyword_only_argument(self):
        mem = MagicMock()
        with pytest.raises(TypeError):
            ConsensusStorage(mem)  # positional not allowed


# ===========================================================================
# Tests: store_consensus_outcome – early exits
# ===========================================================================


class TestStoreConsensusOutcomeEarlyExits:
    def test_returns_none_when_no_consensus_memory(self, storage_no_memory, simple_ctx):
        assert storage_no_memory.store_consensus_outcome(simple_ctx) is None

    def test_returns_none_when_result_is_none(self, storage, simple_ctx):
        simple_ctx.result = None
        assert storage.store_consensus_outcome(simple_ctx) is None

    def test_returns_none_when_result_is_falsy(self, storage, simple_ctx):
        simple_ctx.result = False
        assert storage.store_consensus_outcome(simple_ctx) is None

    def test_returns_none_when_final_answer_is_none(self, storage, simple_ctx):
        simple_ctx.result.final_answer = None
        assert storage.store_consensus_outcome(simple_ctx) is None

    def test_returns_none_when_final_answer_is_empty_string(self, storage, simple_ctx):
        simple_ctx.result.final_answer = ""
        assert storage.store_consensus_outcome(simple_ctx) is None

    def test_no_memory_does_not_call_store(self, storage_no_memory, simple_ctx):
        storage_no_memory.store_consensus_outcome(simple_ctx)
        # No consensus_memory → nothing should be called
        assert True  # reaching here without AttributeError is enough


# ===========================================================================
# Tests: store_consensus_outcome – happy path
# ===========================================================================


class TestStoreConsensusOutcomeHappyPath:
    def test_returns_consensus_record_id(self, storage, simple_ctx):
        result_id = storage.store_consensus_outcome(simple_ctx)
        assert result_id == "consensus-abc"

    def test_calls_store_consensus_with_topic(self, storage, simple_ctx):
        storage.store_consensus_outcome(simple_ctx)
        call_kwargs = storage.consensus_memory.store_consensus.call_args[1]
        assert call_kwargs["topic"] == "Should we do X or Y?"

    def test_calls_store_consensus_with_conclusion_truncated(self, storage):
        result = MockResult(final_answer="x" * 3000)
        ctx = make_ctx(result=result)
        storage.store_consensus_outcome(ctx)
        call_kwargs = storage.consensus_memory.store_consensus.call_args[1]
        assert len(call_kwargs["conclusion"]) == 2000

    def test_calls_store_consensus_with_confidence(self, storage, simple_ctx):
        storage.store_consensus_outcome(simple_ctx)
        call_kwargs = storage.consensus_memory.store_consensus.call_args[1]
        assert call_kwargs["confidence"] == simple_ctx.result.confidence

    def test_calls_store_consensus_with_domain(self, storage):
        result = MockResult()
        ctx = make_ctx(result=result, domain="healthcare")
        storage.store_consensus_outcome(ctx)
        call_kwargs = storage.consensus_memory.store_consensus.call_args[1]
        assert call_kwargs["domain"] == "healthcare"

    def test_calls_store_consensus_with_rounds(self, storage, simple_ctx):
        storage.store_consensus_outcome(simple_ctx)
        call_kwargs = storage.consensus_memory.store_consensus.call_args[1]
        assert call_kwargs["rounds"] == 3

    def test_participating_agents_uses_agent_names(self, storage, simple_ctx):
        storage.store_consensus_outcome(simple_ctx)
        call_kwargs = storage.consensus_memory.store_consensus.call_args[1]
        assert "claude" in call_kwargs["participating_agents"]
        assert "gpt4" in call_kwargs["participating_agents"]

    def test_no_votes_yields_empty_agreeing_dissenting(self, storage):
        result = MockResult(votes=[])
        ctx = make_ctx(result=result)
        storage.store_consensus_outcome(ctx)
        call_kwargs = storage.consensus_memory.store_consensus.call_args[1]
        assert call_kwargs["agreeing_agents"] == []
        assert call_kwargs["dissenting_agents"] == []

    def test_vote_split_into_agreeing_and_dissenting(self, storage):
        votes = [
            MockVote(agent="claude", choice="A"),
            MockVote(agent="gpt4", choice="B"),
            MockVote(agent="gemini", choice="A"),
        ]
        result = MockResult(winner="A", votes=votes)
        ctx = make_ctx(result=result)
        storage.store_consensus_outcome(ctx)
        call_kwargs = storage.consensus_memory.store_consensus.call_args[1]
        assert "claude" in call_kwargs["agreeing_agents"]
        assert "gemini" in call_kwargs["agreeing_agents"]
        assert "gpt4" in call_kwargs["dissenting_agents"]

    def test_choice_mapping_applied_for_vote_classification(self, storage):
        # canonical name for "agent-A" should be "claude"
        votes = [MockVote(agent="claude", choice="agent-A")]
        result = MockResult(winner="claude", votes=votes)
        ctx = make_ctx(
            result=result,
            choice_mapping={"agent-A": "claude"},
        )
        storage.store_consensus_outcome(ctx)
        call_kwargs = storage.consensus_memory.store_consensus.call_args[1]
        assert "claude" in call_kwargs["agreeing_agents"]
        assert call_kwargs["dissenting_agents"] == []

    def test_no_winner_means_no_vote_classification(self, storage):
        votes = [MockVote(agent="claude", choice="A")]
        result = MockResult(winner=None, votes=votes)
        ctx = make_ctx(result=result)
        storage.store_consensus_outcome(ctx)
        call_kwargs = storage.consensus_memory.store_consensus.call_args[1]
        assert call_kwargs["agreeing_agents"] == []
        assert call_kwargs["dissenting_agents"] == []

    def test_belief_cruxes_included_in_key_claims(self, storage):
        result = MockResult(belief_cruxes=["crux1", "crux2"])
        ctx = make_ctx(result=result)
        storage.store_consensus_outcome(ctx)
        call_kwargs = storage.consensus_memory.store_consensus.call_args[1]
        assert "crux1" in call_kwargs["key_claims"]
        assert "crux2" in call_kwargs["key_claims"]

    def test_belief_cruxes_limited_to_10(self, storage):
        result = MockResult(belief_cruxes=list(range(20)))
        ctx = make_ctx(result=result)
        storage.store_consensus_outcome(ctx)
        call_kwargs = storage.consensus_memory.store_consensus.call_args[1]
        assert len(call_kwargs["key_claims"]) == 10

    def test_none_belief_cruxes_treated_as_empty(self, storage):
        result = MockResult(belief_cruxes=None)
        ctx = make_ctx(result=result)
        storage.store_consensus_outcome(ctx)
        call_kwargs = storage.consensus_memory.store_consensus.call_args[1]
        assert call_kwargs["key_claims"] == []

    def test_metadata_set_when_key_claims_present(self, storage):
        result = MockResult(belief_cruxes=["crux1"])
        ctx = make_ctx(result=result)
        storage.store_consensus_outcome(ctx)
        call_kwargs = storage.consensus_memory.store_consensus.call_args[1]
        assert call_kwargs["metadata"] is not None
        assert "belief_cruxes" in call_kwargs["metadata"]

    def test_metadata_none_when_no_key_claims(self, storage):
        result = MockResult(belief_cruxes=[])
        ctx = make_ctx(result=result)
        storage.store_consensus_outcome(ctx)
        call_kwargs = storage.consensus_memory.store_consensus.call_args[1]
        assert call_kwargs["metadata"] is None

    def test_dissent_storage_called_when_dissenting_agents_present(self, storage):
        votes = [
            MockVote(agent="claude", choice="A"),
            MockVote(agent="gpt4", choice="B"),
        ]
        result = MockResult(winner="A", votes=votes)
        ctx = make_ctx(result=result)
        with patch.object(storage, "_store_dissenting_views") as mock_dissent:
            storage.store_consensus_outcome(ctx)
        mock_dissent.assert_called_once_with(ctx, "consensus-abc", ["gpt4"])

    def test_dissent_storage_not_called_when_all_agree(self, storage):
        votes = [MockVote(agent="claude", choice="A")]
        result = MockResult(winner="A", votes=votes)
        ctx = make_ctx(result=result)
        with patch.object(storage, "_store_dissenting_views") as mock_dissent:
            storage.store_consensus_outcome(ctx)
        mock_dissent.assert_not_called()

    def test_strength_value_passed_to_store(self, storage, simple_ctx):
        simple_ctx.result.confidence = 0.95
        storage.store_consensus_outcome(simple_ctx)
        call_kwargs = storage.consensus_memory.store_consensus.call_args[1]
        assert call_kwargs["strength"] == "unanimous"


# ===========================================================================
# Tests: store_consensus_outcome – error handling
# ===========================================================================


class TestStoreConsensusOutcomeErrorHandling:
    def test_import_error_returns_none(self, storage, simple_ctx):
        with patch(
            "aragora.debate.phases.consensus_storage.ConsensusStorage._confidence_to_strength",
            side_effect=ImportError("module not found"),
        ):
            result = storage.store_consensus_outcome(simple_ctx)
        assert result is None

    def test_runtime_error_returns_none(self, storage, simple_ctx):
        storage.consensus_memory.store_consensus.side_effect = RuntimeError("db error")
        result = storage.store_consensus_outcome(simple_ctx)
        assert result is None

    def test_attribute_error_returns_none(self, storage, simple_ctx):
        storage.consensus_memory.store_consensus.side_effect = AttributeError("missing attr")
        result = storage.store_consensus_outcome(simple_ctx)
        assert result is None

    def test_value_error_returns_none(self, storage, simple_ctx):
        storage.consensus_memory.store_consensus.side_effect = ValueError("bad value")
        result = storage.store_consensus_outcome(simple_ctx)
        assert result is None

    def test_os_error_returns_none(self, storage, simple_ctx):
        storage.consensus_memory.store_consensus.side_effect = OSError("disk full")
        result = storage.store_consensus_outcome(simple_ctx)
        assert result is None

    def test_key_error_returns_none(self, storage, simple_ctx):
        storage.consensus_memory.store_consensus.side_effect = KeyError("missing key")
        result = storage.store_consensus_outcome(simple_ctx)
        assert result is None


# ===========================================================================
# Tests: _confidence_to_strength
# ===========================================================================


class TestConfidenceToStrength:
    """Each boundary is tested independently, relying on the real module import."""

    @pytest.fixture(autouse=True)
    def cs(self):
        self.cs = ConsensusStorage()

    def test_unanimous_at_exactly_0_9(self):
        from aragora.memory.consensus import ConsensusStrength

        assert self.cs._confidence_to_strength(0.9) is ConsensusStrength.UNANIMOUS

    def test_unanimous_above_0_9(self):
        from aragora.memory.consensus import ConsensusStrength

        assert self.cs._confidence_to_strength(1.0) is ConsensusStrength.UNANIMOUS

    def test_strong_at_exactly_0_8(self):
        from aragora.memory.consensus import ConsensusStrength

        assert self.cs._confidence_to_strength(0.8) is ConsensusStrength.STRONG

    def test_strong_below_0_9(self):
        from aragora.memory.consensus import ConsensusStrength

        assert self.cs._confidence_to_strength(0.85) is ConsensusStrength.STRONG

    def test_moderate_at_exactly_0_6(self):
        from aragora.memory.consensus import ConsensusStrength

        assert self.cs._confidence_to_strength(0.6) is ConsensusStrength.MODERATE

    def test_moderate_below_0_8(self):
        from aragora.memory.consensus import ConsensusStrength

        assert self.cs._confidence_to_strength(0.7) is ConsensusStrength.MODERATE

    def test_weak_at_exactly_0_5(self):
        from aragora.memory.consensus import ConsensusStrength

        assert self.cs._confidence_to_strength(0.5) is ConsensusStrength.WEAK

    def test_weak_below_0_6(self):
        from aragora.memory.consensus import ConsensusStrength

        assert self.cs._confidence_to_strength(0.55) is ConsensusStrength.WEAK

    def test_split_below_0_5(self):
        from aragora.memory.consensus import ConsensusStrength

        assert self.cs._confidence_to_strength(0.49) is ConsensusStrength.SPLIT

    def test_split_at_zero(self):
        from aragora.memory.consensus import ConsensusStrength

        assert self.cs._confidence_to_strength(0.0) is ConsensusStrength.SPLIT


# ===========================================================================
# Tests: _store_dissenting_views
# ===========================================================================


class TestStoreDissentiingViews:
    def test_calls_store_dissent_for_each_dissenting_vote(self, storage):
        votes = [
            MockVote(agent="gpt4", choice="B", reasoning="B is better"),
            MockVote(agent="gemini", choice="C", reasoning="C is best"),
        ]
        result = MockResult(winner="A", votes=votes)
        ctx = make_ctx(result=result)
        storage._store_dissenting_views(ctx, "cons-id-1", ["gpt4", "gemini"])
        assert storage.consensus_memory.store_dissent.call_count == 2

    def test_skips_votes_not_in_dissenting_list(self, storage):
        votes = [
            MockVote(agent="claude", choice="A", reasoning="A is best"),
            MockVote(agent="gpt4", choice="B", reasoning="B is better"),
        ]
        result = MockResult(winner="A", votes=votes)
        ctx = make_ctx(result=result)
        storage._store_dissenting_views(ctx, "cons-id-1", ["gpt4"])
        assert storage.consensus_memory.store_dissent.call_count == 1
        call_kwargs = storage.consensus_memory.store_dissent.call_args[1]
        assert call_kwargs["agent_id"] == "gpt4"

    def test_reasoning_fallback_when_empty(self, storage):
        votes = [MockVote(agent="gpt4", choice="B", reasoning="")]
        result = MockResult(winner="A", votes=votes)
        ctx = make_ctx(result=result)
        storage._store_dissenting_views(ctx, "cons-id-1", ["gpt4"])
        call_kwargs = storage.consensus_memory.store_dissent.call_args[1]
        assert "B" in call_kwargs["content"]
        assert "A" in call_kwargs["content"]

    def test_reasoning_truncated_to_500_chars(self, storage):
        long_reasoning = "x" * 1000
        votes = [MockVote(agent="gpt4", choice="B", reasoning=long_reasoning)]
        result = MockResult(winner="A", votes=votes)
        ctx = make_ctx(result=result)
        storage._store_dissenting_views(ctx, "cons-id-1", ["gpt4"])
        call_kwargs = storage.consensus_memory.store_dissent.call_args[1]
        assert len(call_kwargs["content"]) == 500

    def test_uses_vote_confidence_attribute(self, storage):
        votes = [MockVote(agent="gpt4", choice="B", reasoning="B", confidence=0.77)]
        result = MockResult(winner="A", votes=votes)
        ctx = make_ctx(result=result)
        storage._store_dissenting_views(ctx, "cons-id-1", ["gpt4"])
        call_kwargs = storage.consensus_memory.store_dissent.call_args[1]
        assert call_kwargs["confidence"] == pytest.approx(0.77)

    def test_store_dissent_error_does_not_propagate(self, storage):
        storage.consensus_memory.store_dissent.side_effect = ValueError("bad")
        votes = [MockVote(agent="gpt4", choice="B", reasoning="B is better")]
        result = MockResult(winner="A", votes=votes)
        ctx = make_ctx(result=result)
        # Should not raise
        storage._store_dissenting_views(ctx, "cons-id-1", ["gpt4"])

    def test_consensus_id_passed_to_store_dissent(self, storage):
        votes = [MockVote(agent="gpt4", choice="B", reasoning="B")]
        result = MockResult(winner="A", votes=votes)
        ctx = make_ctx(result=result)
        storage._store_dissenting_views(ctx, "my-consensus-id", ["gpt4"])
        call_kwargs = storage.consensus_memory.store_dissent.call_args[1]
        assert call_kwargs["debate_id"] == "my-consensus-id"


# ===========================================================================
# Tests: store_cruxes – early exits
# ===========================================================================


class TestStoreCruxesEarlyExits:
    def test_returns_early_when_no_consensus_memory(self, storage_no_memory, simple_ctx):
        # Should not raise and not call anything
        storage_no_memory.store_cruxes(simple_ctx, consensus_id="cid")

    def test_returns_early_when_no_consensus_id_and_no_ctx_attribute(self, storage, simple_ctx):
        del simple_ctx._last_consensus_id  # ensure attribute absent
        storage.store_cruxes(simple_ctx, consensus_id=None)
        storage.consensus_memory.update_cruxes.assert_not_called()

    def test_uses_ctx_last_consensus_id_as_fallback(self, storage):
        result = MockResult(dissenting_views=[], votes=[MockVote("a", "X"), MockVote("b", "Y")])
        ctx = make_ctx(result=result)
        ctx._last_consensus_id = "ctx-cid"
        # Give votes reasoning so a crux is created
        result.votes[0].reasoning = "X is best"
        result.votes[1].reasoning = "Y is best"
        storage.store_cruxes(ctx, consensus_id=None)
        storage.consensus_memory.update_cruxes.assert_called_once()
        args = storage.consensus_memory.update_cruxes.call_args[0]
        assert args[0] == "ctx-cid"

    def test_returns_early_when_result_is_none(self, storage, simple_ctx):
        simple_ctx.result = None
        storage.store_cruxes(simple_ctx, consensus_id="cid")
        storage.consensus_memory.update_cruxes.assert_not_called()

    def test_no_cruxes_means_update_cruxes_not_called(self, storage):
        result = MockResult(dissenting_views=[], votes=[], belief_cruxes=[])
        ctx = make_ctx(result=result)
        storage.store_cruxes(ctx, consensus_id="cid")
        storage.consensus_memory.update_cruxes.assert_not_called()


# ===========================================================================
# Tests: store_cruxes – crux extraction sources
# ===========================================================================


class TestStoreCruxesExtraction:
    def test_cruxes_from_belief_network(self, storage):
        belief_network = MagicMock()
        belief_network.get_cruxes.return_value = [
            {"claim": "claim1", "positions": {"a": "yes"}, "confidence_gap": 0.3},
        ]
        result = MockResult(dissenting_views=[], votes=[])
        ctx = make_ctx(result=result, belief_network=belief_network)
        storage.store_cruxes(ctx, consensus_id="cid")
        storage.consensus_memory.update_cruxes.assert_called_once()
        cruxes = storage.consensus_memory.update_cruxes.call_args[0][1]
        assert any(c["source"] == "belief_network" for c in cruxes)

    def test_belief_network_crux_fields(self, storage):
        belief_network = MagicMock()
        belief_network.get_cruxes.return_value = [
            {"claim": "key claim", "positions": {"a": "yes"}, "confidence_gap": 0.4},
        ]
        result = MockResult(dissenting_views=[], votes=[])
        ctx = make_ctx(result=result, belief_network=belief_network)
        storage.store_cruxes(ctx, consensus_id="cid")
        cruxes = storage.consensus_memory.update_cruxes.call_args[0][1]
        bn_crux = next(c for c in cruxes if c["source"] == "belief_network")
        assert bn_crux["claim"] == "key claim"
        assert bn_crux["confidence_gap"] == pytest.approx(0.4)

    def test_belief_network_error_does_not_propagate(self, storage):
        belief_network = MagicMock()
        belief_network.get_cruxes.side_effect = AttributeError("no method")
        result = MockResult(dissenting_views=[], votes=[MockVote("a", "X"), MockVote("b", "Y")])
        result.votes[0].reasoning = "X reason"
        result.votes[1].reasoning = "Y reason"
        ctx = make_ctx(result=result, belief_network=belief_network)
        # Should not raise; vote cruxes should still be collected
        storage.store_cruxes(ctx, consensus_id="cid")
        storage.consensus_memory.update_cruxes.assert_called_once()

    def test_cruxes_from_high_confidence_dissenting_views(self, storage):
        view = MagicMock()
        view.confidence = 0.8
        view.content = "Alternative approach content"
        view.agent = "gpt4"
        result = MockResult(dissenting_views=[view], votes=[])
        ctx = make_ctx(result=result)
        storage.store_cruxes(ctx, consensus_id="cid")
        cruxes = storage.consensus_memory.update_cruxes.call_args[0][1]
        assert any(c["source"] == "dissent" for c in cruxes)

    def test_low_confidence_dissenting_views_excluded(self, storage):
        view = MagicMock()
        view.confidence = 0.6  # threshold is > 0.7
        view.content = "Low confidence dissent"
        view.agent = "gpt4"
        result = MockResult(dissenting_views=[view], votes=[])
        ctx = make_ctx(result=result)
        storage.store_cruxes(ctx, consensus_id="cid")
        storage.consensus_memory.update_cruxes.assert_not_called()

    def test_only_first_two_dissenting_views_used(self, storage):
        views = []
        for i in range(5):
            v = MagicMock()
            v.confidence = 0.9
            v.content = f"view {i}"
            v.agent = f"agent{i}"
            views.append(v)
        result = MockResult(dissenting_views=views, votes=[])
        ctx = make_ctx(result=result)
        storage.store_cruxes(ctx, consensus_id="cid")
        cruxes = storage.consensus_memory.update_cruxes.call_args[0][1]
        dissent_cruxes = [c for c in cruxes if c["source"] == "dissent"]
        assert len(dissent_cruxes) == 2

    def test_cruxes_from_vote_split(self, storage):
        votes = [
            MockVote(agent="claude", choice="A", reasoning="A is best because of X"),
            MockVote(agent="gpt4", choice="B", reasoning="B is better because of Y"),
        ]
        result = MockResult(dissenting_views=[], votes=votes)
        ctx = make_ctx(result=result)
        storage.store_cruxes(ctx, consensus_id="cid")
        cruxes = storage.consensus_memory.update_cruxes.call_args[0][1]
        assert any(c["source"] == "vote_split" for c in cruxes)

    def test_vote_split_crux_contains_positions(self, storage):
        votes = [
            MockVote(agent="claude", choice="A", reasoning="A reason"),
            MockVote(agent="gpt4", choice="B", reasoning="B reason"),
        ]
        result = MockResult(dissenting_views=[], votes=votes)
        ctx = make_ctx(result=result)
        storage.store_cruxes(ctx, consensus_id="cid")
        cruxes = storage.consensus_memory.update_cruxes.call_args[0][1]
        vote_crux = next(c for c in cruxes if c["source"] == "vote_split")
        assert "A" in vote_crux["positions"]
        assert "B" in vote_crux["positions"]

    def test_single_vote_unanimous_no_vote_split_crux(self, storage):
        votes = [MockVote(agent="claude", choice="A", reasoning="A reason")]
        result = MockResult(dissenting_views=[], votes=votes)
        ctx = make_ctx(result=result)
        storage.store_cruxes(ctx, consensus_id="cid")
        # single vote → only one choice → no split crux
        storage.consensus_memory.update_cruxes.assert_not_called()

    def test_votes_without_reasoning_produce_empty_positions_in_split(self, storage):
        # When votes have no reasoning the choice keys still exist in vote_choices
        # (because len(vote_choices) > 1) but each has an empty reasoning list.
        # The code still appends a vote_split crux; positions values are empty lists.
        votes = [
            MockVote(agent="claude", choice="A", reasoning=""),
            MockVote(agent="gpt4", choice="B", reasoning=""),
        ]
        result = MockResult(dissenting_views=[], votes=votes)
        ctx = make_ctx(result=result)
        storage.store_cruxes(ctx, consensus_id="cid")
        cruxes = storage.consensus_memory.update_cruxes.call_args[0][1]
        vote_crux = next(c for c in cruxes if c["source"] == "vote_split")
        # positions dict is created but filtered by `if votes` → empty, so dict is empty
        assert vote_crux["positions"] == {}

    def test_update_cruxes_called_with_consensus_id(self, storage):
        votes = [
            MockVote(agent="claude", choice="A", reasoning="A reason"),
            MockVote(agent="gpt4", choice="B", reasoning="B reason"),
        ]
        result = MockResult(dissenting_views=[], votes=votes)
        ctx = make_ctx(result=result)
        storage.store_cruxes(ctx, consensus_id="my-consensus-id")
        args = storage.consensus_memory.update_cruxes.call_args[0]
        assert args[0] == "my-consensus-id"

    def test_update_cruxes_error_does_not_propagate(self, storage):
        votes = [
            MockVote(agent="claude", choice="A", reasoning="A reason"),
            MockVote(agent="gpt4", choice="B", reasoning="B reason"),
        ]
        result = MockResult(dissenting_views=[], votes=votes)
        ctx = make_ctx(result=result)
        storage.consensus_memory.update_cruxes.side_effect = RuntimeError("store error")
        # Should not raise
        storage.store_cruxes(ctx, consensus_id="cid")

    def test_reasoning_in_vote_split_truncated_to_150(self, storage):
        long_reasoning = "z" * 200
        votes = [
            MockVote(agent="claude", choice="A", reasoning=long_reasoning),
            MockVote(agent="gpt4", choice="B", reasoning=long_reasoning),
        ]
        result = MockResult(dissenting_views=[], votes=votes)
        ctx = make_ctx(result=result)
        storage.store_cruxes(ctx, consensus_id="cid")
        cruxes = storage.consensus_memory.update_cruxes.call_args[0][1]
        vote_crux = next(c for c in cruxes if c["source"] == "vote_split")
        for choice, reasons in vote_crux["positions"].items():
            for r in reasons:
                assert len(r) <= 150

    def test_multiple_sources_combined(self, storage):
        belief_network = MagicMock()
        belief_network.get_cruxes.return_value = [
            {"claim": "claim1", "positions": {}, "confidence_gap": 0.2},
        ]
        view = MagicMock()
        view.confidence = 0.9
        view.content = "dissent content"
        view.agent = "gpt4"
        votes = [
            MockVote(agent="claude", choice="A", reasoning="A reason"),
            MockVote(agent="gpt4", choice="B", reasoning="B reason"),
        ]
        result = MockResult(dissenting_views=[view], votes=votes)
        ctx = make_ctx(result=result, belief_network=belief_network)
        storage.store_cruxes(ctx, consensus_id="cid")
        cruxes = storage.consensus_memory.update_cruxes.call_args[0][1]
        sources = {c["source"] for c in cruxes}
        assert "belief_network" in sources
        assert "dissent" in sources
        assert "vote_split" in sources
